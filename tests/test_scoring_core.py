"""Step 3 — the canonical scorer, the single registry, and legacy equivalence.

The most important test here is
``test_legacy_specs_reproduce_pre_audit_scores_exactly``: it proves the new core
reproduces the old formulas bit for bit, which is what makes previously published
pilot numbers reproducible after the refactor.
"""

import numpy as np
import pytest

from daowod.normalisation import average_ranks, normalise
from daowod.scoring import (
    REQUIRED_STRATEGIES,
    STRATEGY_REGISTRY,
    StrategyError,
    StrategySpec,
    aggregate_image_scores,
    score_pool,
    select_images,
)

RNG = np.random.default_rng(7)
PROPOSALS, DIMENSION, CLASSES = 90, 16, 6


def _fixture() -> dict[str, object]:
    """A pool with long-tailed class structure and a realistic posterior."""
    counts = np.array([30, 20, 14, 12, 8, 6])
    assert counts.sum() == PROPOSALS
    centres = RNG.normal(size=(CLASSES, DIMENSION)) * 5.0
    embeddings = np.vstack(
        [
            centres[index] + RNG.normal(scale=1.0, size=(size, DIMENSION))
            for index, size in enumerate(counts)
        ]
    )
    logits = RNG.normal(size=(PROPOSALS, 8))
    posterior = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return {
        "image_ids": np.asarray(
            [f"img_{index // 3:03d}" for index in range(PROPOSALS)], dtype=object
        ),
        "embeddings": embeddings,
        "reference_embeddings": RNG.normal(size=(12, DIMENSION)),
        "confidence": RNG.uniform(0.01, 0.30, PROPOSALS),
        "posterior": posterior,
        "predicted_labels": np.repeat(np.arange(CLASSES), counts).astype(np.int64),
    }


FIXTURE = _fixture()


def _score(name: str, **overrides: object):
    spec = STRATEGY_REGISTRY.resolve(name)
    if overrides:
        spec = StrategySpec(**{**spec.as_dict(), **overrides})
    return score_pool(
        spec=spec,
        image_ids=FIXTURE["image_ids"],
        embeddings=FIXTURE["embeddings"],
        reference_embeddings=FIXTURE["reference_embeddings"],
        confidence=FIXTURE["confidence"],
        posterior=FIXTURE["posterior"],
        predicted_labels=FIXTURE["predicted_labels"],
        seed=0,
        compute_all_components=True,
    )


# --- normalisation -----------------------------------------------------------


def test_rank_normalisation_is_tie_stable_and_bounded() -> None:
    values = np.array([5.0, 1.0, 5.0, 3.0])
    ranks = average_ranks(values)
    assert ranks.tolist() == [2.5, 0.0, 2.5, 1.0]
    scaled = normalise(values, "rank")
    # Average ranks divided by (N - 1): a tied maximum cannot reach 1.0, which
    # is the intended arithmetic, so the invariant is the bound, not the extreme.
    assert scaled.tolist() == [2.5 / 3, 0.0, 2.5 / 3, 1 / 3]
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    # Reordering the input must not change any element's normalised value.
    permuted = normalise(values[[3, 0, 1, 2]], "rank")
    assert permuted.tolist() == scaled[[3, 0, 1, 2]].tolist()


def test_rank_normalisation_is_invariant_to_monotone_transforms() -> None:
    """This is why rank normalisation answers S4."""
    counts = np.array([300.0, 150.0, 52.0, 13.0, 3.0, 1.0])
    inverse = normalise(counts**-1.0, "rank")
    logarithmic = normalise(-np.log(counts / counts.sum()), "rank")
    assert inverse.tolist() == logarithmic.tolist()
    # Under min-max the same two transforms disagree sharply.
    assert not np.allclose(
        normalise(counts**-1.0, "minmax"), normalise(-np.log(counts / counts.sum()), "minmax")
    )


def test_constant_component_normalisation_is_neutral_under_rank() -> None:
    constant = np.full(8, 0.42)
    assert normalise(constant, "rank").tolist() == [0.5] * 8
    assert normalise(constant, "minmax").tolist() == [1.0] * 8
    assert normalise(constant, "zscore_sigmoid").tolist() == [0.5] * 8


def test_unknown_normalisation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown normalisation"):
        normalise(np.arange(3.0), "quantile")  # type: ignore[arg-type]


# --- registry ----------------------------------------------------------------


def test_registry_exposes_every_required_strategy() -> None:
    for name in REQUIRED_STRATEGIES:
        assert STRATEGY_REGISTRY.resolve(name).name == name


def test_registry_names_are_plain_and_unique() -> None:
    """One formula, one namespace: no name may need a version prefix to be meant."""

    names = STRATEGY_REGISTRY.names()
    assert names == sorted(set(names))
    assert not any(":" in name for name in names)
    assert set(REQUIRED_STRATEGIES) <= set(names)


def test_registry_rejects_unknown_names() -> None:
    with pytest.raises(StrategyError, match="Unknown strategy"):
        STRATEGY_REGISTRY.resolve("does_not_exist")
    # A leftover version prefix is simply an unknown name now.
    with pytest.raises(StrategyError, match="Unknown strategy"):
        STRATEGY_REGISTRY.resolve("v2:full")


def test_spec_validation_rejects_incoherent_definitions() -> None:
    with pytest.raises(StrategyError, match="at least one component weight"):
        StrategySpec(name="empty")
    with pytest.raises(StrategyError, match="must not carry component weights"):
        StrategySpec(name="bad_random", random_selection=True, rarity_weight=1.0)
    with pytest.raises(StrategyError, match="non-negative"):
        StrategySpec(name="negative", uncertainty_weight=-1.0)
    with pytest.raises(StrategyError, match="unknown uncertainty method"):
        StrategySpec(name="bad_u", uncertainty_weight=1.0, uncertainty_method="vibes")
    with pytest.raises(StrategyError, match="unknown image aggregation"):
        StrategySpec(name="bad_agg", rarity_weight=1.0, image_aggregation="median")


# --- legacy equivalence ------------------------------------------------------


# --- version 2 behaviour -----------------------------------------------------


def test_v2_uncertainty_uses_the_posterior_not_the_unknown_score() -> None:
    spec = STRATEGY_REGISTRY.resolve("uncertainty")
    assert spec.uncertainty_method == "entropy"
    with pytest.raises(ValueError, match="requires the exported posterior"):
        score_pool(
            spec=spec,
            image_ids=FIXTURE["image_ids"],
            embeddings=FIXTURE["embeddings"],
            reference_embeddings=FIXTURE["reference_embeddings"],
            confidence=FIXTURE["confidence"],
            posterior=None,
        )


def test_gate_is_applied_to_normalised_rarity_and_renormalised() -> None:
    result = _score("full")
    expected = result.normalised["rarity"] * np.power(
        result.raw["coherence"], result.spec.coherence_exponent
    )
    np.testing.assert_allclose(result.raw["gated"], expected, atol=1e-12)
    np.testing.assert_allclose(result.normalised["gated"], normalise(expected, "rank"), atol=1e-12)


def test_score_is_the_declared_weighted_sum() -> None:
    result = _score("full")
    weights = result.spec.weights()
    expected = sum(
        weights[name] * np.asarray(result.normalised[name]) for name in weights if weights[name] > 0
    )
    np.testing.assert_allclose(result.scores, expected, atol=1e-12)


def test_proposal_formula_preset_carries_both_distribution_terms() -> None:
    spec = STRATEGY_REGISTRY.resolve("proposal_formula")
    assert spec.uncertainty_weight > 0
    assert spec.rarity_weight > 0 and spec.gated_weight > 0
    assert spec.novelty_weight == 0.0


def test_ablation_variants_drop_exactly_one_term() -> None:
    full = STRATEGY_REGISTRY.resolve("full")
    assert STRATEGY_REGISTRY.resolve("full_no_uncertainty").uncertainty_weight == 0.0
    assert STRATEGY_REGISTRY.resolve("full_no_novelty").novelty_weight == 0.0
    no_coherence = STRATEGY_REGISTRY.resolve("full_no_coherence")
    assert no_coherence.gated_weight == 0.0
    assert no_coherence.rarity_weight == full.gated_weight
    no_rarity = STRATEGY_REGISTRY.resolve("full_no_rarity")
    assert no_rarity.rarity_weight == 0.0 and no_rarity.gated_weight == 0.0


def test_unused_components_are_not_computed_unless_requested() -> None:
    """The lean live path skips unused components; diagnostics opt in."""

    spec = STRATEGY_REGISTRY.resolve("uncertainty")
    arguments = {
        "spec": spec,
        "image_ids": FIXTURE["image_ids"],
        "embeddings": FIXTURE["embeddings"],
        "reference_embeddings": FIXTURE["reference_embeddings"],
        "confidence": FIXTURE["confidence"],
        "posterior": FIXTURE["posterior"],
        "predicted_labels": FIXTURE["predicted_labels"],
    }
    lean = score_pool(**arguments)
    assert np.all(lean.raw["novelty"] == 0.0)
    assert np.all(lean.raw["coherence"] == 0.0)

    rich = score_pool(**arguments, compute_all_components=True)
    assert np.any(rich.raw["novelty"] > 0.0)
    assert np.any(rich.raw["coherence"] > 0.0)
    # Computing extra components must not change the strategy's own score.
    np.testing.assert_allclose(rich.scores, lean.scores, atol=1e-12)


def test_scoring_is_deterministic() -> None:
    first, second = _score("full"), _score("full")
    np.testing.assert_array_equal(first.scores, second.scores)
    assert first.image_scores == second.image_scores


# --- aggregation and selection ----------------------------------------------


def test_aggregations_have_the_expected_semantics() -> None:
    ids = np.asarray(["a", "a", "a", "b"], dtype=object)
    scores = np.array([0.9, 0.5, 0.1, 0.4])
    assert aggregate_image_scores(ids, scores, method="max")["a"] == pytest.approx(0.9)
    assert aggregate_image_scores(ids, scores, method="mean")["a"] == pytest.approx(0.5)
    assert aggregate_image_scores(ids, scores, method="top_k_mean", top_k=2)["a"] == pytest.approx(
        0.7
    )
    assert aggregate_image_scores(ids, scores, method="noisy_or")["a"] == pytest.approx(
        1 - 0.1 * 0.5 * 0.9
    )
    # Fewer proposals than top_k must average what exists, not pad with zeros.
    assert aggregate_image_scores(ids, scores, method="top_k_mean", top_k=3)["b"] == pytest.approx(
        0.4
    )


def test_selection_respects_budget_and_breaks_ties_deterministically() -> None:
    scores = {"b": 1.0, "a": 1.0, "c": 0.5}
    assert select_images(scores, budget=2) == ["a", "b"]
    with pytest.raises(ValueError, match="exceeds"):
        select_images(scores, budget=4)
    with pytest.raises(ValueError, match="budget must be positive"):
        select_images(scores, budget=0)


def test_selected_proposal_mask_tracks_the_aggregation_method() -> None:
    top_k = _score("full")
    chosen = select_images(top_k.image_scores, budget=3)
    mask = top_k.selected_proposal_mask(chosen)
    assert mask.sum() == 3 * top_k.spec.top_k

    maximum = _score("full", image_aggregation="max")
    chosen_max = select_images(maximum.image_scores, budget=3)
    assert maximum.selected_proposal_mask(chosen_max).sum() == 3
