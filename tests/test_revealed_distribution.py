"""Tests for label-anchored distribution estimation.

The properties that matter, and that a wrong implementation would violate
silently:

* the anchored estimators read the oracle **only** through the revealed bank, so
  scrambling the ground truth of unannotated proposals cannot change a score;
* a cold round (nothing revealed yet) is bit-identical to the unsupervised
  baseline, so the experiment isolates one variable;
* support ranks regions resembling confirmed unknowns above regions that do not;
* anchored rarity ranks a class the oracle has confirmed once above a class it has
  confirmed many times;
* the gate keeps its multiplicative form.
"""

from __future__ import annotations

import numpy as np
import pytest

from daowod import annotation, components
from daowod.scoring import STRATEGY_REGISTRY, StrategyError, StrategySpec

DIMENSIONS = 8


def make_pool(count: int = 60) -> annotation.ProposalPool:
    """Two tight semantic groups plus diffuse background, in a known geometry."""

    generator = np.random.default_rng(0)
    group_a = np.tile(np.eye(1, DIMENSIONS, 0), (count // 3, 1)) + generator.normal(
        scale=0.05, size=(count // 3, DIMENSIONS)
    )
    group_b = np.tile(np.eye(1, DIMENSIONS, 1), (count // 3, 1)) + generator.normal(
        scale=0.05, size=(count // 3, DIMENSIONS)
    )
    diffuse = generator.normal(scale=1.0, size=(count - 2 * (count // 3), DIMENSIONS))
    embeddings = np.vstack([group_a, group_b, diffuse])
    posterior = generator.random((embeddings.shape[0], 5)) + 0.01
    size = embeddings.shape[0]
    return annotation.ProposalPool(
        proposal_ids=np.array([f"p{index:03d}" for index in range(size)], dtype=object),
        image_ids=np.array([f"img{index % 6}" for index in range(size)], dtype=object),
        embeddings=embeddings,
        posterior=posterior / posterior.sum(axis=1, keepdims=True),
        confidence=generator.random(size),
        objectness=generator.random(size),
        predicted_labels=np.full(size, 4, dtype=np.int64),
        boxes_cxcywh=np.clip(generator.random((size, 4)) * 0.5 + 0.25, 0.01, 0.99),
    )


# --------------------------------------------------------------------------
# The bank
# --------------------------------------------------------------------------


def test_bank_splits_by_verdict_and_counts_classes() -> None:
    bank = components.RevealedBank()
    bank.add(np.ones(4), is_unknown=True, class_name="kite")
    bank.add(np.ones(4) * 2, is_unknown=True, class_name="kite")
    bank.add(np.ones(4) * 3, is_unknown=True, class_name="vase")
    bank.add(np.zeros(4), is_unknown=False)
    assert bank.unknown_count == 3
    assert bank.negative_count == 1
    assert bank.revealed_class_counts == {"kite": 2, "vase": 1}
    assert bank.report()["singleton_revealed_classes"] == 1


# --------------------------------------------------------------------------
# Support
# --------------------------------------------------------------------------


def test_support_is_neutral_before_anything_is_revealed() -> None:
    values, report = components.support(
        np.random.default_rng(0).normal(size=(10, 4)), components.RevealedBank()
    )
    assert report["cold_start"] is True
    assert values.tolist() == [components.COLD_START_SUPPORT] * 10


def test_support_defers_to_the_unsupervised_value_when_a_fallback_is_given() -> None:
    """What the acquisition loop does, so a cold round matches the baseline."""

    unsupervised = np.linspace(0.1, 0.9, 10)
    values, report = components.support(
        np.random.default_rng(0).normal(size=(10, 4)),
        components.RevealedBank(),
        fallback=unsupervised,
    )
    assert report["cold_start"] is True
    assert report["source"] == "unsupervised coherence fallback"
    assert values.tolist() == unsupervised.tolist()


def test_support_ranks_regions_resembling_confirmed_unknowns_highest() -> None:
    pool = make_pool()
    bank = components.RevealedBank()
    # The oracle confirmed two members of group A (indices 0 and 1).
    bank.add(pool.embeddings[0], is_unknown=True, class_name="kite")
    bank.add(pool.embeddings[1], is_unknown=True, class_name="kite")
    values, report = components.support(pool.embeddings, bank, neighbours=2)
    assert report["cold_start"] is False
    third = pool.size // 3
    assert values[:third].mean() > values[third : 2 * third].mean()
    assert values[:third].mean() > values[2 * third :].mean()
    assert 0.0 <= values.min() and values.max() <= 1.0


def test_support_uses_the_nearest_anchors_not_the_whole_bank() -> None:
    """A second confirmed class must not dilute support for the first."""

    pool = make_pool()
    bank_one = components.RevealedBank()
    bank_one.add(pool.embeddings[0], is_unknown=True, class_name="kite")
    bank_both = components.RevealedBank()
    bank_both.add(pool.embeddings[0], is_unknown=True, class_name="kite")
    for offset in range(pool.size // 3, pool.size // 3 + 6):
        bank_both.add(pool.embeddings[offset], is_unknown=True, class_name="vase")
    near_a = slice(0, pool.size // 3)
    only, _ = components.support(pool.embeddings, bank_one, neighbours=1)
    both, _ = components.support(pool.embeddings, bank_both, neighbours=1)
    assert both[near_a].mean() == pytest.approx(only[near_a].mean(), abs=1e-9)


def test_support_blocking_does_not_change_the_result() -> None:
    pool = make_pool(count=300)
    bank = components.RevealedBank()
    for index in range(12):
        bank.add(pool.embeddings[index], is_unknown=True, class_name=f"c{index % 3}")
    whole, _ = components.support(pool.embeddings, bank, neighbours=3)
    assert np.allclose(whole, components.support(pool.embeddings, bank, neighbours=3)[0])


# --------------------------------------------------------------------------
# Anchored rarity
# --------------------------------------------------------------------------


def test_anchored_rarity_falls_back_until_two_classes_are_revealed() -> None:
    pool = make_pool()
    fallback = np.arange(pool.size, dtype=np.float64)
    bank = components.RevealedBank()
    values, report = components.anchored_rarity(pool.embeddings, bank, fallback=fallback)
    assert report["cold_start"] is True
    assert values.tolist() == fallback.tolist()

    bank.add(pool.embeddings[0], is_unknown=True, class_name="kite")
    values, report = components.anchored_rarity(pool.embeddings, bank, fallback=fallback)
    assert report["cold_start"] is True  # one class is not a distribution
    assert values.tolist() == fallback.tolist()


def test_anchored_rarity_prefers_the_class_revealed_least_often() -> None:
    pool = make_pool()
    third = pool.size // 3
    bank = components.RevealedBank()
    # "common" confirmed four times in group A, "rare" once in group B.
    for index in range(4):
        bank.add(pool.embeddings[index], is_unknown=True, class_name="common")
    bank.add(pool.embeddings[third], is_unknown=True, class_name="rare")
    values, report = components.anchored_rarity(pool.embeddings, bank, fallback=np.zeros(pool.size))
    assert report["cold_start"] is False
    assert report["source"] == "nearest revealed class"
    # Group B resembles the rare class and must score higher than group A.
    assert values[third : 2 * third].mean() > values[:third].mean()


# --------------------------------------------------------------------------
# Integration with the acquisition loop
# --------------------------------------------------------------------------


def _run(
    spec: StrategySpec, pool: annotation.ProposalPool, *, gt_class, gt_unknown, rounds=3, budget=18
):
    return annotation.run_campaign(
        pool=pool,
        spec=spec,
        reference_embeddings=pool.embeddings[:4],
        gt_class=gt_class,
        gt_is_unknown=gt_unknown,
        total_budget=budget,
        rounds=rounds,
        seed=0,
    )


def _truth(pool: annotation.ProposalPool):
    """Group A is class 'kite', group B is class 'vase', the rest is background."""

    third = pool.size // 3
    classes = np.array([""] * pool.size, dtype=object)
    unknown = np.zeros(pool.size, dtype=bool)
    classes[:third] = "kite"
    classes[third : 2 * third] = "vase"
    unknown[: 2 * third] = True
    return classes, unknown


def test_cold_first_round_is_identical_to_the_unsupervised_baseline() -> None:
    """Nothing is revealed before round one, so the first ranking must match."""

    pool = make_pool()
    baseline = STRATEGY_REGISTRY.resolve("full")
    anchored = STRATEGY_REGISTRY.resolve("revealed_full")
    state_b = annotation.initial_state(
        pool_size=pool.size, reference_embeddings=pool.embeddings[:4]
    )
    state_a = annotation.initial_state(
        pool_size=pool.size, reference_embeddings=pool.embeddings[:4]
    )
    first_b = annotation.score_round(pool=pool, spec=baseline, state=state_b, seed=0)
    first_a = annotation.score_round(pool=pool, spec=anchored, state=state_a, seed=0)
    assert first_a.anchored["support"]["cold_start"] is True
    assert first_a.anchored["rarity"]["cold_start"] is True
    assert np.allclose(first_a.scores, first_b.scores)


def test_the_bank_grows_only_with_annotated_regions() -> None:
    pool = make_pool()
    classes, unknown = _truth(pool)
    result = _run(
        STRATEGY_REGISTRY.resolve("revealed_full"),
        pool,
        gt_class=classes,
        gt_unknown=unknown,
        rounds=3,
        budget=18,
    )
    assert result.selection_order.size == 18
    # Reconstructing the bank from the trajectory must reproduce its size exactly.
    state = annotation.initial_state(pool_size=pool.size, reference_embeddings=pool.embeddings[:4])
    state = annotation.reveal(
        state,
        selected=result.selection_order,
        pool=pool,
        pseudo_labels_full=np.zeros(pool.size, dtype=np.int64),
        gt_class=classes,
        gt_is_unknown=unknown,
    )
    assert state.bank.unknown_count + state.bank.negative_count == 18


def test_anchored_scores_ignore_the_ground_truth_of_unannotated_regions() -> None:
    """The leakage test: scramble the unselected labels, get the same trajectory."""

    pool = make_pool()
    classes, unknown = _truth(pool)
    spec = STRATEGY_REGISTRY.resolve("revealed_full")
    honest = _run(spec, pool, gt_class=classes, gt_unknown=unknown)

    scrambled_classes = classes.copy()
    scrambled_unknown = unknown.copy()
    chosen = set(honest.selection_order.tolist())
    generator = np.random.default_rng(7)
    for index in range(pool.size):
        if index in chosen:
            continue
        scrambled_classes[index] = generator.choice(["nonsense", "garbage", ""])
        scrambled_unknown[index] = bool(generator.integers(0, 2))
    scrambled = _run(spec, pool, gt_class=scrambled_classes, gt_unknown=scrambled_unknown)
    assert honest.selection_order.tolist() == scrambled.selection_order.tolist()


def test_anchored_estimator_changes_the_trajectory_once_labels_exist() -> None:
    """A sanity check in the other direction: the term must actually do something."""

    pool = make_pool()
    classes, unknown = _truth(pool)
    baseline = _run(STRATEGY_REGISTRY.resolve("full"), pool, gt_class=classes, gt_unknown=unknown)
    anchored = _run(
        STRATEGY_REGISTRY.resolve("revealed_full"), pool, gt_class=classes, gt_unknown=unknown
    )
    assert baseline.selection_order.tolist() != anchored.selection_order.tolist()
    # ...and the first round, being cold, must agree.
    first = baseline.round_boundaries[0]
    assert baseline.selection_order[:first].tolist() == anchored.selection_order[:first].tolist()


def test_anchored_round_records_its_bank_state() -> None:
    pool = make_pool()
    classes, unknown = _truth(pool)
    spec = STRATEGY_REGISTRY.resolve("revealed_full")
    state = annotation.initial_state(pool_size=pool.size, reference_embeddings=pool.embeddings[:4])
    first = annotation.score_round(pool=pool, spec=spec, state=state, seed=0)
    selected = annotation.select_batch(first.scores, batch_size=12, proposal_ids=pool.proposal_ids)
    state = annotation.reveal(
        state,
        selected=selected,
        pool=pool,
        pseudo_labels_full=np.zeros(pool.size, dtype=np.int64),
        gt_class=classes,
        gt_is_unknown=unknown,
    )
    second = annotation.score_round(pool=pool, spec=spec, state=state, seed=0)
    assert second.anchored["support"]["revealed_unknown_regions"] == state.bank.unknown_count
    report = components.diagnostics(state.bank, support_values=np.linspace(0, 1, 5))
    assert report["support_mean"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_the_baseline_is_untouched_by_the_new_estimator() -> None:
    for name in ("full", "full_no_coherence", "uncertainty", "random"):
        assert STRATEGY_REGISTRY.resolve(name).distribution_estimator == "cluster"


def test_anchored_strategies_share_the_baseline_weights_and_gate_form() -> None:
    baseline = STRATEGY_REGISTRY.resolve("full")
    anchored = STRATEGY_REGISTRY.resolve("revealed_full")
    assert anchored.weights() == baseline.weights()
    assert anchored.coherence_exponent == baseline.coherence_exponent
    assert anchored.distribution_estimator == "revealed"


def test_unknown_estimator_is_rejected() -> None:
    with pytest.raises(StrategyError, match="unknown distribution_estimator"):
        StrategySpec(name="bad", uncertainty_weight=1.0, distribution_estimator="magic")


# --------------------------------------------------------------------------
# The informativeness prior (the free control)
# --------------------------------------------------------------------------


def test_objectness_area_prior_ranks_large_confident_boxes_highest() -> None:
    from daowod import components

    objectness = np.array([0.1, 0.9, 0.9, 0.1])
    boxes = np.array(
        [
            [0.5, 0.5, 0.05, 0.05],  # small, low objectness
            [0.5, 0.5, 0.40, 0.40],  # large, high objectness
            [0.5, 0.5, 0.05, 0.05],  # small, high objectness
            [0.5, 0.5, 0.40, 0.40],  # large, low objectness
        ]
    )
    values = components.compute_uncertainty(
        method="objectness_area_prior", objectness=objectness, boxes_cxcywh=boxes
    )
    assert values.argmax() == 1
    assert values.argmin() == 0
    assert 0.0 <= values.min() and values.max() <= 1.0


def test_objectness_area_prior_requires_both_inputs() -> None:
    from daowod import components

    with pytest.raises(ValueError, match="requires both objectness and boxes"):
        components.compute_uncertainty(
            method="objectness_area_prior", objectness=np.ones(3), boxes_cxcywh=None
        )


def test_prior_strategies_are_scored_end_to_end() -> None:
    pool = make_pool()
    classes, unknown = _truth(pool)
    for name in ("objectness_area_prior", "prior_full", "prior_revealed_full"):
        result = _run(STRATEGY_REGISTRY.resolve(name), pool, gt_class=classes, gt_unknown=unknown)
        assert result.selection_order.size == 18


def test_the_comparison_matrix_holds_baseline_control_and_new_arms() -> None:
    from daowod import study

    families = {study.STRATEGY_FAMILY[name] for name in study.COMPARISON_STRATEGIES}
    assert "baseline" in families
    assert "free-control" in families
    assert "label-anchored" in families
    assert "full" in study.COMPARISON_STRATEGIES  # the baseline is retained
    assert len(study.COMPARISON_STRATEGIES) == 11


def test_per_round_anchored_diagnostics_are_recorded() -> None:
    """A cold bank and an uninformative term must be distinguishable in the output."""

    pool = make_pool()
    classes, unknown = _truth(pool)
    result = _run(
        STRATEGY_REGISTRY.resolve("revealed_full"),
        pool,
        gt_class=classes,
        gt_unknown=unknown,
        rounds=3,
        budget=18,
    )
    reports = [round_result.anchored for round_result in result.rounds]
    assert len(reports) == 3
    assert reports[0]["support"]["cold_start"] is True
    assert reports[0]["support"]["revealed_unknown_regions"] == 0
    # By the last round the oracle has confirmed something, so the term is warm.
    assert reports[-1]["support"]["revealed_unknown_regions"] > 0
    assert reports[-1]["support"]["cold_start"] is False


def test_baseline_rounds_carry_no_anchored_report() -> None:
    pool = make_pool()
    classes, unknown = _truth(pool)
    result = _run(STRATEGY_REGISTRY.resolve("full"), pool, gt_class=classes, gt_unknown=unknown)
    assert all(not round_result.anchored for round_result in result.rounds)
