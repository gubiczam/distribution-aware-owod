"""Steps 4 and 5 — do rarity and coherence mean what Decision 3 says?

The intended interaction is:

    rare + coherent      -> promoted
    rare + isolated      -> suppressed
    frequent + coherent  -> no rarity bonus
    frequent + isolated  -> not promoted

These are synthetic tests of the *definitions*. Whether the real candidate pool
puts coherence in an informative regime is a separate, empirical question that
only the bridge-exported pool can answer.
"""

import numpy as np
import pytest
from fixtures import simulate_pool, structured_regime_pool

from daowod import components
from daowod.oracle import LeakageError, assert_no_ground_truth
from daowod.scoring import STRATEGY_REGISTRY, score_pool

# One well-formed tail class (6 proposals), one big head class, plus genuinely
# isolated proposals. Every class has the same intra-class spread.
REGIME_COUNTS = (120, 60, 30, 12, 6)
EMBEDDINGS, TRUE_CLASS, IMAGE_IDS = structured_regime_pool(
    proposals_per_class=REGIME_COUNTS, isolated_count=4, seed=1
)
HEAD_MASK = TRUE_CLASS == 0
TAIL_MASK = TRUE_CLASS == len(REGIME_COUNTS) - 1
ISOLATED_MASK = TRUE_CLASS >= len(REGIME_COUNTS)


def _coherence(method: str) -> components.CoherenceResult:
    return components.compute_coherence(
        EMBEDDINGS,
        method=method,
        pseudo_labels=TRUE_CLASS,
        neighbour_count=5,
    )


# --- coherence definitions ---------------------------------------------------


def test_legacy_density_coherence_collapses_below_the_neighbour_count() -> None:
    """A sharper statement of S5 than the audit had.

    ``density`` coherence is ``1 / (1 + d_k / median(d_k))`` with ``k = 5``. When
    a pseudo-class has fewer than ``k`` members, its k-th nearest neighbour is
    necessarily in a *different* cluster, so ``d_k`` jumps to the inter-cluster
    scale and coherence collapses. The confound is therefore a step at the
    neighbour count, not a gradient — measured tail/head ratios:

        tail size   2     3     6     20    60
        legacy      0.14  0.14  0.83  0.89  0.93
        relative    0.94  1.00  1.00  0.99  1.00
    """

    neighbours = 5
    for tail_size, expect_collapse in ((2, True), (3, True), (6, False), (20, False)):
        embeddings, labels, _ = structured_regime_pool(
            proposals_per_class=(120, 60, 30, 12, tail_size), seed=1
        )
        head, tail = labels == 0, labels == 4
        legacy = components.compute_coherence(
            embeddings,
            method="density",
            pseudo_labels=labels,
            neighbour_count=neighbours,
        ).coherence
        relative = components.compute_coherence(
            embeddings,
            method="relative_within_cluster",
            pseudo_labels=labels,
            neighbour_count=neighbours,
        ).coherence
        legacy_ratio = legacy[tail].mean() / legacy[head].mean()
        relative_ratio = relative[tail].mean() / relative[head].mean()

        if expect_collapse:
            assert legacy_ratio < 0.3, (tail_size, legacy_ratio)
        else:
            assert legacy_ratio > 0.75, (tail_size, legacy_ratio)
        # The relative measure is scale-free at every size: that is the fix.
        assert relative_ratio > 0.85, (tail_size, relative_ratio)
        assert relative_ratio >= legacy_ratio, (tail_size, relative_ratio, legacy_ratio)


def test_isolated_proposals_are_suppressed_by_every_coherence_method() -> None:
    for method in ("relative_within_cluster", "neighbour_consistency", "density"):
        result = _coherence(method)
        assert result.coherence[ISOLATED_MASK].mean() < 0.25, method
        assert result.isolated[ISOLATED_MASK].all(), method


def test_singleton_and_small_cluster_behaviour_is_explicit() -> None:
    result = _coherence("relative_within_cluster")
    # Singletons get the declared constant, not an unstable division.
    assert np.all(result.coherence[ISOLATED_MASK] == 0.0)
    assert result.details["singleton_clusters"] == int(ISOLATED_MASK.sum())
    assert np.all(np.isfinite(result.coherence))

    custom = components.compute_coherence(
        EMBEDDINGS,
        method="relative_within_cluster",
        pseudo_labels=TRUE_CLASS,
        singleton_coherence=0.25,
    )
    assert np.all(custom.coherence[ISOLATED_MASK] == 0.25)

    # A two-member cluster cannot estimate its own scale and must borrow.
    tiny_embeddings, tiny_labels, _ = structured_regime_pool(proposals_per_class=(40, 2), seed=2)
    tiny = components.compute_coherence(
        tiny_embeddings,
        method="relative_within_cluster",
        pseudo_labels=tiny_labels,
        minimum_cluster_size=3,
    )
    assert tiny.details["proposals_borrowing_pooled_scale"] == 2
    assert np.all(np.isfinite(tiny.coherence))


def test_neighbour_consistency_is_not_penalised_by_small_class_size() -> None:
    """Raw purity would give a 6-member class at most 5/5; the normalisation fixes it."""

    result = _coherence("neighbour_consistency")
    assert result.coherence[TAIL_MASK].mean() == pytest.approx(1.0, abs=1e-9)
    assert result.coherence[HEAD_MASK].mean() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("counts", "label"),
    [
        ((60, 1), "singleton"),
        ((60, 3), "2_to_3"),
        ((60, 5), "4_to_5"),
        ((60, 12), "6_or_more"),
    ],
)
def test_every_cluster_size_regime_produces_finite_defined_values(
    counts: tuple[int, int], label: str
) -> None:
    """The four regimes the audit identified must all be well defined."""

    embeddings, labels, _ = structured_regime_pool(proposals_per_class=counts, seed=3)
    for method in ("relative_within_cluster", "neighbour_consistency", "density"):
        result = components.compute_coherence(
            embeddings, method=method, pseudo_labels=labels, neighbour_count=5
        )
        assert np.all(np.isfinite(result.coherence)), (method, label)
        assert np.all((result.coherence >= 0.0) & (result.coherence <= 1.0))


# --- rarity ------------------------------------------------------------------


def test_rarity_is_continuous_not_a_singleton_indicator() -> None:
    """S4: under rank normalisation rarity must grade, not flag."""

    from daowod.normalisation import normalise

    counts = (120, 60, 30, 12, 6, 3, 1)
    embeddings, labels, _ = structured_regime_pool(proposals_per_class=counts, seed=4)
    raw_legacy = components.compute_rarity(labels, method="inverse_frequency")
    legacy = normalise(raw_legacy, "minmax")
    modern = normalise(components.compute_rarity(labels, method="log_inverse_frequency"), "rank")

    # Legacy: almost everything is crushed towards zero.
    assert (legacy < 0.1).mean() > 0.9
    # Modern: rarity is spread across the pool and strictly ordered by class size.
    assert (modern < 0.1).mean() < 0.35
    assert modern.std() > 0.25
    means = [modern[labels == index].mean() for index in range(len(counts))]
    assert means == sorted(means), "rarity must increase as class size falls"


def test_all_rarity_methods_agree_under_rank_normalisation() -> None:
    from daowod.normalisation import normalise

    _, labels, _ = structured_regime_pool(proposals_per_class=(50, 20, 8, 2), seed=5)
    ranked = [
        normalise(components.compute_rarity(labels, method=method), "rank")
        for method in ("log_inverse_frequency", "inverse_frequency", "negative_count")
    ]
    for other in ranked[1:]:
        np.testing.assert_allclose(ranked[0], other, atol=1e-12)


# --- the four intended interactions -----------------------------------------


def test_gate_promotes_rare_and_coherent_over_rare_and_isolated() -> None:
    """The whole point of Contribution A, tested directly."""

    spec = STRATEGY_REGISTRY.resolve("rarity_coherence")
    result = score_pool(
        spec=spec,
        image_ids=IMAGE_IDS,
        embeddings=EMBEDDINGS,
        reference_embeddings=np.zeros((3, EMBEDDINGS.shape[1])),
        predicted_labels=TRUE_CLASS,
        posterior=None,
        seed=0,
    )
    # Use the true partition so the test isolates the gate, not KMeans.
    scored = score_pool(
        spec=STRATEGY_REGISTRY.resolve("rarity_coherence").__class__(
            **{**spec.as_dict(), "pseudo_label_source": "predicted"}
        ),
        image_ids=IMAGE_IDS,
        embeddings=EMBEDDINGS,
        reference_embeddings=np.zeros((3, EMBEDDINGS.shape[1])),
        predicted_labels=TRUE_CLASS,
        posterior=None,
        seed=0,
    )
    gated = scored.normalised["gated"]
    rarity = scored.normalised["rarity"]

    rare_coherent = gated[TAIL_MASK].mean()
    rare_isolated = gated[ISOLATED_MASK].mean()
    frequent_coherent = gated[HEAD_MASK].mean()

    # rare + coherent is promoted above both rare + isolated and frequent.
    assert rare_coherent > rare_isolated
    assert rare_coherent > frequent_coherent
    # Ungated rarity cannot make that distinction: isolated singletons are the
    # rarest thing in the pool, so they win.
    assert rarity[ISOLATED_MASK].mean() > rarity[TAIL_MASK].mean()
    assert result.spec.name == "rarity_coherence"


def test_frequent_and_isolated_is_not_promoted() -> None:
    spec = STRATEGY_REGISTRY.resolve("full")
    result = score_pool(
        spec=spec.__class__(**{**spec.as_dict(), "pseudo_label_source": "predicted"}),
        image_ids=IMAGE_IDS,
        embeddings=EMBEDDINGS,
        reference_embeddings=EMBEDDINGS[HEAD_MASK][:5],
        predicted_labels=TRUE_CLASS,
        posterior=np.full((EMBEDDINGS.shape[0], 4), 0.25),
        seed=0,
    )
    gated = result.normalised["gated"]
    assert gated[HEAD_MASK].mean() < gated[TAIL_MASK].mean()


# --- diagnostics -------------------------------------------------------------


POOL = simulate_pool(class_count=12, largest_class_images=12, proposals_per_image=8, seed=11)


# --- Step 9: proposal records and the leakage guard --------------------------


def test_leakage_guard_rejects_ground_truth_in_acquisition_artifacts() -> None:
    rows = [{"image_id": "a", "gt_classes": "toaster"}]
    with pytest.raises(LeakageError, match="ground-truth fields"):
        assert_no_ground_truth(rows)
    with pytest.raises(LeakageError):
        assert_no_ground_truth([{"image_id": "a", "true_class": "sofa"}])


def test_simulator_is_deterministic_and_pins_its_calibration() -> None:
    first = simulate_pool(class_count=6, largest_class_images=5, proposals_per_image=4, seed=9)
    second = simulate_pool(class_count=6, largest_class_images=5, proposals_per_image=4, seed=9)
    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    np.testing.assert_array_equal(first.confidence, second.confidence)
    assert 0.09 <= first.objectness.min() <= first.objectness.max() <= 0.71
    assert "not evidence about the real" in str(first.metadata["warning"])
