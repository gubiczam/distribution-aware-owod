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
from daowod.diagnostics import (
    LeakageError,
    assert_no_ground_truth,
    cohens_d,
    coherence_regime,
    component_diagnostics,
    join_ground_truth,
    proposal_table,
    spearman,
    uncertainty_comparison,
)
from daowod.groups import ClassGroups
from daowod.scoring import STRATEGY_REGISTRY, score_pool, select_images

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

    spec = STRATEGY_REGISTRY.resolve("v2:rarity_coherence")
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
        spec=STRATEGY_REGISTRY.resolve("v2:rarity_coherence").__class__(
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
    spec = STRATEGY_REGISTRY.resolve("v2:full")
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


def _pool_result(strategy: str = "v2:full"):
    return score_pool(
        spec=STRATEGY_REGISTRY.resolve(strategy),
        image_ids=POOL.image_ids,
        embeddings=POOL.embeddings,
        reference_embeddings=POOL.reference_embeddings,
        confidence=POOL.confidence,
        posterior=POOL.posterior,
        predicted_labels=POOL.predicted_labels,
        seed=0,
        compute_all_components=True,
    )


def test_coherence_regime_classifier_recognises_each_regime() -> None:
    sizes = np.array([100.0] * 50 + [2.0] * 50)
    assert coherence_regime(np.full(100, 0.5), sizes)["regime"] == "inactive"
    confounded = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])
    assert coherence_regime(confounded, sizes)["regime"] == "frequency_confounded"
    rng = np.random.default_rng(0)
    independent = rng.uniform(0.0, 1.0, 100)
    assert coherence_regime(independent, rng.permutation(sizes))["regime"] in {
        "informative",
        "frequency_confounded",
    }


def test_component_diagnostics_reports_every_required_field() -> None:
    result = _pool_result()
    groups = ClassGroups.from_mapping(
        {
            row["class_name"]: row["group"]  # type: ignore[index]
            for row in POOL.class_stats_rows()
        },
        source="simulated",
    )
    report = component_diagnostics(
        result,
        budget=5,
        image_classes=POOL.image_classes,
        class_groups=groups,
        unknown_classes=list(POOL.class_image_counts),
    )
    for component in ("uncertainty", "rarity", "coherence", "gated"):
        assert "mean" in report[f"raw_{component}"]
        assert "q50" in report[f"norm_{component}"]
        assert report[f"histogram_norm_{component}"]["counts"]
    assert set(report["by_cluster_size_regime"]) == {
        "singleton",
        "2_to_3",
        "4_to_5",
        "6_or_more",
    }
    assert "rarity_vs_coherence" in report["correlations"]
    assert report["coherence_regime"]["regime"] in {
        "informative",
        "frequency_confounded",
        "saturated",
        "inactive",
    }
    assert "selected_image_jaccard" in report["gate_impact"]
    assert set(report["by_ground_truth_group"]) >= {"head", "medium", "tail"}


def test_uncertainty_comparison_detects_the_legacy_identity() -> None:
    """The legacy transform must be flagged as monotone; entropy must not be."""

    report = uncertainty_comparison(
        posterior=POOL.posterior,
        confidence=POOL.confidence,
        image_ids=POOL.image_ids,
        budget=6,
    )
    legacy = report["methods"]["legacy_prob_score"]
    entropy = report["methods"]["entropy"]

    # 1 - |2c - 1| equals 2c for every c < 0.5, so it is monotone in the unknown
    # score except for the handful of proposals that cross the fold at c = 0.5.
    # In this pool 3 of 384 proposals do, giving 0.99998 rather than exactly 1.
    above_fold = float((POOL.confidence > 0.5).mean())
    assert above_fold < 0.02
    assert legacy["spearman_with_unknown_score"] > 0.999
    assert legacy["is_monotone_in_unknown_score"] is True

    assert entropy["is_monotone_in_unknown_score"] is False
    assert abs(entropy["spearman_with_unknown_score"]) < 0.999
    assert "carries ranking information" in report["verdict"]


def test_cohens_d_and_spearman_behave_sanely() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 200)
    assert cohens_d(a, b) > 0.7
    assert spearman(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)
    assert np.isnan(cohens_d([1.0], [2.0]))
    assert np.isnan(spearman([1.0, 1.0], [2.0, 2.0]))


# --- Step 9: proposal records and the leakage guard --------------------------


def test_proposal_table_is_complete_and_ground_truth_free() -> None:
    result = _pool_result()
    chosen = select_images(result.image_scores, budget=4)
    rows = proposal_table(
        result,
        run_id="test-run",
        seed=0,
        round_index=1,
        selected_image_ids=chosen,
        posterior=POOL.posterior,
        confidence=POOL.confidence,
        predicted_labels=POOL.predicted_labels,
    )
    assert len(rows) == result.proposal_count
    required = {
        "run_id",
        "seed",
        "round",
        "strategy",
        "image_id",
        "proposal_index",
        "predicted_class_index",
        "cluster_id",
        "cluster_size",
        "posterior_entropy",
        "posterior_max",
        "posterior_margin",
        "unknown_score",
        "raw_uncertainty",
        "norm_uncertainty",
        "raw_novelty",
        "norm_novelty",
        "raw_rarity",
        "norm_rarity",
        "raw_coherence",
        "norm_coherence",
        "raw_gated",
        "norm_gated",
        "proposal_score",
        "image_score",
        "proposal_selected",
        "image_selected",
        "isolated_outlier",
    }
    assert required <= set(rows[0])
    assert_no_ground_truth(rows)
    assert sum(1 for row in rows if row["image_selected"]) > 0
    assert sum(1 for row in rows if row["proposal_selected"]) == 4 * result.spec.top_k


def test_leakage_guard_rejects_ground_truth_in_acquisition_artifacts() -> None:
    rows = [{"image_id": "a", "gt_classes": "toaster"}]
    with pytest.raises(LeakageError, match="ground-truth fields"):
        assert_no_ground_truth(rows)
    with pytest.raises(LeakageError):
        assert_no_ground_truth([{"image_id": "a", "true_class": "sofa"}])


def test_post_hoc_join_adds_ground_truth_and_tags_the_stage() -> None:
    result = _pool_result()
    chosen = select_images(result.image_scores, budget=4)
    rows = proposal_table(result, run_id="r", seed=0, round_index=1, selected_image_ids=chosen)
    groups = ClassGroups.from_mapping(
        {row["class_name"]: row["group"] for row in POOL.class_stats_rows()},  # type: ignore[index]
        source="simulated",
    )
    joined = join_ground_truth(
        rows,
        image_classes=POOL.image_classes,
        class_groups=groups,
        unknown_classes=list(POOL.class_image_counts),
    )
    assert all(row["analysis_stage"] == "post_hoc" for row in joined)
    assert any(row["gt_has_tail"] for row in joined)
    assert all(row["gt_classes"] for row in joined)
    # Joining twice must be refused: the guard runs on the input.
    with pytest.raises(LeakageError):
        join_ground_truth(
            joined,
            image_classes=POOL.image_classes,
            class_groups=groups,
            unknown_classes=list(POOL.class_image_counts),
        )


def test_simulator_is_deterministic_and_pins_its_calibration() -> None:
    first = simulate_pool(class_count=6, largest_class_images=5, proposals_per_image=4, seed=9)
    second = simulate_pool(class_count=6, largest_class_images=5, proposals_per_image=4, seed=9)
    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    np.testing.assert_array_equal(first.confidence, second.confidence)
    assert 0.09 <= first.objectness.min() <= first.objectness.max() <= 0.71
    assert "not evidence about the real" in str(first.metadata["warning"])
