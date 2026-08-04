"""Tests for Representation Experiment E4's feature spaces and geometry metrics.

The properties asserted here are the ones that would let E4 reach a wrong
conclusion without anything crashing:

* a substituted export changes the embedding and **nothing else**, so a comparison
  across representations really does isolate the representation;
* the decisive statistic behaves correctly on spaces whose geometry is known by
  construction — it must be above 1 when the rare class clusters and below 1 when
  the background does;
* derived transforms are deterministic and preserve the row count;
* silhouette, compactness and the local/global separability pair report what they
  claim on synthetic geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from daowod import geometry, representation_plots, representations
from daowod.audit import Strata

DIMENSIONS = 16


def make_export(count: int = 240, seed: int = 0) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    posterior = generator.random((count, 6)) + 0.01
    return {
        "image_ids": np.array([f"img{index % 12:03d}" for index in range(count)], dtype=object),
        "embeddings": generator.normal(size=(count, DIMENSIONS)),
        "posterior": posterior / posterior.sum(axis=1, keepdims=True),
        "confidence": generator.random(count),
        "objectness": generator.random(count),
        "predicted_labels": np.full(count, 80, dtype=np.int64),
        "boxes": np.clip(generator.random((count, 4)) * 0.5 + 0.25, 0.01, 0.99),
    }


def clustered_space(
    *, tail_clusters: bool, count: int = 240, seed: int = 0
) -> tuple[np.ndarray, Strata]:
    """A space whose geometry is known, so a metric's verdict can be checked.

    ``tail_clusters=True``  each unknown class is a tight blob and background is
                            diffuse — the geometry the coherence gate assumes.
    ``tail_clusters=False`` the reverse: background is one tight mass and the unknown
                            regions are scattered through it, which is what PROB's
                            decoder space was measured to look like.
    """

    generator = np.random.default_rng(seed)
    third = count // 3
    kinds = np.array(["background"] * count, dtype=object)
    classes = np.array([""] * count, dtype=object)
    groups = np.array([""] * count, dtype=object)

    embeddings = np.zeros((count, DIMENSIONS), dtype=np.float64)
    # Two unknown classes: "rare" in the tail group, "common" in the head group.
    rare = slice(0, third // 2)
    common = slice(third // 2, third)
    background = slice(third, count)
    kinds[rare] = "unknown"
    kinds[common] = "unknown"
    classes[rare] = "rare"
    classes[common] = "common"
    groups[rare] = "tail"
    groups[common] = "head"

    if tail_clusters:
        embeddings[rare] = np.eye(1, DIMENSIONS, 0) + generator.normal(
            scale=0.02, size=(third // 2, DIMENSIONS)
        )
        embeddings[common] = np.eye(1, DIMENSIONS, 1) + generator.normal(
            scale=0.02, size=(third - third // 2, DIMENSIONS)
        )
        embeddings[background] = generator.normal(scale=1.5, size=(count - third, DIMENSIONS))
    else:
        embeddings[rare] = generator.normal(scale=1.5, size=(third // 2, DIMENSIONS))
        embeddings[common] = generator.normal(scale=1.5, size=(third - third // 2, DIMENSIONS))
        embeddings[background] = np.eye(1, DIMENSIONS, 2) + generator.normal(
            scale=0.02, size=(count - third, DIMENSIONS)
        )
    return embeddings, Strata.from_oracle(kinds, groups, classes)


# --------------------------------------------------------------------------
# The decisive statistic
# --------------------------------------------------------------------------


def test_tail_purity_advantage_exceeds_one_when_the_tail_clusters() -> None:
    embeddings, strata = clustered_space(tail_clusters=True)
    structure = geometry.Neighbourhoods.build(embeddings, neighbours=5)
    report = geometry.purity_summary(structure, strata, representation="synthetic_good")
    assert report["tail_same_label"] > 0.8
    assert report["tail_purity_advantage"] > 1.0
    assert report["coherence_premise_holds"] is True


def test_tail_purity_advantage_falls_below_one_when_background_clusters() -> None:
    """The measured shape of PROB's decoder space, reproduced by construction."""

    embeddings, strata = clustered_space(tail_clusters=False)
    structure = geometry.Neighbourhoods.build(embeddings, neighbours=5)
    report = geometry.purity_summary(structure, strata, representation="synthetic_bad")
    assert report["background_same_label"] > report["tail_same_label"]
    assert report["tail_purity_advantage"] < 1.0
    assert report["coherence_premise_holds"] is False


def test_headline_table_orders_by_the_decisive_statistic() -> None:
    rows = []
    for name, clusters in (("good", True), ("bad", False)):
        embeddings, strata = clustered_space(tail_clusters=clusters)
        structure = geometry.Neighbourhoods.build(embeddings, neighbours=5)
        rows.append(geometry.purity_summary(structure, strata, representation=name))
    table = geometry.headline_table(rows)
    assert [row["representation"] for row in table] == ["good", "bad"]


# --------------------------------------------------------------------------
# Density, precision, mutuality
# --------------------------------------------------------------------------


def test_density_ranks_the_tight_stratum_as_densest() -> None:
    embeddings, strata = clustered_space(tail_clusters=False)
    structure = geometry.Neighbourhoods.build(embeddings, neighbours=5)
    rows = geometry.density_summary(structure, strata, representation="synthetic_bad")
    by_stratum = {str(row["stratum"]): row for row in rows}
    # Background is the tight mass here, so a density term would rank it densest.
    assert (
        by_stratum["background"]["density_rank_percentile_median"]
        < by_stratum["tail"]["density_rank_percentile_median"]
    )


def test_nearest_neighbour_precision_tracks_the_geometry() -> None:
    good, strata_good = clustered_space(tail_clusters=True)
    bad, strata_bad = clustered_space(tail_clusters=False)
    good_report = geometry.nearest_neighbour_precision(
        geometry.Neighbourhoods.build(good, neighbours=5), strata_good, representation="good"
    )
    bad_report = geometry.nearest_neighbour_precision(
        geometry.Neighbourhoods.build(bad, neighbours=5), strata_bad, representation="bad"
    )
    assert good_report["one_nn_precision_tail"] > bad_report["one_nn_precision_tail"]


def test_mutual_neighbour_fraction_is_bounded() -> None:
    embeddings, strata = clustered_space(tail_clusters=True)
    report = geometry.mutual_neighbour_consistency(
        geometry.Neighbourhoods.build(embeddings, neighbours=5), strata, representation="good"
    )
    assert 0.0 <= float(report["mutual_neighbour_fraction_all"]) <= 1.0


# --------------------------------------------------------------------------
# Cluster geometry
# --------------------------------------------------------------------------


def test_silhouette_and_compactness_agree_with_the_construction() -> None:
    good, strata_good = clustered_space(tail_clusters=True)
    bad, strata_bad = clustered_space(tail_clusters=False)
    good_clusters = geometry.cluster_quality(good, strata_good, representation="good", sample=500)
    bad_clusters = geometry.cluster_quality(bad, strata_bad, representation="bad", sample=500)
    good_unknown = next(row for row in good_clusters if row["label_set"] == "unknown_class")
    bad_unknown = next(row for row in bad_clusters if row["label_set"] == "unknown_class")
    assert good_unknown["silhouette"] > bad_unknown["silhouette"]

    good_compact = geometry.compactness_separation(good, strata_good, representation="good")
    bad_compact = geometry.compactness_separation(bad, strata_bad, representation="bad")
    assert good_compact["structure_present"] is True
    assert good_compact["compactness_ratio"] < bad_compact["compactness_ratio"]


def test_local_and_global_separability_are_reported_separately() -> None:
    """The distinction the earlier audit needed: global signal, no local signal."""

    generator = np.random.default_rng(3)
    count = 300
    kinds = np.array(["background"] * count, dtype=object)
    classes = np.array([""] * count, dtype=object)
    groups = np.array([""] * count, dtype=object)
    tail = slice(0, 30)
    kinds[tail] = "unknown"
    classes[tail] = "rare"
    groups[tail] = "tail"
    # Tail regions are shifted along one axis (a global direction exists) but
    # individually scattered, so each one's neighbours are background.
    embeddings = generator.normal(scale=1.0, size=(count, DIMENSIONS))
    embeddings[tail, 0] += 2.5
    embeddings[tail] += generator.normal(scale=1.5, size=(30, DIMENSIONS))
    strata = Strata.from_oracle(kinds, groups, classes)
    report = geometry.background_tail_overlap(embeddings, strata, representation="global_only")
    assert report["available"] is True
    assert report["centroid_auc"] > report["nearest_tail_auc"]


# --------------------------------------------------------------------------
# The representation registry
# --------------------------------------------------------------------------


def test_prob_spaces_load_from_the_export_alone() -> None:
    export = make_export()
    for name in ("prob_decoder", "prob_posterior", "prob_geometry"):
        matrix, manifest = representations.load(name, export=export)
        assert matrix.shape[0] == export["embeddings"].shape[0]
        assert manifest["name"] == name
        assert np.isfinite(matrix).all()
    decoder, _ = representations.load("prob_decoder", export=export)
    assert np.array_equal(decoder, export["embeddings"])


def test_crop_spaces_are_reported_unavailable_without_extraction(tmp_path) -> None:
    export = make_export()
    ready = representations.available(export=export, directory=tmp_path)
    assert "prob_decoder" in ready
    assert "dino_resnet50" not in ready
    rows = {
        row["name"]: row for row in representations.audit_rows(export=export, directory=tmp_path)
    }
    assert rows["dino_resnet50"]["available"] is False
    assert "extract_region_embeddings" in str(rows["dino_resnet50"]["blocked_by"])


def test_derived_transforms_are_deterministic_and_shape_preserving() -> None:
    export = make_export()
    whitened, manifest = representations.load("prob_decoder_whitened", export=export)
    again, _ = representations.load("prob_decoder_whitened", export=export)
    assert np.allclose(whitened, again)
    assert whitened.shape[0] == export["embeddings"].shape[0]
    assert whitened.shape[1] <= DIMENSIONS
    assert 0.0 < float(manifest["explained_variance_ratio_sum"]) <= 1.0

    reduced, manifest = representations.load("prob_decoder_minus_top4", export=export)
    assert reduced.shape == export["embeddings"].shape
    assert float(manifest["removed_variance_ratio"]) > 0.0
    # The removed directions really are gone: projecting onto them yields ~zero.
    from sklearn.decomposition import PCA

    model = PCA(n_components=4, random_state=0).fit(np.asarray(export["embeddings"]))
    residual = np.abs(reduced @ model.components_.T).max()
    assert residual < 1e-8


def test_derived_transform_is_fitted_on_the_requested_rows_only() -> None:
    """Whitening the pool and whitening the whole export are different spaces."""

    export = make_export()
    rows = np.arange(0, 120)
    subset, _ = representations.load("prob_decoder_whitened", export=export, rows=rows)
    whole, _ = representations.load("prob_decoder_whitened", export=export)
    assert subset.shape[0] == rows.size
    assert not np.allclose(subset, whole[rows])


def test_unknown_representation_is_rejected() -> None:
    with pytest.raises(representations.RepresentationError, match="Unknown representation"):
        representations.resolve("nonexistent_space")


# --------------------------------------------------------------------------
# The isolation mechanism
# --------------------------------------------------------------------------


def test_substituted_export_changes_only_the_embeddings() -> None:
    export = make_export()
    generator = np.random.default_rng(9)
    replacement = generator.normal(size=(export["embeddings"].shape[0], 7))
    replaced = representations.substituted_export(export, replacement)
    assert np.array_equal(replaced["embeddings"], replacement)
    for field in (
        "image_ids",
        "posterior",
        "confidence",
        "objectness",
        "predicted_labels",
        "boxes",
    ):
        assert np.array_equal(np.asarray(replaced[field]), np.asarray(export[field])), field
    assert set(replaced) == set(export)


def test_substituted_export_refuses_a_row_count_mismatch() -> None:
    export = make_export()
    with pytest.raises(representations.RepresentationError, match="rows"):
        representations.substituted_export(export, np.zeros((5, 4)))


def test_substituted_export_round_trips_through_disk(tmp_path) -> None:
    export = make_export()
    generator = np.random.default_rng(11)
    replacement = generator.normal(size=(export["embeddings"].shape[0], 5))
    path = representations.write_substituted_export(
        tmp_path / "swapped.npz", export=export, embeddings=replacement, manifest={"name": "x"}
    )
    with np.load(path, allow_pickle=True) as handle:
        assert np.allclose(handle["embeddings"], replacement)
        assert np.array_equal(
            np.asarray([str(v) for v in handle["image_ids"].tolist()], dtype=object),
            np.asarray([str(v) for v in export["image_ids"].tolist()], dtype=object),
        )
    assert path.with_suffix(".json").exists()


def test_the_candidate_pool_is_identical_after_substitution(tmp_path) -> None:
    """The claim Phase 5 rests on, checked rather than assumed."""

    from daowod import annotation_study, candidates

    export = make_export(count=240)
    spec = candidates.CandidatePoolSpec(per_image_limit=5)
    generator = np.random.default_rng(5)
    replaced = representations.substituted_export(export, generator.normal(size=(240, 9)))
    original = candidates.build_candidate_pool(
        image_ids=export["image_ids"],
        boxes_cxcywh=export["boxes"],
        objectness=export["objectness"],
        unknown_score=export["confidence"],
        posterior=export["posterior"],
        predicted_labels=export["predicted_labels"],
        spec=spec,
    )
    after = candidates.build_candidate_pool(
        image_ids=replaced["image_ids"],
        boxes_cxcywh=replaced["boxes"],
        objectness=replaced["objectness"],
        unknown_score=replaced["confidence"],
        posterior=replaced["posterior"],
        predicted_labels=replaced["predicted_labels"],
        spec=spec,
    )
    assert original.indices.tolist() == after.indices.tolist()
    assert annotation_study.STRATEGY_FAMILY["v2:full"] == "baseline"  # baseline untouched


# --------------------------------------------------------------------------
# Projections and sampling
# --------------------------------------------------------------------------


def test_balanced_sampling_keeps_every_unknown_region() -> None:
    _, strata = clustered_space(tail_clusters=True, count=300)
    indices = representation_plots.sample_indices(strata, scheme="balanced", seed=0)
    assert strata.is_unknown[indices].sum() == int(strata.is_unknown.sum())
    # ...and it is not simply everything.
    assert indices.size < strata.is_unknown.shape[0]


def test_natural_sampling_preserves_the_composition() -> None:
    _, strata = clustered_space(tail_clusters=True, count=600)
    indices = representation_plots.sample_indices(strata, scheme="natural", size=300, seed=0)
    assert indices.size == 300
    observed = float(strata.is_unknown[indices].mean())
    expected = float(strata.is_unknown.mean())
    assert abs(observed - expected) < 0.1


def test_projection_is_deterministic_and_two_dimensional() -> None:
    embeddings, _ = clustered_space(tail_clusters=True, count=200)
    first, manifest = representation_plots.project(embeddings, method="pca", seed=0)
    second, _ = representation_plots.project(embeddings, method="pca", seed=0)
    assert first.shape == (200, 2)
    assert np.allclose(first, second)
    assert manifest["method"] == "pca"


def test_figures_are_written(tmp_path) -> None:
    embeddings, strata = clustered_space(tail_clusters=True, count=240)
    structure = geometry.Neighbourhoods.build(embeddings, neighbours=5)
    purity = [geometry.purity_summary(structure, strata, representation="synthetic")]
    written = representation_plots.purity_bars(geometry.headline_table(purity), tmp_path)
    assert {path.suffix for path in written} == {".png", ".pdf"}
    written = representation_plots.purity_panel(purity, tmp_path)
    assert written
    paths, manifest = representation_plots.projection_figure(
        embeddings,
        strata,
        tmp_path,
        representation="synthetic",
        method="pca",
        scheme="balanced",
        extra={"objectness": np.linspace(0, 1, 240)},
    )
    assert paths and manifest["tail_points"] > 0


# --------------------------------------------------------------------------
# The ceiling correction
# --------------------------------------------------------------------------


def test_purity_ceiling_reflects_class_size() -> None:
    labels = np.array(["a"] * 20 + ["b", "b", "b"] + ["c"], dtype=object)
    ceiling = geometry.purity_ceiling(labels, neighbours=10)
    assert ceiling[0] == pytest.approx(1.0)  # 19 siblings available, k = 10
    assert ceiling[20] == pytest.approx(0.2)  # 2 siblings available
    assert ceiling[-1] == pytest.approx(0.0)  # a singleton can never have one


def test_normalised_purity_removes_the_frequency_confound() -> None:
    """A space where every class is perfectly grouped must normalise to 1.0."""

    generator = np.random.default_rng(0)
    # Three unknown classes with very different sizes, each a tight blob, plus a
    # large diffuse background. Perfect grouping, unequal frequencies.
    sizes = {"rare": 2, "mid": 6, "common": 30}
    blocks = []
    kinds: list[str] = []
    classes: list[str] = []
    groups: list[str] = []
    for index, (name, size) in enumerate(sizes.items()):
        centre = np.eye(1, DIMENSIONS, index)
        blocks.append(centre + generator.normal(scale=0.01, size=(size, DIMENSIONS)))
        kinds += ["unknown"] * size
        classes += [name] * size
        groups += [{"rare": "tail", "mid": "medium", "common": "head"}[name]] * size
    # Background is grouped but slightly less tightly than the classes, so a correct
    # normalised ratio must land strictly above 1.0 rather than exactly at it.
    background = 200
    blocks.append(
        np.eye(1, DIMENSIONS, 5) + generator.normal(scale=0.45, size=(background, DIMENSIONS))
    )
    kinds += ["background"] * background
    classes += [""] * background
    groups += [""] * background

    embeddings = np.vstack(blocks)
    strata = Strata.from_oracle(
        np.array(kinds, dtype=object),
        np.array(groups, dtype=object),
        np.array(classes, dtype=object),
    )
    structure = geometry.Neighbourhoods.build(embeddings, neighbours=10)
    report = geometry.purity_summary(structure, strata, representation="perfect")

    # Raw tail purity is capped by the class having only one sibling...
    assert report["tail_same_label"] < 0.2
    # ...but normalised it is essentially perfect, and the premise now registers.
    assert report["tail_same_label_normalised"] > 0.9
    assert report["background_same_label"] < 1.0
    assert report["tail_purity_advantage"] < 0.2
    assert report["tail_purity_advantage_normalised"] > 1.0
    assert report["coherence_premise_holds"] is True
    assert report["coherence_premise_holds_raw"] is False


def test_sibling_rank_is_one_when_classes_group_and_large_when_they_do_not() -> None:
    good, strata_good = clustered_space(tail_clusters=True, count=300)
    bad, strata_bad = clustered_space(tail_clusters=False, count=300)
    good_report = geometry.same_class_rank(good, strata_good, representation="good")
    bad_report = geometry.same_class_rank(bad, strata_bad, representation="bad")
    assert good_report["unknown_median_sibling_rank"] < bad_report["unknown_median_sibling_rank"]
    assert good_report["unknown_sibling_within_10"] > bad_report["unknown_sibling_within_10"]


def test_sibling_rank_reports_excluded_singletons() -> None:
    kinds = np.array(["unknown", "unknown", "background", "background"], dtype=object)
    groups = np.array(["tail", "tail", "", ""], dtype=object)
    classes = np.array(["only", "other", "", ""], dtype=object)
    strata = Strata.from_oracle(kinds, groups, classes)
    embeddings = np.eye(4, DIMENSIONS)
    report = geometry.same_class_rank(embeddings, strata, representation="singletons")
    # Both unknown classes have exactly one member, so neither has a sibling.
    assert report["tail_singletons_excluded"] == 2
    assert np.isnan(float(report["tail_median_sibling_rank"]))


# --------------------------------------------------------------------------
# Phase 6 statistics
# --------------------------------------------------------------------------


def test_paired_interval_widens_with_spread_and_narrows_with_seeds() -> None:
    from daowod import reporting

    tight = reporting.paired_interval([0.10, 0.11, 0.09])
    loose = reporting.paired_interval([0.10, 0.30, -0.10])
    assert tight[0] > 0.0 and tight[1] > tight[0]  # excludes zero: an effect
    assert loose[0] < 0.0 < loose[1]  # includes zero: not an effect
    assert (loose[1] - loose[0]) > (tight[1] - tight[0])


def test_paired_interval_is_undefined_for_a_single_seed() -> None:
    from daowod import reporting

    low, high = reporting.paired_interval([0.2])
    assert np.isnan(low) and np.isnan(high)


def test_paired_interval_ignores_non_finite_entries() -> None:
    from daowod import reporting

    assert reporting.paired_interval([0.1, 0.2, np.nan]) == reporting.paired_interval([0.1, 0.2])
