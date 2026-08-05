"""Tests for the offline active-annotation study.

The properties asserted here are the ones a wrong result would violate silently:

* the candidate pool is a function of PROB outputs only;
* the oracle's coordinate conversion is right (a wrong one turns every IoU into
  noise and every discovery number into a plausible lie);
* discovery counts distinct objects, not proposals;
* an annotated proposal cannot be bought twice, and the oracle is read only at
  selected positions;
* the long-tail severities are validated, not assumed, to differ;
* the acquisition score is reproducible from its recorded components.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from daowod import (
    active,
    annotation_study,
    candidates,
    discovery,
    export_cache,
    longtail,
    modes,
    oracle,
    pipeline,
    plots,
    reporting,
)
from daowod.components import compute_novelty
from daowod.groups import ClassGroups

# --------------------------------------------------------------------------
# A tiny synthetic export with known geometry. Deliberately not the simulator:
# these tests are about bookkeeping, so hand-placed boxes make the expected
# answer checkable by eye.
# --------------------------------------------------------------------------

IMAGE_WIDTH, IMAGE_HEIGHT = 200, 100


def voc_xml(objects: list[tuple[str, tuple[int, int, int, int]]]) -> str:
    body = "".join(
        f"<object><name>{name}</name><bndbox>"
        f"<xmin>{box[0]}</xmin><ymin>{box[1]}</ymin>"
        f"<xmax>{box[2]}</xmax><ymax>{box[3]}</ymax>"
        "</bndbox></object>"
        for name, box in objects
    )
    return (
        "<annotation><size>"
        f"<width>{IMAGE_WIDTH}</width><height>{IMAGE_HEIGHT}</height><depth>3</depth>"
        f"</size>{body}</annotation>"
    )


@pytest.fixture
def annotations_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "Annotations"
    directory.mkdir()
    # "dog" and "person" are Task-1 known classes; "apple" and "kite" are unknown.
    (directory / "img0.xml").write_text(
        voc_xml([("dog", (11, 11, 51, 51)), ("apple", (111, 11, 151, 51))]),
        encoding="utf-8",
    )
    (directory / "img1.xml").write_text(
        voc_xml([("kite", (11, 11, 51, 51)), ("apple", (111, 11, 191, 91))]),
        encoding="utf-8",
    )
    return directory


def normalised_box(x_min: int, y_min: int, x_max: int, y_max: int) -> list[float]:
    """VOC pixel xyxy -> normalised cxcywh, the export's coordinate space."""

    centre_x = ((x_min - 1) + x_max) / 2.0 / IMAGE_WIDTH
    centre_y = ((y_min - 1) + y_max) / 2.0 / IMAGE_HEIGHT
    width = (x_max - (x_min - 1)) / IMAGE_WIDTH
    height = (y_max - (y_min - 1)) / IMAGE_HEIGHT
    return [centre_x, centre_y, width, height]


# --------------------------------------------------------------------------
# Oracle
# --------------------------------------------------------------------------


def test_annotation_parsing_applies_prob_conventions(annotations_dir: Path) -> None:
    parsed = oracle.read_voc_annotation("img0", annotations_dir)
    assert (parsed.width, parsed.height) == (IMAGE_WIDTH, IMAGE_HEIGHT)
    # xmin/ymin are decremented by one, as OWDetection.load_instances does.
    assert parsed.objects[0].box_xyxy == (10.0, 10.0, 51.0, 51.0)
    assert parsed.objects[0].is_known is True
    assert parsed.objects[1].is_known is False


def test_cocofied_names_are_normalised() -> None:
    assert oracle.canonical_class_name("airplane") == "aeroplane"
    assert oracle.canonical_class_name("dining table") == "diningtable"
    assert oracle.canonical_class_name("kite") == "kite"


def test_a_proposal_on_an_annotation_matches_that_annotation(annotations_dir: Path) -> None:
    """The coordinate contract, checked against the annotation it came from."""

    annotations = oracle.load_annotations(["img0"], annotations_dir)
    table = oracle.match_proposals(
        image_ids=["img0", "img0", "img0"],
        boxes_cxcywh=[
            normalised_box(11, 11, 51, 51),  # exactly the dog
            normalised_box(111, 11, 151, 51),  # exactly the apple
            [0.9, 0.9, 0.02, 0.02],  # empty corner
        ],
        annotations=annotations,
    )
    assert list(table.gt_match_kind) == ["known", "unknown", "background"]
    assert list(table.gt_class) == ["dog", "apple", ""]
    assert table.gt_best_iou[0] == pytest.approx(1.0)
    assert table.gt_best_iou[1] == pytest.approx(1.0)
    assert table.gt_object_index[2] == -1


def test_several_proposals_on_one_object_are_one_discovery(annotations_dir: Path) -> None:
    annotations = oracle.load_annotations(["img0"], annotations_dir)
    table = oracle.match_proposals(
        image_ids=["img0"] * 3,
        boxes_cxcywh=[
            normalised_box(111, 11, 151, 51),
            normalised_box(113, 13, 149, 49),
            normalised_box(109, 9, 153, 53),
        ],
        annotations=annotations,
    )
    assert (table.gt_match_kind == "unknown").all()
    assert len(set(table.gt_object_index.tolist())) == 1


def test_frequency_groups_are_deterministic_and_complete() -> None:
    counts = {"a": 10, "b": 9, "c": 5, "d": 4, "e": 2, "f": 1}
    groups = oracle.assign_frequency_groups(counts)
    assert groups.groups["a"] == "head"
    assert groups.groups["f"] == "tail"
    assert set(groups.counts()) == {"head", "medium", "tail"}
    assert oracle.assign_frequency_groups(counts).groups == groups.groups


def test_reachable_counts_ignore_unreachable_objects(annotations_dir: Path) -> None:
    annotations = oracle.load_annotations(["img0", "img1"], annotations_dir)
    table = oracle.match_proposals(
        image_ids=["img0", "img1"],
        boxes_cxcywh=[normalised_box(111, 11, 151, 51), [0.9, 0.9, 0.02, 0.02]],
        annotations=annotations,
    )
    # Two unknown classes are annotated; only "apple" is reached by a proposal.
    assert oracle.unknown_class_counts(annotations) == {"apple": 2, "kite": 1}
    assert oracle.reachable_class_counts(table) == {"apple": 1}


# --------------------------------------------------------------------------
# Candidate pool: ground-truth free
# --------------------------------------------------------------------------


def test_candidate_pool_signature_takes_no_ground_truth() -> None:
    import inspect

    parameters = set(inspect.signature(candidates.build_candidate_pool).parameters)
    forbidden = {name for name in parameters if name.startswith("gt_")} | (
        parameters & {"annotations", "oracle", "table", "ground_truth", "class_groups"}
    )
    assert not forbidden


def test_nms_removes_duplicate_boxes_and_keeps_the_best() -> None:
    boxes = [
        [0.5, 0.5, 0.2, 0.2],
        [0.51, 0.51, 0.2, 0.2],  # heavy overlap with the first
        [0.1, 0.1, 0.05, 0.05],  # disjoint
    ]
    kept = candidates.class_agnostic_nms(boxes, [0.9, 0.8, 0.7], iou_threshold=0.5)
    assert kept.tolist() == [0, 2]


def test_per_image_limit_and_ranking_are_respected() -> None:
    pool = candidates.build_candidate_pool(
        image_ids=["a"] * 4 + ["b"] * 4,
        boxes_cxcywh=[[0.1 * index, 0.5, 0.05, 0.05] for index in range(8)],
        objectness=[0.1, 0.9, 0.5, 0.3, 0.2, 0.8, 0.4, 0.6],
        unknown_score=[0.5] * 8,
        spec=candidates.CandidatePoolSpec(per_image_limit=2),
    )
    assert pool.size == 4
    assert pool.report["final_images"] == 2
    # Highest objectness per image survives.
    assert 1 in pool.indices.tolist() and 5 in pool.indices.tolist()


def test_empty_pool_fails_loudly_rather_than_returning_nothing() -> None:
    with pytest.raises(candidates.CandidateError, match="removed every proposal"):
        candidates.build_candidate_pool(
            image_ids=["a"],
            boxes_cxcywh=[[0.5, 0.5, 0.1, 0.1]],
            objectness=[0.1],
            unknown_score=[0.1],
            spec=candidates.CandidatePoolSpec(minimum_objectness=0.9),
        )


def test_deterministic_subset_is_order_independent() -> None:
    first = candidates.deterministic_subset(["c", "a", "b", "d"], limit=2, seed=7)
    second = candidates.deterministic_subset(["d", "b", "a", "c"], limit=2, seed=7)
    assert first == second


# --------------------------------------------------------------------------
# Novelty: the memory-bounded path must be numerically identical
# --------------------------------------------------------------------------


def test_chunked_novelty_matches_the_unchunked_computation() -> None:
    generator = np.random.default_rng(0)
    pool = generator.normal(size=(300, 16))
    bank = generator.normal(size=(120, 16))
    chunked = compute_novelty(pool, bank, chunk_elements=64)
    whole = compute_novelty(pool, bank, chunk_elements=10**9)
    # Equal to floating-point noise (BLAS sums a block and a single row in
    # different orders), and — what selection actually depends on — identical in
    # rank order.
    assert np.allclose(chunked, whole, atol=1e-12, rtol=0.0)
    assert np.array_equal(np.argsort(chunked, kind="stable"), np.argsort(whole, kind="stable"))
    assert np.array_equal(compute_novelty(pool, bank), compute_novelty(pool, bank))


def test_novelty_is_one_when_the_bank_is_empty() -> None:
    values = compute_novelty(np.ones((3, 4)), np.zeros((0, 4)))
    assert values.tolist() == [1.0, 1.0, 1.0]


# --------------------------------------------------------------------------
# Long-tail protocol
# --------------------------------------------------------------------------


def make_oracle_table(class_counts: dict[str, int]) -> oracle.OracleTable:
    """One proposal per unknown object, plus one background proposal."""

    objects: list[oracle.GroundTruthObject] = []
    for name, count in class_counts.items():
        for index in range(count):
            objects.append(
                oracle.GroundTruthObject(
                    image_id=f"img{index}",
                    class_name=name,
                    box_xyxy=(0.0, 0.0, 1.0, 1.0),
                    is_known=False,
                )
            )
    total = len(objects) + 1
    kinds = np.array(["unknown"] * len(objects) + ["background"], dtype=object)
    classes = np.array([item.class_name for item in objects] + [""], dtype=object)
    return oracle.OracleTable(
        gt_match_kind=kinds,
        gt_class=classes,
        gt_group=np.array([""] * total, dtype=object),
        gt_is_unknown=np.array([True] * len(objects) + [False]),
        gt_object_index=np.array(list(range(len(objects))) + [-1], dtype=np.int64),
        gt_best_iou=np.ones(total),
        objects=tuple(objects),
        iou_threshold=0.5,
    )


def test_relative_profile_sharpens_where_absolute_cannot() -> None:
    """The measured defect that motivated the second profile.

    A naturally steep distribution already sits below an absolute exponential
    target, so ``min(target, available)`` keeps everything and the severity is a
    no-op. The relative profile multiplies each class's own count instead.
    """

    counts = {"a": 40, "b": 8, "c": 3, "d": 1}
    absolute = longtail.retention_profile(counts, imbalance_ratio=20.0, profile="absolute")
    relative = longtail.retention_profile(counts, imbalance_ratio=20.0, profile="relative")
    assert absolute == counts  # nothing was removed: the request was not binding
    assert relative["a"] == 40
    assert relative["b"] < counts["b"]
    assert sum(relative.values()) < sum(counts.values())


def test_head_cap_flattens() -> None:
    counts = {"a": 40, "b": 8, "c": 3, "d": 1}
    capped = longtail.retention_profile(
        counts, imbalance_ratio=1.5, head_cap_fraction=0.1, profile="absolute"
    )
    assert capped["a"] == 4
    assert capped["d"] == 1


def test_no_class_is_ever_emptied() -> None:
    counts = {"a": 50, "b": 1, "c": 1}
    profile = longtail.retention_profile(counts, imbalance_ratio=1000.0, profile="relative")
    assert min(profile.values()) >= 1


def test_undersampling_removes_every_proposal_of_a_dropped_object() -> None:
    table = make_oracle_table({"a": 6, "b": 2})
    pool = longtail.build_long_tail_pool(
        table,
        spec=longtail.ImbalanceSpec(name="cut", imbalance_ratio=6.0, profile="relative"),
        seed=0,
    )
    kept_objects = {
        int(value) for value in table.gt_object_index[pool.keep_mask & table.gt_is_unknown].tolist()
    }
    assert len(kept_objects) == sum(pool.retained_objects_by_class.values())
    # The background proposal is untouched.
    assert bool(pool.keep_mask[-1])


def test_indistinguishable_severities_raise_with_a_useful_message() -> None:
    reports = [
        {"setting": "one", "head_to_tail_object_ratio": 4.0, "requested_imbalance_ratio": 1.0},
        {"setting": "two", "head_to_tail_object_ratio": 4.05, "requested_imbalance_ratio": 20.0},
    ]
    with pytest.raises(longtail.LongTailError) as error:
        longtail.validate_settings_distinct(reports)
    message = str(error.value)
    assert "indistinguishable pairs" in message
    assert "one" in message and "two" in message
    assert "profile='relative'" in message


def test_distinct_severities_pass_and_report_the_span() -> None:
    reports = [
        {"setting": "flat", "head_to_tail_object_ratio": 2.0},
        {"setting": "natural", "head_to_tail_object_ratio": 6.0},
    ]
    verdict = longtail.validate_settings_distinct(reports)
    assert "flat=2.00" in verdict and "natural=6.00" in verdict


def test_pairwise_collapse_is_caught_even_when_the_extremes_differ() -> None:
    reports = [
        {"setting": "a", "head_to_tail_object_ratio": 2.0},
        {"setting": "b", "head_to_tail_object_ratio": 6.0},
        {"setting": "c", "head_to_tail_object_ratio": 6.1},
    ]
    distinct, message = longtail.settings_are_distinct(reports)
    assert not distinct
    assert "b" in message and "c" in message


def test_budgets_are_clamped_to_the_pool() -> None:
    assert longtail.resolve_budgets([10, 50, 500], pool_size=100) == [10, 50]
    with pytest.raises(longtail.LongTailError):
        longtail.resolve_budgets([500], pool_size=100)


# --------------------------------------------------------------------------
# Acquisition loop
# --------------------------------------------------------------------------


def make_pool(count: int = 40, dimensions: int = 8) -> active.ProposalPool:
    generator = np.random.default_rng(3)
    posterior = generator.random((count, 5)) + 0.01
    return active.ProposalPool(
        proposal_ids=np.array([f"p{index:03d}" for index in range(count)], dtype=object),
        image_ids=np.array([f"img{index % 5}" for index in range(count)], dtype=object),
        embeddings=generator.normal(size=(count, dimensions)),
        posterior=posterior / posterior.sum(axis=1, keepdims=True),
        confidence=generator.random(count),
        objectness=generator.random(count),
        predicted_labels=np.full(count, 4, dtype=np.int64),
        boxes_cxcywh=np.clip(generator.random((count, 4)) * 0.5 + 0.25, 0.01, 0.99),
    )


def test_annotated_proposals_can_never_be_selected_twice() -> None:
    pool = make_pool()
    spec = annotation_study.STRATEGY_REGISTRY.resolve("v2:full")
    state = active.initial_state(pool_size=pool.size, reference_embeddings=pool.embeddings[:5])
    first = active.score_round(pool=pool, spec=spec, state=state, seed=0)
    chosen = active.select_batch(first.scores, batch_size=5, proposal_ids=pool.proposal_ids)
    state = active.reveal(
        state,
        selected=chosen,
        pool=pool,
        pseudo_labels_full=np.zeros(pool.size, dtype=np.int64),
        gt_class=np.array([""] * pool.size, dtype=object),
        gt_is_unknown=np.zeros(pool.size, dtype=bool),
    )
    second = active.score_round(pool=pool, spec=spec, state=state, seed=0)
    assert np.isneginf(second.scores[chosen]).all()
    again = active.select_batch(second.scores, batch_size=5, proposal_ids=pool.proposal_ids)
    assert not set(again.tolist()) & set(chosen.tolist())


def test_reveal_reads_the_oracle_only_at_selected_positions() -> None:
    """Scrambling the unselected oracle entries must not change the state."""

    pool = make_pool()
    selected = np.array([0, 1, 2], dtype=np.int64)
    labels = np.zeros(pool.size, dtype=np.int64)
    truthful = np.array(["apple"] * pool.size, dtype=object)
    scrambled = truthful.copy()
    scrambled[3:] = "nonsense"
    unknown = np.ones(pool.size, dtype=bool)

    def run(classes: np.ndarray) -> tuple[set[str], dict[int, int]]:
        state = active.initial_state(pool_size=pool.size, reference_embeddings=pool.embeddings[:4])
        state = active.reveal(
            state,
            selected=selected,
            pool=pool,
            pseudo_labels_full=labels,
            gt_class=classes,
            gt_is_unknown=unknown,
        )
        return set(state.revealed_classes), dict(state.annotations_per_cluster)

    assert run(truthful) == run(scrambled)


def test_every_round_verifies_its_own_score_against_its_components(monkeypatch) -> None:
    """The leakage identity is checked per round, not once per run."""

    pool = make_pool()
    spec = annotation_study.STRATEGY_REGISTRY.resolve("v2:full")
    state = active.initial_state(pool_size=pool.size, reference_embeddings=pool.embeddings[:5])

    calls: list[int] = []
    real = discovery.assert_selection_is_ground_truth_free

    def counting(**kwargs: object) -> None:
        calls.append(1)
        real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(discovery, "assert_selection_is_ground_truth_free", counting)
    active.score_round(pool=pool, spec=spec, state=state, seed=0)
    assert calls == [1]

    def failing(**_: object) -> None:
        raise discovery.MetricError("an unrecorded term is influencing selection")

    monkeypatch.setattr(discovery, "assert_selection_is_ground_truth_free", failing)
    with pytest.raises(discovery.MetricError):
        active.score_round(pool=pool, spec=spec, state=state, seed=0)


def test_selection_is_deterministic_and_breaks_ties_by_proposal_id() -> None:
    scores = np.array([1.0, 1.0, 1.0, 0.5])
    ids = np.array(["pB", "pA", "pC", "pD"], dtype=object)
    chosen = active.select_batch(scores, batch_size=2, proposal_ids=ids)
    assert sorted(ids[chosen].tolist()) == ["pA", "pB"]


def test_campaign_order_is_a_prefix_chain() -> None:
    pool = make_pool()
    spec = annotation_study.STRATEGY_REGISTRY.resolve("v2:full")
    result = active.run_campaign(
        pool=pool,
        spec=spec,
        reference_embeddings=pool.embeddings[:5],
        gt_class=np.array([""] * pool.size, dtype=object),
        gt_is_unknown=np.zeros(pool.size, dtype=bool),
        total_budget=12,
        rounds=3,
        seed=0,
    )
    assert result.selection_order.size == 12
    assert len(set(result.selection_order.tolist())) == 12
    assert result.prefix(6).tolist() == result.selection_order[:6].tolist()


def test_saturation_downweights_already_bought_clusters() -> None:
    state = active.AcquisitionState(
        annotated=np.zeros(4, dtype=bool),
        reference_embeddings=np.zeros((1, 3)),
        annotations_per_cluster={0: 3},
    )
    weights = state.saturation_weights([0, 0, 1, 1], mode="cluster", strength=1.0)
    assert weights[0] == pytest.approx(0.25)
    assert weights[2] == pytest.approx(1.0)
    neutral = state.saturation_weights([0, 0, 1, 1], mode="none", strength=1.0)
    assert neutral.tolist() == [1.0, 1.0, 1.0, 1.0]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_normalised_auc_is_a_mean_over_the_sweep() -> None:
    assert discovery.normalised_auc([10, 20], [0.2, 0.4]) == pytest.approx(0.3)
    assert discovery.normalised_auc([10], [0.7]) == pytest.approx(0.7)


def test_discovery_recall_counts_objects_not_proposals(annotations_dir: Path) -> None:
    annotations = oracle.load_annotations(["img0"], annotations_dir)
    table = oracle.match_proposals(
        image_ids=["img0"] * 3,
        boxes_cxcywh=[
            normalised_box(111, 11, 151, 51),
            normalised_box(112, 12, 150, 50),
            [0.95, 0.95, 0.02, 0.02],
        ],
        annotations=annotations,
    )
    groups = ClassGroups.from_mapping({"apple": "tail"}, source="test")
    table = oracle.with_class_groups(table, groups)
    targets = discovery.DiscoveryTargets(
        objects_by_group=longtail.discoverable_objects(table, np.ones(3, dtype=bool)),
        classes_by_group=longtail.unknown_classes_present(table, np.ones(3, dtype=bool)),
    )
    pool = make_pool(count=3, dimensions=4)
    row = discovery.budget_row(
        strategy="test",
        seed=0,
        imbalance_setting="natural",
        budget=2,
        selected=[0, 1],
        pool=pool,
        oracle=table,
        targets=targets,
    )
    assert row["all_objects_found"] == 1  # two proposals, one object
    assert row["tail_discovery_recall"] == pytest.approx(1.0)
    assert row["annotation_precision"] == pytest.approx(1.0)
    assert row["background_selection_rate"] == pytest.approx(0.0)


def test_component_rebuild_check_rejects_an_unexplained_term() -> None:
    components = {"uncertainty": np.array([0.1, 0.9]), "novelty": np.array([0.5, 0.5])}
    weights = {"uncertainty": 1.0, "novelty": 0.0}
    discovery.assert_selection_is_ground_truth_free(
        scores=np.array([0.1, 0.9]), components=components, spec_weights=weights
    )
    with pytest.raises(discovery.MetricError, match="do not reproduce"):
        discovery.assert_selection_is_ground_truth_free(
            scores=np.array([0.6, 0.9]), components=components, spec_weights=weights
        )


def test_aggregation_reports_nan_sd_for_a_single_seed() -> None:
    rows = [
        {"strategy": "a", "imbalance_setting": "n", "budget": 10, "value": 0.5},
    ]
    aggregated = discovery.aggregate_over_seeds(rows)
    assert aggregated[0]["value_mean"] == pytest.approx(0.5)
    assert np.isnan(aggregated[0]["value_sd"])


# --------------------------------------------------------------------------
# Modes, runtime planning, reporting
# --------------------------------------------------------------------------


def test_every_mode_is_internally_consistent() -> None:
    for name in modes.MODE_NAMES:
        mode = modes.resolve_mode(name)
        assert mode.total_images > 0
        assert mode.study_config().strategies == mode.strategies
        assert len(mode.imbalance_settings) >= 2
    assert modes.resolve_mode("main").name == "MAIN"
    with pytest.raises(modes.ModeError):
        modes.resolve_mode("HUGE")


def test_main_mode_targets_the_five_required_strategies() -> None:
    mode = modes.resolve_mode("MAIN")
    assert mode.strategies == annotation_study.PRIMARY_STRATEGIES
    assert len(mode.strategies) == 5
    assert mode.research_grade is True


def test_a_fitting_estimate_reports_runtime_disk_and_memory() -> None:
    mode = modes.resolve_mode("MAIN")
    estimate = pipeline.estimate_cost(
        mode=mode,
        seconds_per_image=0.001,
        seconds_per_cell=0.5,
        measured_pool_size=mode.evaluation_images * mode.per_image_limit,
        ablation_specs=18,
        budget_seconds=36_000.0,
    )
    assert estimate.within_budget
    estimate.enforce()  # must not raise
    payload = estimate.as_dict()
    assert payload["study_cells"] == (
        len(mode.imbalance_settings) * len(mode.strategies) * len(mode.seeds)
    )
    # D3: runtime is not the only declared limit.
    assert payload["export_disk_gb"] > 0.0
    assert payload["pool_memory_gb"] > 0.0


def test_an_over_budget_estimate_refuses_and_never_shrinks_the_pool() -> None:
    """The protocol must not change because the machine was slow.

    An earlier version reacted to this case by reducing the evaluation pool until
    the projection fit, which silently moved every reported denominator. The
    estimate is now inert: it reports and refuses.
    """

    mode = modes.resolve_mode("MAIN")
    estimate = pipeline.estimate_cost(
        mode=mode,
        seconds_per_image=1.0,
        seconds_per_cell=1e5,
        measured_pool_size=1000,
        ablation_specs=18,
        budget_seconds=60.0,
    )
    assert not estimate.within_budget
    with pytest.raises(pipeline.RuntimeBudgetExceeded, match="refuses to start"):
        estimate.enforce()
    # The design it was asked about is untouched.
    assert estimate.evaluation_images == mode.evaluation_images
    assert estimate.per_image_limit == mode.per_image_limit
    assert estimate.seeds == tuple(mode.seeds)


def test_a_zero_budget_means_unbounded_rather_than_impossible() -> None:
    estimate = pipeline.estimate_cost(
        mode=modes.resolve_mode("MAIN"),
        seconds_per_image=1.0,
        seconds_per_cell=1e6,
        measured_pool_size=1000,
        ablation_specs=18,
        budget_seconds=0.0,
    )
    assert estimate.within_budget
    estimate.enforce()


def test_cell_cost_scales_superlinearly_in_pool_size() -> None:
    doubled = pipeline.scale_cell_seconds(10.0, measured_pool=1000, target_pool=2000)
    assert doubled > 20.0


def test_split_disjoint_partitions_without_overlap() -> None:
    pools = export_cache.split_disjoint(
        [f"img{index}" for index in range(30)],
        counts={"reference": 10, "pilot": 5, "evaluation": 12},
        seed=1,
    )
    assert [len(pools[name]) for name in ("reference", "pilot", "evaluation")] == [10, 5, 12]
    assert not set(pools["pilot"]) & set(pools["evaluation"])
    assert not set(pools["reference"]) & set(pools["evaluation"])
    assert pools == export_cache.split_disjoint(
        [f"img{index}" for index in reversed(range(30))],
        counts={"reference": 10, "pilot": 5, "evaluation": 12},
        seed=1,
    )


def test_split_disjoint_refuses_to_overdraw() -> None:
    with pytest.raises(export_cache.ExportError, match="lists only"):
        export_cache.split_disjoint(["a", "b"], counts={"reference": 2, "evaluation": 2}, seed=0)


def test_export_fingerprint_changes_with_the_protocol_not_the_interpreter() -> None:
    base = export_cache.BridgeSettings(
        prob_repository="/prob", checkpoint="/ckpt.pth", data_root="/data"
    )
    same = replace(base, python_executable="/other/python", num_workers=8)
    different = replace(base, current_introduced_classes=20)
    assert base.fingerprint(checkpoint_digest="x") == same.fingerprint(checkpoint_digest="x")
    assert base.fingerprint(checkpoint_digest="x") != different.fingerprint(checkpoint_digest="x")
    assert base.fingerprint(checkpoint_digest="x") != base.fingerprint(checkpoint_digest="y")


def test_predict_command_carries_every_protocol_flag() -> None:
    command = export_cache.BridgeSettings(
        prob_repository="/prob", checkpoint="/ckpt.pth", data_root="/data"
    ).predict_command()
    for fragment in (
        "daowod_prob_bridge.py predict",
        "--image-ids {image_ids}",
        "--checkpoint {checkpoint}",
        "--output {proposals}",
        "--dataset OWDETR",
        "--current-introduced-classes 19",
        "--num-classes 81",
        "--max-proposals-per-image 100",
    ):
        assert fragment in command


def test_chunking_is_deterministic_and_deduplicated() -> None:
    chunks = export_cache.chunk_image_ids(["b", "a", "b", "c", "d"], chunk_images=2)
    assert chunks == [["a", "b"], ["c", "d"]]


def test_paired_contrasts_are_computed_per_seed() -> None:
    auc_rows = [
        {"strategy": "v2:full", "seed": 0, "imbalance_setting": "n", "tail_discovery_auc": 0.5},
        {"strategy": "v2:random", "seed": 0, "imbalance_setting": "n", "tail_discovery_auc": 0.2},
        {"strategy": "v2:full", "seed": 1, "imbalance_setting": "n", "tail_discovery_auc": 0.6},
        {"strategy": "v2:random", "seed": 1, "imbalance_setting": "n", "tail_discovery_auc": 0.1},
    ]
    contrasts = reporting.headline_contrasts(auc_rows)
    row = next(item for item in contrasts if item["comparison"] == "gate vs random")
    assert row["paired_seeds"] == 2
    assert row["mean_difference"] == pytest.approx(0.4)
    assert row["all_seeds_positive"] is True


def test_cost_to_reach_reports_a_miss_as_a_miss() -> None:
    rows = [
        {
            "strategy": "s",
            "seed": 0,
            "imbalance_setting": "n",
            "budget": budget,
            "tail_discovery_recall": recall,
        }
        for budget, recall in ((10, 0.1), (20, 0.2))
    ]
    result = reporting.budget_to_reach(rows, target_recall=0.5)
    assert result[0]["reached"] is False
    assert result[0]["budget_to_reach"] is None
    reached = reporting.budget_to_reach(rows, target_recall=0.15)
    assert reached[0]["budget_to_reach"] == 20


def test_summary_states_a_negative_verdict_when_the_gate_loses() -> None:
    contrasts = [
        {
            "imbalance_setting": "n",
            "comparison": "gate vs ungated rarity",
            "metric": "tail_discovery_auc",
            "paired_seeds": 3,
            "mean_difference": -0.1,
            "sd_difference": 0.01,
            "all_seeds_positive": False,
            "all_seeds_negative": True,
        }
    ]
    text = reporting.research_summary(
        mode={"name": "FAST", "description": "d", "research_grade": False},
        pool_report={},
        composition={},
        severity_rows=[],
        auc_rows=[],
        curve_rows=[],
        contrasts=contrasts,
        gate_rows=[],
        cost_rows=[],
        leakage={"components_rebuild_score": True},
        runtime={"total": 1.0},
    )
    assert "not supported" in text
    assert "Not a reportable result" in text


def test_bundle_excludes_the_proposal_cache(tmp_path: Path) -> None:
    (tmp_path / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "figure.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "chunk_0000.npz").write_bytes(b"not a real npz")
    archive = reporting.bundle(tmp_path, archive=tmp_path / "out.zip")
    import zipfile

    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
    assert "table.csv" in names and "figure.png" in names
    assert "chunk_0000.npz" not in names


def test_expected_files_are_verified_not_assumed(tmp_path: Path) -> None:
    (tmp_path / "present.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    rows = reporting.verify_expected_files(tmp_path, ["present.csv", "empty.csv", "absent.csv"])
    statuses = {row["artifact"]: row["status"] for row in rows}
    assert statuses == {"present.csv": "PASS", "empty.csv": "FAIL", "absent.csv": "FAIL"}


def test_strategy_colours_are_keyed_by_entity_not_position() -> None:
    """Filtering the strategy set must not repaint the survivors."""

    for strategy in annotation_study.PRIMARY_STRATEGIES:
        assert strategy in plots.STRATEGY_COLOURS
    assert len(set(plots.STRATEGY_COLOURS.values())) == len(plots.STRATEGY_COLOURS)


def test_dataset_staging_copies_only_what_is_needed_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "drive"
    (source / "Annotations").mkdir(parents=True)
    (source / "JPEGImages").mkdir(parents=True)
    (source / "ImageSets" / "OWDETR").mkdir(parents=True)
    for index in range(4):
        (source / "Annotations" / f"img{index}.xml").write_text("<annotation/>", encoding="utf-8")
        (source / "JPEGImages" / f"img{index}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (source / "ImageSets" / "OWDETR" / "split.txt").write_text("img0\nimg1\n", encoding="utf-8")

    destination = tmp_path / "local"
    report = export_cache.stage_dataset(
        source=source, destination=destination, image_ids=["img0", "img1"], dataset="OWDETR"
    )
    assert report["images"] == 2
    assert report["files_copied"] == 4  # two annotations, two images
    assert (destination / "JPEGImages" / "img0.jpg").exists()
    assert not (destination / "JPEGImages" / "img2.jpg").exists()
    assert report["splits"] == ["split.txt"]

    again = export_cache.stage_dataset(
        source=source, destination=destination, image_ids=["img0", "img1"], dataset="OWDETR"
    )
    assert again["files_copied"] == 0
    assert again["files_already_present"] == 4


def test_dataset_staging_names_the_missing_file(tmp_path: Path) -> None:
    source = tmp_path / "drive"
    (source / "Annotations").mkdir(parents=True)
    (source / "JPEGImages").mkdir(parents=True)
    (source / "Annotations" / "img0.xml").write_text("<annotation/>", encoding="utf-8")
    with pytest.raises(export_cache.ExportError, match="missing at the source"):
        export_cache.stage_dataset(
            source=source, destination=tmp_path / "local", image_ids=["img0"], dataset="OWDETR"
        )


def _coherence_rows(tail: float, background: float) -> list[dict[str, object]]:
    return [
        {"strategy": "v2:full", "component": "coherence", "stratum": stratum, "median": median}
        for stratum, median in (
            ("true_tail", tail),
            ("true_head", 0.46),
            ("background", background),
            ("isolated_outlier", 0.11),
        )
    ]


def _summary_with(rows: list[dict[str, object]]) -> str:
    return reporting.research_summary(
        mode={"name": "MAIN", "description": "d", "research_grade": True},
        pool_report={},
        composition={},
        severity_rows=[],
        auc_rows=[],
        curve_rows=[],
        contrasts=[],
        gate_rows=[{"imbalance_setting": "natural", "seed": 0, "suppressed_isolated": 166}],
        cost_rows=[],
        leakage={},
        runtime={},
        distribution_rows=rows,
    )


def test_coherence_separation_diagnoses_a_coherent_background() -> None:
    """A negative contrast must come with the reason, not just the number."""

    rows = _coherence_rows(tail=0.44, background=0.56)
    report = reporting.coherence_separation(rows)
    assert report["available"] is True
    assert report["separates_tail_from_isolated_outliers"] is True
    assert report["separates_tail_from_background"] is False
    assert report["tail_indistinguishable_from_background"] is False
    assert "Background is at least as coherent as tail regions" in _summary_with(rows)


def test_a_one_percent_coherence_difference_is_not_a_separation() -> None:
    """The measured case: tail 0.559 versus background 0.553 is noise, not signal."""

    rows = _coherence_rows(tail=0.559, background=0.553)
    report = reporting.coherence_separation(rows)
    assert report["separates_tail_from_background"] is False
    assert report["tail_indistinguishable_from_background"] is True
    text = _summary_with(rows)
    assert "indistinguishable in coherence" in text


def test_a_real_separation_is_reported_as_one() -> None:
    rows = _coherence_rows(tail=0.80, background=0.40)
    report = reporting.coherence_separation(rows)
    assert report["separates_tail_from_background"] is True
    assert report["tail_indistinguishable_from_background"] is False
    assert "more coherent than background" in _summary_with(rows)


def test_coherence_separation_is_absent_when_nothing_was_recorded() -> None:
    assert reporting.coherence_separation([]) == {"available": False}
