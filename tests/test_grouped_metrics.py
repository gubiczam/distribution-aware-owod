"""Step 1 — grouped head / medium / tail long-tail metrics.

Covers the failure modes the audit identified: a group mapping that silently
drops an evaluated class, per-group AP that depends on the other groups'
prevalence, and detections whose category space does not match the ground truth.
"""

import json
from pathlib import Path

import pytest

from daowod.dataset import build_long_tail_pool
from daowod.groups import ClassGroups, GroupError
from daowod.metrics import (
    Detection,
    GroundTruth,
    MetricConsistencyError,
    grouped_detection_metrics,
    grouped_unknown_recall,
    load_detection_json,
    match_detections,
    require_consistent_category_space,
    validate_grouped_metrics,
)

UNKNOWN_CLASSES = ("aeroplane", "sofa", "toaster")
GROUPS = ClassGroups.from_mapping(
    {"aeroplane": "head", "sofa": "medium", "toaster": "tail"}, source="test-fixture"
)


def _ground_truth() -> list[GroundTruth]:
    return [
        GroundTruth("img1", "aeroplane", (0, 0, 10, 10)),
        GroundTruth("img1", "aeroplane", (20, 20, 30, 30)),
        GroundTruth("img2", "aeroplane", (0, 0, 10, 10)),
        GroundTruth("img2", "sofa", (20, 20, 30, 30)),
        GroundTruth("img3", "toaster", (0, 0, 10, 10)),
    ]


def _detections() -> list[Detection]:
    # The highest-scoring detection hits a *tail* object. When the head group is
    # scored it must be ignored, not counted as a head false positive.
    return [
        Detection("img3", "unknown", 0.95, (0, 0, 10, 10)),
        Detection("img1", "unknown", 0.90, (0, 0, 10, 10)),
        Detection("img1", "unknown", 0.50, (100, 100, 110, 110)),
    ]


def test_grouped_metrics_report_per_group_recall_and_support() -> None:
    metrics = grouped_detection_metrics(
        _ground_truth(),
        _detections(),
        unknown_classes=UNKNOWN_CLASSES,
        class_groups=GROUPS,
    )
    assert metrics["unknown_gt_total"] == 5
    assert metrics["unknown_gt_head"] == 3
    assert metrics["unknown_gt_medium"] == 1
    assert metrics["unknown_gt_tail"] == 1
    assert metrics["U_Recall_head"] == pytest.approx(1 / 3)
    assert metrics["U_Recall_medium"] == pytest.approx(0.0)
    assert metrics["U_Recall_tail"] == pytest.approx(1.0)
    assert metrics["U_Recall_grouped"] == pytest.approx(2 / 5)
    validate_grouped_metrics(metrics)


def test_per_group_ap_uses_ignore_semantics() -> None:
    """A detection matching another group's object must not deflate this group.

    Without ignore regions the head AP would be 1/6 instead of 1/3 purely
    because a tail object happened to be detected first.
    """
    ground_truth, detections = _ground_truth(), _detections()
    head = [item for item in ground_truth if item.class_name == "aeroplane"]
    others = [item for item in ground_truth if item.class_name != "aeroplane"]

    with_ignore = match_detections(detections, head, ignore=others)
    without_ignore = match_detections(detections, head)

    assert with_ignore.average_precision() == pytest.approx(1 / 3)
    assert without_ignore.average_precision() == pytest.approx(1 / 6)
    assert with_ignore.ignored == 1

    metrics = grouped_detection_metrics(
        ground_truth, detections, unknown_classes=UNKNOWN_CLASSES, class_groups=GROUPS
    )
    assert metrics["unknown_AP50_head"] == pytest.approx(1 / 3)
    assert metrics["unknown_ignored_head"] == 1


def test_empty_group_reports_nan_rather_than_zero() -> None:
    groups = ClassGroups.from_mapping(
        {"aeroplane": "head", "sofa": "head", "toaster": "head"}, source="all-head"
    )
    metrics = grouped_detection_metrics(
        _ground_truth(),
        _detections(),
        unknown_classes=UNKNOWN_CLASSES,
        class_groups=groups,
    )
    assert metrics["unknown_gt_medium"] == 0
    assert metrics["U_Recall_medium"] != metrics["U_Recall_medium"]  # NaN
    validate_grouped_metrics(metrics)


def test_missing_group_fails_loudly_instead_of_dropping_the_class() -> None:
    partial = ClassGroups.from_mapping({"aeroplane": "head", "sofa": "medium"}, source="partial")
    with pytest.raises(GroupError, match="toaster"):
        grouped_detection_metrics(
            _ground_truth(),
            _detections(),
            unknown_classes=UNKNOWN_CLASSES,
            class_groups=partial,
        )


def test_class_groups_reject_duplicate_and_unknown_groups(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("class_name,group\naeroplane,head\naeroplane,tail\n", encoding="utf-8")
    with pytest.raises(GroupError, match="exactly"):
        ClassGroups.from_class_stats_csv(duplicate)

    bad_group = tmp_path / "bad.csv"
    bad_group.write_text("class_name,group\naeroplane,enormous\n", encoding="utf-8")
    with pytest.raises(GroupError, match="unsupported frequency groups"):
        ClassGroups.from_class_stats_csv(bad_group)

    missing_column = tmp_path / "missing.csv"
    missing_column.write_text("class_name\naeroplane\n", encoding="utf-8")
    with pytest.raises(GroupError, match="group"):
        ClassGroups.from_class_stats_csv(missing_column)

    with pytest.raises(GroupError, match="Missing"):
        ClassGroups.from_class_stats_csv(tmp_path / "absent.csv")


def test_known_side_metrics_are_class_aware_and_opt_in() -> None:
    ground_truth = [
        GroundTruth("img1", "car", (0, 0, 10, 10)),
        GroundTruth("img1", "bus", (20, 20, 30, 30)),
    ]
    detections = [
        Detection("img1", "car", 0.9, (0, 0, 10, 10)),
        Detection("img1", "car", 0.8, (20, 20, 30, 30)),  # wrong class for the bus
    ]
    known_groups = ClassGroups.from_mapping({"car": "head", "bus": "tail"}, source="known-fixture")

    without = grouped_detection_metrics(
        ground_truth,
        detections,
        unknown_classes=(),
        class_groups=GROUPS,
        known_classes=("car", "bus"),
    )
    assert "Recall_head" not in without
    assert "not defined" in str(without["known_group_metrics"])

    with_known = grouped_detection_metrics(
        ground_truth,
        detections,
        unknown_classes=(),
        class_groups=GROUPS,
        known_classes=("car", "bus"),
        known_class_groups=known_groups,
    )
    assert with_known["Recall_head"] == pytest.approx(1.0)
    assert with_known["Recall_tail"] == pytest.approx(0.0)
    validate_grouped_metrics(with_known)


def test_validate_grouped_metrics_detects_inconsistency() -> None:
    metrics = grouped_detection_metrics(
        _ground_truth(),
        _detections(),
        unknown_classes=UNKNOWN_CLASSES,
        class_groups=GROUPS,
    )
    tampered = dict(metrics)
    tampered["unknown_gt_head"] = 2
    with pytest.raises(MetricConsistencyError, match="sum to"):
        validate_grouped_metrics(tampered)

    tampered = dict(metrics)
    tampered["U_Recall_tail"] = 0.5
    with pytest.raises(MetricConsistencyError, match="matched/support"):
        validate_grouped_metrics(tampered)


def test_category_space_mismatch_is_detected() -> None:
    ground_truth = _ground_truth()
    numeric = [Detection("img1", "42", 0.9, (0, 0, 10, 10))]
    with pytest.raises(GroupError, match="outside the declared category space"):
        require_consistent_category_space(
            ground_truth,
            numeric,
            known_classes=("car",),
            unknown_classes=UNKNOWN_CLASSES,
        )
    report = require_consistent_category_space(
        ground_truth,
        _detections(),
        known_classes=("car",),
        unknown_classes=UNKNOWN_CLASSES,
    )
    assert report["unknown_classes_present"] == 3
    assert report["unknown_predictions_present"] is True


def test_legacy_grouped_unknown_recall_is_preserved() -> None:
    legacy = grouped_unknown_recall(
        _ground_truth(),
        _detections(),
        unknown_classes=UNKNOWN_CLASSES,
        class_groups=GROUPS.as_dict(),
    )
    assert legacy["U_Recall_head"] == pytest.approx(1 / 3)
    assert legacy["U_Recall_tail"] == pytest.approx(1.0)
    assert legacy["U_Recall_grouped"] == pytest.approx(2 / 5)


def _write_voc(path: Path, classes: list[str]) -> None:
    objects = "".join(
        f"<object><name>{name}</name>"
        "<bndbox><xmin>1</xmin><ymin>1</ymin><xmax>11</xmax><ymax>11</ymax></bndbox>"
        "</object>"
        for name in classes
    )
    path.write_text(f"<annotation>{objects}</annotation>", encoding="utf-8")


def test_integration_long_tail_groups_drive_grouped_metrics(tmp_path: Path) -> None:
    """End to end: long-tail pool -> class_stats.csv -> detections -> metrics.

    This is the path the live campaign now uses, exercised on a fixture small
    enough to reason about by hand.
    """
    annotations = tmp_path / "Annotations"
    annotations.mkdir()
    layout = {
        "a1": ["aeroplane"],
        "a2": ["aeroplane"],
        "a3": ["aeroplane"],
        "s1": ["sofa"],
        "s2": ["sofa"],
        "t1": ["toaster"],
    }
    for image_id, classes in layout.items():
        _write_voc(annotations / f"{image_id}.xml", classes)
    split = tmp_path / "split.txt"
    split.write_text("\n".join(layout) + "\n", encoding="utf-8")

    pool = build_long_tail_pool(
        annotation_dir=annotations,
        source_split=split,
        task_class_names=list(UNKNOWN_CLASSES),
        output_dir=tmp_path / "pool",
        imbalance_ratio=3.0,
        seed=0,
    )
    groups = ClassGroups.from_class_stats_csv(pool["class_stats_path"])
    assert set(groups.as_dict()) == set(UNKNOWN_CLASSES)
    assert groups.counts() == {"head": 1, "medium": 1, "tail": 1}
    groups.require_covers(UNKNOWN_CLASSES, context="integration")

    artifact = tmp_path / "detections.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "daowod_detections_v1",
                "ground_truth": [
                    {
                        "image_id": item.image_id,
                        "class_name": item.class_name,
                        "box": list(item.box),
                    }
                    for item in _ground_truth()
                ],
                "detections": [
                    {
                        "image_id": item.image_id,
                        "class_name": item.class_name,
                        "score": item.score,
                        "box": list(item.box),
                    }
                    for item in _detections()
                ],
            }
        ),
        encoding="utf-8",
    )
    ground_truth, detections = load_detection_json(artifact)
    metrics = grouped_detection_metrics(
        ground_truth,
        detections,
        unknown_classes=UNKNOWN_CLASSES,
        class_groups=groups,
    )
    validate_grouped_metrics(metrics)
    assert metrics["U_Recall_tail"] == pytest.approx(1.0)
    assert metrics["class_group_source"] == str(pool["class_stats_path"])


def test_load_detection_json_rejects_incomplete_artifact(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"detections": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="ground_truth"):
        load_detection_json(path)
