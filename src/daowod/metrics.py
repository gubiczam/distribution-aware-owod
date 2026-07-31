"""OWOD detection metrics, including head / medium / tail long-tail diagnostics.

The official PROB evaluator remains the source of truth for aggregate
``known_mAP``, ``U_Recall``, ``WI`` and ``A_OSE``. This module adds the grouped
long-tail metrics the research question needs, computed from an explicit
detections artifact so that nothing depends on an unreachable code path.

Per-group AP uses *ignore* semantics: when scoring the head group, ground-truth
objects of the medium and tail groups are neither targets nor false positives.
Counting them as false positives would make per-group AP depend on the other
groups' prevalence, which is exactly the confound these metrics exist to remove.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from daowod.groups import GROUP_NAMES, ClassGroups, GroupError

UNKNOWN_PREDICTION_NAME = "unknown"


@dataclass(frozen=True)
class GroundTruth:
    image_id: str
    class_name: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    image_id: str
    class_name: str
    score: float
    box: tuple[float, float, float, float]


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection over union of two xyxy boxes."""

    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


@dataclass(frozen=True)
class MatchResult:
    """Greedy score-ordered matching of detections against target ground truth."""

    matched: np.ndarray
    true_positive: np.ndarray
    false_positive: np.ndarray
    ignored: int

    @property
    def target_count(self) -> int:
        return int(self.matched.size)

    @property
    def matched_count(self) -> int:
        return int(self.matched.sum())

    def recall(self) -> float:
        if self.target_count == 0:
            return float("nan")
        return self.matched_count / self.target_count

    def average_precision(self) -> float:
        """All-point interpolated area under the precision-recall curve."""

        if self.target_count == 0:
            return float("nan")
        if self.true_positive.size == 0:
            return 0.0
        cumulative_tp = np.cumsum(self.true_positive)
        cumulative_fp = np.cumsum(self.false_positive)
        recall = cumulative_tp / self.target_count
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        # Monotone envelope, then exact area of the step function.
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        recall = np.concatenate(([0.0], recall))
        return float(np.sum(np.diff(recall) * precision))


def match_detections(
    detections: Sequence[Detection],
    target: Sequence[GroundTruth],
    *,
    ignore: Sequence[GroundTruth] = (),
    iou_threshold: float = 0.5,
    class_aware: bool = False,
) -> MatchResult:
    """Match score-ordered detections to targets, absorbing ignore-region hits.

    ``class_aware`` requires the detection class to equal the ground-truth class;
    it is used for known-class metrics. Unknown-side metrics are class-agnostic
    because every unknown ground-truth object may only be recalled as
    ``unknown``.
    """

    ordered = sorted(detections, key=lambda item: (-item.score, item.image_id))
    matched = np.zeros(len(target), dtype=np.bool_)
    true_positive = np.zeros(len(ordered), dtype=np.float64)
    false_positive = np.zeros(len(ordered), dtype=np.float64)
    ignored = 0

    target_by_image: dict[str, list[int]] = {}
    for index, item in enumerate(target):
        target_by_image.setdefault(item.image_id, []).append(index)
    ignore_by_image: dict[str, list[GroundTruth]] = {}
    for item in ignore:
        ignore_by_image.setdefault(item.image_id, []).append(item)

    for position, prediction in enumerate(ordered):
        candidates = [
            index
            for index in target_by_image.get(prediction.image_id, ())
            if not matched[index]
            and (not class_aware or target[index].class_name == prediction.class_name)
        ]
        best_index, best_iou = None, 0.0
        for index in candidates:
            overlap = box_iou(prediction.box, target[index].box)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None and best_iou >= iou_threshold:
            matched[best_index] = True
            true_positive[position] = 1.0
            continue

        absorbed = any(
            box_iou(prediction.box, item.box) >= iou_threshold
            and (not class_aware or item.class_name == prediction.class_name)
            for item in ignore_by_image.get(prediction.image_id, ())
        )
        if absorbed:
            ignored += 1
        else:
            false_positive[position] = 1.0

    return MatchResult(
        matched=matched,
        true_positive=true_positive,
        false_positive=false_positive,
        ignored=ignored,
    )


def grouped_unknown_recall(
    ground_truth: Sequence[GroundTruth],
    detections: Sequence[Detection],
    *,
    unknown_classes: Sequence[str],
    class_groups: Mapping[str, str],
    unknown_prediction_name: str = UNKNOWN_PREDICTION_NAME,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """U-Recall split by head / medium / tail unknown class (legacy signature)."""

    unknown_set = set(unknown_classes)
    gt_unknown = [item for item in ground_truth if item.class_name in unknown_set]
    predictions = [item for item in detections if item.class_name == unknown_prediction_name]
    overall = match_detections(
        predictions, gt_unknown, iou_threshold=iou_threshold, class_aware=False
    )

    result: dict[str, float] = {}
    for group in GROUP_NAMES:
        indices = [
            index
            for index, target in enumerate(gt_unknown)
            if class_groups.get(target.class_name) == group
        ]
        result[f"U_Recall_{group}"] = (
            float(overall.matched[indices].mean()) if indices else float("nan")
        )
    result["U_Recall_grouped"] = overall.recall()
    return result


def grouped_detection_metrics(
    ground_truth: Sequence[GroundTruth],
    detections: Sequence[Detection],
    *,
    unknown_classes: Sequence[str],
    class_groups: ClassGroups,
    known_classes: Sequence[str] = (),
    known_class_groups: ClassGroups | None = None,
    unknown_prediction_name: str = UNKNOWN_PREDICTION_NAME,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Full grouped long-tail metric block for one evaluation.

    Unknown side (always computed when unknown ground truth exists):
      ``U_Recall_{group}``      recall of that group's unknown objects
      ``unknown_AP50_{group}``  AP of ``unknown`` detections on that group
      ``unknown_gt_{group}`` / ``unknown_matched_{group}`` support counts

    Known side (only when ``known_class_groups`` is supplied):
      ``Recall_{group}`` / ``AP50_{group}`` class-aware, over known classes.
    """

    unknown_set = set(unknown_classes)
    gt_unknown = [item for item in ground_truth if item.class_name in unknown_set]
    if gt_unknown:
        class_groups.require_covers(
            (item.class_name for item in gt_unknown),
            context="grouped unknown metrics",
        )
    unknown_predictions = [
        item for item in detections if item.class_name == unknown_prediction_name
    ]

    metrics: dict[str, object] = {
        "iou_threshold": float(iou_threshold),
        "unknown_prediction_name": unknown_prediction_name,
        "class_group_source": class_groups.source,
        "unknown_gt_total": len(gt_unknown),
        "unknown_detections_total": len(unknown_predictions),
    }

    overall = match_detections(
        unknown_predictions, gt_unknown, iou_threshold=iou_threshold, class_aware=False
    )
    metrics["U_Recall_grouped"] = overall.recall()
    metrics["unknown_AP50_grouped"] = overall.average_precision()

    for group in GROUP_NAMES:
        members = set(class_groups.members(group))
        target = [item for item in gt_unknown if item.class_name in members]
        ignore = [item for item in gt_unknown if item.class_name not in members]
        result = match_detections(
            unknown_predictions,
            target,
            ignore=ignore,
            iou_threshold=iou_threshold,
            class_aware=False,
        )
        metrics[f"U_Recall_{group}"] = result.recall()
        metrics[f"unknown_AP50_{group}"] = result.average_precision()
        metrics[f"unknown_gt_{group}"] = result.target_count
        metrics[f"unknown_matched_{group}"] = result.matched_count
        metrics[f"unknown_ignored_{group}"] = result.ignored
        metrics[f"unknown_classes_{group}"] = len(members)

    if known_class_groups is None:
        metrics["known_group_metrics"] = (
            "not defined: no known-class frequency-group mapping was supplied"
        )
        return metrics

    known_set = set(known_classes)
    gt_known = [item for item in ground_truth if item.class_name in known_set]
    if gt_known:
        known_class_groups.require_covers(
            (item.class_name for item in gt_known), context="grouped known metrics"
        )
    known_predictions = [
        item
        for item in detections
        if item.class_name != unknown_prediction_name and item.class_name in known_set
    ]
    metrics["known_group_metrics"] = "computed"
    metrics["known_gt_total"] = len(gt_known)
    for group in GROUP_NAMES:
        members = set(known_class_groups.members(group))
        target = [item for item in gt_known if item.class_name in members]
        ignore = [item for item in gt_known if item.class_name not in members]
        result = match_detections(
            known_predictions,
            target,
            ignore=ignore,
            iou_threshold=iou_threshold,
            class_aware=True,
        )
        metrics[f"Recall_{group}"] = result.recall()
        metrics[f"AP50_{group}"] = result.average_precision()
        metrics[f"known_gt_{group}"] = result.target_count
        metrics[f"known_matched_{group}"] = result.matched_count
    return metrics


class MetricConsistencyError(ValueError):
    """Raised when grouped metrics disagree with their own support counts."""


def validate_grouped_metrics(metrics: Mapping[str, object], *, tolerance: float = 1e-9) -> None:
    """Assert grouped metrics are internally consistent with the aggregate.

    Checks that (a) the per-group ground-truth counts partition the total,
    (b) every reported recall equals matched / support, and (c) a group with no
    support reports NaN rather than a number that would silently average in.
    """

    problems: list[str] = []
    for prefix, total_key, recall_key in (
        ("unknown", "unknown_gt_total", "U_Recall"),
        ("known", "known_gt_total", "Recall"),
    ):
        if total_key not in metrics:
            continue
        total = int(metrics[total_key])  # type: ignore[arg-type]
        counts = [metrics.get(f"{prefix}_gt_{group}") for group in GROUP_NAMES]
        if any(value is None for value in counts):
            continue
        grouped_total = sum(int(value) for value in counts)  # type: ignore[arg-type]
        if grouped_total != total:
            problems.append(
                f"{prefix}: per-group ground-truth counts sum to {grouped_total} "
                f"but {total_key} is {total}; a class was dropped or double counted."
            )
        for group in GROUP_NAMES:
            support = int(metrics[f"{prefix}_gt_{group}"])  # type: ignore[arg-type]
            matched = int(metrics[f"{prefix}_matched_{group}"])  # type: ignore[arg-type]
            recall = float(metrics[f"{recall_key}_{group}"])  # type: ignore[arg-type]
            if support == 0:
                if not math.isnan(recall):
                    problems.append(
                        f"{prefix} {group}: no ground truth but recall is {recall}; "
                        "empty groups must report NaN."
                    )
                continue
            if matched > support:
                problems.append(f"{prefix} {group}: matched {matched} exceeds support {support}.")
            expected = matched / support
            if not math.isclose(recall, expected, rel_tol=1e-9, abs_tol=tolerance):
                problems.append(f"{prefix} {group}: recall {recall} != matched/support {expected}.")
    if problems:
        raise MetricConsistencyError("; ".join(problems))


def load_detection_json(
    path: str | Path,
) -> tuple[list[GroundTruth], list[Detection]]:
    """Load the standard detection JSON emitted by the PROB bridge."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("ground_truth", "detections"):
        if key not in raw:
            raise ValueError(f"{path}: detection JSON has no {key!r} entry.")
    ground_truth = [
        GroundTruth(
            image_id=str(item["image_id"]),
            class_name=str(item["class_name"]),
            box=tuple(float(value) for value in item["box"]),
        )
        for item in raw["ground_truth"]
    ]
    detections = [
        Detection(
            image_id=str(item["image_id"]),
            class_name=str(item["class_name"]),
            score=float(item["score"]),
            box=tuple(float(value) for value in item["box"]),
        )
        for item in raw["detections"]
    ]
    return ground_truth, detections


def require_consistent_category_space(
    ground_truth: Sequence[GroundTruth],
    detections: Sequence[Detection],
    *,
    known_classes: Sequence[str],
    unknown_classes: Sequence[str],
    unknown_prediction_name: str = UNKNOWN_PREDICTION_NAME,
) -> dict[str, object]:
    """Verify detections and ground truth speak the same category language.

    A silent mismatch here (integer ids on one side, names on the other, or a
    different cocofication) produces zero recall that looks like a model result.
    """

    known_set, unknown_set = set(known_classes), set(unknown_classes)
    vocabulary = known_set | unknown_set
    gt_names = {item.class_name for item in ground_truth}
    detection_names = {item.class_name for item in detections}

    unmapped_gt = sorted(gt_names - vocabulary)
    unmapped_detections = sorted(detection_names - vocabulary - {unknown_prediction_name})
    if unmapped_gt:
        raise GroupError(
            f"Ground-truth classes outside the declared category space: {unmapped_gt[:20]}"
        )
    if unmapped_detections:
        raise GroupError(
            f"Detection classes outside the declared category space: {unmapped_detections[:20]}"
        )
    if gt_names and not (gt_names & vocabulary):
        raise GroupError(
            "No ground-truth class matches the declared category space; the "
            "detection and ground-truth category mappings differ."
        )
    return {
        "ground_truth_classes": len(gt_names),
        "detection_classes": len(detection_names),
        "known_classes_present": len(gt_names & known_set),
        "unknown_classes_present": len(gt_names & unknown_set),
        "unknown_predictions_present": unknown_prediction_name in detection_names,
    }
