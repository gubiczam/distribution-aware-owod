"""OWOD metric helpers, including head/medium/tail unknown recall."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


def grouped_unknown_recall(
    ground_truth: Sequence[GroundTruth],
    detections: Sequence[Detection],
    *,
    unknown_classes: Sequence[str],
    class_groups: Mapping[str, str],
    unknown_prediction_name: str = "unknown",
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute U-Recall separately for head, medium and tail unknowns."""

    unknown_set = set(unknown_classes)
    gt_unknown = [item for item in ground_truth if item.class_name in unknown_set]
    unknown_predictions = sorted(
        (item for item in detections if item.class_name == unknown_prediction_name),
        key=lambda item: item.score,
        reverse=True,
    )

    matched = np.zeros(len(gt_unknown), dtype=np.bool_)
    for prediction in unknown_predictions:
        candidates = [
            index
            for index, target in enumerate(gt_unknown)
            if not matched[index] and target.image_id == prediction.image_id
        ]
        if not candidates:
            continue
        best_index = max(
            candidates,
            key=lambda index: box_iou(prediction.box, gt_unknown[index].box),
        )
        if box_iou(prediction.box, gt_unknown[best_index].box) >= iou_threshold:
            matched[best_index] = True

    result: dict[str, float] = {}
    for group in ("head", "medium", "tail"):
        indices = [
            index
            for index, target in enumerate(gt_unknown)
            if class_groups.get(target.class_name) == group
        ]
        result[f"U_Recall_{group}"] = float(matched[indices].mean()) if indices else float("nan")
    result["U_Recall_grouped"] = float(matched.mean()) if matched.size else float("nan")
    return result


def load_detection_json(
    path: str | Path,
) -> tuple[list[GroundTruth], list[Detection]]:
    """Load the standard detection JSON emitted by the PROB bridge."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
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
