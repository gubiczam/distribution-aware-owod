"""Candidate-pool construction from a raw PROB export. Ground-truth free.

``daowod_prob_bridge predict`` writes every decoder query (PROB uses 100, and
``--max-proposals-per-image 100`` keeps all of them), so a raw export is ~85 %
background on real S-OWODB data. Scoring all of it wastes the annotation
budget's discriminative range on boxes no annotator would ever be shown, and it
is also what makes pseudo-class rarity unstable: the audit probe measured rarity
rank stability rising from 0.736 to 0.991 once the pool was restricted to
object-like proposals.

Every filter here is a function of PROB outputs only — objectness, unknown
score, predicted label, box geometry. No annotation is read. That is what makes
the pool a legitimate acquisition input rather than an oracle-assisted shortcut.

Filter choice is measured, not assumed
--------------------------------------
On the real 500-image Task-1 export (50,000 proposals, 322 of them on an unknown
object at IoU 0.5), per-image ``top-M`` selection retains true unknowns as
follows:

===================  ==========  =================  ===============
ranking signal       M           unknown retained   on-object rate
===================  ==========  =================  ===============
objectness           20          0.519              0.394
objectness           40          0.711              0.291
unknown score        20          0.438              0.265
unknown score        40          0.683              0.225
===================  ==========  =================  ===============

Objectness dominates the unknown score at equal pool size on both axes, which is
consistent with its higher AUC for "sits on an object" (0.879 vs 0.711), so
:data:`DEFAULT_RANKING` is ``objectness``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ObjectArray = NDArray[np.object_]

RankingSignal = Literal["objectness", "unknown_score"]
RANKING_SIGNALS: tuple[str, ...] = ("objectness", "unknown_score")

#: Measured best on real data; see the module docstring.
DEFAULT_RANKING: RankingSignal = "objectness"


class CandidateError(ValueError):
    """Raised when a candidate-pool specification is impossible or unsafe."""


@dataclass(frozen=True)
class CandidatePoolSpec:
    """Every decision that defines "what counts as a candidate proposal".

    ``require_unknown_prediction`` restricts the pool to proposals PROB itself
    labels unknown (``predicted_labels == num_classes - 1``).
    ``maximum_known_confidence`` additionally admits proposals whose best *known*
    posterior mass is low, which is the "low known-class certainty" half of the
    plan's candidate definition; set it to ``1.0`` to disable that relaxation.
    """

    ranking: RankingSignal = DEFAULT_RANKING
    per_image_limit: int = 20
    nms_iou_threshold: float = 0.7
    minimum_objectness: float = 0.0
    minimum_unknown_score: float = 0.0
    require_unknown_prediction: bool = False
    maximum_known_confidence: float = 1.0
    unknown_label: int = 80
    total_limit: int = 0

    def __post_init__(self) -> None:
        if self.ranking not in RANKING_SIGNALS:
            raise CandidateError(
                f"Unknown ranking signal {self.ranking!r}. Supported: {list(RANKING_SIGNALS)}"
            )
        if self.per_image_limit < 1:
            raise CandidateError("per_image_limit must be positive.")
        if not 0.0 < self.nms_iou_threshold <= 1.0:
            raise CandidateError("nms_iou_threshold must lie in (0, 1].")
        if not 0.0 <= self.minimum_objectness <= 1.0:
            raise CandidateError("minimum_objectness must lie in [0, 1].")
        if not 0.0 <= self.minimum_unknown_score <= 1.0:
            raise CandidateError("minimum_unknown_score must lie in [0, 1].")
        if not 0.0 < self.maximum_known_confidence <= 1.0:
            raise CandidateError("maximum_known_confidence must lie in (0, 1].")
        if self.total_limit < 0:
            raise CandidateError("total_limit must be non-negative (0 disables it).")

    def as_dict(self) -> dict[str, object]:
        return {
            "ranking": self.ranking,
            "per_image_limit": self.per_image_limit,
            "nms_iou_threshold": self.nms_iou_threshold,
            "minimum_objectness": self.minimum_objectness,
            "minimum_unknown_score": self.minimum_unknown_score,
            "require_unknown_prediction": self.require_unknown_prediction,
            "maximum_known_confidence": self.maximum_known_confidence,
            "unknown_label": self.unknown_label,
            "total_limit": self.total_limit,
        }


def class_agnostic_nms(
    boxes_cxcywh: ArrayLike, scores: ArrayLike, *, iou_threshold: float
) -> IntArray:
    """Greedy class-agnostic NMS returning kept indices, highest score first.

    Operates in the export's normalised coordinate space. IoU is scale-invariant
    under a common affine rescaling of both boxes, so per-image NMS in normalised
    ``cxcywh`` gives the same result as in pixels; no image size is needed and
    therefore no annotation is touched.
    """

    boxes = np.asarray(boxes_cxcywh, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_cxcywh must have shape (N, 4).")
    if values.shape != (boxes.shape[0],):
        raise ValueError("scores must be parallel to boxes_cxcywh.")
    if not 0.0 < iou_threshold <= 1.0:
        raise CandidateError("iou_threshold must lie in (0, 1].")
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    half_w, half_h = boxes[:, 2] / 2.0, boxes[:, 3] / 2.0
    x1, y1 = boxes[:, 0] - half_w, boxes[:, 1] - half_h
    x2, y2 = boxes[:, 0] + half_w, boxes[:, 1] + half_h
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-values, kind="stable")
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        inter_x1 = np.maximum(x1[current], x1[rest])
        inter_y1 = np.maximum(y1[current], y1[rest])
        inter_x2 = np.minimum(x2[current], x2[rest])
        inter_y2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
        union = areas[current] + areas[rest] - intersection
        iou = np.where(union > 0.0, intersection / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


@dataclass(frozen=True)
class CandidatePool:
    """Indices into the raw export that survived the candidate filters."""

    indices: IntArray
    spec: CandidatePoolSpec
    report: Mapping[str, object]

    @property
    def size(self) -> int:
        return int(self.indices.size)


def build_candidate_pool(
    *,
    image_ids: ArrayLike,
    boxes_cxcywh: ArrayLike,
    objectness: ArrayLike,
    unknown_score: ArrayLike,
    posterior: ArrayLike | None = None,
    predicted_labels: ArrayLike | None = None,
    spec: CandidatePoolSpec | None = None,
) -> CandidatePool:
    """Select the candidate proposals, using PROB outputs only.

    Order of operations, all ground-truth free:

    1. drop proposals below ``minimum_objectness`` / ``minimum_unknown_score``;
    2. optionally require an unknown prediction, or low known-class confidence;
    3. per image, run class-agnostic NMS on the ranking signal to remove
       duplicate boxes on the same object;
    4. per image, keep the top ``per_image_limit`` survivors;
    5. optionally thin the pool to ``total_limit`` by global ranking.

    The returned indices are sorted ascending so that downstream arrays keep the
    export's stable order, which makes every seeded run reproducible.
    """

    settings = spec or CandidatePoolSpec()
    ids = np.asarray([str(value) for value in np.asarray(image_ids, dtype=object)], dtype=object)
    boxes = np.asarray(boxes_cxcywh, dtype=np.float64)
    objectness_values = np.asarray(objectness, dtype=np.float64)
    unknown_values = np.asarray(unknown_score, dtype=np.float64)
    count = ids.shape[0]
    for name, array in (
        ("boxes_cxcywh", boxes),
        ("objectness", objectness_values),
        ("unknown_score", unknown_values),
    ):
        if array.shape[0] != count:
            raise CandidateError(f"{name} must be parallel to image_ids.")

    keep = np.ones(count, dtype=np.bool_)
    keep &= objectness_values >= settings.minimum_objectness
    keep &= unknown_values >= settings.minimum_unknown_score
    after_score = int(keep.sum())

    unknown_predicted: BoolArray | None = None
    if predicted_labels is not None:
        labels = np.asarray(predicted_labels, dtype=np.int64)
        if labels.shape != (count,):
            raise CandidateError("predicted_labels must be parallel to image_ids.")
        unknown_predicted = labels == settings.unknown_label

    low_known: BoolArray | None = None
    if posterior is not None and settings.maximum_known_confidence < 1.0:
        probabilities = np.asarray(posterior, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[0] != count:
            raise CandidateError("posterior must have shape (N, K).")
        totals = np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        normalised = probabilities / totals
        # The exported posterior is ordered [known..., unknown]; the last column
        # is PROB's unknown class, so the known mass is everything before it.
        best_known = (
            normalised[:, :-1].max(axis=1)
            if normalised.shape[1] > 1
            else np.zeros(count, dtype=np.float64)
        )
        low_known = best_known <= settings.maximum_known_confidence

    if settings.require_unknown_prediction:
        if unknown_predicted is None:
            raise CandidateError(
                "require_unknown_prediction needs predicted_labels from the export."
            )
        admitted = unknown_predicted if low_known is None else (unknown_predicted | low_known)
        keep &= admitted
    elif low_known is not None:
        keep &= low_known
    after_class = int(keep.sum())
    if after_class == 0:
        raise CandidateError(
            "The candidate filters removed every proposal. Relax "
            "minimum_objectness / require_unknown_prediction."
        )

    ranking = objectness_values if settings.ranking == "objectness" else unknown_values
    survivors = np.flatnonzero(keep)
    selected: list[IntArray] = []
    order = np.argsort(ids[survivors].astype(str), kind="stable")
    grouped = survivors[order]
    group_keys = ids[grouped].astype(str)
    boundaries = np.flatnonzero(np.r_[True, group_keys[1:] != group_keys[:-1]])
    for start, stop in zip(boundaries, np.r_[boundaries[1:], group_keys.size], strict=True):
        block = grouped[start:stop]
        kept_local = class_agnostic_nms(
            boxes[block], ranking[block], iou_threshold=settings.nms_iou_threshold
        )
        selected.append(block[kept_local[: settings.per_image_limit]])
    indices = np.sort(np.concatenate(selected)) if selected else np.zeros(0, dtype=np.int64)
    after_nms = int(indices.size)

    if settings.total_limit and indices.size > settings.total_limit:
        best = np.argsort(-ranking[indices], kind="stable")[: settings.total_limit]
        indices = np.sort(indices[best])

    report = {
        "raw_proposals": count,
        "raw_images": int(np.unique(ids).size),
        "after_score_thresholds": after_score,
        "after_class_filters": after_class,
        "after_nms_and_per_image_limit": after_nms,
        "final_pool": int(indices.size),
        "final_images": int(np.unique(ids[indices]).size) if indices.size else 0,
        "spec": settings.as_dict(),
    }
    return CandidatePool(indices=indices, spec=settings, report=report)


def pool_composition(match_kinds: ArrayLike, groups: ArrayLike | None = None) -> dict[str, float]:
    """Post-hoc composition of a pool: what the oracle says it actually holds.

    Diagnostic only — this reads oracle output and must never inform selection.
    """

    kinds = np.asarray(match_kinds, dtype=object)
    total = max(int(kinds.size), 1)
    report = {
        "proposals": int(kinds.size),
        "unknown_rate": float((kinds == "unknown").sum() / total),
        "known_rate": float((kinds == "known").sum() / total),
        "background_rate": float((kinds == "background").sum() / total),
    }
    if groups is not None:
        group_values = np.asarray(groups, dtype=object)
        for name in ("head", "medium", "tail"):
            report[f"{name}_rate"] = float((group_values == name).sum() / total)
    return report


def summarise_per_image(image_ids: ArrayLike) -> dict[str, float]:
    """Pool shape per image, used by the runtime estimate and the redundancy plot."""

    ids = np.asarray([str(value) for value in np.asarray(image_ids, dtype=object)], dtype=object)
    if ids.size == 0:
        return {"images": 0, "mean_per_image": 0.0, "max_per_image": 0.0}
    _, counts = np.unique(ids, return_counts=True)
    return {
        "images": int(counts.size),
        "mean_per_image": float(counts.mean()),
        "max_per_image": float(counts.max()),
    }


def deterministic_subset(image_ids: Sequence[str], *, limit: int, seed: int = 0) -> list[str]:
    """A reproducible image subset, used to size a run down to its time budget.

    Sampling is seeded and order-independent (IDs are sorted first), so the same
    ``(limit, seed)`` always yields the same images regardless of how the caller
    happened to order the split file.
    """

    unique = sorted(dict.fromkeys(str(value) for value in image_ids))
    if limit <= 0 or limit >= len(unique):
        return unique
    generator = np.random.default_rng(seed)
    chosen = generator.choice(len(unique), size=limit, replace=False)
    return [unique[index] for index in sorted(chosen.tolist())]
