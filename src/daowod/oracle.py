"""Proposal-level oracle: what object, if any, does a PROB proposal sit on?

The library previously joined ground truth only at *image* level
(the old image-level join said so in its own docstring: "a
proposal's box is not matched to an object here"). Every real-data claim in
``docs/`` therefore reduces to "the selected image contained a tail class
somewhere", which cannot distinguish an annotation spent on a tail object from
one spent on background in the same image. Contribution A is a statement about
*regions*, so the oracle has to be a region-level oracle.

Coordinate contract
-------------------
``daowod_prob_bridge.predict`` writes ``outputs["pred_boxes"]`` unchanged, i.e.
Deformable-DETR's **normalised cxcywh** in ``[0, 1]`` relative to the padded,
resized input. VOC XML boxes are absolute pixels with PROB's ``xmin/ymin -= 1``
convention (see ``OWDetection.load_instances``). :func:`boxes_to_pixel_xyxy`
performs exactly that conversion and nothing else; getting it wrong silently
turns every IoU into noise, so :func:`match_proposals` is covered by a test that
matches a proposal against a box derived from the same annotation.

Leakage contract
----------------
Nothing in this module may be called before or during scoring. The returned
:class:`OracleTable` carries only ``gt_``-prefixed fields precisely so that
:func:`assert_no_ground_truth`, defined at the bottom of this module, rejects it
if it is ever handed to an acquisition-time artifact. The guard lives here rather
than in a module of its own because it is the enforcement half of this module's
own contract: the oracle is the only source of ground truth, so the assertion
that ground truth has not leaked belongs beside it.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ObjectArray = NDArray[np.object_]

#: PROB's COCO-name -> VOC-name normalisation, copied from
#: ``datasets/torchvision_datasets/open_world.py`` (``VOC_CLASS_NAMES_COCOFIED``
#: -> ``BASE_VOC_CLASS_NAMES``). Without it "airplane" and "aeroplane" count as
#: two classes and the known/unknown split is wrong for six categories.
COCOFIED_TO_VOC: Mapping[str, str] = {
    "airplane": "aeroplane",
    "dining table": "diningtable",
    "motorcycle": "motorbike",
    "potted plant": "pottedplant",
    "couch": "sofa",
    "tv": "tvmonitor",
}

#: S-OWODB / OWDETR Task-1 known classes, i.e. ``T1_CLASS_NAMES`` upstream.
#: Everything else present in the annotations is unknown at Task 1.
OWDETR_TASK1_KNOWN: tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bus",
    "car",
    "cat",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "sheep",
    "train",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "truck",
    "person",
)

#: What a proposal was matched to. ``background`` means "no ground-truth object
#: reaches the IoU threshold", which is the class an isolated false positive
#: falls into.
MATCH_KINDS: tuple[str, ...] = ("unknown", "known", "background")


class OracleError(ValueError):
    """Raised when annotations are missing, malformed or inconsistent."""


def canonical_class_name(name: str) -> str:
    """Normalise one annotation class name to PROB's VOC vocabulary."""

    cleaned = str(name).strip()
    return COCOFIED_TO_VOC.get(cleaned, cleaned)


@dataclass(frozen=True)
class GroundTruthObject:
    """One annotated object in absolute pixel ``xyxy`` coordinates."""

    image_id: str
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    is_known: bool


@dataclass(frozen=True)
class ImageAnnotation:
    """Every annotated object in one image, plus the image size."""

    image_id: str
    width: int
    height: int
    objects: tuple[GroundTruthObject, ...]


def read_voc_annotation(
    image_id: str,
    annotations_dir: str | Path,
    *,
    known_classes: Sequence[str] = OWDETR_TASK1_KNOWN,
) -> ImageAnnotation:
    """Parse one VOC XML file with PROB's box and naming conventions.

    ``xmin``/``ymin`` are decremented by one to match
    ``OWDetection.load_instances``; ``xmax``/``ymax`` are used as written.
    """

    path = Path(annotations_dir) / f"{image_id}.xml"
    if not path.exists():
        raise OracleError(f"Missing annotation file: {path}")
    root = ElementTree.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise OracleError(f"{path}: annotation has no <size> element.")
    width = int(float(_require_text(size, "width", path)))
    height = int(float(_require_text(size, "height", path)))
    if width <= 0 or height <= 0:
        raise OracleError(f"{path}: non-positive image size {width}x{height}.")

    known = set(known_classes)
    objects: list[GroundTruthObject] = []
    for element in root.findall("object"):
        name_element = element.find("name")
        if name_element is None or not (name_element.text or "").strip():
            raise OracleError(f"{path}: <object> without a <name>.")
        class_name = canonical_class_name(name_element.text or "")
        box = element.find("bndbox")
        if box is None:
            raise OracleError(f"{path}: object {class_name!r} has no <bndbox>.")
        x_min = float(_require_text(box, "xmin", path)) - 1.0
        y_min = float(_require_text(box, "ymin", path)) - 1.0
        x_max = float(_require_text(box, "xmax", path))
        y_max = float(_require_text(box, "ymax", path))
        objects.append(
            GroundTruthObject(
                image_id=str(image_id),
                class_name=class_name,
                box_xyxy=(x_min, y_min, x_max, y_max),
                is_known=class_name in known,
            )
        )
    return ImageAnnotation(
        image_id=str(image_id),
        width=width,
        height=height,
        objects=tuple(objects),
    )


def _require_text(element: ElementTree.Element, tag: str, path: Path) -> str:
    child = element.find(tag)
    if child is None or child.text is None:
        raise OracleError(f"{path}: missing <{tag}>.")
    return child.text


def load_annotations(
    image_ids: Sequence[str],
    annotations_dir: str | Path,
    *,
    known_classes: Sequence[str] = OWDETR_TASK1_KNOWN,
) -> dict[str, ImageAnnotation]:
    """Parse the annotations for a set of images, failing loudly on gaps."""

    unique = list(dict.fromkeys(str(value) for value in image_ids))
    return {
        image_id: read_voc_annotation(image_id, annotations_dir, known_classes=known_classes)
        for image_id in unique
    }


def boxes_to_pixel_xyxy(
    boxes_cxcywh: ArrayLike, widths: ArrayLike, heights: ArrayLike
) -> FloatArray:
    """Convert normalised ``cxcywh`` proposals to absolute pixel ``xyxy``."""

    boxes = np.asarray(boxes_cxcywh, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_cxcywh must have shape (N, 4).")
    width = np.asarray(widths, dtype=np.float64)
    height = np.asarray(heights, dtype=np.float64)
    if width.shape != (boxes.shape[0],) or height.shape != (boxes.shape[0],):
        raise ValueError("widths and heights must be parallel to boxes_cxcywh.")
    centre_x, centre_y, box_w, box_h = boxes.T
    half_w, half_h = box_w / 2.0, box_h / 2.0
    return np.stack(
        [
            (centre_x - half_w) * width,
            (centre_y - half_h) * height,
            (centre_x + half_w) * width,
            (centre_y + half_h) * height,
        ],
        axis=1,
    )


def pairwise_iou(box: Sequence[float], targets: FloatArray) -> FloatArray:
    """IoU of one ``xyxy`` box against an ``(M, 4)`` array of ``xyxy`` boxes."""

    if targets.size == 0:
        return np.zeros(0, dtype=np.float64)
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    inter_x1 = np.maximum(x_min, targets[:, 0])
    inter_y1 = np.maximum(y_min, targets[:, 1])
    inter_x2 = np.minimum(x_max, targets[:, 2])
    inter_y2 = np.minimum(y_max, targets[:, 3])
    intersection = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    target_area = np.maximum(0.0, targets[:, 2] - targets[:, 0]) * np.maximum(
        0.0, targets[:, 3] - targets[:, 1]
    )
    union = area + target_area - intersection
    return np.where(union > 0.0, intersection / np.maximum(union, 1e-12), 0.0)


@dataclass(frozen=True)
class OracleTable:
    """Per-proposal oracle verdicts, aligned to the proposal arrays.

    ``gt_object_index`` indexes :attr:`objects` and is ``-1`` for background, so
    "how many *distinct* objects has this budget discovered" is a set count over
    non-negative entries rather than a proposal count. That distinction is the
    whole point of a region-level oracle: 40 proposals on one dog are one
    discovery, not forty.
    """

    gt_match_kind: ObjectArray
    gt_class: ObjectArray
    gt_group: ObjectArray
    gt_is_unknown: BoolArray
    gt_object_index: IntArray
    gt_best_iou: FloatArray
    objects: tuple[GroundTruthObject, ...]
    iou_threshold: float

    def __post_init__(self) -> None:
        sizes = {
            self.gt_match_kind.shape,
            self.gt_class.shape,
            self.gt_group.shape,
            self.gt_is_unknown.shape,
            self.gt_object_index.shape,
            self.gt_best_iou.shape,
        }
        if len(sizes) != 1:
            raise OracleError("OracleTable columns must be parallel vectors.")

    @property
    def proposal_count(self) -> int:
        return int(self.gt_match_kind.size)

    def unknown_object_indices(self) -> IntArray:
        """Indices of the unknown ground-truth objects, in :attr:`objects`."""

        return np.array(
            [index for index, item in enumerate(self.objects) if not item.is_known],
            dtype=np.int64,
        )

    def as_columns(self) -> dict[str, NDArray[np.generic]]:
        """The oracle as post-hoc, ``gt_``-prefixed dataframe columns."""

        return {
            "gt_match_kind": self.gt_match_kind,
            "gt_class": self.gt_class,
            "gt_group": self.gt_group,
            "gt_is_unknown": self.gt_is_unknown,
            "gt_object_index": self.gt_object_index,
            "gt_best_iou": self.gt_best_iou,
        }


def match_proposals(
    *,
    image_ids: ArrayLike,
    boxes_cxcywh: ArrayLike,
    annotations: Mapping[str, ImageAnnotation],
    class_groups: ClassGroups | None = None,
    iou_threshold: float = 0.5,
) -> OracleTable:
    """Match every proposal to its best ground-truth object by IoU.

    A proposal is assigned to the highest-IoU object in its own image provided
    that IoU reaches ``iou_threshold``; otherwise it is ``background``. Matching
    is deliberately *not* one-to-one: several proposals may point at the same
    object, because the metric that matters counts distinct discovered objects
    (see :attr:`OracleTable.gt_object_index`) and the number of proposals spent
    reaching them is exactly the annotation waste we want to measure.

    ``class_groups`` supplies head/medium/tail for unknown classes. Unknown
    classes absent from the mapping get group ``""`` rather than a guess, and
    :func:`assign_frequency_groups` exists to build the mapping from the pool
    itself.
    """

    ids = np.asarray([str(value) for value in np.asarray(image_ids, dtype=object)], dtype=object)
    pixel_boxes = _pixel_boxes_for(ids, boxes_cxcywh, annotations)
    if not 0.0 < iou_threshold <= 1.0:
        raise OracleError("iou_threshold must lie in (0, 1].")

    flat_objects: list[GroundTruthObject] = []
    object_slices: dict[str, tuple[int, FloatArray]] = {}
    for image_id, annotation in annotations.items():
        start = len(flat_objects)
        flat_objects.extend(annotation.objects)
        boxes = (
            np.asarray([item.box_xyxy for item in annotation.objects], dtype=np.float64)
            if annotation.objects
            else np.zeros((0, 4), dtype=np.float64)
        )
        object_slices[image_id] = (start, boxes)

    count = ids.shape[0]
    kinds = np.empty(count, dtype=object)
    classes = np.empty(count, dtype=object)
    groups = np.empty(count, dtype=object)
    is_unknown = np.zeros(count, dtype=np.bool_)
    object_index = np.full(count, -1, dtype=np.int64)
    best_iou = np.zeros(count, dtype=np.float64)
    group_map = class_groups.groups if class_groups is not None else {}

    for position in range(count):
        image_id = str(ids[position])
        start, boxes = object_slices[image_id]
        ious = pairwise_iou(pixel_boxes[position], boxes)
        if ious.size == 0:
            kinds[position], classes[position], groups[position] = "background", "", ""
            continue
        best = int(np.argmax(ious))
        best_iou[position] = float(ious[best])
        if ious[best] < iou_threshold:
            kinds[position], classes[position], groups[position] = "background", "", ""
            continue
        matched = flat_objects[start + best]
        object_index[position] = start + best
        classes[position] = matched.class_name
        if matched.is_known:
            kinds[position], groups[position] = "known", ""
            continue
        kinds[position] = "unknown"
        is_unknown[position] = True
        groups[position] = str(group_map.get(matched.class_name, ""))

    return OracleTable(
        gt_match_kind=kinds,
        gt_class=classes,
        gt_group=groups,
        gt_is_unknown=is_unknown,
        gt_object_index=object_index,
        gt_best_iou=best_iou,
        objects=tuple(flat_objects),
        iou_threshold=float(iou_threshold),
    )


def _pixel_boxes_for(
    ids: ObjectArray,
    boxes_cxcywh: ArrayLike,
    annotations: Mapping[str, ImageAnnotation],
) -> FloatArray:
    missing = sorted({str(value) for value in ids.tolist()} - set(annotations))
    if missing:
        raise OracleError(
            f"{len(missing)} proposal image(s) have no parsed annotation, e.g. {missing[:5]}."
        )
    widths = np.array([annotations[str(value)].width for value in ids.tolist()], dtype=np.float64)
    heights = np.array([annotations[str(value)].height for value in ids.tolist()], dtype=np.float64)
    return boxes_to_pixel_xyxy(boxes_cxcywh, widths, heights)


def unknown_class_counts(
    annotations: Mapping[str, ImageAnnotation],
) -> dict[str, int]:
    """Object-level frequency of every unknown class in the annotated images."""

    counts: dict[str, int] = {}
    for annotation in annotations.values():
        for item in annotation.objects:
            if item.is_known:
                continue
            counts[item.class_name] = counts.get(item.class_name, 0) + 1
    return counts


def reachable_class_counts(
    table: OracleTable, keep_mask: ArrayLike | None = None
) -> dict[str, int]:
    """Unknown-class frequency counted over *pool-reachable* objects.

    This, not the raw annotation frequency, is the distribution an acquisition
    function actually faces. Measured on the real 500-image Task-1 export: the
    annotations hold 50 unknown classes, but PROB's proposals reach only 22 of
    them, and the classes it misses are exactly the rarest — so head/medium/tail
    thirds taken over annotation frequency put almost every *reachable* object in
    "head" and leave the tail group with 2 objects, which cannot resolve a
    discovery curve. Grouping over reachable objects keeps all three groups
    populated and makes the long-tail structure the one the scorer is exposed to.
    """

    mask = (
        np.ones(table.proposal_count, dtype=np.bool_)
        if keep_mask is None
        else np.asarray(keep_mask, dtype=np.bool_)
    )
    selected = mask & table.gt_is_unknown
    seen: dict[str, set[int]] = {}
    for class_name, object_index in zip(
        table.gt_class[selected].tolist(), table.gt_object_index[selected].tolist(), strict=True
    ):
        name = str(class_name)
        if not name or int(object_index) < 0:
            continue
        seen.setdefault(name, set()).add(int(object_index))
    return {name: len(indices) for name, indices in seen.items()}


def with_class_groups(table: OracleTable, class_groups: ClassGroups) -> OracleTable:
    """Re-attach frequency groups to an already-matched oracle table.

    Matching has to happen before the groups can be derived from reachable
    objects, so the group column is filled in afterwards rather than the pipeline
    guessing a grouping up front.
    """

    groups = np.empty(table.proposal_count, dtype=object)
    mapping = class_groups.groups
    for position in range(table.proposal_count):
        if not bool(table.gt_is_unknown[position]):
            groups[position] = ""
            continue
        groups[position] = str(mapping.get(str(table.gt_class[position]), ""))
    return OracleTable(
        gt_match_kind=table.gt_match_kind,
        gt_class=table.gt_class,
        gt_group=groups,
        gt_is_unknown=table.gt_is_unknown,
        gt_object_index=table.gt_object_index,
        gt_best_iou=table.gt_best_iou,
        objects=table.objects,
        iou_threshold=table.iou_threshold,
    )


def assign_frequency_groups(
    class_counts: Mapping[str, int],
    *,
    head_fraction: float = 1 / 3,
    tail_fraction: float = 1 / 3,
    source: str = "unknown class object frequency",
) -> ClassGroups:
    """Split unknown classes into head/medium/tail by descending frequency.

    Classes are ordered by count (ties broken by name, so the split is
    deterministic) and cut into contiguous thirds *by class count*, not by object
    count: the research question is about how many distinct rare classes an
    annotation budget reaches, so each group must contain a comparable number of
    classes. Fractions are configurable because the plan fixes the existence of
    three groups, not their boundaries.
    """

    if not class_counts:
        raise OracleError("Cannot assign frequency groups to an empty class set.")
    if head_fraction <= 0 or tail_fraction <= 0 or head_fraction + tail_fraction >= 1:
        raise OracleError(
            "head_fraction and tail_fraction must be positive and leave a non-empty medium group."
        )
    ordered = sorted(class_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    total = len(ordered)
    head_size = max(1, int(round(total * head_fraction)))
    tail_size = max(1, int(round(total * tail_fraction)))
    if head_size + tail_size >= total:
        head_size = max(1, total // 3)
        tail_size = max(1, total // 3)
        if head_size + tail_size >= total and total >= 3:
            head_size = tail_size = 1
    mapping: dict[str, str] = {}
    for position, (class_name, _) in enumerate(ordered):
        if position < head_size:
            mapping[class_name] = "head"
        elif position >= total - tail_size:
            mapping[class_name] = "tail"
        else:
            mapping[class_name] = "medium"
    present = {group for group in mapping.values()}
    if total >= 3 and present != set(GROUP_NAMES):
        raise OracleError(
            f"Frequency grouping produced only {sorted(present)}; expected {list(GROUP_NAMES)}."
        )
    return ClassGroups.from_mapping(mapping, source=source)


# --- leakage guard -----------------------------------------------------------
#
# Salvaged from the deleted `diagnostics` module, which was otherwise
# synthetic-pool instrumentation. This is the automated half of the ground-truth
# discipline documented in `docs/reproduction.md` section 3.

#: Field names that may only ever appear in a *post hoc* artifact. Any `gt_`
#: prefix is rejected too, so a new oracle field cannot silently bypass the guard
#: by not being on this list.
GROUND_TRUTH_FIELDS: tuple[str, ...] = (
    "gt_class",
    "gt_classes",
    "gt_group",
    "gt_unknown",
    "ground_truth",
    "true_class",
    "label",
)


class LeakageError(AssertionError):
    """Raised when an acquisition-time artifact contains ground truth."""


def assert_no_ground_truth(rows: Sequence[Mapping[str, object]]) -> None:
    """Reject acquisition-time records that carry oracle information.

    Cheap and name-based: it proves that no ground-truth *field* reached the
    artifact, not that the score is free of oracle influence. The strong check is
    :func:`daowod.discovery.assert_selection_is_ground_truth_free`, which
    re-derives every score from its recorded components and so constrains
    arithmetic rather than naming. Both run in the pipeline.
    """

    if not rows:
        return
    present = sorted(set().union(*(set(row) for row in rows)))
    offending = [
        field for field in present if field in GROUND_TRUTH_FIELDS or field.startswith("gt_")
    ]
    if offending:
        raise LeakageError(
            f"Acquisition-time proposal records contain ground-truth fields {offending}. "
            "Ground truth may only be joined after selection, in a post-hoc artifact."
        )


# =============================================================================
# Frequency groups (head / medium / tail)
#
# Merged here because group assignment is already an oracle operation:
# assign_frequency_groups above derives the groups from the oracle's own class
# counts. Groups are loaded once, validated loudly, then passed to the metric
# functions; no metric may invent a group or skip a class it cannot place.
# =============================================================================

GROUP_NAMES: tuple[str, ...] = ("head", "medium", "tail")


class GroupError(ValueError):
    """Raised when a frequency-group mapping is missing, ambiguous or partial."""


@dataclass(frozen=True)
class ClassGroups:
    """An immutable, validated ``class name -> frequency group`` mapping."""

    groups: Mapping[str, str]
    source: str

    def __post_init__(self) -> None:
        unknown = sorted({group for group in self.groups.values() if group not in GROUP_NAMES})
        if unknown:
            raise GroupError(
                f"{self.source}: unsupported frequency groups {unknown}; "
                f"expected only {list(GROUP_NAMES)}."
            )
        if not self.groups:
            raise GroupError(f"{self.source}: the frequency-group mapping is empty.")

    @classmethod
    def from_class_stats_csv(cls, path: str | Path) -> ClassGroups:
        """Load the mapping from a ``class_stats.csv`` long-tail artifact."""

        source = str(path)
        rows = _read_class_stats(Path(path), source)
        groups: dict[str, str] = {}
        for class_name, group in rows:
            previous = groups.get(class_name)
            if previous is not None and previous != group:
                raise GroupError(
                    f"{source}: class {class_name!r} is assigned to both "
                    f"{previous!r} and {group!r}; every class must have exactly "
                    "one group."
                )
            if previous is not None:
                raise GroupError(f"{source}: class {class_name!r} appears more than once.")
            groups[class_name] = group
        return cls(groups=groups, source=source)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str], *, source: str) -> ClassGroups:
        return cls(groups=dict(mapping), source=source)

    def group_of(self, class_name: str) -> str:
        try:
            return self.groups[class_name]
        except KeyError as error:
            raise GroupError(
                f"{self.source}: class {class_name!r} has no frequency group."
            ) from error

    def members(self, group: str) -> list[str]:
        if group not in GROUP_NAMES:
            raise GroupError(f"Unsupported frequency group: {group!r}")
        return sorted(name for name, value in self.groups.items() if value == group)

    def require_covers(self, class_names: Iterable[str], *, context: str) -> None:
        """Fail loudly when an evaluated class cannot be placed in a group.

        Silently dropping such a class is the failure mode this guards against:
        it would make grouped recall look complete while quietly excluding part
        of the ground truth.
        """

        requested = sorted(set(class_names))
        missing = [name for name in requested if name not in self.groups]
        if missing:
            raise GroupError(
                f"{context}: {len(missing)} evaluated class(es) have no frequency "
                f"group in {self.source}: {missing}. Regenerate the long-tail "
                "protocol or pass the matching class_stats.csv."
            )

    def counts(self) -> dict[str, int]:
        return {
            group: sum(1 for value in self.groups.values() if value == group)
            for group in GROUP_NAMES
        }

    def as_dict(self) -> dict[str, str]:
        return dict(self.groups)


def _read_class_stats(path: Path, source: str) -> list[tuple[str, str]]:
    if not path.exists():
        raise GroupError(f"Missing long-tail class statistics: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        for required in ("class_name", "group"):
            if required not in fieldnames:
                raise GroupError(f"{source}: column {required!r} is missing; found {fieldnames}.")
        rows: list[tuple[str, str]] = []
        for number, row in enumerate(reader, start=2):
            class_name = (row.get("class_name") or "").strip()
            group = (row.get("group") or "").strip()
            if not class_name:
                raise GroupError(f"{source}: line {number} has an empty class_name.")
            if not group:
                raise GroupError(f"{source}: line {number} has no group for {class_name!r}.")
            rows.append((class_name, group))
    if not rows:
        raise GroupError(f"{source}: no class rows were found.")
    return rows
