"""VOC-style OWOD dataset handling and controlled long-tail pools."""

import json
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DatasetState:
    """Current labelled set and annotation pool."""

    labelled_ids: list[str]
    pool_ids: list[str]

    @classmethod
    def initialise(
        cls,
        image_ids: Sequence[str],
        *,
        initial_images: int,
        seed: int,
    ) -> "DatasetState":
        unique_ids = list(dict.fromkeys(str(image_id) for image_id in image_ids))
        if len(unique_ids) != len(image_ids):
            raise ValueError("image_ids must not contain duplicates.")
        if initial_images < 1 or initial_images > len(unique_ids):
            raise ValueError("Invalid initial_images value.")

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(unique_ids))
        labelled = [unique_ids[index] for index in order[:initial_images]]
        pool = [unique_ids[index] for index in order[initial_images:]]
        return cls(labelled_ids=labelled, pool_ids=pool)

    def reveal(self, selected_image_ids: Sequence[str]) -> None:
        """Move selected images from the pool to the labelled set."""

        selected = list(dict.fromkeys(str(value) for value in selected_image_ids))
        missing = set(selected) - set(self.pool_ids)
        if missing:
            raise ValueError(f"Images are not in the pool: {sorted(missing)}")
        selected_set = set(selected)
        self.labelled_ids.extend(selected)
        self.pool_ids = [image_id for image_id in self.pool_ids if image_id not in selected_set]


def read_image_ids(path: str | Path) -> list[str]:
    """Read a PROB/VOC ImageSets text file."""

    values = [
        line.strip().split()[0]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return list(dict.fromkeys(values))


def read_voc_classes(image_id: str, annotations_dir: str | Path) -> list[str]:
    """Return class names found in one VOC XML annotation."""

    path = Path(annotations_dir) / f"{image_id}.xml"
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation: {path}")

    root = ET.parse(path).getroot()
    return [str(node.text).strip() for node in root.findall("./object/name") if node.text]


def unknown_class_counts(
    image_ids: Sequence[str],
    annotations_dir: str | Path,
    unknown_classes: Sequence[str],
) -> dict[str, int]:
    """Count unknown-class objects in the candidate pool."""

    allowed = set(unknown_classes)
    counts: Counter[str] = Counter()
    for image_id in image_ids:
        counts.update(
            name for name in read_voc_classes(image_id, annotations_dir) if name in allowed
        )
    return dict(counts)


def frequency_groups(
    class_counts: Mapping[str, int],
    *,
    tail_max: int,
    head_min: int,
) -> dict[str, str]:
    """Assign unknown classes to head, medium and tail groups."""

    if tail_max < 0 or head_min <= tail_max:
        raise ValueError("Invalid frequency thresholds.")

    groups: dict[str, str] = {}
    for class_name, count in class_counts.items():
        if count <= tail_max:
            group = "tail"
        elif count >= head_min:
            group = "head"
        else:
            group = "medium"
        groups[class_name] = group
    return groups


def build_long_tail_pool(
    image_ids: Sequence[str],
    *,
    annotations_dir: str | Path,
    unknown_classes: Sequence[str],
    tail_max: int,
    head_min: int,
    head_retention: float,
    medium_retention: float,
    tail_retention: float,
    seed: int,
    manifest_path: str | Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Create a deterministic image-level long-tail pool.

    Every retained image keeps all annotations. An image is assigned to the
    rarest unknown-class group it contains, avoiding partial annotations that
    would silently turn omitted objects into background.
    """

    counts = unknown_class_counts(image_ids, annotations_dir, unknown_classes)
    groups = frequency_groups(counts, tail_max=tail_max, head_min=head_min)
    rates = {
        "head": head_retention,
        "medium": medium_retention,
        "tail": tail_retention,
    }
    if any(rate < 0 or rate > 1 for rate in rates.values()):
        raise ValueError("Retention rates must be in [0, 1].")

    priority = {"tail": 0, "medium": 1, "head": 2}
    rng = np.random.default_rng(seed)
    retained: list[str] = []
    image_groups: dict[str, str] = {}
    unknown_set = set(unknown_classes)

    for image_id in image_ids:
        present = [
            name
            for name in read_voc_classes(image_id, annotations_dir)
            if name in unknown_set and name in groups
        ]
        group = min((groups[name] for name in present), key=priority.get) if present else "head"
        image_groups[image_id] = group
        if rng.random() < rates[group]:
            retained.append(image_id)

    if manifest_path is not None:
        manifest = {
            "seed": seed,
            "class_counts": counts,
            "class_groups": groups,
            "image_groups": image_groups,
            "retained_image_ids": retained,
        }
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return retained, groups
