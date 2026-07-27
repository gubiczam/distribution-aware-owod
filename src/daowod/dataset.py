"""VOC-style OWOD dataset handling and controlled long-tail pools."""

import csv
import hashlib
import json
import random
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

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing image set: {path}")
    values = [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return list(dict.fromkeys(values))


def _annotation_path(image_id: str, annotations_dir: str | Path) -> Path:
    path = Path(annotations_dir) / f"{image_id}.xml"
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation: {path}")
    return path


def _read_voc_classes_from_path(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [str(node.text).strip() for node in root.findall("./object/name") if node.text]


def read_voc_classes(image_id: str, annotations_dir: str | Path) -> list[str]:
    """Return class names found in one VOC XML annotation."""

    return _read_voc_classes_from_path(_annotation_path(image_id, annotations_dir))


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


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_group(index: int, class_count: int) -> str:
    """Split ranked classes as [0, ceil(C/3)), [ceil(C/3), ceil(2C/3)), rest."""

    head_end = (class_count + 2) // 3
    medium_end = (2 * class_count + 2) // 3
    if index < head_end:
        return "head"
    if index < medium_end:
        return "medium"
    return "tail"


def _group_average(values: Mapping[str, int], class_names: Sequence[str]) -> float | None:
    if not class_names:
        return None
    return sum(values[name] for name in class_names) / len(class_names)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def build_long_tail_pool(
    annotation_dir: str | Path,
    source_split: str | Path,
    task_class_names: Sequence[str],
    output_dir: str | Path,
    imbalance_ratio: float = 50.0,
    seed: int = 0,
) -> dict[str, object]:
    """Create a deterministic image-level controlled long-tail pool."""

    if imbalance_ratio < 1:
        raise ValueError("imbalance_ratio must be >= 1.")

    task_classes = list(dict.fromkeys(str(name) for name in task_class_names))
    if len(task_classes) != len(task_class_names):
        raise ValueError("task_class_names must not contain duplicates.")
    if not task_classes:
        raise ValueError("task_class_names must not be empty.")

    split_path = Path(source_split)
    annotations_path = Path(annotation_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_ids = read_image_ids(split_path)
    task_set = set(task_classes)
    image_classes: dict[str, set[str]] = {}
    source_frequency = dict.fromkeys(task_classes, 0)
    eligible_ids: list[str] = []

    for image_id in image_ids:
        present = {
            class_name
            for class_name in _read_voc_classes_from_path(
                _annotation_path(image_id, annotations_path)
            )
            if class_name in task_set
        }
        image_classes[image_id] = present
        if not present:
            continue
        eligible_ids.append(image_id)
        for class_name in present:
            source_frequency[class_name] += 1

    class_order = sorted(task_classes, key=lambda name: (-source_frequency[name], name))
    max_target = max(source_frequency.values(), default=0)
    class_count = len(class_order)
    target_frequency: dict[str, int] = {}
    class_groups: dict[str, str] = {}
    grouped_classes = {"head": [], "medium": [], "tail": []}

    for rank, class_name in enumerate(class_order):
        source_count = source_frequency[class_name]
        if source_count == 0:
            target = 0
        else:
            scheduled = round(max_target * imbalance_ratio ** (-rank / max(class_count - 1, 1)))
            target = min(source_count, max(1, scheduled))
        target_frequency[class_name] = target
        group = _class_group(rank, class_count)
        class_groups[class_name] = group
        grouped_classes[group].append(class_name)

    rng = random.Random(seed)
    tie_priority = {image_id: rng.random() for image_id in eligible_ids}
    eligible_index = {image_id: index for index, image_id in enumerate(eligible_ids)}
    realised_frequency = dict.fromkeys(task_classes, 0)
    selected_set: set[str] = set()

    while True:
        best_id: str | None = None
        best_score: tuple[int, int, float, int] | None = None
        for image_id in eligible_ids:
            if image_id in selected_set:
                continue
            present = image_classes[image_id]
            contribution = sum(
                1
                for class_name in present
                if realised_frequency[class_name] < target_frequency[class_name]
            )
            if contribution == 0:
                continue
            overflow = sum(
                max(0, realised_frequency[class_name] + 1 - target_frequency[class_name])
                for class_name in present
            )
            score = (contribution, -overflow, -tie_priority[image_id], -eligible_index[image_id])
            if best_score is None or score > best_score:
                best_id = image_id
                best_score = score

        if best_id is None:
            break

        selected_set.add(best_id)
        for class_name in image_classes[best_id]:
            realised_frequency[class_name] += 1

    selected_ids = [image_id for image_id in image_ids if image_id in selected_set]

    pool_path = output_path / "pool_ids.txt"
    class_stats_path = output_path / "class_stats.csv"
    manifest_path = output_path / "protocol_manifest.json"

    pool_text = "\n".join(selected_ids)
    if selected_ids:
        pool_text += "\n"
    pool_path.write_text(pool_text, encoding="utf-8")

    with class_stats_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "class_name",
            "rank",
            "group",
            "source_frequency",
            "target_frequency",
            "realised_frequency",
            "absolute_error",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for rank, class_name in enumerate(class_order):
            writer.writerow(
                {
                    "class_name": class_name,
                    "rank": rank,
                    "group": class_groups[class_name],
                    "source_frequency": source_frequency[class_name],
                    "target_frequency": target_frequency[class_name],
                    "realised_frequency": realised_frequency[class_name],
                    "absolute_error": abs(
                        target_frequency[class_name] - realised_frequency[class_name]
                    ),
                }
            )

    requested_head = _group_average(target_frequency, grouped_classes["head"])
    requested_tail = _group_average(target_frequency, grouped_classes["tail"])
    realised_head = _group_average(realised_frequency, grouped_classes["head"])
    realised_tail = _group_average(realised_frequency, grouped_classes["tail"])
    manifest = {
        "protocol_version": 1,
        "source_split": str(split_path),
        "source_split_sha256": file_sha256(split_path),
        "annotation_dir": str(annotations_path),
        "seed": seed,
        "imbalance_ratio": imbalance_ratio,
        "source_image_count": len(image_ids),
        "eligible_image_count": len(eligible_ids),
        "excluded_image_count": len(image_ids) - len(eligible_ids),
        "selected_image_count": len(selected_ids),
        "task_class_count": len(task_classes),
        "class_order": class_order,
        "head_classes": grouped_classes["head"],
        "medium_classes": grouped_classes["medium"],
        "tail_classes": grouped_classes["tail"],
        "requested_head_to_tail_ratio": _ratio(requested_head, requested_tail),
        "realised_head_to_tail_ratio": _ratio(realised_head, realised_tail),
        "pool_ids_sha256": file_sha256(pool_path),
        "algorithm": "seeded_greedy_image_quota_v1",
        "class_group_rule": "ceil_thirds_contiguous_rank_groups",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "selected_image_ids": selected_ids,
        "class_groups": class_groups,
        "manifest_path": manifest_path,
        "class_stats_path": class_stats_path,
        "pool_split_path": pool_path,
    }
