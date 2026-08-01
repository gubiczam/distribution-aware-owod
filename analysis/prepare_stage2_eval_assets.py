#!/usr/bin/env python3
"""Stage the canonical OWDETR evaluation split and its local assets.

This script intentionally does not choose or rewrite the evaluation split. It
copies PROB's pinned `OWDETR/owdetr_test.txt` byte-for-byte into the staged data
root, generates VOC XML annotations from the local COCO val2017 annotation file
through PROB's official `datasets/coco2voc.py` conversion function, and symlinks
missing JPEGs from the local COCO val image directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-repo", type=Path, default=Path("/Users/gubiczam/Documents/PROB"))
    parser.add_argument("--data-root", type=Path, default=Path("/Users/gubiczam/owod_stage"))
    parser.add_argument(
        "--coco-val-annotations",
        type=Path,
        default=Path("/Users/gubiczam/Downloads/active_learning_data/instances_val2017.json"),
    )
    parser.add_argument(
        "--coco-val-images",
        type=Path,
        default=Path("/Users/gubiczam/Downloads/active_learning_data/images"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2_plan"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    values = [
        line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    deduped = list(dict.fromkeys(values))
    if len(values) != len(deduped):
        raise ValueError(f"Duplicate IDs in {path}")
    return deduped


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def import_official_converter(prob_repo: Path) -> Any:
    sys.path.insert(0, str(prob_repo / "datasets"))
    import coco2voc  # type: ignore[import-not-found]

    return coco2voc


def main() -> int:
    args = parse_args()
    official_split = args.prob_repo / "data" / "OWOD" / "ImageSets" / "OWDETR" / "owdetr_test.txt"
    staged_split = args.data_root / "ImageSets" / "OWDETR" / "owdetr_test.txt"
    generated_root = args.output_dir / "generated_eval_voc"
    generated_annotations = generated_root / "Annotations"
    staged_annotations = args.data_root / "Annotations"
    staged_images = args.data_root / "JPEGImages"

    for required in (
        args.prob_repo,
        official_split,
        args.coco_val_annotations,
        args.coco_val_images,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    staged_split.parent.mkdir(parents=True, exist_ok=True)
    staged_annotations.mkdir(parents=True, exist_ok=True)
    staged_images.mkdir(parents=True, exist_ok=True)

    if staged_split.exists() and sha256(staged_split) != sha256(official_split):
        raise RuntimeError(f"Refusing to overwrite divergent staged split: {staged_split}")
    shutil.copy2(official_split, staged_split)
    eval_ids = read_ids(staged_split)

    if not generated_annotations.exists():
        converter = import_official_converter(args.prob_repo)
        converter.coco_to_voc_detection(str(args.coco_val_annotations), str(generated_root))

    copied_annotations = 0
    linked_images = 0
    missing_generated_annotations: list[str] = []
    missing_source_images: list[str] = []
    for image_id in eval_ids:
        source_xml = generated_annotations / f"{image_id}.xml"
        target_xml = staged_annotations / f"{image_id}.xml"
        if not source_xml.exists():
            missing_generated_annotations.append(image_id)
        elif not target_xml.exists():
            shutil.copy2(source_xml, target_xml)
            copied_annotations += 1

        target_image = staged_images / f"{image_id}.jpg"
        source_image = args.coco_val_images / f"{image_id}.jpg"
        if not source_image.exists():
            missing_source_images.append(image_id)
        elif not target_image.exists():
            target_image.symlink_to(source_image)
            linked_images += 1

    missing_annotations = [
        image_id for image_id in eval_ids if not (staged_annotations / f"{image_id}.xml").exists()
    ]
    missing_images = [
        image_id for image_id in eval_ids if not (staged_images / f"{image_id}.jpg").exists()
    ]
    manifest = {
        "schema": "stage2_eval_asset_preparation_v1",
        "canonical_split": {
            "name": "owdetr_test",
            "source_path": str(official_split),
            "staged_path": str(staged_split),
            "image_count": len(eval_ids),
            "sha256": sha256(staged_split),
        },
        "official_source": {
            "prob_repo": str(args.prob_repo),
            "prob_commit": git_commit(args.prob_repo),
            "converter": str(args.prob_repo / "datasets" / "coco2voc.py"),
            "conversion_function": "coco_to_voc_detection",
            "source_annotations": str(args.coco_val_annotations),
            "source_annotations_sha256": sha256(args.coco_val_annotations),
            "source_images": str(args.coco_val_images),
        },
        "actions": {
            "copied_split": True,
            "generated_xml_root": str(generated_annotations),
            "copied_annotations": copied_annotations,
            "symlinked_images": linked_images,
        },
        "postcheck": {
            "missing_generated_annotations": missing_generated_annotations[:50],
            "missing_generated_annotations_count": len(missing_generated_annotations),
            "missing_source_images": missing_source_images[:50],
            "missing_source_images_count": len(missing_source_images),
            "missing_staged_annotations": missing_annotations[:50],
            "missing_staged_annotations_count": len(missing_annotations),
            "missing_staged_images": missing_images[:50],
            "missing_staged_images_count": len(missing_images),
        },
        "reproduction_command": (
            "PYTHONPATH=src /Users/gubiczam/Documents/PROB/.venv/bin/python "
            "analysis/prepare_stage2_eval_assets.py"
        ),
    }
    (args.output_dir / "evaluation_asset_preparation.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["postcheck"], indent=2, sort_keys=True))
    if missing_annotations or missing_images:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
