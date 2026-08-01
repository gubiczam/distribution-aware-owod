#!/usr/bin/env python3
"""Prepare leak-free Stage 1B proposal exports from official Task-1 train assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from daowod.dataset import file_sha256, read_image_ids, read_voc_classes

UNKNOWN_CLASSES = tuple(
    "traffic light|fire hydrant|stop sign|parking meter|bench|chair|diningtable|pottedplant|"
    "backpack|umbrella|handbag|tie|suitcase|microwave|oven|toaster|sink|refrigerator|"
    "bed|toilet|sofa|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|"
    "skateboard|surfboard|tennis racket|banana|apple|sandwich|orange|broccoli|carrot|"
    "hot dog|pizza|donut|cake|laptop|mouse|remote|keyboard|cell phone|book|clock|vase|"
    "scissors|teddy bear|hair drier|toothbrush|wine glass|cup|fork|knife|spoon|bowl|"
    "tvmonitor|bottle".split("|")
)
KNOWN_CLASSES = set(
    "aeroplane bicycle bird boat bus car cat cow dog horse motorbike sheep train elephant bear "
    "zebra giraffe truck person".split()
)
CLASS_ALIASES = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
    "couch": "sofa",
    "dining table": "diningtable",
    "potted plant": "pottedplant",
    "tv": "tvmonitor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-t1-split",
        type=Path,
        default=Path(
            "/Users/gubiczam/Documents/PROB/data/OWOD/ImageSets/OWDETR/owdetr_t1_train.txt"
        ),
    )
    parser.add_argument(
        "--official-eval-split",
        type=Path,
        default=Path("/Users/gubiczam/owod_stage/ImageSets/OWDETR/owdetr_test.txt"),
    )
    parser.add_argument(
        "--source-ids",
        type=Path,
        default=Path("/Users/gubiczam/owod_stage/ImageSets/OWDETR/pilot_t1_train_4000.txt"),
    )
    parser.add_argument(
        "--source-proposals",
        type=Path,
        default=Path("outputs/real_stage1/reference_proposals.npz"),
    )
    parser.add_argument(
        "--source-proposals-metadata",
        type=Path,
        default=Path("outputs/real_stage1/reference_proposals.json"),
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=Path("/Users/gubiczam/owod_stage/Annotations")
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("/Users/gubiczam/owod_stage/JPEGImages")
    )
    parser.add_argument("--candidate-size", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1b"))
    return parser.parse_args()


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def proposal_indices(all_image_ids: np.ndarray, selected: set[str]) -> np.ndarray:
    mask = np.asarray([str(image_id) in selected for image_id in all_image_ids], dtype=bool)
    return np.flatnonzero(mask)


def write_subset_npz(source: Path, output: Path, selected_ids: list[str]) -> dict[str, Any]:
    selected = set(selected_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=True) as data:
        image_ids = np.asarray(data["image_ids"], dtype=object)
        indices = proposal_indices(image_ids, selected)
        unique = list(dict.fromkeys(str(image_ids[index]) for index in indices))
        if unique != selected_ids:
            raise ValueError(
                f"Proposal coverage mismatch for {output}: expected first IDs "
                f"{selected_ids[:5]}, observed {unique[:5]}"
            )
        arrays = {name: np.asarray(data[name])[indices] for name in data.files}
    np.savez_compressed(output, **arrays)
    return {
        "path": str(output),
        "image_count": len(selected_ids),
        "proposal_count": int(indices.size),
        "proposals_per_image": int(indices.size // len(selected_ids)) if selected_ids else 0,
        "sha256": file_sha256(output),
    }


def support_counts(ids: list[str], annotations_dir: Path) -> dict[str, Any]:
    unknown_set = set(UNKNOWN_CLASSES)
    counts: dict[str, Any] = {
        "image_count": len(ids),
        "known_objects": 0,
        "unknown_objects": 0,
        "known_images": 0,
        "unknown_images": 0,
        "unknown_classes_present": [],
    }
    unknown_classes: set[str] = set()
    for image_id in ids:
        classes = [
            CLASS_ALIASES.get(name, name) for name in read_voc_classes(image_id, annotations_dir)
        ]
        has_known = any(name in KNOWN_CLASSES for name in classes)
        has_unknown = any(name in unknown_set for name in classes)
        counts["known_images"] += int(has_known)
        counts["unknown_images"] += int(has_unknown)
        for name in classes:
            if name in KNOWN_CLASSES:
                counts["known_objects"] += 1
            if name in unknown_set:
                counts["unknown_objects"] += 1
                unknown_classes.add(name)
    counts["unknown_classes_present"] = sorted(unknown_classes)
    counts["unknown_class_count"] = len(unknown_classes)
    return counts


def main() -> int:
    args = parse_args()
    official_t1 = set(read_image_ids(args.official_t1_split))
    official_eval = set(read_image_ids(args.official_eval_split))
    source_ids = read_image_ids(args.source_ids)
    if len(source_ids) < args.candidate_size + 1:
        raise ValueError("Source ID bank is too small for Stage 1B.")
    not_official_t1 = sorted(set(source_ids) - official_t1)
    eval_overlap = sorted(set(source_ids) & official_eval)
    if not_official_t1:
        raise ValueError(f"Source bank contains non-T1 IDs: {not_official_t1[:20]}")
    if eval_overlap:
        raise ValueError(f"Source bank overlaps eval IDs: {eval_overlap[:20]}")

    candidate_ids = source_ids[: args.candidate_size]
    reference_ids = source_ids[args.candidate_size :]
    initial_labelled_ids: list[str] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_split = args.output_dir / "stage1b_candidate_500.txt"
    reference_split = args.output_dir / "stage1b_reference_3500.txt"
    initial_split = args.output_dir / "stage1b_initial_labelled_empty.txt"
    write_ids(candidate_split, candidate_ids)
    write_ids(reference_split, reference_ids)
    initial_split.write_text("", encoding="utf-8")

    candidate_proposals = write_subset_npz(
        args.source_proposals, args.output_dir / "candidate_proposals.npz", candidate_ids
    )
    reference_proposals = write_subset_npz(
        args.source_proposals, args.output_dir / "reference_proposals.npz", reference_ids
    )
    for proposal in (candidate_proposals, reference_proposals):
        metadata_path = Path(proposal["path"]).with_suffix(".json")
        metadata = {
            "schema": "stage1b_sliced_real_prob_export_v1",
            "source_proposals": str(args.source_proposals),
            "source_proposals_sha256": file_sha256(args.source_proposals),
            "source_metadata": str(args.source_proposals_metadata),
            "official_t1_split": str(args.official_t1_split),
            "official_t1_split_sha256": file_sha256(args.official_t1_split),
            "official_eval_split": str(args.official_eval_split),
            "official_eval_split_sha256": file_sha256(args.official_eval_split),
            "proposal_export": proposal,
            "detector_checkpoint": "/Users/gubiczam/Downloads/results/SOWODB/t1.pth",
            "detector_checkpoint_sha256": "dba5390bffdfdf63058a995f241696df8d06b7fb859aecc8292d9ea02d459a22",
            "export_semantics": (
                "Real PROB Task-1 proposals sliced from the existing 4,000-image "
                "official-train-side export; no synthetic scores and no detector training."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if args.source_proposals_metadata.exists():
        shutil.copy2(
            args.source_proposals_metadata, args.output_dir / "source_reference_proposals.json"
        )

    missing_annotations = [
        image_id
        for image_id in source_ids
        if not (args.annotations_dir / f"{image_id}.xml").exists()
    ]
    missing_images = [
        image_id for image_id in source_ids if not (args.images_dir / f"{image_id}.jpg").exists()
    ]
    manifest = {
        "schema": "stage1b_split_manifest_v1",
        "split_policy": (
            "Deterministic official-order split of the existing 4,000-image "
            "official OWDETR Task-1 train-side PROB export: first 500 candidate, "
            "remaining 3,500 fixed representation reference, empty initial-labelled set."
        ),
        "oracle_semantics": {
            "acquisition_inputs": "PROB proposals only; no ground-truth class/group/oracle fields.",
            "posthoc_diagnostics": "VOC XML annotations used only after scoring for support and tail-lift diagnostics.",
            "training_oracle": (
                "If later training is run, selected image IDs are passed to PROB. "
                "Official Task-1 training keeps current known-class annotations and removes unknown-class objects."
            ),
            "unknown_semantics": (
                "Unknown objects are present in raw annotations and official evaluation; they are hidden "
                "from acquisition and not labelled as trainable known classes in Task 1."
            ),
        },
        "official_sources": {
            "t1_train_split": str(args.official_t1_split),
            "t1_train_split_sha256": file_sha256(args.official_t1_split),
            "eval_split": str(args.official_eval_split),
            "eval_split_sha256": file_sha256(args.official_eval_split),
            "source_id_bank": str(args.source_ids),
            "source_id_bank_sha256": file_sha256(args.source_ids),
        },
        "splits": {
            "initial_labelled": {
                "path": str(initial_split),
                "image_count": len(initial_labelled_ids),
                "sha256": file_sha256(initial_split),
            },
            "candidate_pool": {
                "path": str(candidate_split),
                "image_count": len(candidate_ids),
                "sha256": file_sha256(candidate_split),
                "support": support_counts(candidate_ids, args.annotations_dir),
            },
            "reference_bank": {
                "path": str(reference_split),
                "image_count": len(reference_ids),
                "sha256": file_sha256(reference_split),
                "support": support_counts(reference_ids, args.annotations_dir),
            },
        },
        "overlap": {
            "candidate_reference": len(set(candidate_ids) & set(reference_ids)),
            "candidate_eval": len(set(candidate_ids) & official_eval),
            "reference_eval": len(set(reference_ids) & official_eval),
            "initial_candidate": len(set(initial_labelled_ids) & set(candidate_ids)),
            "initial_reference": len(set(initial_labelled_ids) & set(reference_ids)),
            "initial_eval": len(set(initial_labelled_ids) & official_eval),
        },
        "asset_status": {
            "missing_annotations": missing_annotations[:50],
            "missing_annotations_count": len(missing_annotations),
            "missing_images": missing_images[:50],
            "missing_images_count": len(missing_images),
        },
        "proposal_exports": {
            "candidate": candidate_proposals,
            "reference": reference_proposals,
        },
    }
    (args.output_dir / "stage1b_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "overlap": manifest["overlap"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
