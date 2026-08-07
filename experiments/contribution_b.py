#!/usr/bin/env python3
"""Contribution B entrypoint — the exemplar-allocation sweep.

    python experiments/contribution_b.py --config configs/contribution_b.yaml

Computes `m_c` proportional to `n_c ** alpha` over the declared alpha grid and
writes the resulting buffers, one row per (alpha, class).

**What this is not.** It does not train, evaluate, or estimate forgetting. The
research question — the optimal alpha and its dependence on tail severity (H-B1) —
requires real incremental model updates with PROB retraining, which is a separate
future step (`docs/research_design.md` section 8). What this produces is the input
to that experiment: the buffers whose effect remains to be measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from daowod.memory import GRANULARITIES, AllocationError, allocate, allocate_images  # noqa: E402
from daowod.oracle import load_annotations, unknown_class_counts  # noqa: E402


def read_class_counts(path: Path) -> dict[str, int]:
    """Read a `class_name,count` CSV."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    missing = {"class_name", "count"} - set(rows[0])
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    return {str(row["class_name"]): int(row["count"]) for row in rows}


def counts_from_annotations(annotations_dir: Path, split_file: Path) -> dict[str, int]:
    """Per-class unknown-object counts over a real split, via the oracle."""

    image_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    if not image_ids:
        raise SystemExit(f"{split_file}: no image IDs")
    return unknown_class_counts(load_annotations(image_ids, annotations_dir))


def image_classes_from_annotations(annotations_dir: Path, split_file: Path) -> dict[str, list[str]]:
    """Image ID -> the unknown classes of the objects it contains."""

    image_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    annotations = load_annotations(image_ids, annotations_dir)
    return {
        image_id: [obj.class_name for obj in annotation.objects if not obj.is_known]
        for image_id, annotation in annotations.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "contribution_b.yaml"))
    parser.add_argument("--total-memory", type=int, default=None, help="Override the budget.")
    parser.add_argument("--granularity", choices=GRANULARITIES, default=None)
    parser.add_argument("--output", default=None, help="Override output_dir.")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    allocation = config.get("allocation") or {}
    source = config.get("source") or {}

    total_memory = args.total_memory or int(allocation.get("total_memory", 0))
    alphas = [float(value) for value in allocation.get("alphas") or ()]
    granularity = args.granularity or str(allocation.get("granularity", "object"))
    respect = bool(allocation.get("respect_availability", True))
    output_dir = Path(args.output or config.get("output_dir") or "outputs/contribution_b")
    if not alphas:
        raise SystemExit(f"{args.config}: allocation.alphas is empty")

    csv_path = str(source.get("class_counts_csv") or "")
    annotations_dir = str(source.get("annotations_dir") or "")
    split_file = str(source.get("split_file") or "")

    if csv_path:
        class_counts = read_class_counts(Path(csv_path))
        image_classes: dict[str, list[str]] | None = None
    elif annotations_dir and split_file:
        class_counts = counts_from_annotations(Path(annotations_dir), Path(split_file))
        image_classes = (
            image_classes_from_annotations(Path(annotations_dir), Path(split_file))
            if granularity == "image"
            else None
        )
    else:
        raise SystemExit(
            f"{args.config}: set source.class_counts_csv, or both source.annotations_dir "
            "and source.split_file. There is no default: an allocation over invented "
            "counts would describe nothing."
        )

    if granularity == "image" and image_classes is None:
        raise SystemExit("image granularity needs source.annotations_dir and source.split_file")

    print(f"classes      : {len(class_counts)}")
    print(f"objects      : {sum(class_counts.values())}")
    print(f"total memory : {total_memory} {'objects' if granularity == 'object' else 'images'}")
    print(f"alphas       : {alphas}")

    rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for alpha in alphas:
        try:
            if granularity == "image":
                assert image_classes is not None
                result = allocate_images(image_classes, total_memory=total_memory, alpha=alpha)
            else:
                result = allocate(
                    class_counts,
                    total_memory=total_memory,
                    alpha=alpha,
                    respect_availability=respect,
                )
        except AllocationError as error:
            raise SystemExit(f"alpha={alpha}: {error}") from error
        manifests.append(result.as_dict())
        for class_name, count in sorted(result.counts.items()):
            rows.append(
                {
                    "alpha": alpha,
                    "granularity": result.granularity,
                    "class_name": class_name,
                    "class_objects": class_counts.get(class_name, 0),
                    "exemplars": count,
                    "capped": class_name in result.capped_classes,
                    "shortfall": (result.shortfall or {}).get(class_name, 0),
                }
            )
        capped = f", capped {len(result.capped_classes)}" if result.capped_classes else ""
        print(f"  alpha={alpha:+.2f} -> {sum(result.counts.values())} exemplars{capped}")

    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / "allocations.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = output_dir / "allocation_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config": str(args.config),
                "granularity": granularity,
                "total_memory": total_memory,
                "class_counts": dict(sorted(class_counts.items())),
                "allocations": manifests,
                "claim": (
                    "Allocation only. Optimal alpha (H-B1) requires incremental model "
                    "updates and is not measured here."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {table}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
