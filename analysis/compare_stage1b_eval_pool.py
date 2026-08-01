#!/usr/bin/env python3
"""Compare leak-free Stage 1B diagnostics with the disqualified eval-pool run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

STRATEGIES = (
    "v2:random",
    "v2:uncertainty_objectness_weighted_entropy",
    "v2:full",
    "v2:rarity_coherence",
)
METRICS = (
    "tail_image_lift",
    "object_positive_image_rate",
    "distinct_gt_classes_selected",
    "distinct_tail_classes_selected",
    "background_only_selection_rate",
)
BUDGETS = (20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=Path("outputs/real_stage1"))
    parser.add_argument("--stage1b-dir", type=Path, default=Path("outputs/stage1b_real"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1b_real"))
    return parser.parse_args()


def read_metrics(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["strategy"], int(row["budget"])): row for row in rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    old = read_metrics(args.old_dir / "real_selection_metrics.csv")
    stage1b = read_metrics(args.stage1b_dir / "real_selection_metrics.csv")
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for strategy in STRATEGIES:
            old_row = old[(strategy, budget)]
            new_row = stage1b[(strategy, budget)]
            out: dict[str, Any] = {"strategy": strategy, "budget": budget}
            for metric in METRICS:
                old_value = float(old_row[metric])
                new_value = float(new_row[metric])
                out[f"old_eval_pool_{metric}"] = old_value
                out[f"stage1b_{metric}"] = new_value
                out[f"delta_{metric}"] = new_value - old_value
            rows.append(out)
    summary = {
        "schema": "stage1b_vs_eval_pool_comparison_v1",
        "old_eval_pool_status": "permanently_disqualified_from_acquisition_and_training",
        "reason": "The old 500-image Stage 1 candidate pool is a subset of official owdetr_test.",
        "stage1b_status": "leak_free_official_task1_train_side_pool",
        "budgets": list(BUDGETS),
        "strategies": list(STRATEGIES),
        "rows": rows,
    }
    write_csv(args.output_dir / "stage1b_vs_eval_pool_comparison.csv", rows)
    (args.output_dir / "stage1b_vs_eval_pool_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage 1B vs Old Eval-Pool Diagnostics",
        "",
        "The old eval-pool diagnostics are retained only as a contaminated baseline. They are not admissible acquisition or training evidence because the 500-image pool is a subset of `owdetr_test`.",
        "",
        "| Strategy | Budget | Old tail lift | Stage 1B tail lift | Old object-positive | Stage 1B object-positive |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {strategy} | {budget} | {old_tail:.3f} | {new_tail:.3f} | {old_obj:.3f} | {new_obj:.3f} |".format(
                strategy=row["strategy"],
                budget=row["budget"],
                old_tail=row["old_eval_pool_tail_image_lift"],
                new_tail=row["stage1b_tail_image_lift"],
                old_obj=row["old_eval_pool_object_positive_image_rate"],
                new_obj=row["stage1b_object_positive_image_rate"],
            )
        )
    (args.output_dir / "stage1b_vs_eval_pool_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
