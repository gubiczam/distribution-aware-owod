"""Representation Experiment E4, Phase 5: the same strategies, a different space.

    python analysis/run_e4_active_learning.py \\
        --export outputs/real_stage1/reference_proposals.npz \\
        --annotations ~/owod_stage/Annotations \\
        --representations outputs/e4_representations \\
        --output outputs/e4_active_learning

For each feature space this writes a substituted export — every field identical to
the original except ``embeddings`` — and runs the existing pipeline on it. Nothing
about the acquisition changes: the same strategy definitions, the same weights, the
same coherence exponent, the same candidate pool, the same three severities, the
same seeds, the same budgets, the same oracle, the same metrics.

Two of the arms are **representation-invariant by construction**: ``random``
ignores the score entirely and ``objectness_area_prior`` reads only objectness
and box geometry. Their results must therefore be identical across every space, and
the report checks that. If they ever differ, something other than the representation
changed and the comparison is void — so the invariance check is the experiment's own
correctness test rather than a nicety.

The frozen experiments are untouched: this writes to its own output directory and
uses its own execution mode.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from daowod import annotation_study as study
from daowod import modes, reporting, representations
from daowod.modes import resolve_mode
from daowod.pipeline import PipelineConfig, run_pipeline

#: The arms Phase 5 runs. Deliberately smaller than the eleven-arm follow-up: the
#: variable under test is the representation, so the set is the frozen ladder plus
#: the two invariance controls plus the label-anchored gate, and every space gets
#: exactly the same six.
E4_STRATEGIES: tuple[str, ...] = (
    "random",
    "objectness_area_prior",
    "uncertainty_novelty",
    "full_no_coherence",
    "full",
    "revealed_full",
)

#: Arms whose ranking cannot depend on the embedding. Used as the correctness check.
INVARIANT_STRATEGIES: tuple[str, ...] = ("random", "objectness_area_prior")

E4_MODE_NAME = "E4REPRESENTATION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--representations", default="outputs/e4_representations")
    parser.add_argument("--output", default="outputs/e4_active_learning")
    parser.add_argument("--base-mode", default="MAINREVEALED")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only", action="append", help="Restrict to these representation names; repeatable."
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached stage results.")
    return parser.parse_args()


def register_mode(base_name: str) -> None:
    """The E4 mode: the base protocol with the six-arm set and no ablation grid."""

    base = resolve_mode(base_name)
    modes.register(
        replace(
            base,
            name=E4_MODE_NAME,
            description=(
                "Representation Experiment E4: the frozen strategy ladder plus two "
                "representation-invariant controls, run once per feature space on one "
                "shared candidate pool."
            ),
            strategies=E4_STRATEGIES,
            run_ablations=False,
        ),
        replace_existing=True,
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    register_mode(args.base_mode)
    directory = Path(args.representations)

    export = study.load_export(args.export)
    ready = representations.available(export=export, directory=directory)
    if args.only:
        wanted = set(args.only)
        ready = [name for name in ready if name in wanted]
    ready = representations.sequence(ready)
    print(f"Phase 5 over {len(ready)} representations: {ready}", flush=True)

    curve_rows: list[dict[str, object]] = []
    auc_rows: list[dict[str, object]] = []
    counts_rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []

    for name in ready:
        started = time.time()
        matrix, manifest = representations.load(
            name, export=export, directory=directory, seed=args.seed
        )
        if name == representations.BASELINE_REPRESENTATION:
            export_path = Path(args.export)
        else:
            export_path = output / "exports" / f"{name}.npz"
            if args.force or not export_path.exists():
                representations.write_substituted_export(
                    export_path, export=export, embeddings=matrix, manifest=manifest
                )
                print(f"  wrote substituted export {export_path}", flush=True)
        run_directory = output / name
        print(f"  running {name} ({matrix.shape[1]}-d) -> {run_directory}", flush=True)
        result = run_pipeline(
            PipelineConfig(
                mode=E4_MODE_NAME,
                data_root=str(Path(args.annotations).expanduser().parent),
                split_file="",
                existing_export=str(export_path),
                output_dir=str(run_directory),
                cache_dir=str(run_directory / "cache"),
                require_gpu=False,
                seed=args.seed,
                force=args.force,
            ),
            progress=None,
        )
        for row in result.outputs.strategy_rows:
            entry = dict(row)
            entry["representation"] = name
            entry["representation_dimensions"] = int(matrix.shape[1])
            curve_rows.append(entry)
        for row in result.outputs.auc_rows:
            entry = dict(row)
            entry["representation"] = name
            auc_rows.append(entry)
        for row in reporting.discovery_counts(result.outputs.strategy_rows):
            entry = dict(row)
            entry["representation"] = name
            counts_rows.append(entry)
        runs.append(
            {
                "representation": name,
                "dimensions": int(matrix.shape[1]),
                "export": str(export_path),
                "output_dir": str(run_directory),
                "seconds": round(time.time() - started, 1),
                "pool_size": int(result.pool_report.get("final_pool", 0)),
                "manifest": manifest,
            }
        )
        print(f"  done in {(time.time() - started) / 60:.1f} min", flush=True)

    reporting.write_csv(output / "e4_budget_curves.csv", curve_rows)
    reporting.write_csv(output / "e4_strategy_auc.csv", auc_rows)
    reporting.write_csv(output / "e4_discovery_counts.csv", counts_rows)

    invariance = _invariance_check(counts_rows)
    reporting.write_csv(output / "e4_invariance_check.csv", invariance)
    comparison = _representation_comparison(auc_rows, counts_rows)
    reporting.write_csv(output / "e4_representation_comparison.csv", comparison)

    from daowod import representation_plots

    figures = [
        str(path) for path in representation_plots.active_learning_comparison(counts_rows, output)
    ]

    manifest = {
        "export": args.export,
        "base_mode": args.base_mode,
        "mode": E4_MODE_NAME,
        "strategies": list(E4_STRATEGIES),
        "invariant_strategies": list(INVARIANT_STRATEGIES),
        "seed": args.seed,
        "runs": runs,
        "invariance_violations": [row for row in invariance if not row["consistent"]],
        "figures": figures,
    }
    reporting.write_json(output / "e4_active_learning_manifest.json", manifest)
    (output / "e4_active_learning_summary.md").write_text(
        _summary(manifest, comparison, invariance), encoding="utf-8"
    )
    print(json.dumps({"invariance_violations": manifest["invariance_violations"]}, indent=2))
    print(f"\nwrote {output}")


def _invariance_check(counts_rows) -> list[dict[str, object]]:
    """Do the representation-invariant arms give identical results everywhere?"""

    rows: list[dict[str, object]] = []
    for strategy in INVARIANT_STRATEGIES:
        for severity in sorted({str(row["imbalance_setting"]) for row in counts_rows}):
            values = {
                str(row["representation"]): float(row["unknown_objects_found_mean"])
                for row in counts_rows
                if str(row["strategy"]) == strategy and str(row["imbalance_setting"]) == severity
            }
            if not values:
                continue
            spread = max(values.values()) - min(values.values())
            rows.append(
                {
                    "strategy": strategy,
                    "imbalance_setting": severity,
                    "representations": len(values),
                    "min": min(values.values()),
                    "max": max(values.values()),
                    "spread": spread,
                    "consistent": bool(spread < 1e-9),
                    "values": json.dumps(values, sort_keys=True),
                }
            )
    return rows


def _representation_comparison(auc_rows, counts_rows) -> list[dict[str, object]]:
    """Each (representation, strategy) cell against the same strategy in the baseline space."""

    baseline = representations.BASELINE_REPRESENTATION
    auc_by_cell: dict[tuple[str, str, str, int], float] = {}
    for row in auc_rows:
        try:
            value = float(row["all_discovery_auc"])
        except (KeyError, TypeError, ValueError):
            continue
        auc_by_cell[
            (
                str(row["representation"]),
                str(row["imbalance_setting"]),
                str(row["strategy"]),
                int(row["seed"]),
            )
        ] = value

    counts_by_cell = {
        (
            str(row["representation"]),
            str(row["imbalance_setting"]),
            str(row["strategy"]),
        ): float(row["unknown_objects_found_mean"])
        for row in counts_rows
    }

    rows: list[dict[str, object]] = []
    severities = sorted({key[1] for key in auc_by_cell})
    strategies = sorted({key[2] for key in auc_by_cell})
    reps = sorted({key[0] for key in auc_by_cell})
    for representation in reps:
        for severity in severities:
            for strategy in strategies:
                seeds = sorted(
                    {
                        key[3]
                        for key in auc_by_cell
                        if key[:3] == (representation, severity, strategy)
                    }
                )
                own = [auc_by_cell[(representation, severity, strategy, seed)] for seed in seeds]
                paired = [
                    auc_by_cell[(representation, severity, strategy, seed)]
                    - auc_by_cell[(baseline, severity, strategy, seed)]
                    for seed in seeds
                    if (baseline, severity, strategy, seed) in auc_by_cell
                ]
                if not own:
                    continue
                differences = np.asarray(paired, dtype=np.float64)
                mean = float(np.mean(own))
                # Paired-difference confidence interval from the t distribution. With
                # three seeds this is wide, and it is reported precisely so that a
                # difference smaller than the interval is not read as an effect.
                interval = reporting.paired_interval(differences)
                rows.append(
                    {
                        "representation": representation,
                        "imbalance_setting": severity,
                        "strategy": strategy,
                        "is_baseline_space": representation == baseline,
                        "seeds": len(own),
                        "unknown_discovery_auc_mean": mean,
                        "unknown_discovery_auc_sd": (
                            float(np.std(own, ddof=1)) if len(own) > 1 else float("nan")
                        ),
                        "unknown_objects_found": counts_by_cell.get(
                            (representation, severity, strategy), float("nan")
                        ),
                        "paired_seeds": int(differences.size),
                        "mean_difference_vs_baseline_space": (
                            float(differences.mean()) if differences.size else float("nan")
                        ),
                        "ci95_low": interval[0],
                        "ci95_high": interval[1],
                        "significant_at_95": bool(
                            differences.size > 1
                            and np.isfinite(interval[0])
                            and (interval[0] > 0 or interval[1] < 0)
                        ),
                        "all_seeds_better": bool(differences.size and np.all(differences > 0)),
                        "all_seeds_worse": bool(differences.size and np.all(differences < 0)),
                    }
                )
    return rows


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt(value, digits=4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def _summary(manifest, comparison, invariance) -> str:
    lines = ["# E4 Phase 5 — active learning across representations", ""]
    lines.append(
        "The same six arms, the same pool, the same severities, seeds and budgets; "
        "only the embedding differs."
    )
    lines.append("")
    lines.append("## Correctness check: the representation-invariant arms")
    lines.append("")
    lines.append(
        "`random` and `objectness_area_prior` cannot depend on the embedding. "
        "Their spread across representations must be exactly zero."
    )
    lines.append("")
    lines.append(
        _table(
            ["strategy", "severity", "spaces", "min", "max", "spread", "consistent"],
            [
                [
                    row["strategy"],
                    row["imbalance_setting"],
                    row["representations"],
                    _fmt(row["min"], 1),
                    _fmt(row["max"], 1),
                    _fmt(row["spread"], 6),
                    "yes" if row["consistent"] else "**NO**",
                ]
                for row in invariance
            ],
        )
    )
    lines.append("")
    if manifest["invariance_violations"]:
        lines.append(
            "> **The invariance check failed.** Something other than the "
            "representation differed between runs, so the comparison below is not "
            "valid and must not be interpreted."
        )
        lines.append("")

    lines.append("## Unknown objects discovered at the largest budget")
    lines.append("")
    severities = sorted({str(row["imbalance_setting"]) for row in comparison})
    for severity in severities:
        lines.append(f"### severity: {severity}")
        lines.append("")
        strategies = sorted({str(row["strategy"]) for row in comparison})
        reps = sorted({str(row["representation"]) for row in comparison})
        lines.append(
            _table(
                ["strategy", *reps],
                [
                    [
                        strategy,
                        *[
                            _fmt(
                                next(
                                    (
                                        row["unknown_objects_found"]
                                        for row in comparison
                                        if str(row["strategy"]) == strategy
                                        and str(row["representation"]) == representation
                                        and str(row["imbalance_setting"]) == severity
                                    ),
                                    float("nan"),
                                ),
                                1,
                            )
                            for representation in reps
                        ],
                    ]
                    for strategy in strategies
                ],
            )
        )
        lines.append("")

    lines.append("## Paired difference against the baseline space, with 95 % intervals")
    lines.append("")
    lines.append(
        "Unknown-discovery AUC. Paired by seed and severity, since every space shares "
        "the pool and the seeds. `significant` means the interval excludes zero."
    )
    lines.append("")
    interesting = [
        row
        for row in comparison
        if not row["is_baseline_space"]
        and str(row["strategy"]) not in manifest["invariant_strategies"]
    ]
    lines.append(
        _table(
            ["space", "severity", "strategy", "mean diff", "95 % CI", "significant"],
            [
                [
                    row["representation"],
                    row["imbalance_setting"],
                    row["strategy"],
                    _fmt(row["mean_difference_vs_baseline_space"]),
                    f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]",
                    "**yes**" if row["significant_at_95"] else "no",
                ]
                for row in interesting
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
