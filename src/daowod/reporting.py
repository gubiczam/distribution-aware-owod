"""Turn the study's row dictionaries into the files a reader actually opens.

Three kinds of output, deliberately separated:

* **CSV** — one file per logical table, long format, one row per measured cell.
  Long format because every table is grouped by (strategy, severity, budget, seed)
  and a wide layout would need a new column per strategy each time one is added.
* **JSON** — manifests: configuration, environment, runtime plan, pool reports,
  leakage verdicts. Everything needed to reproduce or audit a run.
* **Markdown** — the research summary: the headline comparison, the mechanism
  evidence, and the limitations, written from the numbers rather than asserted.

The summary states a verdict per hypothesis. It is allowed to say the hypothesis
was not supported: a report that can only confirm is not evidence.
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from daowod.annotation_study import STRATEGY_ROLES, StudyOutputs

#: The strategy the headline comparison is read against, and the two rungs below
#: it that isolate what each added term contributes.
BASELINE_STRATEGY = "random"
UNCERTAINTY_STRATEGY = "uncertainty"
NOVELTY_STRATEGY = "uncertainty_novelty"
UNGATED_STRATEGY = "full_no_coherence"
GATED_STRATEGY = "full"


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write long-format rows, unioning keys so a sparse column is still written."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return target


def write_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return target


def write_study_outputs(outputs: StudyOutputs, directory: str | Path) -> dict[str, Path]:
    """Every table the study produced, one CSV each, plus per-strategy splits."""

    root = Path(directory)
    tables = {
        "budget_curves": outputs.strategy_rows,
        "budget_curves_aggregated": outputs.aggregated_rows,
        "strategy_auc": outputs.auc_rows,
        "selected_proposals": outputs.selected_rows,
        "component_distributions": outputs.distribution_rows,
        "gate_suppression": outputs.outlier_rows,
        "long_tail_pools": outputs.pool_rows,
        "class_frequency": outputs.class_frequency_rows,
        "anchored_rounds": outputs.anchored_rows,
    }
    written: dict[str, Path] = {}
    for name, rows in tables.items():
        written[name] = write_csv(root / f"{name}.csv", rows)
    written["runtime"] = write_json(root / "study_runtime.json", outputs.runtime)
    written.update(write_per_strategy(outputs, root / "per_strategy"))
    return written


def write_per_strategy(outputs: StudyOutputs, directory: str | Path) -> dict[str, Path]:
    """One CSV per strategy: its budget curve, its AUCs, its selected regions.

    The long-format tables are the ones to analyse — a new strategy adds rows, not
    columns. These per-strategy files exist because a reader who wants to look at
    exactly one strategy should not have to filter, and because a strategy's whole
    record (curve, AUC, and the regions it bought) is then one directory.
    """

    root = Path(directory)
    strategies = sorted(
        {str(row.get("strategy", "")) for row in outputs.strategy_rows if row.get("strategy")}
    )
    written: dict[str, Path] = {}
    for strategy in strategies:
        slug = strategy.replace(":", "_")
        for label, rows in (
            ("curve", outputs.strategy_rows),
            ("auc", outputs.auc_rows),
            ("selected", outputs.selected_rows),
        ):
            subset = [row for row in rows if str(row.get("strategy", "")) == strategy]
            if subset:
                written[f"{slug}_{label}"] = write_csv(root / f"{slug}_{label}.csv", subset)
    return written


def _numeric(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def summarise_strategies(
    auc_rows: Sequence[Mapping[str, object]],
    *,
    metric: str = "tail_discovery_auc",
) -> list[dict[str, object]]:
    """Mean and sample sd of one AUC per (severity, strategy), over seeds."""

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in auc_rows:
        key = (str(row.get("imbalance_setting", "")), str(row.get("strategy", "")))
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, object]] = []
    for (severity, strategy), members in sorted(grouped.items()):
        values = _numeric(members, metric)
        final = _numeric(members, "final_tail_discovery_recall")
        summary.append(
            {
                "imbalance_setting": severity,
                "strategy": strategy,
                "role": STRATEGY_ROLES.get(strategy, ""),
                "seeds": len(members),
                f"{metric}_mean": float(np.mean(values)) if values else float("nan"),
                f"{metric}_sd": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
                ),
                "final_tail_discovery_recall_mean": (
                    float(np.mean(final)) if final else float("nan")
                ),
            }
        )
    return summary


def headline_contrasts(
    auc_rows: Sequence[Mapping[str, object]],
    *,
    metric: str = "tail_discovery_auc",
) -> list[dict[str, object]]:
    """Paired differences against each rung of the strategy ladder.

    Differences are computed **per seed and per severity**, then averaged, because
    the strategies share a pool, an export and a seed: a paired difference removes
    the pool-to-pool variance that an unpaired difference of means would leave in.
    """

    by_cell: dict[tuple[str, int, str], float] = {}
    for row in auc_rows:
        try:
            value = float(row[metric])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            continue
        by_cell[
            (
                str(row.get("imbalance_setting", "")),
                int(row.get("seed", 0)),
                str(row.get("strategy", "")),
            )
        ] = value

    severities = sorted({key[0] for key in by_cell})
    contrasts: list[dict[str, object]] = []
    comparisons = (
        (GATED_STRATEGY, BASELINE_STRATEGY, "gate vs random"),
        (GATED_STRATEGY, UNCERTAINTY_STRATEGY, "gate vs uncertainty"),
        (GATED_STRATEGY, NOVELTY_STRATEGY, "gate vs uncertainty+novelty"),
        (GATED_STRATEGY, UNGATED_STRATEGY, "gate vs ungated rarity"),
    )
    for severity in severities:
        seeds = sorted({key[1] for key in by_cell if key[0] == severity})
        for treatment, control, label in comparisons:
            paired = [
                by_cell[(severity, seed, treatment)] - by_cell[(severity, seed, control)]
                for seed in seeds
                if (severity, seed, treatment) in by_cell and (severity, seed, control) in by_cell
            ]
            if not paired:
                continue
            differences = np.asarray(paired, dtype=np.float64)
            mean = float(differences.mean())
            sd = float(differences.std(ddof=1)) if differences.size > 1 else float("nan")
            contrasts.append(
                {
                    "imbalance_setting": severity,
                    "comparison": label,
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    "paired_seeds": int(differences.size),
                    "mean_difference": mean,
                    "sd_difference": sd,
                    "all_seeds_positive": bool(np.all(differences > 0)),
                    "all_seeds_negative": bool(np.all(differences < 0)),
                }
            )
    return contrasts


#: Two-sided 95 % t critical values for the small seed counts this protocol uses.
#: Tabulated rather than imported so the library keeps its numpy/sklearn-only
#: dependency set; anything above the table falls back to the normal approximation.
_T_CRITICAL_95: Mapping[int, float] = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}


def paired_interval(differences: ArrayLike, *, confidence: float = 0.95) -> tuple[float, float]:
    """95 % interval for a paired mean difference, or NaNs when n < 2.

    Paired because every arm and every representation shares one candidate pool, one
    export and one seed set, so the seed-to-seed variation is common to all of them
    and an unpaired interval would report it as uncertainty about the contrast. With
    three seeds the interval is wide, which is the point: a difference smaller than
    its interval is not an effect, and reporting the interval makes that visible
    instead of leaving a reader to guess.
    """

    array = np.asarray(differences, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return (float("nan"), float("nan"))
    if confidence != 0.95:
        raise ValueError("Only the 95 % interval is tabulated.")
    mean = float(array.mean())
    error = float(array.std(ddof=1) / np.sqrt(array.size))
    critical = _T_CRITICAL_95.get(int(array.size), 1.96)
    return (mean - critical * error, mean + critical * error)


def arm_comparison(
    auc_rows: Sequence[Mapping[str, object]],
    *,
    metric: str = "all_discovery_auc",
    baseline: str = GATED_STRATEGY,
) -> list[dict[str, object]]:
    """Every arm against the designated baseline, paired by seed and severity.

    Pairing matters more here than in a two-arm comparison: eleven arms share one
    pool, one export and one seed set, so the seed-to-seed variation is common to
    all of them and an unpaired difference of means would report it as uncertainty
    about the contrast.
    """

    by_cell: dict[tuple[str, int, str], float] = {}
    families: dict[str, str] = {}
    for row in auc_rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        strategy = str(row.get("strategy", ""))
        by_cell[(str(row.get("imbalance_setting", "")), int(row.get("seed", 0)), strategy)] = value
        families[strategy] = str(row.get("strategy_family", "")) or ""

    severities = sorted({key[0] for key in by_cell})
    arms = sorted({key[2] for key in by_cell})
    rows: list[dict[str, object]] = []
    for severity in severities:
        seeds = sorted({key[1] for key in by_cell if key[0] == severity})
        for arm in arms:
            paired = [
                by_cell[(severity, seed, arm)] - by_cell[(severity, seed, baseline)]
                for seed in seeds
                if (severity, seed, arm) in by_cell and (severity, seed, baseline) in by_cell
            ]
            values = [
                by_cell[(severity, seed, arm)] for seed in seeds if (severity, seed, arm) in by_cell
            ]
            if not values:
                continue
            differences = np.asarray(paired, dtype=np.float64)
            rows.append(
                {
                    "imbalance_setting": severity,
                    "strategy": arm,
                    "strategy_family": families.get(arm, ""),
                    "role": STRATEGY_ROLES.get(arm, ""),
                    "metric": metric,
                    "seeds": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                    "baseline": baseline,
                    "paired_seeds": int(differences.size),
                    "mean_difference_vs_baseline": (
                        float(differences.mean()) if differences.size else float("nan")
                    ),
                    "sd_difference": (
                        float(differences.std(ddof=1)) if differences.size > 1 else float("nan")
                    ),
                    "all_seeds_better": bool(differences.size and np.all(differences > 0)),
                    "all_seeds_worse": bool(differences.size and np.all(differences < 0)),
                    "is_baseline": arm == baseline,
                }
            )
    return rows


def discovery_counts(
    curve_rows: Sequence[Mapping[str, object]],
    *,
    budget: int | None = None,
) -> list[dict[str, object]]:
    """Absolute discovered-object counts per arm at one budget.

    Counts rather than recalls because the tail denominator is 26 objects: a recall
    of 0.038 is one object, and a reader deserves to see the integer. Absolute
    counts are also what makes the campaign arms comparable with the static
    free-heuristic reference, which has no notion of a severity-restricted
    denominator.
    """

    budgets = sorted({int(row["budget"]) for row in curve_rows if "budget" in row})
    if not budgets:
        return []
    target = int(budget) if budget is not None else budgets[-1]
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in curve_rows:
        if int(row.get("budget", -1)) != target:
            continue
        grouped.setdefault(
            (str(row.get("imbalance_setting", "")), str(row.get("strategy", ""))), []
        ).append(row)
    rows: list[dict[str, object]] = []
    for (severity, strategy), members in sorted(grouped.items()):
        rows.append(
            {
                "imbalance_setting": severity,
                "strategy": strategy,
                "strategy_family": str(members[0].get("strategy_family", "")),
                "role": STRATEGY_ROLES.get(strategy, ""),
                "budget": target,
                "seeds": len(members),
                "unknown_objects_found_mean": float(
                    np.mean([float(row["all_objects_found"]) for row in members])
                ),
                "unknown_objects_reachable": int(members[0].get("all_objects_reachable", 0)),
                "tail_objects_found_mean": float(
                    np.mean([float(row["tail_objects_found"]) for row in members])
                ),
                "tail_objects_reachable": int(members[0].get("tail_objects_reachable", 0)),
                "unique_classes_mean": float(
                    np.mean([float(row["all_unique_classes"]) for row in members])
                ),
                "annotation_precision_mean": float(
                    np.mean([float(row["annotation_precision"]) for row in members])
                ),
            }
        )
    return rows


def budget_to_reach(
    rows: Sequence[Mapping[str, object]],
    *,
    target_recall: float,
    metric: str = "tail_discovery_recall",
) -> list[dict[str, object]]:
    """Smallest budget at which each strategy first reaches ``target_recall``.

    This is the plan's headline framing — "the same tail level for fewer oracle
    calls" — expressed as a cost rather than as a recall. A strategy that never
    reaches the target reports ``None`` instead of the largest budget, so a miss
    cannot read as a tie.
    """

    grouped: dict[tuple[str, str, int], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("imbalance_setting", "")),
            str(row.get("strategy", "")),
            int(row.get("seed", 0)),
        )
        grouped.setdefault(key, []).append(row)
    results: list[dict[str, object]] = []
    for (severity, strategy, seed), members in sorted(grouped.items()):
        ordered = sorted(members, key=lambda item: int(item["budget"]))  # type: ignore[arg-type]
        reached: int | None = None
        for row in ordered:
            try:
                value = float(row[metric])  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value) and value >= float(target_recall):
                reached = int(row["budget"])  # type: ignore[arg-type]
                break
        results.append(
            {
                "imbalance_setting": severity,
                "strategy": strategy,
                "seed": seed,
                "metric": metric,
                "target": float(target_recall),
                "budget_to_reach": reached,
                "reached": reached is not None,
                "largest_budget": int(ordered[-1]["budget"]) if ordered else 0,  # type: ignore[arg-type]
            }
        )
    return results


def coherence_separation(
    rows: Sequence[Mapping[str, object]],
    *,
    strategy: str = GATED_STRATEGY,
    component: str = "coherence",
    minimum_relative_margin: float = 0.1,
) -> dict[str, object]:
    """Does coherence actually separate true tail regions from background?

    This is the gate's *necessary condition*, and it is worth stating separately
    from the outcome. The gate multiplies rarity by coherence in order to prefer
    "rare and locally supported" over "rare and isolated". That helps only if
    coherence ranks true objects above background. If background proposals are at
    least as locally coherent as tail objects — which is plausible, since generic
    texture patches cluster tightly in a decoder's feature space — then the gate
    can suppress isolated outliers perfectly and still buy nothing useful, because
    its competition for the budget was never the isolated outlier.

    Returns the medians it compared and a verdict, so a negative result is
    diagnosed rather than merely observed.
    """

    medians: dict[str, list[float]] = {}
    for row in rows:
        if str(row.get("strategy")) != strategy or str(row.get("component")) != component:
            continue
        try:
            value = float(row["median"])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            medians.setdefault(str(row.get("stratum", "")), []).append(value)
    if not medians:
        return {"available": False}

    def median_of(stratum: str) -> float:
        values = medians.get(stratum, [])
        return float(np.mean(values)) if values else float("nan")

    tail = median_of("true_tail")
    background = median_of("background")
    isolated = median_of("isolated_outlier")
    head = median_of("true_head")
    # A strict ``>`` fires on noise: measured tail 0.559 against background 0.553 is
    # a 1 % difference and would be reported as "coherence separates them", which is
    # exactly the false reassurance this diagnostic exists to prevent. A separation
    # has to clear a margin to count.
    margin = float(minimum_relative_margin)
    separates_isolated = (
        np.isfinite(tail) and np.isfinite(isolated) and tail > isolated * (1.0 + margin)
    )
    separates_background = (
        np.isfinite(tail) and np.isfinite(background) and tail > background * (1.0 + margin)
    )
    indistinguishable_from_background = (
        np.isfinite(tail)
        and np.isfinite(background)
        and abs(tail - background) <= background * margin
    )
    return {
        "available": True,
        "component": component,
        "strategy": strategy,
        "median_true_tail": tail,
        "median_true_head": head,
        "median_background": background,
        "median_isolated_outlier": isolated,
        "separates_tail_from_isolated_outliers": bool(separates_isolated),
        "separates_tail_from_background": bool(separates_background),
        "tail_indistinguishable_from_background": bool(indistinguishable_from_background),
        "minimum_relative_margin": margin,
    }


def _format_float(value: object, digits: int = 3) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def research_summary(
    *,
    mode: Mapping[str, object],
    pool_report: Mapping[str, object],
    composition: Mapping[str, object],
    severity_rows: Sequence[Mapping[str, object]],
    auc_rows: Sequence[Mapping[str, object]],
    curve_rows: Sequence[Mapping[str, object]],
    contrasts: Sequence[Mapping[str, object]],
    gate_rows: Sequence[Mapping[str, object]],
    cost_rows: Sequence[Mapping[str, object]],
    leakage: Mapping[str, object],
    runtime: Mapping[str, object],
    pilot: Mapping[str, object] | None = None,
    ablation_rows: Sequence[Mapping[str, object]] = (),
    distribution_rows: Sequence[Mapping[str, object]] = (),
    arm_rows: Sequence[Mapping[str, object]] = (),
    limitations: Sequence[str] = (),
) -> str:
    """The markdown research summary, written from the measured rows."""

    research_grade = bool(mode.get("research_grade"))
    lines: list[str] = []
    lines.append("# Contribution A — distribution-aware active annotation")
    lines.append("")
    lines.append(f"Execution mode **{mode.get('name')}** — {mode.get('description')}")
    if not research_grade:
        lines.append("")
        lines.append(
            "> **Not a reportable result.** This mode exists to validate the "
            "pipeline. Its pool and seed count are too small to separate "
            "strategies; run `MAIN_MODE` for the reported numbers."
        )
    lines.append("")

    lines.append("## What was measured")
    lines.append("")
    lines.append(
        "An offline annotation simulation over real PROB proposals. Each strategy "
        "spends the same region-level annotation budget on the same candidate pool; "
        "the oracle reveals a region's true class only after it has been selected. "
        "The score is"
    )
    lines.append("")
    lines.append("```text")
    lines.append("s(x) = alpha * uncertainty(x) + beta * novelty(x)")
    lines.append("     + gamma * rarity(x) * coherence(x)**p")
    lines.append("```")
    lines.append("")
    lines.append(
        "with coherence entering only as a multiplicative gate on rarity, so an "
        "isolated proposal keeps its uncertainty and novelty but loses its rarity "
        "bonus."
    )
    lines.append("")

    lines.append("## Pool")
    lines.append("")
    lines.append(
        _table(
            ["quantity", "value"],
            [
                ["raw proposals", pool_report.get("raw_proposals", "n/a")],
                ["raw images", pool_report.get("raw_images", "n/a")],
                ["candidate pool", pool_report.get("final_pool", "n/a")],
                ["images in pool", pool_report.get("final_images", "n/a")],
                ["true-unknown rate", _format_float(composition.get("unknown_rate"))],
                ["known-object rate", _format_float(composition.get("known_rate"))],
                ["background rate", _format_float(composition.get("background_rate"))],
            ],
        )
    )
    lines.append("")
    lines.append(
        "The unknown rate is low by construction: PROB emits one proposal per "
        "decoder query, most of which sit on background. That is the pool an "
        "annotator would actually be shown, and the reason annotation precision is "
        "reported alongside discovery."
    )
    lines.append("")

    lines.append("## Long-tail severities")
    lines.append("")
    lines.append(
        _table(
            [
                "setting",
                "profile",
                "requested ratio",
                "head:tail objects",
                "unknown objects",
                "tail objects",
            ],
            [
                [
                    row.get("setting", "?"),
                    (row.get("spec") or {}).get("profile", "absolute"),
                    row.get("requested_imbalance_ratio", "?"),
                    _format_float(row.get("head_to_tail_object_ratio"), 2),
                    f"{row.get('unknown_objects_before', '?')} -> "
                    f"{row.get('unknown_objects_after', '?')}",
                    row.get("tail_objects_after", "?"),
                ]
                for row in severity_rows
            ],
        )
    )
    lines.append("")

    metric = "tail_discovery_auc"
    summary = summarise_strategies(auc_rows, metric=metric)
    lines.append("## Headline: tail discovery")
    lines.append("")
    lines.append(
        "Tail discovery AUC is the mean fraction of *reachable tail objects* found "
        "over the budget sweep. Distinct objects, not proposals: forty boxes on one "
        "object count once."
    )
    lines.append("")
    lines.append(
        _table(
            ["severity", "strategy", "role", "seeds", "tail AUC", "sd", "final tail recall"],
            [
                [
                    row["imbalance_setting"],
                    row["strategy"],
                    row["role"],
                    row["seeds"],
                    _format_float(row[f"{metric}_mean"]),
                    _format_float(row[f"{metric}_sd"]),
                    _format_float(row["final_tail_discovery_recall_mean"]),
                ]
                for row in summary
            ],
        )
    )
    lines.append("")

    lines.append("### Paired contrasts")
    lines.append("")
    lines.append(
        "Differences are paired by seed and severity, since every strategy sees the "
        "same pool and the same export."
    )
    lines.append("")
    lines.append(
        _table(
            ["severity", "comparison", "mean difference", "sd", "seeds", "consistent sign"],
            [
                [
                    row["imbalance_setting"],
                    row["comparison"],
                    _format_float(row["mean_difference"]),
                    _format_float(row["sd_difference"]),
                    row["paired_seeds"],
                    (
                        "all positive"
                        if row["all_seeds_positive"]
                        else ("all negative" if row["all_seeds_negative"] else "mixed")
                    ),
                ]
                for row in contrasts
            ],
        )
    )
    lines.append("")
    lines.append(_verdict_section(contrasts))
    lines.append("")

    if cost_rows:
        reached = [row for row in cost_rows if row.get("reached")]
        lines.append("## Annotation cost to reach a fixed tail level")
        lines.append("")
        if reached:
            grouped: dict[tuple[str, str], list[int]] = {}
            for row in reached:
                grouped.setdefault(
                    (str(row["imbalance_setting"]), str(row["strategy"])), []
                ).append(int(row["budget_to_reach"]))
            lines.append(
                _table(
                    ["severity", "strategy", "median budget", "seeds reaching target"],
                    [
                        [
                            severity,
                            strategy,
                            int(np.median(values)),
                            len(values),
                        ]
                        for (severity, strategy), values in sorted(grouped.items())
                    ],
                )
            )
            target = _format_float(reached[0].get("target"), 2)
            lines.append("")
            lines.append(f"Target tail discovery recall: {target}.")
        else:
            lines.append(
                "No strategy reached the target tail recall inside the largest "
                "budget, so the cost comparison is undefined for this run. The "
                "budget curve is still valid; the target was simply set above what "
                "this pool supports."
            )
        lines.append("")

    if gate_rows:
        lines.append("## Mechanism: what the gate actually suppresses")
        lines.append("")
        lines.append(
            "Counterfactual over the same pool: rank by ungated rarity, rank by the "
            "gated interaction, and count what changes hands in the top-K."
        )
        lines.append("")
        lines.append(
            _table(
                [
                    "severity",
                    "seed",
                    "suppressed isolated",
                    "promoted isolated",
                    "net isolated removed",
                    "net true unknown gained",
                    "net tail gained",
                ],
                [
                    [
                        row.get("imbalance_setting", "?"),
                        row.get("seed", "?"),
                        row.get("suppressed_isolated", "?"),
                        row.get("promoted_isolated", "?"),
                        row.get("isolated_suppression_gain", "?"),
                        row.get("true_unknown_gain", "?"),
                        row.get("tail_gain", "?"),
                    ]
                    for row in gate_rows
                ],
            )
        )
        lines.append("")

    separation = coherence_separation(distribution_rows)
    if separation.get("available"):
        lines.append("### Mechanism: does coherence separate tail regions from background?")
        lines.append("")
        lines.append(
            "The gate helps only if coherence ranks true objects above background. "
            "Suppressing isolated outliers perfectly buys nothing if the budget's "
            "real competition is *coherent background* — and generic texture patches "
            "can cluster tightly in a decoder's feature space."
        )
        lines.append("")
        lines.append(
            _table(
                ["stratum", "median coherence"],
                [
                    ["true tail", _format_float(separation["median_true_tail"])],
                    ["true head", _format_float(separation["median_true_head"])],
                    ["background", _format_float(separation["median_background"])],
                    ["isolated outlier", _format_float(separation["median_isolated_outlier"])],
                ],
            )
        )
        lines.append("")
        if separation["separates_tail_from_isolated_outliers"]:
            lines.append(
                "- Tail regions are more coherent than isolated outliers, so the gate's "
                "suppression target is correctly identified."
            )
        else:
            lines.append(
                "- **Tail regions are not more coherent than isolated outliers.** The "
                "gate's premise does not hold on this pool."
            )
        if separation["separates_tail_from_background"]:
            lines.append(
                "- Tail regions are also more coherent than background, so the gate can "
                "shift the budget toward true objects."
            )
        elif separation["tail_indistinguishable_from_background"]:
            lines.append(
                "- **Tail regions and background are indistinguishable in coherence** "
                f"({_format_float(separation['median_true_tail'])} versus "
                f"{_format_float(separation['median_background'])}, inside the "
                f"{separation['minimum_relative_margin']:.0%} margin). The gate has "
                "nothing to steer with: whatever it removes, what it promotes instead "
                "is as likely to be background as a rare object."
            )
        else:
            lines.append(
                "- **Background is at least as coherent as tail regions.** The gate can "
                "still remove isolated outliers, but what it promotes instead is "
                "coherent background, so a tail-discovery gain is not expected — which "
                "is the diagnosis to read alongside any negative contrast above."
            )
        lines.append("")

    if pilot:
        lines.append("## Pilot hyperparameter choice")
        lines.append("")
        lines.append(
            f"Coherence definition `{pilot.get('chosen_coherence_method')}` with "
            f"k={pilot.get('chosen_neighbour_count')}, chosen on a disjoint pilot pool "
            f"of {pilot.get('pilot_pool_size')} proposals by: {pilot.get('criterion')}. "
            "The evaluation pool was untouched at selection time."
        )
        lines.append("")

    if ablation_rows:
        lines.append("## Ablations")
        lines.append("")
        by_form: dict[str, list[float]] = {}
        for row in ablation_rows:
            try:
                value = float(row.get("tail_discovery_auc", float("nan")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                by_form.setdefault(str(row.get("gate_form", "?")), []).append(value)
        lines.append(
            _table(
                ["gate form", "cells", "mean tail AUC", "best tail AUC"],
                [
                    [
                        form,
                        len(values),
                        _format_float(float(np.mean(values))),
                        _format_float(float(np.max(values))),
                    ]
                    for form, values in sorted(by_form.items())
                ],
            )
        )
        lines.append("")
        lines.append(
            "`multiplicative_gate` is the proposed composition, `additive` splits the "
            "same weight between rarity and coherence, and `no_coherence` drops the "
            "gate. The comparison against `additive` is the one that isolates the "
            "gate's *form* rather than the presence of a coherence signal."
        )
        lines.append("")

    if arm_rows:
        lines.append("## Every arm against the baseline")
        lines.append("")
        lines.append(
            "Paired by seed and severity. `consistent sign` means the arm beat or lost "
            "to the baseline in *every* seed of that severity."
        )
        lines.append("")
        lines.append(
            _table(
                ["severity", "arm", "family", "mean", "vs baseline", "consistent sign"],
                [
                    [
                        row["imbalance_setting"],
                        row["strategy"],
                        row["strategy_family"],
                        _format_float(row["mean"]),
                        _format_float(row["mean_difference_vs_baseline"]),
                        (
                            "baseline"
                            if row["is_baseline"]
                            else (
                                "all better"
                                if row["all_seeds_better"]
                                else ("all worse" if row["all_seeds_worse"] else "mixed")
                            )
                        ),
                    ]
                    for row in arm_rows
                ],
            )
        )
        lines.append("")

    lines.append("## Leakage controls")
    lines.append("")
    lines.append(
        _table(
            ["control", "result"],
            [[str(key), str(value)] for key, value in sorted(leakage.items())],
        )
    )
    lines.append("")
    lines.append(
        "The strongest of these re-derives every acquisition score from its recorded "
        "components: an unrecorded ground-truth term would break the identity and "
        "the run would stop. Ground truth enters only through the oracle, after "
        "selection, and through the protocol step that builds the long-tail pool."
    )
    lines.append("")

    lines.append("## Runtime")
    lines.append("")
    lines.append(
        _table(
            ["stage", "seconds"],
            [[str(key), _format_float(value, 1)] for key, value in sorted(runtime.items())],
        )
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for item in limitations:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _verdict_section(contrasts: Sequence[Mapping[str, object]]) -> str:
    """A stated verdict per comparison, including "not supported"."""

    if not contrasts:
        return "No paired contrast could be computed, so no verdict is stated."
    lines = ["**Verdict.**"]
    for label in (
        "gate vs random",
        "gate vs uncertainty",
        "gate vs uncertainty+novelty",
        "gate vs ungated rarity",
    ):
        rows = [row for row in contrasts if row.get("comparison") == label]
        if not rows:
            continue
        positives = sum(1 for row in rows if float(row["mean_difference"]) > 0)  # type: ignore[arg-type]
        consistent = sum(1 for row in rows if row.get("all_seeds_positive"))
        if positives == len(rows) and consistent == len(rows):
            verdict = "supported in every severity, with a consistent sign across seeds"
        elif positives == len(rows):
            verdict = "positive in every severity, but the sign varies across seeds"
        elif positives == 0:
            verdict = "not supported: the difference is negative in every severity"
        else:
            verdict = f"mixed: positive in {positives} of {len(rows)} severities"
        lines.append(f"- {label}: {verdict}.")
    return "\n".join(lines)


def bundle(
    directory: str | Path,
    *,
    archive: str | Path,
    patterns: Sequence[str] = ("*.csv", "*.json", "*.md", "*.png", "*.pdf"),
) -> Path:
    """ZIP the report directory, excluding the proposal cache.

    The export can be a gigabyte; it is reproducible from the checkpoint and the
    split file, so it stays out of the archive while every derived artifact goes
    in. The archive is what a reader needs, not what the run needed.
    """

    root = Path(directory)
    target = Path(archive)
    target.parent.mkdir(parents=True, exist_ok=True)
    members: list[Path] = []
    for pattern in patterns:
        members.extend(sorted(root.rglob(pattern)))
    unique = sorted({item.resolve() for item in members if item.is_file()})
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for item in unique:
            handle.write(item, arcname=str(item.relative_to(root.resolve())))
    return target


def verify_expected_files(directory: str | Path, names: Sequence[str]) -> list[dict[str, object]]:
    """Existence and size of every promised artifact, as rows."""

    root = Path(directory)
    report: list[dict[str, object]] = []
    for name in names:
        path = root / name
        report.append(
            {
                "artifact": name,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "status": "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL",
            }
        )
    return report
