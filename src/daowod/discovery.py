"""Discovery, annotation-efficiency, outlier-robustness and diversity metrics.

``daowod.metrics`` decomposes the *detector's* official numbers (known mAP,
U-Recall, WI, A-OSE) into frequency groups. This module measures something the
repository has never measured: the quality of the **annotation set itself**, as a
function of how many regions were annotated.

Discovery counts objects, not proposals
---------------------------------------
Every recall here is over distinct ground-truth object indices
(:attr:`daowod.oracle.OracleTable.gt_object_index`). Forty proposals on one dog
are one discovery. Counting proposals instead would reward a strategy for
flooding a single easy object, which is precisely the redundancy the diversity
term is supposed to prevent, so a proposal-counting metric would score the
failure mode as a success.

Denominators come from the pool, not the dataset
------------------------------------------------
An unknown object that no candidate proposal covers is unreachable for every
strategy, so it belongs in neither numerator nor denominator; including it would
scale all recalls down by the same constant and compress the contrast being
measured. :func:`daowod.longtail.discoverable_objects` supplies the reachable
sets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from daowod.active import CampaignResult, ProposalPool
from daowod.groups import GROUP_NAMES
from daowod.oracle import OracleTable

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

#: Groups reported by every discovery metric, plus the pooled total.
REPORT_GROUPS: tuple[str, ...] = ("all", *GROUP_NAMES)


class MetricError(ValueError):
    """Raised when a metric is asked for something its inputs cannot support."""


@dataclass(frozen=True)
class DiscoveryTargets:
    """The reachable denominators for one pool."""

    objects_by_group: Mapping[str, set[int]]
    classes_by_group: Mapping[str, set[str]]

    def object_total(self, group: str) -> int:
        return len(self.objects_by_group.get(group, set()))

    def class_total(self, group: str) -> int:
        return len(self.classes_by_group.get(group, set()))


def normalised_auc(budgets: Sequence[int], values: Sequence[float]) -> float:
    """Trapezoidal area under a budget curve, divided by the budget range.

    Normalising by the range makes the number a *mean recall over the budget
    sweep* in [0, 1] rather than an area whose magnitude depends on the largest
    budget, so AUCs from runs with different budget grids remain comparable — and
    a single-budget curve degenerates to that budget's value instead of zero.
    """

    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.shape != y.shape:
        raise MetricError("budgets and values must be parallel.")
    if x.size == 0:
        raise MetricError("Cannot integrate an empty curve.")
    if x.size == 1:
        return float(y[0])
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = float(x[-1] - x[0])
    if span <= 0:
        return float(y.mean())
    return float(np.trapezoid(y, x) / span)


def budget_row(
    *,
    strategy: str,
    seed: int,
    imbalance_setting: str,
    budget: int,
    selected: ArrayLike,
    pool: ProposalPool,
    oracle: OracleTable,
    targets: DiscoveryTargets,
    isolated: ArrayLike | None = None,
) -> dict[str, object]:
    """Every required metric for one (strategy, seed, severity, budget) cell."""

    indices = np.asarray(selected, dtype=np.int64)
    count = int(indices.size)
    if count == 0:
        raise MetricError("Cannot score an empty selection.")

    kinds = oracle.gt_match_kind[indices]
    groups = oracle.gt_group[indices]
    classes = oracle.gt_class[indices]
    object_indices = oracle.gt_object_index[indices]
    unknown_mask = oracle.gt_is_unknown[indices]

    row: dict[str, object] = {
        "strategy": strategy,
        "seed": int(seed),
        "imbalance_setting": str(imbalance_setting),
        "budget": int(budget),
        "annotated_proposals": count,
    }

    # --- discovery: distinct objects reached, per group ----------------------
    for group in REPORT_GROUPS:
        reachable = targets.objects_by_group.get(group, set())
        if group == "all":
            found = {
                int(value)
                for value, flag in zip(object_indices.tolist(), unknown_mask.tolist(), strict=True)
                if flag and int(value) >= 0
            }
        else:
            found = {
                int(value)
                for value, name in zip(object_indices.tolist(), groups.tolist(), strict=True)
                if str(name) == group and int(value) >= 0
            }
        found &= reachable
        total = len(reachable)
        row[f"{group}_objects_found"] = len(found)
        row[f"{group}_objects_reachable"] = total
        row[f"{group}_discovery_recall"] = float(len(found) / total) if total else float("nan")

    # --- unique classes discovered ------------------------------------------
    for group in REPORT_GROUPS:
        if group == "all":
            names = {
                str(name)
                for name, flag in zip(classes.tolist(), unknown_mask.tolist(), strict=True)
                if flag and str(name)
            }
        else:
            names = {
                str(name)
                for name, item in zip(classes.tolist(), groups.tolist(), strict=True)
                if str(item) == group and str(name)
            }
        names &= targets.classes_by_group.get(group, set())
        total = targets.class_total(group)
        row[f"{group}_unique_classes"] = len(names)
        row[f"{group}_class_coverage"] = float(len(names) / total) if total else float("nan")

    # --- annotation efficiency ----------------------------------------------
    true_unknown = int(unknown_mask.sum())
    row["true_unknown_proposals"] = true_unknown
    row["annotation_precision"] = float(true_unknown / count)
    row["known_object_proposals"] = int((kinds == "known").sum())
    row["on_object_precision"] = float((kinds != "background").mean())
    for group in GROUP_NAMES:
        row[f"{group}_instances_annotated"] = int((groups == group).sum())

    # --- outlier robustness -------------------------------------------------
    row["background_selection_rate"] = float((kinds == "background").mean())
    row["false_positive_selection_rate"] = float((kinds == "background").mean())
    if isolated is not None:
        isolated_flags = np.asarray(isolated, dtype=np.bool_)
        if isolated_flags.shape[0] != pool.size:
            raise MetricError("isolated must be parallel to the pool.")
        chosen_isolated = isolated_flags[indices]
        row["isolated_selection_rate"] = float(chosen_isolated.mean())
        row["isolated_background_selection_rate"] = float(
            (chosen_isolated & (kinds == "background")).mean()
        )
    else:
        row["isolated_selection_rate"] = float("nan")
        row["isolated_background_selection_rate"] = float("nan")

    # --- diversity ----------------------------------------------------------
    row.update(diversity_metrics(pool=pool, selected=indices))
    return row


def diversity_metrics(*, pool: ProposalPool, selected: ArrayLike) -> dict[str, float]:
    """Spread of a selection in embedding space and across images.

    ``mean_pairwise_distance`` is computed on L2-normalised embeddings, so it is a
    cosine distance in [0, 2]. For large selections the pairwise matrix is
    subsampled deterministically to keep the metric O(1) in memory; the cap is
    reported so the number is never silently an estimate.
    """

    indices = np.asarray(selected, dtype=np.int64)
    if indices.size == 0:
        raise MetricError("Cannot measure the diversity of an empty selection.")
    vectors = pool.embeddings[indices]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-12)

    cap = 2000
    if unit.shape[0] > cap:
        step = unit.shape[0] / cap
        sample = np.unique((np.arange(cap) * step).astype(np.int64))
        sampled = unit[sample]
        subsampled = True
    else:
        sampled = unit
        subsampled = False
    if sampled.shape[0] < 2:
        mean_distance = 0.0
    else:
        similarity = sampled @ sampled.T
        upper = similarity[np.triu_indices(sampled.shape[0], k=1)]
        mean_distance = float(np.mean(1.0 - upper))

    image_ids = np.asarray([str(value) for value in pool.image_ids[indices].tolist()], dtype=object)
    unique_images, counts = np.unique(image_ids, return_counts=True)
    return {
        "mean_pairwise_distance": mean_distance,
        "pairwise_distance_subsampled": float(subsampled),
        "distinct_images": float(unique_images.size),
        "proposals_per_image": float(indices.size / max(unique_images.size, 1)),
        "max_proposals_from_one_image": float(counts.max()),
        "image_redundancy": float(1.0 - unique_images.size / indices.size),
    }


def cluster_concentration(*, selected: ArrayLike, pseudo_labels: ArrayLike) -> dict[str, float]:
    """How concentrated a selection is across pseudo-clusters.

    Reported as a normalised Herfindahl index: 0 means the selection is spread
    evenly over the clusters it touches, 1 means it all sits in one cluster.
    """

    indices = np.asarray(selected, dtype=np.int64)
    labels = np.asarray(pseudo_labels, dtype=np.int64)[indices]
    valid = labels[labels >= 0]
    if valid.size == 0:
        return {"clusters_touched": 0.0, "cluster_concentration": float("nan")}
    _, counts = np.unique(valid, return_counts=True)
    shares = counts / counts.sum()
    herfindahl = float((shares**2).sum())
    clusters = float(counts.size)
    lower = 1.0 / clusters
    normalised = 0.0 if clusters <= 1 else float((herfindahl - lower) / (1.0 - lower))
    return {
        "clusters_touched": clusters,
        "cluster_concentration": max(0.0, min(1.0, normalised)),
    }


def campaign_rows(
    *,
    result: CampaignResult,
    pool: ProposalPool,
    oracle: OracleTable,
    targets: DiscoveryTargets,
    budgets: Sequence[int],
    isolated: ArrayLike | None = None,
) -> list[dict[str, object]]:
    """Budget-curve rows for one campaign, plus its AUC summary row fields."""

    rows: list[dict[str, object]] = []
    for budget in sorted(int(value) for value in budgets):
        selected = result.prefix(budget)
        if selected.size == 0:
            continue
        row = budget_row(
            strategy=result.strategy,
            seed=result.seed,
            imbalance_setting=result.imbalance_setting,
            budget=budget,
            selected=selected,
            pool=pool,
            oracle=oracle,
            targets=targets,
            isolated=isolated,
        )
        if result.rounds:
            row.update(
                cluster_concentration(
                    selected=selected, pseudo_labels=result.rounds[0].pseudo_labels
                )
            )
        row["realised_budget"] = int(selected.size)
        rows.append(row)
    return rows


def auc_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Collapse one campaign's budget curve into the headline AUCs."""

    if not rows:
        raise MetricError("Cannot summarise an empty curve.")
    budgets = [int(row["budget"]) for row in rows]
    summary: dict[str, float] = {}
    for group in REPORT_GROUPS:
        key = f"{group}_discovery_recall"
        values = [float(row[key]) for row in rows]
        if not any(np.isnan(values)):
            summary[f"{group}_discovery_auc"] = normalised_auc(budgets, values)
        coverage = f"{group}_class_coverage"
        values = [float(row[coverage]) for row in rows]
        if not any(np.isnan(values)):
            summary[f"{group}_class_coverage_auc"] = normalised_auc(budgets, values)
    for key in (
        "annotation_precision",
        "background_selection_rate",
        "isolated_selection_rate",
        "mean_pairwise_distance",
    ):
        values = [float(row[key]) for row in rows]
        if not any(np.isnan(values)):
            summary[f"{key}_auc"] = normalised_auc(budgets, values)
    largest = max(budgets)
    final = next(row for row in rows if int(row["budget"]) == largest)
    for group in REPORT_GROUPS:
        summary[f"final_{group}_discovery_recall"] = float(final[f"{group}_discovery_recall"])
        summary[f"final_{group}_unique_classes"] = float(final[f"{group}_unique_classes"])
    summary["final_annotation_precision"] = float(final["annotation_precision"])
    summary["final_background_selection_rate"] = float(final["background_selection_rate"])
    summary["final_isolated_selection_rate"] = float(final["isolated_selection_rate"])
    return summary


def aggregate_over_seeds(
    rows: Sequence[Mapping[str, object]],
    *,
    keys: Sequence[str] = ("strategy", "imbalance_setting", "budget"),
) -> list[dict[str, object]]:
    """Mean, sample standard deviation and n over seeds, per cell.

    Sample sd (``ddof=1``) is used because the seeds are a sample of the
    acquisition's randomness, not the population; with a single seed the sd is
    reported as NaN rather than 0, so an under-powered cell cannot be mistaken
    for a perfectly stable one.
    """

    if not rows:
        return []
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(row)
    numeric = [
        name
        for name, value in rows[0].items()
        if name not in keys and isinstance(value, (int, float, np.floating, np.integer))
    ]
    aggregated: list[dict[str, object]] = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        entry: dict[str, object] = dict(zip(keys, key, strict=True))
        entry["seeds"] = len(members)
        for name in numeric:
            values = np.array(
                [float(member[name]) for member in members if member.get(name) is not None],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                entry[f"{name}_mean"] = float("nan")
                entry[f"{name}_sd"] = float("nan")
                continue
            entry[f"{name}_mean"] = float(values.mean())
            entry[f"{name}_sd"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        aggregated.append(entry)
    return aggregated


def score_distribution_rows(
    *,
    strategy: str,
    seed: int,
    imbalance_setting: str,
    components: Mapping[str, ArrayLike],
    oracle: OracleTable,
    isolated: ArrayLike,
    available: ArrayLike | None = None,
) -> list[dict[str, object]]:
    """Rarity/coherence distributions per oracle stratum.

    The plan asks specifically for the score distribution over *true tail, head,
    background and isolated outlier* proposals, because the mechanism's claim is
    about the ordering between those strata, not about an aggregate.
    """

    isolated_flags = np.asarray(isolated, dtype=np.bool_)
    kinds = oracle.gt_match_kind
    groups = oracle.gt_group
    strata: dict[str, BoolArray] = {
        "true_head": (groups == "head"),
        "true_medium": (groups == "medium"),
        "true_tail": (groups == "tail"),
        "known_object": (kinds == "known"),
        "background": (kinds == "background") & ~isolated_flags,
        "isolated_outlier": (kinds == "background") & isolated_flags,
    }
    if available is not None:
        scope = np.zeros(isolated_flags.shape[0], dtype=np.bool_)
        scope[np.asarray(available, dtype=np.int64)] = True
        strata = {name: mask & scope for name, mask in strata.items()}

    rows: list[dict[str, object]] = []
    for component, values in components.items():
        array = np.asarray(values, dtype=np.float64)
        for stratum, mask in strata.items():
            selected = array[mask]
            selected = selected[np.isfinite(selected)]
            rows.append(
                {
                    "strategy": strategy,
                    "seed": int(seed),
                    "imbalance_setting": str(imbalance_setting),
                    "component": component,
                    "stratum": stratum,
                    "n": int(selected.size),
                    "mean": float(selected.mean()) if selected.size else float("nan"),
                    "median": float(np.median(selected)) if selected.size else float("nan"),
                    "p10": float(np.quantile(selected, 0.1)) if selected.size else float("nan"),
                    "p90": float(np.quantile(selected, 0.9)) if selected.size else float("nan"),
                }
            )
    return rows


def gate_suppression(
    *,
    rarity: ArrayLike,
    coherence: ArrayLike,
    gated: ArrayLike,
    oracle: OracleTable,
    isolated: ArrayLike,
    budget: int,
    available: ArrayLike | None = None,
) -> dict[str, object]:
    """How many high-rarity isolated outliers the gate removed from the top-K.

    This is the counterfactual the plan asks for: rank the pool by ungated rarity
    and by the gated interaction, then count the isolated / background proposals
    that the ungated ranking would have bought and the gated ranking does not.
    A gate that suppresses nothing is not gating.
    """

    rarity_values = np.asarray(rarity, dtype=np.float64)
    gated_values = np.asarray(gated, dtype=np.float64)
    coherence_values = np.asarray(coherence, dtype=np.float64)
    isolated_flags = np.asarray(isolated, dtype=np.bool_)
    scope = (
        np.asarray(available, dtype=np.int64)
        if available is not None
        else np.flatnonzero(np.isfinite(rarity_values))
    )
    take = int(min(budget, scope.size))
    if take < 1:
        raise MetricError("gate_suppression needs a positive budget.")

    def top(values: FloatArray) -> set[int]:
        candidate = scope[np.isfinite(values[scope])]
        order = candidate[np.argsort(-values[candidate], kind="stable")]
        return {int(value) for value in order[:take].tolist()}

    ungated_top = top(rarity_values)
    gated_top = top(gated_values)
    removed = np.array(sorted(ungated_top - gated_top), dtype=np.int64)
    added = np.array(sorted(gated_top - ungated_top), dtype=np.int64)

    def profile(indices: IntArray, prefix: str) -> dict[str, object]:
        if indices.size == 0:
            return {
                f"{prefix}_count": 0,
                f"{prefix}_isolated": 0,
                f"{prefix}_background": 0,
                f"{prefix}_true_unknown": 0,
                f"{prefix}_tail": 0,
            }
        return {
            f"{prefix}_count": int(indices.size),
            f"{prefix}_isolated": int(isolated_flags[indices].sum()),
            f"{prefix}_background": int((oracle.gt_match_kind[indices] == "background").sum()),
            f"{prefix}_true_unknown": int(oracle.gt_is_unknown[indices].sum()),
            f"{prefix}_tail": int((oracle.gt_group[indices] == "tail").sum()),
        }

    report: dict[str, object] = {
        "budget": int(take),
        "mean_coherence_of_removed": (
            float(coherence_values[removed].mean()) if removed.size else float("nan")
        ),
        "mean_coherence_of_added": (
            float(coherence_values[added].mean()) if added.size else float("nan")
        ),
        "overlap": int(len(ungated_top & gated_top)),
    }
    report.update(profile(removed, "suppressed"))
    report.update(profile(added, "promoted"))
    report["isolated_suppression_gain"] = int(
        report["suppressed_isolated"] - report["promoted_isolated"]
    )
    report["true_unknown_gain"] = int(
        report["promoted_true_unknown"] - report["suppressed_true_unknown"]
    )
    report["tail_gain"] = int(report["promoted_tail"] - report["suppressed_tail"])
    return report


def assert_selection_is_ground_truth_free(
    *,
    scores: ArrayLike,
    components: Mapping[str, ArrayLike],
    spec_weights: Mapping[str, float],
    tolerance: float = 1e-8,
) -> None:
    """Re-derive the score from its components and refuse any unexplained term.

    A ground-truth term smuggled into the acquisition score would make the
    recorded components fail to reproduce the score. This is a stronger check
    than :func:`daowod.oracle.assert_no_ground_truth`, which only inspects
    column *names*.
    """

    values = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(values)
    rebuilt = np.zeros(values.shape[0], dtype=np.float64)
    for name, weight in spec_weights.items():
        if weight <= 0:
            continue
        if name not in components:
            raise MetricError(f"Weighted component {name!r} was not recorded.")
        component = np.asarray(components[name], dtype=np.float64)
        rebuilt = rebuilt + float(weight) * np.nan_to_num(component, nan=0.0)
    difference = np.abs(rebuilt[finite] - values[finite])
    if difference.size and float(difference.max()) > tolerance:
        raise MetricError(
            "The recorded components do not reproduce the acquisition score "
            f"(max deviation {float(difference.max()):.3e} > {tolerance:.1e}). "
            "An unrecorded term is influencing selection."
        )
