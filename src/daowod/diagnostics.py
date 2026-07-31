"""Scientific diagnostics for the acquisition score.

Four questions from the audit, one module:

* **S1 / Step 2** — how far apart are the strategies, and how many seeds would be
  needed to tell them apart? :func:`strategy_separation`, :func:`power_estimate`.
* **S4 / S5 / Step 4** — is rarity continuous, and is coherence informative,
  frequency-confounded, saturated or inactive? :func:`component_diagnostics`,
  :func:`coherence_regime`.
* **S2 / Step 5** — does posterior entropy carry information the PROB unknown
  score does not? :func:`uncertainty_comparison`.
* **Step 9** — the per-proposal record. :func:`proposal_table`, and
  :func:`assert_no_ground_truth`, which is the automated leakage guard.

Ground-truth separation is structural, not conventional: :func:`proposal_table`
cannot accept ground truth (there is no parameter for it), and
:func:`join_ground_truth` is the only way to add it, producing rows tagged as
post-hoc.
"""

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from daowod.groups import GROUP_NAMES, ClassGroups
from daowod.normalisation import average_ranks
from daowod.scoring import ScoringResult, aggregate_image_scores, select_images

FloatArray = NDArray[np.float64]

#: Fields that must never appear in an acquisition-time artifact.
GROUND_TRUTH_FIELDS: tuple[str, ...] = (
    "gt_class",
    "gt_classes",
    "gt_group",
    "gt_unknown",
    "ground_truth",
    "true_class",
    "label",
)

#: Cluster-size regimes the audit identified for coherence behaviour (S5).
CLUSTER_SIZE_REGIMES: tuple[tuple[str, int, int], ...] = (
    ("singleton", 1, 1),
    ("2_to_3", 2, 3),
    ("4_to_5", 4, 5),
    ("6_or_more", 6, 1 << 30),
)


# --- small statistics helpers (numpy only, no new dependency) ----------------


def pearson(first: ArrayLike, second: ArrayLike) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(first: ArrayLike, second: ArrayLike) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    return pearson(average_ranks(a), average_ranks(b))


def cohens_d(first: ArrayLike, second: ArrayLike) -> float:
    """Standardised mean difference with a pooled standard deviation."""

    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = math.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)
    )
    if pooled < 1e-12:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def jaccard(first: Sequence[str], second: Sequence[str]) -> float:
    left, right = set(map(str, first)), set(map(str, second))
    union = left | right
    return float(len(left & right) / len(union)) if union else float("nan")


def summarise(
    values: ArrayLike, *, quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    summary: dict[str, float] = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "distinct": int(np.unique(array).size),
        "fraction_near_zero": float((array <= 0.01).mean()),
        "fraction_near_one": float((array >= 0.99).mean()),
        "fraction_below_0_1": float((array < 0.1).mean()),
    }
    for quantile in quantiles:
        summary[f"q{int(quantile * 100):02d}"] = float(np.quantile(array, quantile))
    return summary


def histogram(values: ArrayLike, *, bins: int = 20) -> dict[str, list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(array, bins=bins)
    return {"edges": [float(v) for v in edges], "counts": [int(v) for v in counts]}


# --- Step 9: the per-proposal record ----------------------------------------


def proposal_table(
    result: ScoringResult,
    *,
    run_id: str,
    seed: int,
    round_index: int,
    selected_image_ids: Sequence[str],
    posterior: ArrayLike | None = None,
    confidence: ArrayLike | None = None,
    predicted_labels: ArrayLike | None = None,
) -> list[dict[str, object]]:
    """One row per candidate proposal, acquisition-time only.

    This function has no ground-truth parameter by design; see
    :func:`join_ground_truth`.
    """

    selected_images = {str(value) for value in selected_image_ids}
    selected_proposals = result.selected_proposal_mask(selected_image_ids)
    image_scores = result.image_scores
    spec = result.spec

    posterior_array = np.asarray(posterior, dtype=np.float64) if posterior is not None else None
    if posterior_array is not None:
        mass = posterior_array.sum(axis=1, keepdims=True)
        probabilities = posterior_array / np.maximum(mass, 1e-12)
        posterior_max = probabilities.max(axis=1)
        posterior_argmax = probabilities.argmax(axis=1)
        ordered = np.sort(probabilities, axis=1)
        posterior_margin = ordered[:, -1] - ordered[:, -2]
        posterior_entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1) / math.log(
            probabilities.shape[1]
        )
    else:
        posterior_max = posterior_argmax = posterior_margin = posterior_entropy = None

    confidence_array = np.asarray(confidence, dtype=np.float64) if confidence is not None else None
    predicted = (
        np.asarray(predicted_labels, dtype=np.int64) if predicted_labels is not None else None
    )

    rows: list[dict[str, object]] = []
    for index in range(result.proposal_count):
        image_id = str(result.image_ids[index])
        row: dict[str, object] = {
            "run_id": run_id,
            "seed": seed,
            "round": round_index,
            "strategy": spec.name,
            "semantics_version": spec.semantics_version,
            "image_id": image_id,
            "proposal_index": int(result.proposal_index[index]),
            "cluster_id": int(result.pseudo_labels[index]),
            "cluster_size": int(result.cluster_sizes[index]),
            "cluster_size_regime": cluster_size_regime(int(result.cluster_sizes[index])),
            "predicted_class_index": (int(predicted[index]) if predicted is not None else ""),
            "unknown_score": (
                float(confidence_array[index]) if confidence_array is not None else ""
            ),
            "posterior_max": (float(posterior_max[index]) if posterior_max is not None else ""),
            "posterior_argmax": (
                int(posterior_argmax[index]) if posterior_argmax is not None else ""
            ),
            "posterior_margin": (
                float(posterior_margin[index]) if posterior_margin is not None else ""
            ),
            "posterior_entropy": (
                float(posterior_entropy[index]) if posterior_entropy is not None else ""
            ),
            "kth_neighbour_distance": float(result.kth_distance[index]),
            "isolated_outlier": bool(result.isolated[index]),
            "proposal_score": float(result.scores[index]),
            "image_score": float(image_scores.get(image_id, float("nan"))),
            "proposal_selected": bool(selected_proposals[index]),
            "image_selected": image_id in selected_images,
        }
        for component in ("uncertainty", "novelty", "rarity", "coherence", "gated"):
            row[f"raw_{component}"] = float(result.raw[component][index])
            row[f"norm_{component}"] = float(result.normalised[component][index])
        rows.append(row)
    return rows


def cluster_size_regime(size: int) -> str:
    for name, low, high in CLUSTER_SIZE_REGIMES:
        if low <= size <= high:
            return name
    return "unknown"


class LeakageError(AssertionError):
    """Raised when an acquisition-time artifact contains ground truth."""


def assert_no_ground_truth(rows: Sequence[Mapping[str, object]]) -> None:
    """Automated leakage guard for acquisition-time artifacts."""

    if not rows:
        return
    present = sorted(set().union(*(set(row) for row in rows)))
    offending = [
        field for field in present if field in GROUND_TRUTH_FIELDS or field.startswith("gt_")
    ]
    if offending:
        raise LeakageError(
            "Acquisition-time proposal records contain ground-truth fields "
            f"{offending}. Ground truth may only be joined post hoc via "
            "join_ground_truth()."
        )


def join_ground_truth(
    rows: Sequence[Mapping[str, object]],
    *,
    image_classes: Mapping[str, Sequence[str]],
    class_groups: ClassGroups,
    unknown_classes: Sequence[str],
) -> list[dict[str, object]]:
    """Post-hoc analysis join. Never call this before or during selection.

    Ground truth is attached at *image* level because a proposal's box is not
    matched to an object here; the resulting fields answer "what was in the image
    this proposal came from", which is what the selection analysis needs.
    """

    assert_no_ground_truth(rows)
    unknown_set = set(unknown_classes)
    joined: list[dict[str, object]] = []
    for row in rows:
        image_id = str(row["image_id"])
        classes = [str(name) for name in image_classes.get(image_id, ())]
        relevant = [name for name in classes if name in unknown_set]
        groups = [class_groups.groups.get(name, "") for name in relevant]
        joined.append(
            {
                **dict(row),
                "analysis_stage": "post_hoc",
                "gt_classes": "|".join(sorted(set(relevant))),
                "gt_class_count": len(relevant),
                "gt_unknown_present": bool(relevant),
                "gt_groups": "|".join(sorted(set(group for group in groups if group))),
                "gt_has_tail": "tail" in groups,
                "gt_has_medium": "medium" in groups,
                "gt_has_head": "head" in groups,
            }
        )
    return joined


def write_rows(path: str | Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write dict rows to CSV with a stable, union-of-keys header."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])
    return target


# --- Step 4: rarity and coherence diagnostics -------------------------------


def coherence_regime(
    coherence: ArrayLike,
    cluster_sizes: ArrayLike,
    *,
    saturation_spread: float = 0.05,
    confound_correlation: float = 0.3,
    inactive_spread: float = 0.01,
) -> dict[str, object]:
    """Classify what the coherence term is actually doing in this pool.

    ``inactive``              spread below ``inactive_spread``: a constant factor
                              that only rescales the gate's weight
    ``saturated``             spread below ``saturation_spread``: nearly constant,
                              so gating barely reorders anything
    ``frequency_confounded``  |Spearman(coherence, cluster size)| above
                              ``confound_correlation``: coherence is acting as a
                              class-frequency proxy, which is S5
    ``informative``           varies materially and is not frequency-driven
    """

    values = np.asarray(coherence, dtype=np.float64)
    sizes = np.asarray(cluster_sizes, dtype=np.float64)
    if values.size == 0:
        return {"regime": "empty"}
    spread = float(values.std(ddof=1)) if values.size > 1 else 0.0
    correlation = spearman(values, sizes)
    if spread < inactive_spread:
        regime = "inactive"
    elif spread < saturation_spread:
        regime = "saturated"
    elif not math.isnan(correlation) and abs(correlation) > confound_correlation:
        regime = "frequency_confounded"
    else:
        regime = "informative"
    return {
        "regime": regime,
        "spread": spread,
        "spearman_with_cluster_size": correlation,
        "mean": float(values.mean()),
        "thresholds": {
            "inactive_spread": inactive_spread,
            "saturation_spread": saturation_spread,
            "confound_correlation": confound_correlation,
        },
    }


def gate_impact(
    rarity_normalised: ArrayLike,
    gated_normalised: ArrayLike,
    *,
    image_ids: ArrayLike | None = None,
    budget: int | None = None,
    top_k: int = 3,
) -> dict[str, object]:
    """How much does the coherence gate actually change the ranking?"""

    rarity = np.asarray(rarity_normalised, dtype=np.float64)
    gated = np.asarray(gated_normalised, dtype=np.float64)
    if rarity.size == 0:
        return {"proposals": 0}
    rarity_order = np.argsort(-rarity, kind="stable")
    gated_order = np.argsort(-gated, kind="stable")
    ranks_changed = float(np.mean(rarity_order != gated_order))
    top = max(1, rarity.size // 10)
    report: dict[str, object] = {
        "proposals": int(rarity.size),
        "spearman_rarity_vs_gated": spearman(rarity, gated),
        "fraction_of_positions_changed": ranks_changed,
        "top_decile_overlap": float(
            len(set(rarity_order[:top].tolist()) & set(gated_order[:top].tolist())) / top
        ),
        "mean_absolute_change": float(np.mean(np.abs(gated - rarity))),
    }
    if image_ids is not None and budget:
        rarity_images = select_images(
            aggregate_image_scores(image_ids, rarity, method="top_k_mean", top_k=top_k),
            budget=budget,
        )
        gated_images = select_images(
            aggregate_image_scores(image_ids, gated, method="top_k_mean", top_k=top_k),
            budget=budget,
        )
        report["selected_image_jaccard"] = jaccard(rarity_images, gated_images)
        report["selected_images_changed"] = int(len(set(gated_images) - set(rarity_images)))
    return report


def component_diagnostics(
    result: ScoringResult,
    *,
    budget: int | None = None,
    image_classes: Mapping[str, Sequence[str]] | None = None,
    class_groups: ClassGroups | None = None,
    unknown_classes: Sequence[str] = (),
) -> dict[str, object]:
    """Full real-pool diagnostic block for one scoring pass."""

    report: dict[str, object] = {
        "strategy": result.spec.name,
        "semantics_version": result.spec.semantics_version,
        "proposals": result.proposal_count,
        "images": len(result.image_scores),
        "coherence_method": result.spec.coherence_method,
        "rarity_method": result.spec.rarity_method,
        "normalisation": dict(result.diagnostics["normalisation"]),  # type: ignore[arg-type]
        "isolated_proposals": int(result.isolated.sum()),
        "isolated_fraction": float(result.isolated.mean())
        if result.proposal_count
        else float("nan"),
    }

    for component in ("uncertainty", "novelty", "rarity", "coherence", "gated"):
        report[f"raw_{component}"] = summarise(result.raw[component])
        report[f"norm_{component}"] = summarise(result.normalised[component])
        report[f"histogram_raw_{component}"] = histogram(result.raw[component])
        report[f"histogram_norm_{component}"] = histogram(result.normalised[component])

    sizes = result.cluster_sizes.astype(np.float64)
    report["cluster_sizes"] = summarise(sizes, quantiles=(0.1, 0.5, 0.9))
    # Decisive statistic for the legacy density coherence: when a pseudo-class
    # has fewer members than `neighbour_count`, its k-th nearest neighbour is
    # necessarily in another cluster, so absolute density collapses. Measured
    # tail/head coherence ratio is ~0.14 below the threshold and ~0.85+ above it.
    below = result.cluster_sizes <= result.spec.neighbour_count
    report["clusters_below_neighbour_count"] = {
        "neighbour_count": result.spec.neighbour_count,
        "proposals": int(below.sum()),
        "fraction_of_proposals": float(below.mean()) if below.size else float("nan"),
        "pseudo_classes": int(np.unique(result.pseudo_labels[below]).size if below.any() else 0),
        "mean_raw_coherence_below": float(result.raw["coherence"][below].mean())
        if below.any()
        else float("nan"),
        "mean_raw_coherence_above": float(result.raw["coherence"][~below].mean())
        if (~below).any()
        else float("nan"),
        "note": (
            "a large fraction here means the legacy 'density' coherence would be "
            "severely frequency-confounded in this pool"
        ),
    }
    report["by_cluster_size_regime"] = {}
    for name, low, high in CLUSTER_SIZE_REGIMES:
        mask = (result.cluster_sizes >= low) & (result.cluster_sizes <= high)
        if not mask.any():
            report["by_cluster_size_regime"][name] = {"proposals": 0}  # type: ignore[index]
            continue
        report["by_cluster_size_regime"][name] = {  # type: ignore[index]
            "proposals": int(mask.sum()),
            "mean_raw_coherence": float(result.raw["coherence"][mask].mean()),
            "mean_norm_coherence": float(result.normalised["coherence"][mask].mean()),
            "mean_norm_rarity": float(result.normalised["rarity"][mask].mean()),
            "mean_norm_gated": float(result.normalised["gated"][mask].mean()),
        }

    report["correlations"] = {
        "coherence_vs_cluster_size": spearman(result.raw["coherence"], sizes),
        "rarity_vs_cluster_size": spearman(result.normalised["rarity"], sizes),
        "rarity_vs_coherence": spearman(result.normalised["rarity"], result.raw["coherence"]),
        "uncertainty_vs_coherence": spearman(
            result.normalised["uncertainty"], result.raw["coherence"]
        ),
        "gated_vs_rarity": spearman(result.normalised["gated"], result.normalised["rarity"]),
    }
    report["coherence_regime"] = coherence_regime(result.raw["coherence"], result.cluster_sizes)
    report["gate_impact"] = gate_impact(
        result.normalised["rarity"],
        result.normalised["gated"],
        image_ids=result.image_ids,
        budget=budget,
        top_k=result.spec.top_k,
    )

    if image_classes is not None and class_groups is not None:
        report["by_ground_truth_group"] = _component_means_by_group(
            result,
            image_classes=image_classes,
            class_groups=class_groups,
            unknown_classes=unknown_classes,
        )
    return report


def _component_means_by_group(
    result: ScoringResult,
    *,
    image_classes: Mapping[str, Sequence[str]],
    class_groups: ClassGroups,
    unknown_classes: Sequence[str],
) -> dict[str, object]:
    """Post-hoc: component values split by the ground-truth frequency group."""

    unknown_set = set(unknown_classes)
    labels: list[str] = []
    for image_id in result.image_ids.tolist():
        classes = [name for name in image_classes.get(str(image_id), ()) if name in unknown_set]
        groups = {class_groups.groups.get(name, "") for name in classes}
        if "tail" in groups:
            labels.append("tail")
        elif "medium" in groups:
            labels.append("medium")
        elif "head" in groups:
            labels.append("head")
        else:
            labels.append("none")
    label_array = np.asarray(labels, dtype=object)
    output: dict[str, object] = {}
    for group in (*GROUP_NAMES, "none"):
        mask = label_array == group
        if not mask.any():
            output[group] = {"proposals": 0}
            continue
        output[group] = {
            "proposals": int(mask.sum()),
            **{
                f"mean_norm_{component}": float(result.normalised[component][mask].mean())
                for component in ("uncertainty", "novelty", "rarity", "coherence", "gated")
            },
            "mean_raw_coherence": float(result.raw["coherence"][mask].mean()),
            "mean_proposal_score": float(result.scores[mask].mean()),
            "isolated_fraction": float(result.isolated[mask].mean()),
        }
    for component in ("rarity", "coherence", "gated"):
        head = label_array == "head"
        tail = label_array == "tail"
        if head.any() and tail.any():
            output[f"tail_over_head_{component}"] = float(
                result.normalised[component][tail].mean()
                / max(result.normalised[component][head].mean(), 1e-12)
            )
            output[f"cohens_d_tail_vs_head_{component}"] = cohens_d(
                result.normalised[component][tail], result.normalised[component][head]
            )
    return output


# --- Step 5: uncertainty comparison -----------------------------------------


def uncertainty_comparison(
    *,
    posterior: ArrayLike,
    confidence: ArrayLike,
    image_ids: ArrayLike,
    budget: int | None = None,
    top_k: int = 3,
    top_fraction: float = 0.1,
) -> dict[str, object]:
    """Does posterior uncertainty carry information the unknown score does not?

    The audit's finding was Spearman(legacy uncertainty, unknown score) = +1.000.
    Any replacement must break that identity to be worth having.
    """

    from daowod import components  # local import keeps the module import-light

    unknown_score = np.asarray(confidence, dtype=np.float64)
    methods = {
        method: components.compute_uncertainty(
            method=method, posterior=posterior, confidence=unknown_score
        )
        for method in ("entropy", "margin", "one_minus_max", "legacy_prob_score")
    }

    top = max(1, int(round(unknown_score.size * top_fraction)))
    score_top = set(np.argsort(-unknown_score, kind="stable")[:top].tolist())
    report: dict[str, object] = {
        "proposals": int(unknown_score.size),
        "top_fraction": top_fraction,
        "unknown_score": summarise(unknown_score),
        "methods": {},
    }
    for method, values in methods.items():
        entry: dict[str, object] = {
            "distribution": summarise(values),
            "spearman_with_unknown_score": spearman(values, unknown_score),
            "pearson_with_unknown_score": pearson(values, unknown_score),
            "top_decile_overlap_with_unknown_score": float(
                len(set(np.argsort(-values, kind="stable")[:top].tolist()) & score_top) / top
            ),
            "is_monotone_in_unknown_score": bool(abs(spearman(values, unknown_score)) > 0.999),
        }
        if budget:
            method_images = select_images(
                aggregate_image_scores(image_ids, values, method="top_k_mean", top_k=top_k),
                budget=budget,
            )
            score_images = select_images(
                aggregate_image_scores(image_ids, unknown_score, method="top_k_mean", top_k=top_k),
                budget=budget,
            )
            entry["selected_image_jaccard_with_unknown_score"] = jaccard(
                method_images, score_images
            )
            entry["selected_images"] = method_images
        report["methods"][method] = entry  # type: ignore[index]

    report["entropy_vs_margin_spearman"] = spearman(methods["entropy"], methods["margin"])
    report["entropy_vs_legacy_spearman"] = spearman(
        methods["entropy"], methods["legacy_prob_score"]
    )
    report["verdict"] = (
        "entropy is a monotone rescaling of the unknown score; it adds no ranking information"
        if abs(spearman(methods["entropy"], unknown_score)) > 0.999
        else "entropy carries ranking information the unknown score does not"
    )
    return report


# --- Step 2: strategy separation and power ----------------------------------


def strategy_separation(
    selections: Mapping[str, Sequence[str]],
    *,
    scores: Mapping[str, ArrayLike] | None = None,
) -> list[dict[str, object]]:
    """Pairwise separation between strategies at equal budget."""

    names = list(selections)
    rows: list[dict[str, object]] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            left_ids = [str(value) for value in selections[left]]
            right_ids = [str(value) for value in selections[right]]
            shared = set(left_ids) & set(right_ids)
            row: dict[str, object] = {
                "left": left,
                "right": right,
                "budget": len(left_ids),
                "overlap": len(shared),
                "jaccard": jaccard(left_ids, right_ids),
                "differing_images": len(set(left_ids) - shared),
                "percent_differing": (
                    100.0 * len(set(left_ids) - shared) / len(left_ids)
                    if left_ids
                    else float("nan")
                ),
            }
            if scores is not None and left in scores and right in scores:
                row["proposal_score_spearman"] = spearman(scores[left], scores[right])
            rows.append(row)
    return rows


def component_effect_sizes(
    results: Mapping[str, ScoringResult],
    *,
    selections: Mapping[str, Sequence[str]],
    reference: str,
) -> list[dict[str, object]]:
    """Cohen's d of each component between selected and unselected proposals.

    A strategy that claims to prefer rare, coherent proposals should show a
    positive effect on those components in what it actually selected.
    """

    rows: list[dict[str, object]] = []
    for name, result in results.items():
        chosen = selections.get(name)
        if chosen is None:
            continue
        mask = result.selected_proposal_mask(chosen)
        if not mask.any() or mask.all():
            continue
        for component in ("uncertainty", "novelty", "rarity", "coherence", "gated"):
            values = result.normalised[component]
            rows.append(
                {
                    "strategy": name,
                    "component": component,
                    "reference_strategy": reference,
                    "selected_mean": float(values[mask].mean()),
                    "unselected_mean": float(values[~mask].mean()),
                    "cohens_d_selected_vs_unselected": cohens_d(values[mask], values[~mask]),
                }
            )
    return rows


#: Two-sided alpha = 0.05, power = 0.80 normal-approximation constants.
_Z_ALPHA_TWO_SIDED = 1.959963985
_Z_POWER = 0.841621234


def power_estimate(
    *,
    effect: float,
    noise_std: float,
    alpha_z: float = _Z_ALPHA_TWO_SIDED,
    power_z: float = _Z_POWER,
) -> dict[str, object]:
    """Seeds per arm needed to resolve ``effect`` against ``noise_std``.

    Normal approximation for two independent arms:
        n >= 2 * (z_alpha + z_power)^2 * sigma^2 / delta^2

    Assumptions, stated because they matter: seed-to-seed metric variation is
    approximately normal and independent, the two arms share a variance, and
    ``noise_std`` was estimated from enough seeds to be meaningful. With one seed
    it cannot be estimated at all, which is itself the finding.
    """

    if not math.isfinite(effect) or abs(effect) < 1e-12:
        return {
            "effect": effect,
            "noise_std": noise_std,
            "seeds_per_arm": float("inf"),
            "note": "no observable effect; no number of seeds resolves it",
        }
    if not math.isfinite(noise_std):
        return {
            "effect": effect,
            "noise_std": noise_std,
            "seeds_per_arm": float("nan"),
            "note": "noise not estimable (needs at least two seeds)",
        }
    required = 2.0 * (alpha_z + power_z) ** 2 * noise_std**2 / effect**2
    return {
        "effect": float(effect),
        "noise_std": float(noise_std),
        "standardised_effect": float(effect / noise_std) if noise_std > 0 else float("inf"),
        "seeds_per_arm": float(math.ceil(required)) if math.isfinite(required) else required,
        "alpha": 0.05,
        "power": 0.80,
        "assumptions": (
            "normal, independent, equal-variance seed noise; two-arm comparison; "
            "no multiplicity correction"
        ),
    }


def power_report(
    metric_series: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    reference: str,
    metric_directions: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """Effect size and required seeds for every strategy against a reference.

    ``metric_series`` is ``{metric: {strategy: [value per seed]}}``.
    """

    rows: list[dict[str, object]] = []
    for metric, per_strategy in metric_series.items():
        if reference not in per_strategy:
            continue
        baseline = np.asarray(per_strategy[reference], dtype=np.float64)
        pooled = [
            np.asarray(values, dtype=np.float64).std(ddof=1)
            for values in per_strategy.values()
            if len(values) > 1
        ]
        noise = float(np.mean(pooled)) if pooled else float("nan")
        for strategy, values in per_strategy.items():
            if strategy == reference:
                continue
            arm = np.asarray(values, dtype=np.float64)
            effect = float(arm.mean() - baseline.mean())
            rows.append(
                {
                    "metric": metric,
                    "direction": (metric_directions or {}).get(metric, "unknown"),
                    "strategy": strategy,
                    "reference": reference,
                    "seeds": int(arm.size),
                    "strategy_mean": float(arm.mean()),
                    "reference_mean": float(baseline.mean()),
                    **power_estimate(effect=effect, noise_std=noise),
                }
            )
    return rows
