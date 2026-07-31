"""Offline multi-seed diagnostics over exported proposals.

This is the cheap protocol the audit recommended for S1: one proposal export from
a frozen checkpoint is enough to compare every strategy over many seeds, so
retraining is spent only on the variants that actually select different images.

Everything here scores through :func:`daowod.scoring.score_pool`; there is no
separate offline formula.
"""

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from daowod import diagnostics as diag
from daowod.groups import ClassGroups
from daowod.prob_adapter import ProposalBatch
from daowod.scoring import (
    REQUIRED_STRATEGIES,
    STRATEGY_REGISTRY,
    ScoringResult,
    StrategySpec,
    score_pool,
    select_images,
)

DEFAULT_STRATEGIES: tuple[str, ...] = REQUIRED_STRATEGIES


def read_image_classes(
    image_ids: Sequence[str], annotations_dir: str | Path
) -> dict[str, list[str]]:
    """Ground-truth class names per image, for post-hoc analysis only."""

    directory = Path(annotations_dir)
    result: dict[str, list[str]] = {}
    for image_id in dict.fromkeys(str(value) for value in image_ids):
        path = directory / f"{image_id}.xml"
        if not path.exists():
            continue
        result[image_id] = [
            str(node.text).strip()
            for node in ET.parse(path).getroot().findall("./object/name")
            if node.text
        ]
    return result


def score_all_strategies(
    *,
    candidates: ProposalBatch,
    references: ProposalBatch,
    strategies: Sequence[str],
    seed: int,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, ScoringResult]:
    """Score one pool with every requested strategy at one seed."""

    results: dict[str, ScoringResult] = {}
    for name in strategies:
        spec = STRATEGY_REGISTRY.resolve(name)
        if overrides and spec.semantics_version == 2 and not spec.random_selection:
            spec = StrategySpec(**{**spec.as_dict(), **dict(overrides)})
        if spec.needs_posterior() and candidates.posterior is None:
            continue
        results[name] = score_pool(
            spec=spec,
            image_ids=candidates.image_ids,
            embeddings=candidates.embeddings,
            reference_embeddings=references.embeddings,
            confidence=candidates.confidence,
            posterior=candidates.posterior,
            predicted_labels=candidates.predicted_labels,
            seed=seed,
            compute_all_components=True,
        )
    return results


def _random_selection(image_ids: Sequence[str], *, budget: int, seed: int) -> list[str]:
    import random as _random

    unique = list(dict.fromkeys(str(value) for value in image_ids))
    order = unique.copy()
    _random.Random(f"offline:{seed}").shuffle(order)
    return order[:budget]


def diagnose_pool(
    *,
    candidate_proposals: str | Path | ProposalBatch,
    reference_proposals: str | Path | ProposalBatch,
    output_dir: str | Path,
    strategies: Sequence[str] | None = None,
    budget: int = 10,
    seeds: Sequence[int] = (0, 1, 2),
    class_stats_path: str | Path | None = None,
    annotations_dir: str | Path | None = None,
    unknown_classes: Sequence[str] = (),
    overrides: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Run the full offline diagnostic campaign and write every artifact.

    Answers, per seed and pooled: does the new uncertainty differ from the
    unknown score; is rarity continuous; what regime is coherence in; does the
    gate change the ranking; which strategies select different images; and how
    large is the strategy separation relative to seed-to-seed variation.
    """

    names = tuple(strategies) if strategies else DEFAULT_STRATEGIES
    candidates = (
        candidate_proposals
        if isinstance(candidate_proposals, ProposalBatch)
        else ProposalBatch.load(candidate_proposals)
    )
    references = (
        reference_proposals
        if isinstance(reference_proposals, ProposalBatch)
        else ProposalBatch.load(reference_proposals)
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    class_groups = ClassGroups.from_class_stats_csv(class_stats_path) if class_stats_path else None
    image_classes = (
        read_image_classes(candidates.image_ids.tolist(), annotations_dir)
        if annotations_dir
        else None
    )

    unique_images = list(dict.fromkeys(str(v) for v in candidates.image_ids.tolist()))
    effective_budget = min(budget, len(unique_images))

    proposal_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    separation_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    selections_by_seed: dict[int, dict[str, list[str]]] = {}
    diagnostics_by_seed: dict[int, dict[str, Any]] = {}
    uncertainty_report: dict[str, Any] = {}

    for seed in seeds:
        results = score_all_strategies(
            candidates=candidates,
            references=references,
            strategies=names,
            seed=seed,
            overrides=overrides,
        )
        selections: dict[str, list[str]] = {}
        scores: dict[str, np.ndarray] = {}
        for name, result in results.items():
            if result.spec.random_selection:
                selections[name] = _random_selection(
                    candidates.image_ids, budget=effective_budget, seed=seed
                )
            else:
                selections[name] = select_images(result.image_scores, budget=effective_budget)
                scores[name] = result.scores
            rows = diag.proposal_table(
                result,
                run_id=f"offline-seed{seed}",
                seed=seed,
                round_index=0,
                selected_image_ids=selections[name],
                posterior=candidates.posterior,
                confidence=candidates.confidence,
                predicted_labels=candidates.predicted_labels,
            )
            diag.assert_no_ground_truth(rows)
            if image_classes is not None and class_groups is not None:
                rows = diag.join_ground_truth(
                    rows,
                    image_classes=image_classes,
                    class_groups=class_groups,
                    unknown_classes=unknown_classes,
                )
            proposal_rows.extend(rows)

            report = diag.component_diagnostics(
                result,
                budget=effective_budget,
                image_classes=image_classes,
                class_groups=class_groups,
                unknown_classes=unknown_classes,
            )
            diagnostics_by_seed.setdefault(seed, {})[name] = report
            component_rows.append(
                {
                    "seed": seed,
                    "strategy": name,
                    "semantics_version": result.spec.semantics_version,
                    "coherence_method": result.spec.coherence_method,
                    "coherence_regime": report["coherence_regime"]["regime"],
                    "coherence_spread": report["coherence_regime"]["spread"],
                    "coherence_vs_cluster_size": report["correlations"][
                        "coherence_vs_cluster_size"
                    ],
                    "rarity_vs_coherence": report["correlations"]["rarity_vs_coherence"],
                    "rarity_fraction_below_0_1": report["norm_rarity"]["fraction_below_0_1"],
                    "rarity_distinct_values": report["norm_rarity"]["distinct"],
                    "isolated_fraction": report["isolated_fraction"],
                    "clusters_below_neighbour_count": report["clusters_below_neighbour_count"][
                        "fraction_of_proposals"
                    ],
                    "gate_spearman_rarity_vs_gated": report["gate_impact"][
                        "spearman_rarity_vs_gated"
                    ],
                    "gate_selected_image_jaccard": report["gate_impact"].get(
                        "selected_image_jaccard"
                    ),
                    **{
                        f"tail_over_head_{component}": (
                            report.get("by_ground_truth_group", {}).get(
                                f"tail_over_head_{component}"
                            )
                        )
                        for component in ("rarity", "coherence", "gated")
                    },
                }
            )

        selections_by_seed[seed] = selections
        for row in diag.strategy_separation(selections, scores=scores):
            separation_rows.append({"seed": seed, **row})
        effect_rows.extend(
            {"seed": seed, **row}
            for row in diag.component_effect_sizes(
                results, selections=selections, reference="v2:random"
            )
        )

        if candidates.posterior is not None and not uncertainty_report:
            uncertainty_report = diag.uncertainty_comparison(
                posterior=candidates.posterior,
                confidence=candidates.confidence,
                image_ids=candidates.image_ids,
                budget=effective_budget,
            )

    # Seed-to-seed stability of each strategy's own selection.
    stability_rows: list[dict[str, object]] = []
    for name in names:
        per_seed = [
            selections_by_seed[seed][name]
            for seed in seeds
            if name in selections_by_seed.get(seed, {})
        ]
        pairwise = [
            diag.jaccard(per_seed[i], per_seed[j])
            for i in range(len(per_seed))
            for j in range(i + 1, len(per_seed))
        ]
        stability_rows.append(
            {
                "strategy": name,
                "seeds": len(per_seed),
                "mean_self_jaccard_across_seeds": float(np.mean(pairwise))
                if pairwise
                else float("nan"),
                "min_self_jaccard_across_seeds": float(np.min(pairwise))
                if pairwise
                else float("nan"),
            }
        )

    diag.write_rows(directory / "offline_proposals.csv", proposal_rows)
    diag.write_rows(directory / "offline_component_diagnostics.csv", component_rows)
    diag.write_rows(directory / "offline_strategy_separation.csv", separation_rows)
    diag.write_rows(directory / "offline_component_effect_sizes.csv", effect_rows)
    diag.write_rows(directory / "offline_seed_stability.csv", stability_rows)
    (directory / "offline_uncertainty_comparison.json").write_text(
        json.dumps(uncertainty_report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (directory / "offline_component_diagnostics.json").write_text(
        json.dumps(diagnostics_by_seed, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (directory / "offline_selections.json").write_text(
        json.dumps(
            {str(seed): value for seed, value in selections_by_seed.items()},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    headline = _headline(
        component_rows=component_rows,
        separation_rows=separation_rows,
        stability_rows=stability_rows,
        uncertainty_report=uncertainty_report,
        proposals=int(candidates.embeddings.shape[0]),
        images=len(unique_images),
        budget=effective_budget,
        seeds=list(seeds),
        strategies=list(names),
    )
    (directory / "offline_headline.json").write_text(
        json.dumps(headline, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {
        "headline": headline,
        "component_rows": component_rows,
        "separation_rows": separation_rows,
        "stability_rows": stability_rows,
        "effect_rows": effect_rows,
        "uncertainty": uncertainty_report,
        "selections": selections_by_seed,
        "output_dir": directory,
    }


def _headline(
    *,
    component_rows: Sequence[Mapping[str, Any]],
    separation_rows: Sequence[Mapping[str, Any]],
    stability_rows: Sequence[Mapping[str, Any]],
    uncertainty_report: Mapping[str, Any],
    proposals: int,
    images: int,
    budget: int,
    seeds: Sequence[int],
    strategies: Sequence[str],
) -> dict[str, Any]:
    """The six questions the diagnostic campaign exists to answer."""

    def mean_of(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and np.isfinite(float(row[key]))
        ]
        return float(np.mean(values)) if values else float("nan")

    v2_rows = [row for row in component_rows if row.get("semantics_version") == 2]
    v1_rows = [row for row in component_rows if row.get("semantics_version") == 1]
    regimes = {str(row["coherence_regime"]) for row in v2_rows}

    full_vs_no_coherence = [
        row
        for row in separation_rows
        if {row["left"], row["right"]} == {"v2:full", "v2:full_no_coherence"}
    ]
    full_vs_random = [
        row for row in separation_rows if {row["left"], row["right"]} == {"v2:full", "v2:random"}
    ]
    stability = {str(row["strategy"]): row for row in stability_rows}
    entropy = (uncertainty_report.get("methods") or {}).get("entropy", {})

    return {
        "pool": {
            "proposals": proposals,
            "images": images,
            "budget": budget,
            "seeds": list(seeds),
            "strategies": list(strategies),
        },
        "q1_uncertainty_differs_from_unknown_score": {
            "entropy_spearman_with_unknown_score": entropy.get("spearman_with_unknown_score"),
            "entropy_is_monotone_in_unknown_score": entropy.get("is_monotone_in_unknown_score"),
            "entropy_selected_image_jaccard_with_unknown_score": entropy.get(
                "selected_image_jaccard_with_unknown_score"
            ),
            "verdict": uncertainty_report.get("verdict"),
        },
        "q2_rarity_is_continuous": {
            "v2_fraction_below_0_1": mean_of(v2_rows, "rarity_fraction_below_0_1"),
            "v2_distinct_values": mean_of(v2_rows, "rarity_distinct_values"),
            "v1_fraction_below_0_1": mean_of(v1_rows, "rarity_fraction_below_0_1"),
        },
        "q3_coherence_regime": {
            "v2_regimes_observed": sorted(regimes),
            "v2_mean_spread": mean_of(v2_rows, "coherence_spread"),
            "v2_mean_spearman_with_cluster_size": mean_of(v2_rows, "coherence_vs_cluster_size"),
            "v1_mean_spearman_with_cluster_size": mean_of(v1_rows, "coherence_vs_cluster_size"),
            "mean_fraction_of_proposals_in_small_clusters": mean_of(
                component_rows, "clusters_below_neighbour_count"
            ),
        },
        "q4_gate_changes_ranking": {
            "mean_spearman_rarity_vs_gated": mean_of(v2_rows, "gate_spearman_rarity_vs_gated"),
            "mean_selected_image_jaccard": mean_of(v2_rows, "gate_selected_image_jaccard"),
            "full_vs_full_no_coherence_mean_jaccard": mean_of(full_vs_no_coherence, "jaccard"),
            "full_vs_full_no_coherence_mean_percent_differing": mean_of(
                full_vs_no_coherence, "percent_differing"
            ),
        },
        "q5_strategies_select_different_images": {
            "mean_pairwise_jaccard": mean_of(separation_rows, "jaccard"),
            "full_vs_random_mean_jaccard": mean_of(full_vs_random, "jaccard"),
            "least_separated_pair": min(
                (
                    {
                        "left": row["left"],
                        "right": row["right"],
                        "jaccard": float(row["jaccard"]),
                    }
                    for row in separation_rows
                    if np.isfinite(float(row["jaccard"]))
                ),
                key=lambda row: -row["jaccard"],
                default=None,
            ),
        },
        "q6_seed_stability": {
            "mean_self_jaccard_across_seeds": mean_of(
                stability_rows, "mean_self_jaccard_across_seeds"
            ),
            "full": stability.get("v2:full"),
            "random": stability.get("v2:random"),
            "interpretation": (
                "A strategy whose own selection changes more between seeds than it "
                "differs from a competing strategy cannot be separated from that "
                "competitor at this budget."
            ),
        },
        "caveat": (
            "Offline selection behaviour only. It bounds which strategies could "
            "differ downstream; it does not measure detector performance."
        ),
    }
