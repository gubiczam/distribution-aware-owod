"""Scientific validation of Contribution A as it currently stands.

Measurement only. This script introduces no scoring component, changes no
default, and calls only existing library functions. It answers:

  Phase 1  does each component carry unique information?
  Phase 2  what does each component contribute to the selection?
  Phase 3  is the method stable under its hyper-parameters?
  Phase 4  what experiment schedule do the measured effects justify?

Every number is written to CSV/JSON under the output directory so the report can
cite it, and every sweep is repeated over several simulator configurations so a
conclusion that depends on an unverifiable simulator assumption is visible as
such.

Usage:  python analysis/validate_contribution_a.py OUTPUT_DIR
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from daowod.diagnostics import (
    cohens_d,
    jaccard,
    pearson,
    power_estimate,
    spearman,
    summarise,
    write_rows,
)
from daowod.normalisation import average_ranks, normalise
from daowod.scoring import (
    STRATEGY_REGISTRY,
    StrategySpec,
    aggregate_image_scores,
    combine_components,
    score_pool,
    select_images,
)
from daowod.simulation import simulate_pool

COMPONENTS = ("uncertainty", "novelty", "rarity", "coherence", "gated")

#: Two *distinct* sources of variation, which must never be pooled.
#:
#: ``POOL_REALISATIONS`` redraws the proposal embeddings, i.e. it asks "would this
#: conclusion hold for a different detector state / dataset sample". In a real
#: campaign the detector is fixed, so this is a robustness axis, not noise.
#:
#: ``SCORING_SEEDS`` fixes the embeddings and varies only the acquisition's own
#: randomness (the KMeans partition). This *is* the noise a multi-seed campaign
#: fights, and it is what a signal-to-noise ratio must be measured against.
#:
#: An earlier version of this script varied both together and reported a
#: signal-to-noise of 0.28 for the coherence gate. That number was meaningless:
#: redrawing every embedding makes two selections nearly disjoint (self-Jaccard
#: 0.08), so it measured "does a different pool select different images" — which
#: it trivially does — rather than the gate's reproducibility.
POOL_REALISATIONS = (0, 1, 2)
SCORING_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_BUDGET = 10

#: The ablation matrix Phase 2 requires. R+C is reported in both forms because
#: the additive and gated readings of "rarity + coherence" are different models.
ABLATION = (
    "v2:random",
    "v2:uncertainty",
    "v2:rarity",
    "v2:coherence",
    "v2:uncertainty_rarity",
    "v2:uncertainty_coherence",
    "v2:rarity_coherence",
    "v2:rarity_plus_coherence",
    "v2:full",
    "v2:full_no_coherence",
    "v2:full_no_rarity",
    "v2:full_no_uncertainty",
)

#: Simulator configurations. `reference` mirrors the pilot protocol; the others
#: vary the assumptions least justified by real data, so a conclusion that only
#: holds in one of them is not a conclusion about the method.
POOL_CONFIGURATIONS: dict[str, dict[str, object]] = {
    "reference": {},
    "few_on_object": {"on_object_fraction": 0.20},
    "many_on_object": {"on_object_fraction": 0.60},
    "ten_proposals": {"proposals_per_image": 10},
    "mild_imbalance": {"imbalance_ratio": 10.0},
    "severe_imbalance": {"imbalance_ratio": 50.0},
    "more_outliers": {"planted_outlier_fraction": 0.15},
}
BASE_POOL: dict[str, object] = {
    "class_count": 20,
    "largest_class_images": 30,
    "imbalance_ratio": 20.0,
    "proposals_per_image": 20,
    "on_object_fraction": 0.35,
    "planted_outlier_fraction": 0.05,
    "embedding_dimension": 256,
    "known_class_count": 40,
    "reference_images": 30,
}


# --------------------------------------------------------------------------- #
# statistics                                                                   #
# --------------------------------------------------------------------------- #


def mutual_information(x: np.ndarray, y: np.ndarray, *, bins: int = 12) -> tuple[float, float]:
    """MI in nats and normalised MI, on equal-frequency bins (scale-free)."""

    def discretise(values: np.ndarray) -> tuple[np.ndarray, int]:
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
        if edges.size < 2:
            return np.zeros(values.size, dtype=np.int64), 1
        index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)
        return index.astype(np.int64), edges.size - 1

    xi, nx = discretise(np.asarray(x, dtype=np.float64))
    yi, ny = discretise(np.asarray(y, dtype=np.float64))
    joint = np.zeros((nx, ny), dtype=np.float64)
    np.add.at(joint, (xi, yi), 1.0)
    total = joint.sum()
    if total <= 0:
        return 0.0, 0.0
    joint /= total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    independent = px @ py
    mask = joint > 0
    information = float((joint[mask] * np.log(joint[mask] / independent[mask])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    floor = max(min(hx, hy), 1e-12)
    return information, information / floor


def _least_squares_residual(target: np.ndarray, predictors: np.ndarray) -> np.ndarray:
    if predictors.size == 0 or predictors.shape[1] == 0:
        return target - target.mean()
    design = np.column_stack([np.ones(predictors.shape[0]), predictors])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coefficients


def semi_partial_correlation(
    component: np.ndarray, score: np.ndarray, others: Sequence[np.ndarray]
) -> float:
    """Correlation of a component with the score after removing the others.

    Computed on ranks, so it is the rank-based semi-partial correlation: the part
    of the score's variation that only this component explains.
    """

    ranked_component = average_ranks(component)
    ranked_score = average_ranks(score)
    matrix = (
        np.column_stack([average_ranks(other) for other in others])
        if others
        else np.empty((ranked_component.size, 0))
    )
    residual = _least_squares_residual(ranked_component, matrix)
    if residual.std() < 1e-12:
        return 0.0
    return pearson(residual, ranked_score)


def incremental_r2(score: np.ndarray, predictors: Mapping[str, np.ndarray], name: str) -> float:
    """R^2 lost when this component is dropped from a rank-linear model."""

    ranked_score = average_ranks(score)
    variance = ranked_score.var()
    if variance < 1e-12:
        return 0.0

    def r_squared(keys: Sequence[str]) -> float:
        if not keys:
            return 0.0
        matrix = np.column_stack([average_ranks(predictors[key]) for key in keys])
        residual = _least_squares_residual(ranked_score, matrix)
        return 1.0 - residual.var() / variance

    every = list(predictors)
    return r_squared(every) - r_squared([key for key in every if key != name])


# --------------------------------------------------------------------------- #
# scoring helpers (existing library functions only)                            #
# --------------------------------------------------------------------------- #


def build_pool(name: str, seed: int):
    settings = {**BASE_POOL, **POOL_CONFIGURATIONS[name]}
    return simulate_pool(**settings, seed=seed)  # type: ignore[arg-type]


def score(pool, spec: StrategySpec, seed: int):
    return score_pool(
        spec=spec,
        image_ids=pool.image_ids,
        embeddings=pool.embeddings,
        reference_embeddings=pool.reference_embeddings,
        confidence=pool.confidence,
        posterior=pool.posterior,
        predicted_labels=pool.predicted_labels,
        seed=seed,
        compute_all_components=True,
    )


def component_bank(pool, seed: int, *, cluster_count: int = 20, neighbour_count: int = 5):
    """Compute every component once so weight sweeps need no rescoring.

    Uses the same seed for every strategy, matching the shared-clustering rule in
    ``daowod.experiment.derive_pool_seed``.
    """

    base = STRATEGY_REGISTRY.resolve("v2:full")
    spec = StrategySpec(
        **{
            **base.as_dict(),
            "cluster_count": cluster_count,
            "neighbour_count": neighbour_count,
        }
    )
    result = score(pool, spec, seed)
    return result


def recombined_scores(
    result,
    *,
    uncertainty: float = 0.0,
    novelty: float = 0.0,
    rarity: float = 0.0,
    gated: float = 0.0,
    coherence: float = 0.0,
    exponent: float | None = None,
) -> np.ndarray:
    """Re-weight already-computed components through the canonical combiner."""

    spec = StrategySpec(
        name="sweep",
        uncertainty_weight=uncertainty,
        novelty_weight=novelty,
        rarity_weight=rarity,
        gated_weight=gated,
        coherence_weight=coherence,
        coherence_exponent=result.spec.coherence_exponent if exponent is None else exponent,
    )
    values = dict(result.normalised)
    if exponent is not None and exponent != result.spec.coherence_exponent:
        regated = result.normalised["rarity"] * np.power(result.raw["coherence"], exponent)
        values["gated"] = normalise(regated, spec.normalisation_for("gated"))
    return combine_components(spec, values)


def selection_from(
    result, scores: np.ndarray, *, budget: int, top_k: int | None = None
) -> list[str]:
    image_scores = aggregate_image_scores(
        result.image_ids,
        scores,
        method=result.spec.image_aggregation,
        top_k=result.spec.top_k if top_k is None else top_k,
    )
    return select_images(image_scores, budget=budget)


# --------------------------------------------------------------------------- #
# Phase 1                                                                      #
# --------------------------------------------------------------------------- #


def phase_one(output: Path) -> dict[str, object]:
    distribution_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    unique_rows: list[dict[str, object]] = []

    for configuration in POOL_CONFIGURATIONS:
        for realisation in POOL_REALISATIONS:
            pool = build_pool(configuration, realisation)
            seed = SCORING_SEEDS[0]
            result = component_bank(pool, seed)
            raw = result.raw
            norm = result.normalised

            for name in COMPONENTS:
                distribution_rows.append(
                    {
                        "configuration": configuration,
                        "realisation": realisation,
                        "component": name,
                        **{f"raw_{k}": v for k, v in summarise(raw[name]).items()},
                        **{f"norm_{k}": v for k, v in summarise(norm[name]).items()},
                    }
                )

            for index, left in enumerate(COMPONENTS):
                for right in COMPONENTS[index + 1 :]:
                    information, normalised_information = mutual_information(
                        norm[left], norm[right]
                    )
                    correlation_rows.append(
                        {
                            "configuration": configuration,
                            "realisation": realisation,
                            "left": left,
                            "right": right,
                            "spearman": spearman(norm[left], norm[right]),
                            "pearson": pearson(norm[left], norm[right]),
                            "mutual_information_nats": information,
                            "normalised_mutual_information": normalised_information,
                        }
                    )

            # Unique contribution to the full score.
            weighted = {name: norm[name] for name in ("uncertainty", "novelty", "gated")}
            full_scores = result.scores
            for name in weighted:
                others = [weighted[key] for key in weighted if key != name]
                unique_rows.append(
                    {
                        "configuration": configuration,
                        "realisation": realisation,
                        "component": name,
                        "spearman_with_score": spearman(weighted[name], full_scores),
                        "semi_partial_correlation": semi_partial_correlation(
                            weighted[name], full_scores, others
                        ),
                        "incremental_r2": incremental_r2(full_scores, weighted, name),
                        "mutual_information_with_score": mutual_information(
                            weighted[name], full_scores
                        )[0],
                    }
                )

    write_rows(output / "phase1_distributions.csv", distribution_rows)
    write_rows(output / "phase1_correlations.csv", correlation_rows)
    write_rows(output / "phase1_unique_information.csv", unique_rows)

    def mean_by(rows, key_fields, value):
        table: dict[tuple, list[float]] = {}
        for row in rows:
            v = row.get(value)
            if v is None or not np.isfinite(float(v)):
                continue
            table.setdefault(tuple(row[k] for k in key_fields), []).append(float(v))
        return {key: float(np.mean(values)) for key, values in table.items()}

    return {
        "mean_abs_spearman": {
            f"{a}|{b}": abs(v)
            for (a, b), v in mean_by(correlation_rows, ("left", "right"), "spearman").items()
        },
        "mean_normalised_mutual_information": {
            f"{a}|{b}": v
            for (a, b), v in mean_by(
                correlation_rows, ("left", "right"), "normalised_mutual_information"
            ).items()
        },
        "mean_semi_partial": {
            a: v
            for (a,), v in mean_by(unique_rows, ("component",), "semi_partial_correlation").items()
        },
        "mean_incremental_r2": {
            a: v for (a,), v in mean_by(unique_rows, ("component",), "incremental_r2").items()
        },
        "reference_only_spearman": {
            f"{row['left']}|{row['right']}": row["spearman"]
            for row in correlation_rows
            if row["configuration"] == "reference" and row["realisation"] == 0
        },
    }


# --------------------------------------------------------------------------- #
# Phase 2                                                                      #
# --------------------------------------------------------------------------- #


def phase_two(output: Path) -> dict[str, object]:
    """Complete ablations.

    Signal (how much two strategies differ) and noise (how much one strategy
    moves with its own randomness) are both measured on a *fixed* pool, so the
    ratio is interpretable. Pool realisations are then used only to ask whether
    the ratio is stable.
    """

    pair_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    noise_rows: list[dict[str, object]] = []

    for configuration in POOL_CONFIGURATIONS:
        for realisation in POOL_REALISATIONS:
            pool = build_pool(configuration, realisation)
            unique_images = list(dict.fromkeys(str(v) for v in pool.image_ids.tolist()))
            selections_by_seed: dict[int, dict[str, list[str]]] = {}

            for seed in SCORING_SEEDS:
                results: dict[str, object] = {}
                selections: dict[str, list[str]] = {}
                proposal_scores: dict[str, np.ndarray] = {}
                image_scores: dict[str, dict[str, float]] = {}

                for name in ABLATION:
                    spec = STRATEGY_REGISTRY.resolve(name)
                    if spec.random_selection:
                        rng = np.random.default_rng(seed)
                        selections[name] = [
                            unique_images[i]
                            for i in rng.permutation(len(unique_images))[:DEFAULT_BUDGET]
                        ]
                        continue
                    result = score(pool, spec, seed)
                    results[name] = result
                    proposal_scores[name] = result.scores
                    image_scores[name] = result.image_scores
                    selections[name] = select_images(result.image_scores, budget=DEFAULT_BUDGET)

                selections_by_seed[seed] = selections

                names = list(ABLATION)
                for index, left in enumerate(names):
                    for right in names[index + 1 :]:
                        row: dict[str, object] = {
                            "configuration": configuration,
                            "realisation": realisation,
                            "scoring_seed": seed,
                            "left": left,
                            "right": right,
                            "image_overlap": len(set(selections[left]) & set(selections[right])),
                            "image_jaccard": jaccard(selections[left], selections[right]),
                        }
                        if left in proposal_scores and right in proposal_scores:
                            row["proposal_score_spearman"] = spearman(
                                proposal_scores[left], proposal_scores[right]
                            )
                            ordered = sorted(image_scores[left])
                            row["image_score_spearman"] = spearman(
                                [image_scores[left][k] for k in ordered],
                                [image_scores[right][k] for k in ordered],
                            )
                            top = max(1, proposal_scores[left].size // 20)
                            left_top = set(
                                np.argsort(-proposal_scores[left], kind="stable")[:top].tolist()
                            )
                            right_top = set(
                                np.argsort(-proposal_scores[right], kind="stable")[:top].tolist()
                            )
                            row["proposal_top5pct_overlap"] = len(left_top & right_top) / top
                            for budget in (5, 10, 20, 40):
                                row[f"jaccard_budget_{budget}"] = jaccard(
                                    select_images(image_scores[left], budget=budget),
                                    select_images(image_scores[right], budget=budget),
                                )
                        pair_rows.append(row)

                for name, result in results.items():
                    mask = result.selected_proposal_mask(selections[name])  # type: ignore[attr-defined]
                    if not mask.any() or mask.all():
                        continue
                    for component in COMPONENTS:
                        values = result.normalised[component]  # type: ignore[attr-defined]
                        effect_rows.append(
                            {
                                "configuration": configuration,
                                "realisation": realisation,
                                "scoring_seed": seed,
                                "strategy": name,
                                "component": component,
                                "selected_mean": float(values[mask].mean()),
                                "unselected_mean": float(values[~mask].mean()),
                                "cohens_d": cohens_d(values[mask], values[~mask]),
                            }
                        )

            # Clustering noise: same pool, different acquisition randomness.
            for name in ABLATION:
                per_seed = [selections_by_seed[s][name] for s in SCORING_SEEDS]
                pairs = [
                    jaccard(per_seed[i], per_seed[j])
                    for i in range(len(per_seed))
                    for j in range(i + 1, len(per_seed))
                ]
                noise_rows.append(
                    {
                        "configuration": configuration,
                        "realisation": realisation,
                        "strategy": name,
                        "scoring_seeds": len(per_seed),
                        "mean_self_jaccard": float(np.mean(pairs)),
                        "clustering_noise": 1.0 - float(np.mean(pairs)),
                    }
                )

    write_rows(output / "phase2_pairwise.csv", pair_rows)
    write_rows(output / "phase2_component_effects.csv", effect_rows)
    write_rows(output / "phase2_clustering_noise.csv", noise_rows)

    def pair_values(left: str, right: str, field: str, configuration: str | None = None):
        return [
            float(row[field])
            for row in pair_rows
            if {row["left"], row["right"]} == {left, right}
            and row.get(field) not in (None, "")
            and np.isfinite(float(row[field]))
            and (configuration is None or row["configuration"] == configuration)
        ]

    def pair_mean(left, right, field, configuration=None) -> float:
        values = pair_values(left, right, field, configuration)
        return float(np.mean(values)) if values else float("nan")

    def noise_mean(strategy: str, configuration: str | None = None) -> float:
        values = [
            float(row["clustering_noise"])
            for row in noise_rows
            if row["strategy"] == strategy
            and np.isfinite(float(row["clustering_noise"]))
            and (configuration is None or row["configuration"] == configuration)
        ]
        return float(np.mean(values)) if values else float("nan")

    contrasts = {
        "full_vs_random": ("v2:full", "v2:random"),
        "full_vs_no_coherence": ("v2:full", "v2:full_no_coherence"),
        "full_vs_no_rarity": ("v2:full", "v2:full_no_rarity"),
        "full_vs_no_uncertainty": ("v2:full", "v2:full_no_uncertainty"),
        "uncertainty_vs_rarity": ("v2:uncertainty", "v2:rarity"),
        "rarity_vs_coherence": ("v2:rarity", "v2:coherence"),
        "gated_vs_additive_rc": ("v2:rarity_coherence", "v2:rarity_plus_coherence"),
        "uncertainty_vs_coherence": ("v2:uncertainty", "v2:coherence"),
    }
    noise = noise_mean("v2:full")
    summary: dict[str, object] = {}
    for label, (left, right) in contrasts.items():
        difference = 1.0 - pair_mean(left, right, "image_jaccard")
        per_configuration = {
            configuration: 1.0 - pair_mean(left, right, "image_jaccard", configuration)
            for configuration in POOL_CONFIGURATIONS
        }
        spearman_value = pair_mean(left, right, "proposal_score_spearman")
        summary[label] = {
            "mean_image_jaccard": pair_mean(left, right, "image_jaccard"),
            "selection_difference": difference,
            "proposal_score_spearman": None if np.isnan(spearman_value) else spearman_value,
            "clustering_noise_of_reference_arm": noise_mean(left),
            "signal_to_clustering_noise": (
                difference / noise_mean(left) if noise_mean(left) > 1e-12 else float("inf")
            ),
            "per_configuration_difference": per_configuration,
            "difference_range_across_configurations": [
                min(per_configuration.values()),
                max(per_configuration.values()),
            ],
        }
    return {
        "contrasts": summary,
        "clustering_noise_full": noise,
        "clustering_noise_by_strategy": {
            row["strategy"]: float(row["clustering_noise"])
            for row in noise_rows
            if row["configuration"] == "reference" and row["realisation"] == 0
        },
        "budget_trend_full_vs_no_coherence": {
            f"budget_{budget}": 1.0
            - pair_mean("v2:full", "v2:full_no_coherence", f"jaccard_budget_{budget}")
            for budget in (5, 10, 20, 40)
        },
        "mean_pairwise_jaccard_all_strategies": float(
            np.mean([float(row["image_jaccard"]) for row in pair_rows])
        ),
    }


# --------------------------------------------------------------------------- #
# Phase 3                                                                      #
# --------------------------------------------------------------------------- #


def phase_three(output: Path) -> dict[str, object]:
    surface_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []

    reference_spec = STRATEGY_REGISTRY.resolve("v2:full")
    lambda_grid = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    gamma_grid = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    exponent_grid = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    budget_grid = (5, 10, 20, 40)

    # One fixed pool; only the acquisition's own randomness varies.
    reference_pool = build_pool("reference", POOL_REALISATIONS[0])
    banks: dict[int, object] = {
        seed: component_bank(reference_pool, seed) for seed in SCORING_SEEDS[:3]
    }

    # --- surface 1: ungated rarity weight (lambda) x gated weight (gamma) ----
    for lam in lambda_grid:
        for gam in gamma_grid:
            if lam == 0.0 and gam == 0.0:
                continue
            selections: dict[int, list[str]] = {}
            for seed, bank in banks.items():
                scores = recombined_scores(
                    bank, uncertainty=0.3, novelty=0.2, rarity=lam, gated=gam
                )
                selections[seed] = selection_from(bank, scores, budget=DEFAULT_BUDGET)
            reference = {
                seed: select_images(bank.image_scores, budget=DEFAULT_BUDGET)
                for seed, bank in banks.items()
            }
            surface_rows.append(
                {
                    "surface": "lambda_x_gamma",
                    "x_name": "rarity_weight_lambda",
                    "x": lam,
                    "y_name": "gated_weight_gamma",
                    "y": gam,
                    "jaccard_with_default_full": float(
                        np.mean([jaccard(selections[s], reference[s]) for s in selections])
                    ),
                    "self_jaccard_across_seeds": float(
                        np.mean(
                            [
                                jaccard(selections[a], selections[b])
                                for a in selections
                                for b in selections
                                if a < b
                            ]
                        )
                    ),
                }
            )

    # --- surface 2: coherence exponent x budget ------------------------------
    for exponent in exponent_grid:
        for budget in budget_grid:
            selections, ungated = {}, {}
            for seed, bank in banks.items():
                gated_scores = recombined_scores(
                    bank, uncertainty=0.3, novelty=0.2, gated=0.5, exponent=exponent
                )
                ungated_scores = recombined_scores(bank, uncertainty=0.3, novelty=0.2, rarity=0.5)
                selections[seed] = selection_from(bank, gated_scores, budget=budget)
                ungated[seed] = selection_from(bank, ungated_scores, budget=budget)
            difference = float(
                np.mean([1.0 - jaccard(selections[s], ungated[s]) for s in selections])
            )
            noise = float(
                np.mean(
                    [
                        1.0 - jaccard(selections[a], selections[b])
                        for a in selections
                        for b in selections
                        if a < b
                    ]
                )
            )
            surface_rows.append(
                {
                    "surface": "exponent_x_budget",
                    "x_name": "coherence_exponent",
                    "x": exponent,
                    "y_name": "budget",
                    "y": budget,
                    "gate_effect_vs_ungated": difference,
                    "seed_noise": noise,
                    "signal_to_noise": difference / noise if noise > 1e-12 else float("inf"),
                }
            )

    # --- 1-D sweeps ----------------------------------------------------------
    def one_dimensional(name: str, values: Sequence[object], build) -> None:
        for value in values:
            selections, differences = {}, []
            for seed in SCORING_SEEDS[:3]:
                bank, gated_scores, ungated_scores, budget, top_k = build(
                    reference_pool, seed, value
                )
                selections[seed] = selection_from(bank, gated_scores, budget=budget, top_k=top_k)
                differences.append(
                    1.0
                    - jaccard(
                        selections[seed],
                        selection_from(bank, ungated_scores, budget=budget, top_k=top_k),
                    )
                )
            noise = float(
                np.mean(
                    [
                        1.0 - jaccard(selections[a], selections[b])
                        for a in selections
                        for b in selections
                        if a < b
                    ]
                )
            )
            gate_effect = float(np.mean(differences))
            sweep_rows.append(
                {
                    "parameter": name,
                    "value": value,
                    "gate_effect_vs_ungated": gate_effect,
                    "seed_noise": noise,
                    "signal_to_noise": gate_effect / noise if noise > 1e-12 else float("inf"),
                }
            )

    def cluster_build(pool, seed, value):
        bank = component_bank(pool, seed, cluster_count=int(value))
        return (
            bank,
            recombined_scores(bank, uncertainty=0.3, novelty=0.2, gated=0.5),
            recombined_scores(bank, uncertainty=0.3, novelty=0.2, rarity=0.5),
            DEFAULT_BUDGET,
            None,
        )

    def neighbour_build(pool, seed, value):
        bank = component_bank(pool, seed, neighbour_count=int(value))
        return (
            bank,
            recombined_scores(bank, uncertainty=0.3, novelty=0.2, gated=0.5),
            recombined_scores(bank, uncertainty=0.3, novelty=0.2, rarity=0.5),
            DEFAULT_BUDGET,
            None,
        )

    def top_k_build(pool, seed, value):
        bank = banks[seed]
        return (
            bank,
            recombined_scores(bank, uncertainty=0.3, novelty=0.2, gated=0.5),
            recombined_scores(bank, uncertainty=0.3, novelty=0.2, rarity=0.5),
            DEFAULT_BUDGET,
            int(value),
        )

    one_dimensional("cluster_count", (5, 10, 20, 40, 60, 80), cluster_build)
    one_dimensional("neighbour_count", (2, 3, 5, 10, 20), neighbour_build)
    one_dimensional("top_k", (1, 2, 3, 5, 10), top_k_build)

    write_rows(output / "phase3_surfaces.csv", surface_rows)
    write_rows(output / "phase3_sweeps.csv", sweep_rows)

    exponent_rows = [row for row in surface_rows if row["surface"] == "exponent_x_budget"]
    reference_exponent = [
        row for row in exponent_rows if row["y"] == DEFAULT_BUDGET and row["x"] != 0.0
    ]
    return {
        "lambda_gamma_range_of_jaccard_with_default": {
            "min": min(
                float(row["jaccard_with_default_full"])
                for row in surface_rows
                if row["surface"] == "lambda_x_gamma"
            ),
            "max": max(
                float(row["jaccard_with_default_full"])
                for row in surface_rows
                if row["surface"] == "lambda_x_gamma"
            ),
        },
        "gate_effect_by_exponent_at_default_budget": {
            str(row["x"]): float(row["gate_effect_vs_ungated"]) for row in reference_exponent
        },
        "gate_signal_to_noise_by_budget": {
            str(row["y"]): float(row["signal_to_noise"]) for row in exponent_rows if row["x"] == 1.0
        },
        "sweeps": {
            parameter: {
                str(row["value"]): {
                    "gate_effect": float(row["gate_effect_vs_ungated"]),
                    "signal_to_noise": float(row["signal_to_noise"]),
                }
                for row in sweep_rows
                if row["parameter"] == parameter
            }
            for parameter in ("cluster_count", "neighbour_count", "top_k")
        },
        "reference_spec": reference_spec.as_dict(),
    }


# --------------------------------------------------------------------------- #
# Phase 4                                                                      #
# --------------------------------------------------------------------------- #


def phase_four(output: Path, phase2: Mapping[str, object], phase3: Mapping[str, object]) -> dict:
    """Design nomogram.

    What is measurable here is *selection*: how much two strategies' chosen image
    sets differ, and how much one strategy's own choice moves with the seed. What
    a thesis needs is a *metric* difference, and that requires training. The
    honest output is therefore a nomogram: required seeds as a function of the
    assumed detector-metric effect, together with the selection-level ceiling that
    bounds it.
    """

    contrasts = phase2["contrasts"]  # type: ignore[index]
    rows: list[dict[str, object]] = []
    # Plausible per-seed standard deviations of known mAP / U-Recall on a
    # 200-image evaluation split. Stated as assumptions, not measurements.
    for metric_std in (0.005, 0.01, 0.02, 0.03):
        for effect in (0.002, 0.005, 0.01, 0.02, 0.05):
            estimate = power_estimate(effect=effect, noise_std=metric_std)
            rows.append(
                {
                    "assumed_metric_std_per_seed": metric_std,
                    "assumed_metric_effect": effect,
                    "standardised_effect": estimate["standardised_effect"],
                    "seeds_per_arm_for_80pct_power": estimate["seeds_per_arm"],
                }
            )
    write_rows(output / "phase4_power_nomogram.csv", rows)

    budget_signal = phase3["gate_signal_to_noise_by_budget"]  # type: ignore[index]
    budget_trend = phase2["budget_trend_full_vs_no_coherence"]  # type: ignore[index]
    return {
        "selection_level_ceiling": {
            "full_vs_no_coherence_selection_difference": contrasts["full_vs_no_coherence"][
                "selection_difference"
            ],
            "full_vs_random_selection_difference": contrasts["full_vs_random"][
                "selection_difference"
            ],
            "clustering_noise_full": phase2["clustering_noise_full"],
            "signal_to_clustering_noise": contrasts["full_vs_no_coherence"][
                "signal_to_clustering_noise"
            ],
        },
        "gate_signal_to_noise_by_budget": budget_signal,
        "selection_difference_by_budget": budget_trend,
        "nomogram_rows": rows,
        "assumptions": (
            "Metric noise is normal, independent across seeds and equal between "
            "arms; no multiplicity correction; the selection-level difference "
            "bounds but does not determine the metric difference."
        ),
    }


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #


def figures(output: Path) -> list[str]:
    import csv as _csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "text.color": INK,
            "legend.frameon": False,
            "figure.dpi": 130,
            "lines.linewidth": 2.0,
        }
    )
    ramp = LinearSegmentedColormap.from_list(
        "daowod_blue",
        ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    )

    def load(name: str) -> list[dict[str, str]]:
        with (output / name).open(newline="", encoding="utf-8") as handle:
            return list(_csv.DictReader(handle))

    saved: list[str] = []

    def save(figure, stem: str) -> None:
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"
            figure.savefig(path)
            saved.append(path.name)
        plt.close(figure)

    # Figure 1: component correlation and mutual-information matrices.
    rows = [r for r in load("phase1_correlations.csv") if r["configuration"] == "reference"]
    names = list(COMPONENTS)
    corr = np.full((len(names), len(names)), np.nan)
    info = np.full((len(names), len(names)), np.nan)
    for row in rows:
        i, j = names.index(row["left"]), names.index(row["right"])
        for matrix, field in ((corr, "spearman"), (info, "normalised_mutual_information")):
            value = float(row[field])
            previous = matrix[i, j]
            matrix[i, j] = matrix[j, i] = value if np.isnan(previous) else (previous + value) / 2
    np.fill_diagonal(corr, 1.0)
    np.fill_diagonal(info, 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, matrix, title, limits in (
        (axes[0], corr, "Spearman correlation", (-1, 1)),
        (axes[1], info, "normalised mutual information", (0, 1)),
    ):
        image = axis.imshow(
            matrix,
            cmap="RdBu_r" if limits[0] < 0 else ramp,
            vmin=limits[0],
            vmax=limits[1],
        )
        axis.set_xticks(range(len(names)))
        axis.set_yticks(range(len(names)))
        axis.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axis.set_yticklabels(names, fontsize=8)
        axis.grid(False)
        axis.set_title(title, fontsize=10)
        for i in range(len(names)):
            for j in range(len(names)):
                if np.isfinite(matrix[i, j]):
                    axis.annotate(
                        f"{matrix[i, j]:.2f}",
                        (j, i),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=SURFACE if abs(matrix[i, j]) > 0.6 else INK,
                    )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        "Phase 1: component redundancy on the reference pool (PROB-calibrated synthetic)",
        fontsize=11,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save(figure, "figure1_component_redundancy")

    # Figure 2: ablation selection-overlap heatmap.
    pair_rows = [
        r
        for r in load("phase2_pairwise.csv")
        if r["configuration"] == "reference" and r["realisation"] == "0"
    ]
    labels = [n.replace("v2:", "") for n in ABLATION]
    overlap = np.full((len(labels), len(labels)), np.nan)
    for row in pair_rows:
        i = ABLATION.index(row["left"])
        j = ABLATION.index(row["right"])
        value = float(row["image_jaccard"])
        previous = overlap[i, j]
        overlap[i, j] = overlap[j, i] = value if np.isnan(previous) else (previous + value) / 2
    np.fill_diagonal(overlap, 1.0)
    figure, axis = plt.subplots(figsize=(8.4, 7.2))
    image = axis.imshow(overlap, cmap=ramp, vmin=0, vmax=1)
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axis.set_yticklabels(labels, fontsize=8)
    axis.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(overlap[i, j]):
                axis.annotate(
                    f"{overlap[i, j]:.2f}",
                    (j, i),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=SURFACE if overlap[i, j] > 0.55 else INK,
                )
    axis.set_title(
        f"Phase 2: selected-image Jaccard, budget {DEFAULT_BUDGET}, mean over {len(SCORING_SEEDS)} scoring seeds",
        fontsize=11,
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04).set_label(
        "Jaccard", color=INK2, fontsize=9
    )
    figure.tight_layout()
    save(figure, "figure2_ablation_overlap")

    # Figure 3: response surfaces.
    surfaces = load("phase3_surfaces.csv")
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    lam_rows = [r for r in surfaces if r["surface"] == "lambda_x_gamma"]
    lambdas = sorted({float(r["x"]) for r in lam_rows})
    gammas = sorted({float(r["y"]) for r in lam_rows})
    grid = np.full((len(gammas), len(lambdas)), np.nan)
    for row in lam_rows:
        grid[gammas.index(float(row["y"])), lambdas.index(float(row["x"]))] = float(
            row["jaccard_with_default_full"]
        )
    image = axes[0].imshow(grid, cmap=ramp, vmin=0, vmax=1, origin="lower", aspect="auto")
    axes[0].set_xticks(range(len(lambdas)))
    axes[0].set_xticklabels([f"{v:g}" for v in lambdas], fontsize=8)
    axes[0].set_yticks(range(len(gammas)))
    axes[0].set_yticklabels([f"{v:g}" for v in gammas], fontsize=8)
    axes[0].set_xlabel("ungated rarity weight  lambda")
    axes[0].set_ylabel("gated weight  gamma")
    axes[0].set_title("Jaccard with the default full score", fontsize=10)
    axes[0].grid(False)
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    exp_rows = [r for r in surfaces if r["surface"] == "exponent_x_budget"]
    exponents = sorted({float(r["x"]) for r in exp_rows})
    budgets = sorted({int(float(r["y"])) for r in exp_rows})
    for budget in budgets:
        series = [
            float(r["gate_effect_vs_ungated"])
            for exponent in exponents
            for r in exp_rows
            if float(r["x"]) == exponent and int(float(r["y"])) == budget
        ]
        axes[1].plot(exponents, series, marker="o", label=f"budget {budget}")
    noise_rows = [
        float(r["seed_noise"])
        for exponent in exponents
        for r in exp_rows
        if float(r["x"]) == exponent and int(float(r["y"])) == DEFAULT_BUDGET
    ]
    axes[1].plot(
        exponents,
        noise_rows,
        linestyle="--",
        color=INK2,
        marker="x",
        label=f"seed noise (budget {DEFAULT_BUDGET})",
    )
    axes[1].set_xlabel("coherence exponent  p")
    axes[1].set_ylabel("selection difference vs ungated")
    axes[1].set_title("gate effect against its own seed noise", fontsize=10)
    axes[1].legend(fontsize=8)
    figure.suptitle("Phase 3: response surfaces", fontsize=11, fontweight="semibold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save(figure, "figure3_response_surfaces")

    # Figure 4: 1-D stability sweeps.
    sweeps = load("phase3_sweeps.csv")
    parameters = ("cluster_count", "neighbour_count", "top_k")
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, parameter in zip(axes, parameters, strict=True):
        rows_p = [r for r in sweeps if r["parameter"] == parameter]
        x = [float(r["value"]) for r in rows_p]
        axis.plot(
            x, [float(r["gate_effect_vs_ungated"]) for r in rows_p], marker="o", label="gate effect"
        )
        axis.plot(
            x,
            [float(r["seed_noise"]) for r in rows_p],
            marker="x",
            linestyle="--",
            color=INK2,
            label="seed noise",
        )
        axis.set_xlabel(parameter)
        axis.set_title(parameter, fontsize=10)
    axes[0].set_ylabel("selection difference")
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Phase 3: hyper-parameter stability of the coherence gate",
        fontsize=11,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    save(figure, "figure4_hyperparameter_stability")

    return saved


# --------------------------------------------------------------------------- #


def main(argv: Sequence[str]) -> int:
    output = Path(argv[1] if len(argv) > 1 else "outputs/validation")
    output.mkdir(parents=True, exist_ok=True)
    print(f"output: {output}")

    print("Phase 1 — component independence ...")
    phase1 = phase_one(output)
    print("Phase 2 — ablations ...")
    phase2 = phase_two(output)
    print("Phase 3 — sensitivity ...")
    phase3 = phase_three(output)
    print("Phase 4 — design ...")
    phase4 = phase_four(output, phase2, phase3)
    print("figures ...")
    saved = figures(output)

    summary = {
        "provenance": {
            "pool": "PROB-calibrated synthetic (daowod.simulation)",
            "configurations": list(POOL_CONFIGURATIONS),
            "base": BASE_POOL,
            "pool_realisations": list(POOL_REALISATIONS),
            "scoring_seeds": list(SCORING_SEEDS),
            "budget": DEFAULT_BUDGET,
            "ablation": list(ABLATION),
            "caveat": "no real M-OWODB proposals were used; see docs/diagnostics_report.md §8",
        },
        "phase1_component_independence": phase1,
        "phase2_ablations": phase2,
        "phase3_sensitivity": phase3,
        "phase4_design": phase4,
        "figures": saved,
    }
    (output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["phase2_ablations"]["contrasts"], indent=2, default=str)[:1200])
    print(f"\nwrote {len(list(output.iterdir()))} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
