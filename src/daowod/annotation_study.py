"""Orchestration for the offline active-annotation simulation.

All research logic lives here so the Colab notebook is a thin driver: it resolves
paths, calls PROB, and calls :func:`run_study`. Nothing in this module imports
torch, reads a notebook global, or prints progress that a script would not want,
which is what makes the study runnable head-less from ``pytest`` and from the CLI.

Pilot / evaluation separation
----------------------------
:func:`select_hyperparameters` runs on a *pilot* pool and
:func:`run_study` on a disjoint *evaluation* pool. The audit's own criticism of
the earlier work was that the weight sweep and the reported numbers came from the
same ground truth; splitting the images once, up front, is what removes that.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from daowod import candidates as candidate_module
from daowod import discovery, longtail, oracle
from daowod.active import CampaignResult, ProposalPool, initial_state, run_campaign, score_round
from daowod.groups import ClassGroups
from daowod.longtail import ImbalanceSpec
from daowod.scoring import STRATEGY_REGISTRY, StrategySpec

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ObjectArray = NDArray[np.object_]

#: The five strategies the plan requires, in report order. Names resolve through
#: the existing registry so the arithmetic is the repository's canonical scorer.
PRIMARY_STRATEGIES: tuple[str, ...] = (
    "random",
    "uncertainty",
    "uncertainty_novelty",
    "full_no_coherence",
    "full",
)

#: The label-anchored variants, which differ from the baseline in exactly one
#: respect: where the distribution-aware term's rarity and coherence come from.
#: See :mod:`daowod.revealed` for the measurements that motivate them.
ANCHORED_STRATEGIES: tuple[str, ...] = (
    "revealed_support_only",
    "revealed_no_gate",
    "revealed_full",
)

#: The free-heuristic control and its combinations. ``objectness_area_prior`` is
#: the arm the audit says every semantic strategy must be read against; the other
#: two ask whether the distribution term adds anything once the informativeness
#: term actually works.
PRIOR_STRATEGIES: tuple[str, ...] = (
    "objectness_area_prior",
    "prior_full",
    "prior_revealed_full",
)

#: Baseline, control and new method in one matrix. Running them together is what
#: makes the comparison paired: identical pool, identical severities, identical
#: seeds, identical budgets, one export.
COMPARISON_STRATEGIES: tuple[str, ...] = (
    *PRIMARY_STRATEGIES,
    *PRIOR_STRATEGIES,
    *ANCHORED_STRATEGIES,
)

#: Human-readable mapping onto the plan's five numbered strategies, plus the
#: label-anchored follow-ups.
STRATEGY_ROLES: Mapping[str, str] = {
    "random": "1. Random",
    "uncertainty": "2. Uncertainty only",
    "uncertainty_novelty": "3. Uncertainty + Novelty",
    "full_no_coherence": "4. Uncertainty + Novelty + Rarity (ungated)",
    "full": "5. Coherence-aware distribution selection (gated) [BASELINE]",
    "revealed_support_only": "6. Revealed-unknown support only [NEW]",
    "revealed_no_gate": "7. Revealed-class rarity, ungated [NEW]",
    "revealed_full": "8. Label-anchored coherence gate [NEW]",
    "objectness_area_prior": "C. Objectness x box scale prior [FREE CONTROL]",
    "prior_full": "9. Prior + cluster-based gate [NEW]",
    "prior_revealed_full": "10. Prior + label-anchored gate [NEW]",
}

#: Which family each strategy belongs to, so every report can separate the
#: baseline from the new method without a reader having to know the names.
STRATEGY_FAMILY: Mapping[str, str] = {
    **{name: "baseline" for name in PRIMARY_STRATEGIES},
    **{name: "label-anchored" for name in ANCHORED_STRATEGIES},
    "objectness_area_prior": "free-control",
    "prior_full": "prior+cluster",
    "prior_revealed_full": "prior+anchored",
}


class StudyError(RuntimeError):
    """Raised when a study configuration cannot produce a valid measurement."""


@dataclass(frozen=True)
class StudyConfig:
    """Everything that defines one run of the study."""

    budgets: tuple[int, ...] = (25, 50, 100, 200, 500)
    rounds: int = 5
    seeds: tuple[int, ...] = (0, 1, 2)
    strategies: tuple[str, ...] = PRIMARY_STRATEGIES
    imbalance_settings: tuple[ImbalanceSpec, ...] = longtail.DEFAULT_IMBALANCE_SETTINGS
    iou_threshold: float = 0.5
    saturation_mode: str = "cluster"
    saturation_strength: float = 1.0
    candidate_spec: candidate_module.CandidatePoolSpec = field(
        default_factory=candidate_module.CandidatePoolSpec
    )
    head_fraction: float = 1 / 3
    tail_fraction: float = 1 / 3
    reference_limit: int = 20000
    coherence_method_override: str = ""
    neighbour_count_override: int = 0

    def __post_init__(self) -> None:
        if not self.budgets:
            raise StudyError("At least one annotation budget is required.")
        if min(self.budgets) < 1:
            raise StudyError("Budgets must be positive.")
        if not self.seeds:
            raise StudyError("At least one seed is required.")
        if len(self.seeds) < 3:
            # Not fatal: FAST_MODE deliberately runs fewer. The study reports the
            # seed count, and aggregate_over_seeds emits NaN sd for n = 1, so an
            # under-powered run cannot be mistaken for a stable one.
            pass
        if not self.strategies:
            raise StudyError("At least one strategy is required.")

    @property
    def total_budget(self) -> int:
        return int(max(self.budgets))

    def as_dict(self) -> dict[str, object]:
        return {
            "budgets": list(self.budgets),
            "rounds": self.rounds,
            "seeds": list(self.seeds),
            "strategies": list(self.strategies),
            "imbalance_settings": [spec.as_dict() for spec in self.imbalance_settings],
            "iou_threshold": self.iou_threshold,
            "saturation_mode": self.saturation_mode,
            "saturation_strength": self.saturation_strength,
            "candidate_spec": self.candidate_spec.as_dict(),
            "head_fraction": self.head_fraction,
            "tail_fraction": self.tail_fraction,
            "reference_limit": self.reference_limit,
            "coherence_method_override": self.coherence_method_override,
            "neighbour_count_override": self.neighbour_count_override,
        }

    def resolve_strategy(self, name: str) -> StrategySpec:
        """Registry lookup, with the pilot's coherence choice applied.

        The overrides exist so the *pilot* can fix the coherence definition and
        the main run then use it everywhere, without a second copy of the strategy
        table. They are recorded in :meth:`as_dict`, so a reported number always
        names the coherence definition that produced it.
        """

        spec = STRATEGY_REGISTRY.resolve(name)
        updates: dict[str, object] = {}
        if self.coherence_method_override:
            updates["coherence_method"] = self.coherence_method_override
        if self.neighbour_count_override > 0:
            updates["neighbour_count"] = int(self.neighbour_count_override)
            updates["minimum_samples"] = max(2, int(self.neighbour_count_override) // 2)
        return replace(spec, **updates) if updates else spec


@dataclass(frozen=True)
class PreparedPool:
    """A candidate pool with its oracle, groups and reachable denominators."""

    pool: ProposalPool
    table: oracle.OracleTable
    class_groups: ClassGroups
    targets: discovery.DiscoveryTargets
    pool_report: Mapping[str, object]
    composition: Mapping[str, float]

    @property
    def size(self) -> int:
        return self.pool.size


def load_export(path: str) -> dict[str, NDArray[np.generic]]:
    """Load a ``daowod_prob_bridge predict`` NPZ, validating the schema.

    The bridge's own contract (README "PROB Schemas") requires ``image_ids``,
    ``confidence`` and ``embeddings``; this study additionally needs ``boxes``
    for the oracle, ``posterior`` for entropy and ``objectness`` for the pool
    filter, so all six are required here rather than optional.
    """

    required = ("image_ids", "confidence", "embeddings", "posterior", "boxes", "objectness")
    with np.load(path, allow_pickle=True) as handle:
        missing = [name for name in required if name not in handle.files]
        if missing:
            raise StudyError(
                f"{path}: proposal export is missing {missing}. Re-export with "
                "daowod_prob_bridge.py predict from a checkpoint whose model "
                "emits pred_features."
            )
        data = {name: handle[name] for name in handle.files}
    if data["embeddings"].ndim != 2:
        raise StudyError(f"{path}: embeddings must be 2-D.")
    if data["boxes"].shape[1] != 4:
        raise StudyError(f"{path}: boxes must have four columns (cxcywh).")
    return data


def prepare_pool(
    *,
    export: Mapping[str, NDArray[np.generic]],
    annotations_dir: str,
    config: StudyConfig,
    restrict_to_images: Sequence[str] | None = None,
    class_groups: ClassGroups | None = None,
) -> PreparedPool:
    """Build the candidate pool, match it against ground truth, derive groups.

    ``class_groups`` may be supplied to reuse the *evaluation* pool's grouping on
    a pilot pool (or vice versa) so both see the same head/medium/tail definition;
    when omitted the grouping is derived from this pool's own reachable class
    frequencies.
    """

    image_ids = np.asarray([str(value) for value in export["image_ids"].tolist()], dtype=object)
    keep = np.ones(image_ids.shape[0], dtype=np.bool_)
    if restrict_to_images is not None:
        wanted = {str(value) for value in restrict_to_images}
        keep = np.array([str(value) in wanted for value in image_ids.tolist()], dtype=np.bool_)
        if not keep.any():
            raise StudyError("restrict_to_images selected no proposal from the export.")

    subset = np.flatnonzero(keep)
    exported_labels = export.get("predicted_labels")
    selection = candidate_module.build_candidate_pool(
        image_ids=image_ids[subset],
        boxes_cxcywh=export["boxes"][subset],
        objectness=export["objectness"][subset],
        unknown_score=export["confidence"][subset],
        posterior=export["posterior"][subset],
        predicted_labels=None if exported_labels is None else exported_labels[subset],
        spec=config.candidate_spec,
    )
    chosen = subset[selection.indices]
    pool = ProposalPool(
        proposal_ids=np.asarray([f"p{int(index):08d}" for index in chosen.tolist()], dtype=object),
        image_ids=image_ids[chosen],
        embeddings=np.asarray(export["embeddings"][chosen], dtype=np.float64),
        posterior=np.asarray(export["posterior"][chosen], dtype=np.float64),
        confidence=np.asarray(export["confidence"][chosen], dtype=np.float64),
        objectness=np.asarray(export["objectness"][chosen], dtype=np.float64),
        predicted_labels=np.asarray(
            export.get("predicted_labels", np.zeros(image_ids.shape[0]))[chosen], dtype=np.int64
        ),
        boxes_cxcywh=np.asarray(export["boxes"][chosen], dtype=np.float64),
    )

    annotations = oracle.load_annotations(pool.image_ids, annotations_dir)
    unmatched = oracle.match_proposals(
        image_ids=pool.image_ids,
        boxes_cxcywh=pool.boxes_cxcywh,
        annotations=annotations,
        iou_threshold=config.iou_threshold,
    )
    groups = class_groups or oracle.assign_frequency_groups(
        oracle.reachable_class_counts(unmatched),
        head_fraction=config.head_fraction,
        tail_fraction=config.tail_fraction,
        source="pool-reachable unknown object frequency",
    )
    table = oracle.with_class_groups(unmatched, groups)
    full = np.ones(pool.size, dtype=np.bool_)
    targets = discovery.DiscoveryTargets(
        objects_by_group=longtail.discoverable_objects(table, full),
        classes_by_group=longtail.unknown_classes_present(table, full),
    )
    return PreparedPool(
        pool=pool,
        table=table,
        class_groups=groups,
        targets=targets,
        pool_report=selection.report,
        composition=candidate_module.pool_composition(table.gt_match_kind, table.gt_group),
    )


def restrict(prepared: PreparedPool, keep_mask: ArrayLike) -> PreparedPool:
    """Restrict a prepared pool to a mask, recomputing the denominators.

    Used to apply a long-tail severity: the pool shrinks and the reachable object
    and class sets shrink with it, so recall denominators always describe the pool
    the strategies actually searched.
    """

    mask = np.asarray(keep_mask, dtype=np.bool_)
    if mask.shape != (prepared.size,):
        raise StudyError("keep_mask must be parallel to the pool.")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise StudyError("A long-tail severity removed every proposal.")
    table = oracle.OracleTable(
        gt_match_kind=prepared.table.gt_match_kind[indices],
        gt_class=prepared.table.gt_class[indices],
        gt_group=prepared.table.gt_group[indices],
        gt_is_unknown=prepared.table.gt_is_unknown[indices],
        gt_object_index=prepared.table.gt_object_index[indices],
        gt_best_iou=prepared.table.gt_best_iou[indices],
        objects=prepared.table.objects,
        iou_threshold=prepared.table.iou_threshold,
    )
    pool = prepared.pool.subset(indices)
    full = np.ones(pool.size, dtype=np.bool_)
    return PreparedPool(
        pool=pool,
        table=table,
        class_groups=prepared.class_groups,
        targets=discovery.DiscoveryTargets(
            objects_by_group=longtail.discoverable_objects(table, full),
            classes_by_group=longtail.unknown_classes_present(table, full),
        ),
        pool_report=prepared.pool_report,
        composition=candidate_module.pool_composition(table.gt_match_kind, table.gt_group),
    )


def reference_bank(export: Mapping[str, NDArray[np.generic]], *, limit: int) -> FloatArray:
    """The fixed representation bank novelty is measured against.

    Truncation is deterministic (the export's own order) and reported, because
    novelty is the one component whose value depends on the bank's contents.
    """

    embeddings = np.asarray(export["embeddings"], dtype=np.float64)
    if embeddings.shape[0] == 0:
        raise StudyError("The reference export contains no proposals.")
    return embeddings[: int(limit)] if limit and embeddings.shape[0] > limit else embeddings


def _resolve(name: str) -> StrategySpec:
    return STRATEGY_REGISTRY.resolve(name)


@dataclass
class StudyOutputs:
    """Every table the study produces, ready to be written to disk."""

    strategy_rows: list[dict[str, object]] = field(default_factory=list)
    aggregated_rows: list[dict[str, object]] = field(default_factory=list)
    auc_rows: list[dict[str, object]] = field(default_factory=list)
    selected_rows: list[dict[str, object]] = field(default_factory=list)
    distribution_rows: list[dict[str, object]] = field(default_factory=list)
    outlier_rows: list[dict[str, object]] = field(default_factory=list)
    pool_rows: list[dict[str, object]] = field(default_factory=list)
    class_frequency_rows: list[dict[str, object]] = field(default_factory=list)
    anchored_rows: list[dict[str, object]] = field(default_factory=list)
    runtime: dict[str, object] = field(default_factory=dict)


def run_study(
    *,
    prepared: PreparedPool,
    reference_embeddings: ArrayLike,
    config: StudyConfig,
    progress: object | None = None,
) -> StudyOutputs:
    """Run every (severity, strategy, seed) cell and collect all metrics.

    One PROB export supports the whole matrix: the loop is pure NumPy/sklearn, so
    adding seeds costs CPU seconds rather than GPU minutes. Budget curves are
    prefixes of a single trajectory per cell, which is what makes the curves
    monotone in cost and the comparison paired across strategies.
    """

    outputs = StudyOutputs()
    started = time.time()
    references = np.asarray(reference_embeddings, dtype=np.float64)
    cells = 0

    for spec in config.imbalance_settings:
        severity = longtail.build_long_tail_pool(
            prepared.table,
            spec=spec,
            seed=0,
            class_groups=prepared.class_groups.groups,
        )
        outputs.pool_rows.append(dict(severity.report))
        outputs.class_frequency_rows.extend(
            longtail.class_frequency_rows(severity, class_groups=prepared.class_groups.groups)
        )
        scoped = restrict(prepared, severity.keep_mask)
        budgets = longtail.resolve_budgets(config.budgets, pool_size=scoped.size)
        total = max(budgets)

        for strategy_name in config.strategies:
            strategy = config.resolve_strategy(strategy_name)
            for seed in config.seeds:
                cells += 1
                if progress is not None:
                    progress(
                        f"{spec.name} / {strategy_name} / seed {seed} "
                        f"(pool {scoped.size}, budget {total})"
                    )
                result = run_campaign(
                    pool=scoped.pool,
                    spec=strategy,
                    reference_embeddings=references,
                    gt_class=scoped.table.gt_class,
                    gt_is_unknown=scoped.table.gt_is_unknown,
                    total_budget=total,
                    rounds=config.rounds,
                    seed=seed,
                    imbalance_setting=spec.name,
                    saturation_mode=config.saturation_mode,
                    saturation_strength=config.saturation_strength,
                )
                isolated = (
                    result.rounds[0].isolated if result.rounds else np.zeros(scoped.size, bool)
                )
                rows = discovery.campaign_rows(
                    result=result,
                    pool=scoped.pool,
                    oracle=scoped.table,
                    targets=scoped.targets,
                    budgets=budgets,
                    isolated=isolated,
                )
                # The campaign labels itself with the spec's short name ("full");
                # every table joins on the registry name ("full"), so the
                # canonical name is stamped here rather than in two readers.
                for row in rows:
                    row["strategy"] = strategy_name
                    row["strategy_role"] = STRATEGY_ROLES.get(strategy_name, "")
                    row["strategy_family"] = STRATEGY_FAMILY.get(strategy_name, "other")
                    row["distribution_estimator"] = strategy.distribution_estimator
                outputs.strategy_rows.extend(rows)
                summary = discovery.auc_summary(rows)
                outputs.auc_rows.append(
                    {
                        "strategy": strategy_name,
                        "role": STRATEGY_ROLES.get(strategy_name, ""),
                        "strategy_family": STRATEGY_FAMILY.get(strategy_name, "other"),
                        "distribution_estimator": strategy.distribution_estimator,
                        "seed": seed,
                        "imbalance_setting": spec.name,
                        "pool_size": scoped.size,
                        **summary,
                    }
                )
                rows_for_selection = _selection_rows(result, scoped, budgets)
                for row in rows_for_selection:
                    row["strategy"] = strategy_name
                outputs.selected_rows.extend(rows_for_selection)
                for round_index, round_result in enumerate(result.rounds):
                    if not round_result.anchored:
                        continue
                    support_report = dict(round_result.anchored.get("support", {}))
                    rarity_report = dict(round_result.anchored.get("rarity", {}))
                    outputs.anchored_rows.append(
                        {
                            "strategy": strategy_name,
                            "seed": seed,
                            "imbalance_setting": spec.name,
                            "round_index": round_index,
                            "support_cold_start": support_report.get("cold_start"),
                            "revealed_unknown_regions": support_report.get(
                                "revealed_unknown_regions"
                            ),
                            "support_neighbours_used": support_report.get("neighbours_used"),
                            "mean_support": support_report.get("mean_support"),
                            "rarity_cold_start": rarity_report.get("cold_start"),
                            "revealed_unknown_classes": rarity_report.get(
                                "revealed_unknown_classes"
                            ),
                            "rarity_source": rarity_report.get("source"),
                        }
                    )
                if result.rounds:
                    first = result.rounds[0]
                    # Component distributions are a property of the pool and the
                    # first round's scoring, which is seed-dependent only through
                    # the clustering; one seed per (strategy, severity) keeps the
                    # table small without hiding variance that matters.
                    if seed == config.seeds[0]:
                        outputs.distribution_rows.extend(
                            discovery.score_distribution_rows(
                                strategy=strategy_name,
                                seed=seed,
                                imbalance_setting=spec.name,
                                components={
                                    name: first.components[name]
                                    for name in ("rarity", "coherence", "gated", "uncertainty")
                                    if name in first.components
                                },
                                oracle=scoped.table,
                                isolated=first.isolated,
                            )
                        )
                    # The gate counterfactual *is* seed-dependent (the clustering
                    # sets rarity and coherence), so it is recorded for every seed;
                    # a single-seed version could not show whether suppression is
                    # stable.
                    if strategy.gated_weight > 0:
                        outputs.outlier_rows.append(
                            {
                                "strategy": strategy_name,
                                "seed": seed,
                                "imbalance_setting": spec.name,
                                "coherence_method": strategy.coherence_method,
                                "neighbour_count": strategy.neighbour_count,
                                **discovery.gate_suppression(
                                    rarity=first.components["rarity"],
                                    coherence=first.components["coherence"],
                                    gated=first.components["gated"],
                                    oracle=scoped.table,
                                    isolated=first.isolated,
                                    budget=min(total, scoped.size),
                                ),
                            }
                        )

    outputs.aggregated_rows = discovery.aggregate_over_seeds(outputs.strategy_rows)
    outputs.runtime = {
        "cells": cells,
        "seconds": round(time.time() - started, 2),
        "config": config.as_dict(),
    }
    return outputs


def _selection_rows(
    result: CampaignResult, scoped: PreparedPool, budgets: Sequence[int]
) -> list[dict[str, object]]:
    """Selected regions with their oracle verdict attached post hoc."""

    from daowod.active import selection_frame_rows

    rows = selection_frame_rows(result, pool=scoped.pool, budgets=budgets)
    for row in rows:
        index = int(row["pool_index"])
        row["gt_match_kind"] = str(scoped.table.gt_match_kind[index])
        row["gt_class"] = str(scoped.table.gt_class[index])
        row["gt_group"] = str(scoped.table.gt_group[index])
        row["gt_best_iou"] = float(scoped.table.gt_best_iou[index])
        row["gt_object_index"] = int(scoped.table.gt_object_index[index])
    return rows


# --------------------------------------------------------------------------
# Ablations and pilot hyperparameter selection
# --------------------------------------------------------------------------


def ablation_specs(
    *,
    gammas: Sequence[float] = (0.25, 0.5, 0.75),
    coherence_methods: Sequence[str] = ("relative_within_cluster", "radius_core"),
    neighbour_counts: Sequence[int] = (3, 10),
    base: str = "full",
) -> list[StrategySpec]:
    """The ablation grid: gate form x coherence definition x neighbourhood size.

    Includes the three gate forms the plan contrasts — no coherence, additive
    coherence, and the multiplicative gate — so the comparison is against both
    weaker composition rules rather than only against "no coherence".
    """

    root = _resolve(base)
    specs: list[StrategySpec] = []
    for gamma in gammas:
        specs.append(
            replace(
                root,
                name=f"ablation_no_coherence_g{gamma}",
                rarity_weight=float(gamma),
                gated_weight=0.0,
                coherence_weight=0.0,
                description=f"Ungated rarity, gamma={gamma}.",
            )
        )
        specs.append(
            replace(
                root,
                name=f"ablation_additive_g{gamma}",
                rarity_weight=float(gamma) / 2.0,
                coherence_weight=float(gamma) / 2.0,
                gated_weight=0.0,
                description=f"Additive rarity + coherence, gamma={gamma}.",
            )
        )
        for method in coherence_methods:
            for neighbours in neighbour_counts:
                specs.append(
                    replace(
                        root,
                        name=f"ablation_gated_g{gamma}_{method}_k{neighbours}",
                        rarity_weight=0.0,
                        coherence_weight=0.0,
                        gated_weight=float(gamma),
                        coherence_method=method,
                        neighbour_count=int(neighbours),
                        minimum_samples=max(2, int(neighbours) // 2),
                        description=(f"Gated rarity x {method}**p, gamma={gamma}, k={neighbours}."),
                    )
                )
    return specs


def run_ablations(
    *,
    prepared: PreparedPool,
    reference_embeddings: ArrayLike,
    config: StudyConfig,
    specs: Sequence[StrategySpec],
    imbalance: ImbalanceSpec | None = None,
    seeds: Sequence[int] | None = None,
    progress: object | None = None,
) -> list[dict[str, object]]:
    """Run the ablation grid on one severity and return one row per cell."""

    setting = imbalance or config.imbalance_settings[0]
    severity = longtail.build_long_tail_pool(
        prepared.table, spec=setting, seed=0, class_groups=prepared.class_groups.groups
    )
    scoped = restrict(prepared, severity.keep_mask)
    budgets = longtail.resolve_budgets(config.budgets, pool_size=scoped.size)
    total = max(budgets)
    references = np.asarray(reference_embeddings, dtype=np.float64)
    used_seeds = tuple(seeds) if seeds is not None else config.seeds

    rows: list[dict[str, object]] = []
    for spec in specs:
        for seed in used_seeds:
            if progress is not None:
                progress(f"ablation {spec.name} / seed {seed}")
            result = run_campaign(
                pool=scoped.pool,
                spec=spec,
                reference_embeddings=references,
                gt_class=scoped.table.gt_class,
                gt_is_unknown=scoped.table.gt_is_unknown,
                total_budget=total,
                rounds=config.rounds,
                seed=seed,
                imbalance_setting=setting.name,
                saturation_mode=config.saturation_mode,
                saturation_strength=config.saturation_strength,
            )
            isolated = result.rounds[0].isolated if result.rounds else np.zeros(scoped.size, bool)
            curve = discovery.campaign_rows(
                result=result,
                pool=scoped.pool,
                oracle=scoped.table,
                targets=scoped.targets,
                budgets=budgets,
                isolated=isolated,
            )
            rows.append(
                {
                    "ablation": spec.name,
                    "seed": seed,
                    "imbalance_setting": setting.name,
                    "gate_form": _gate_form(spec),
                    # gamma is the *total* weight on the distribution term, so the
                    # three gate forms are comparable at equal gamma: the additive
                    # form splits it across rarity and coherence, the gated form
                    # puts all of it on the interaction. Reading gamma off a single
                    # field instead would place the additive variants on their own
                    # rows and leave the ablation grid full of holes.
                    "gamma": float(spec.gated_weight + spec.rarity_weight + spec.coherence_weight),
                    "coherence_method": spec.coherence_method,
                    "neighbour_count": spec.neighbour_count,
                    "coherence_exponent": spec.coherence_exponent,
                    **discovery.auc_summary(curve),
                }
            )
    return rows


def _gate_form(spec: StrategySpec) -> str:
    if spec.gated_weight > 0:
        return "multiplicative_gate"
    if spec.coherence_weight > 0 and spec.rarity_weight > 0:
        return "additive"
    if spec.rarity_weight > 0:
        return "no_coherence"
    return "other"


def select_hyperparameters(
    *,
    pilot: PreparedPool,
    reference_embeddings: ArrayLike,
    config: StudyConfig,
    candidate_methods: Sequence[str] = ("relative_within_cluster", "radius_core"),
    candidate_neighbours: Sequence[int] = (3, 5, 10),
    seeds: Sequence[int] = (0,),
    progress: object | None = None,
) -> dict[str, object]:
    """Choose the main run's coherence definition on a *pilot* pool.

    Selection criterion is tail discovery AUC, broken by the lower background
    selection rate. Deliberately cheap: one severity, few seeds, small budget
    grid — its only job is to fix a configuration before the evaluation pool is
    touched, so that the reported numbers are not selected on their own data.
    """

    root = _resolve("full")
    specs = [
        replace(
            root,
            name=f"pilot_{method}_k{neighbours}",
            coherence_method=method,
            neighbour_count=int(neighbours),
            minimum_samples=max(2, int(neighbours) // 2),
        )
        for method in candidate_methods
        for neighbours in candidate_neighbours
    ]
    rows = run_ablations(
        prepared=pilot,
        reference_embeddings=reference_embeddings,
        config=config,
        specs=specs,
        seeds=seeds,
        progress=progress,
    )
    if not rows:
        raise StudyError("The pilot produced no ablation rows.")

    def rank_key(row: Mapping[str, object]) -> tuple[float, float, str]:
        # A missing or NaN objective must sort *last*, not participate in an
        # undefined comparison: NaN is neither greater nor smaller than a float, so
        # relying on it would make the choice depend on the input order.
        primary = _finite(row.get("tail_discovery_auc"), default=-np.inf)
        secondary = _finite(row.get("background_selection_rate_auc"), default=np.inf)
        return (-primary, secondary, str(row.get("ablation", "")))

    ranked = sorted(rows, key=rank_key)
    best = ranked[0]
    if not np.isfinite(_finite(best.get("tail_discovery_auc"), default=-np.inf)):
        raise StudyError(
            "No pilot configuration produced a finite tail discovery AUC — the "
            "pilot pool has no reachable tail objects, so it cannot choose a "
            "coherence definition. Enlarge the pilot split."
        )
    return {
        "chosen_coherence_method": best["coherence_method"],
        "chosen_neighbour_count": int(best["neighbour_count"]),
        "criterion": "max tail_discovery_auc, ties broken by min background_selection_rate_auc",
        "pilot_rows": rows,
        "pilot_pool_size": pilot_size(pilot),
    }


def _finite(value: object, *, default: float) -> float:
    """A float that is safe to sort on, or ``default`` when the value is unusable."""

    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def pilot_size(pilot: PreparedPool) -> int:
    """Pool size of the pilot, recorded so the choice is reproducible."""

    return int(pilot.size)


def leakage_check(
    *,
    prepared: PreparedPool,
    reference_embeddings: ArrayLike,
    config: StudyConfig,
    strategy: str = "full",
    seed: int = 0,
) -> dict[str, object]:
    """Prove the acquisition score is reproducible from non-oracle components.

    Three independent assertions:

    1. :func:`daowod.discovery.assert_selection_is_ground_truth_free` rebuilds the
       score from the recorded components; an unrecorded oracle term would break
       the identity;
    2. :func:`daowod.active.score_round` takes no ground-truth parameter at all,
       checked by introspection rather than by trust, so the *only* way ground
       truth could enter is through the pool arrays;
    3. the pool arrays handed to the scorer carry no ``gt_`` field, checked by the
       repository's existing :func:`daowod.oracle.assert_no_ground_truth`.
    """

    spec = _resolve(strategy)
    references = np.asarray(reference_embeddings, dtype=np.float64)
    state = initial_state(pool_size=prepared.size, reference_embeddings=references)
    scored = score_round(
        pool=prepared.pool,
        spec=spec,
        state=state,
        seed=seed,
        saturation_mode=config.saturation_mode,
        saturation_strength=config.saturation_strength,
    )
    discovery.assert_selection_is_ground_truth_free(
        scores=scored.scores[scored.available],
        components={name: values for name, values in scored.components.items()},
        spec_weights=spec.weights(),
    )

    import inspect

    from daowod.oracle import assert_no_ground_truth

    signature = inspect.signature(score_round)
    oracle_parameters = sorted(
        name
        for name in signature.parameters
        if name.startswith("gt_") or name in {"oracle", "table", "annotations", "ground_truth"}
    )
    if oracle_parameters:
        raise StudyError(
            f"score_round accepts oracle parameters {oracle_parameters}; the "
            "scorer must be unable to see ground truth."
        )

    # What the scorer actually receives, as records, so the name-level guard runs
    # against the real payload rather than a hand-written list.
    sample = min(prepared.size, 64)
    assert_no_ground_truth(
        [
            {
                "proposal_id": str(prepared.pool.proposal_ids[index]),
                "image_id": str(prepared.pool.image_ids[index]),
                "objectness": float(prepared.pool.objectness[index]),
                "unknown_score": float(prepared.pool.confidence[index]),
                "predicted_label": int(prepared.pool.predicted_labels[index]),
            }
            for index in range(sample)
        ]
    )

    repeat = score_round(
        pool=prepared.pool,
        spec=spec,
        state=initial_state(pool_size=prepared.size, reference_embeddings=references),
        seed=seed,
        saturation_mode=config.saturation_mode,
        saturation_strength=config.saturation_strength,
    )
    deterministic = bool(
        np.array_equal(
            np.nan_to_num(scored.scores, neginf=-1.0),
            np.nan_to_num(repeat.scores, neginf=-1.0),
        )
    )
    return {
        "components_rebuild_score": True,
        "scorer_has_no_oracle_parameter": True,
        "acquisition_records_have_no_gt_field": True,
        "scoring_is_deterministic_at_fixed_seed": deterministic,
        "strategy": strategy,
        "seed": seed,
    }
