"""Measure first, then decide how big the run may be.

The Colab session is the binding constraint, so the pipeline does not guess: it
times a small detector export and one real acquisition cell, extrapolates, and —
if the projection exceeds the budget — shrinks the *pool*, never the number of
strategies, severities or seeds. Dropping a strategy would remove a comparison;
dropping a seed would remove the only estimate of variance. Shrinking the pool
costs resolution in a way that stays visible in the reported denominators
(``*_objects_reachable`` in every metrics row), which is the honest place to pay.

Extrapolation model
-------------------
Detector export is linear in images: each image is one forward pass.

An acquisition cell is dominated by three terms measured on the real export —
k-means over the pool (linear), nearest-reference novelty (linear in pool x bank,
and blocked so it stays in memory) and the k-nearest-neighbour coherence search
(super-linear). Fitting those together, cell cost grows a little faster than
linearly in pool size; :data:`POOL_COST_EXPONENT` is the exponent used to predict
the cost of a *resized* pool. It is only ever used to choose a smaller size —
after resizing, the pipeline re-measures, so an inaccurate exponent costs one
extra measurement rather than a wrong runtime claim.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike

from daowod.active import run_campaign
from daowod.annotation_study import PreparedPool, StudyConfig
from daowod.longtail import resolve_budgets
from daowod.modes import ExecutionMode, scaled
from daowod.scoring import STRATEGY_REGISTRY

#: Empirical growth exponent of one acquisition cell in pool size. 1.0 would be
#: k-means and novelty alone; the neighbour search pushes it above that.
POOL_COST_EXPONENT = 1.15

#: Never shrink the evaluation pool below this many images. Below it the reachable
#: unknown denominator falls under about 20 objects, at which point a discovery
#: recall moves in visible steps of 5 % and the comparison stops being meaningful.
MINIMUM_EVALUATION_IMAGES = 150

#: Never shrink the per-image candidate limit below this. Measured on the real
#: export, per-image top-20 retains 71 % of true unknown proposals and top-10
#: about half; below that the pool stops containing the objects the strategies
#: are supposed to find.
MINIMUM_PER_IMAGE_LIMIT = 8


class RuntimeBudgetError(RuntimeError):
    """Raised when no admissible run fits the time budget."""


@dataclass(frozen=True)
class Timing:
    """One measured rate, kept with the sample it came from."""

    label: str
    units: float
    seconds: float

    def __post_init__(self) -> None:
        if self.units <= 0:
            raise ValueError(f"{self.label}: units must be positive.")

    @property
    def seconds_per_unit(self) -> float:
        return float(self.seconds) / float(self.units)

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "units": self.units,
            "seconds": round(float(self.seconds), 3),
            "seconds_per_unit": round(self.seconds_per_unit, 4),
        }


def time_call(label: str, units: float, function: Callable[[], object]) -> tuple[Timing, object]:
    """Run ``function`` once and return its wall-clock rate with its result."""

    started = time.perf_counter()
    result = function()
    return Timing(label=label, units=units, seconds=time.perf_counter() - started), result


def count_study_cells(mode: ExecutionMode) -> int:
    """Cells in the main matrix: severities x strategies x seeds."""

    return len(mode.imbalance_settings) * len(mode.strategies) * len(mode.seeds)


def count_ablation_cells(mode: ExecutionMode, *, specs: int) -> int:
    """Cells in the ablation grid, which runs on one severity only."""

    return 0 if not mode.run_ablations else int(specs) * len(mode.ablation_seeds)


def measure_cell_seconds(
    *,
    prepared: PreparedPool,
    reference_embeddings: ArrayLike,
    config: StudyConfig,
    strategy: str = "v2:full",
    seed: int = 0,
) -> Timing:
    """Time one real campaign on the real pool.

    ``v2:full`` is timed because it is the most expensive strategy — it is the
    only one that needs both the neighbour search and the gate — so the projection
    is an upper bound per cell rather than an average that under-counts the
    variant the run exists to measure.
    """

    spec = STRATEGY_REGISTRY.resolve(strategy)
    budgets = resolve_budgets(config.budgets, pool_size=prepared.size)
    references = np.asarray(reference_embeddings, dtype=np.float64)
    timing, _ = time_call(
        f"cell:{strategy}",
        1.0,
        lambda: run_campaign(
            pool=prepared.pool,
            spec=spec,
            reference_embeddings=references,
            gt_class=prepared.table.gt_class,
            gt_is_unknown=prepared.table.gt_is_unknown,
            total_budget=max(budgets),
            rounds=config.rounds,
            seed=seed,
            saturation_mode=config.saturation_mode,
            saturation_strength=config.saturation_strength,
            keep_round_components=True,
        ),
    )
    return timing


def scale_cell_seconds(
    seconds: float, *, measured_pool: int, target_pool: int, exponent: float = POOL_COST_EXPONENT
) -> float:
    """Predict a cell's cost at a different pool size."""

    if measured_pool < 1 or target_pool < 1:
        raise ValueError("Pool sizes must be positive.")
    return float(seconds) * (float(target_pool) / float(measured_pool)) ** float(exponent)


@dataclass
class RuntimePlan:
    """The projection, the verdict, and what was changed to make it fit."""

    mode: ExecutionMode
    seconds_per_image: float
    seconds_per_cell: float
    measured_pool_size: int
    study_cells: int
    ablation_cells: int
    export_seconds: float
    study_seconds: float
    ablation_seconds: float
    overhead_seconds: float
    budget_seconds: float
    actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return (
            self.export_seconds + self.study_seconds + self.ablation_seconds + self.overhead_seconds
        )

    @property
    def within_budget(self) -> bool:
        return self.total_seconds <= self.budget_seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.name,
            "seconds_per_image": round(self.seconds_per_image, 4),
            "seconds_per_cell": round(self.seconds_per_cell, 3),
            "measured_pool_size": self.measured_pool_size,
            "study_cells": self.study_cells,
            "ablation_cells": self.ablation_cells,
            "export_seconds": round(self.export_seconds, 1),
            "study_seconds": round(self.study_seconds, 1),
            "ablation_seconds": round(self.ablation_seconds, 1),
            "overhead_seconds": round(self.overhead_seconds, 1),
            "total_seconds": round(self.total_seconds, 1),
            "total_hours": round(self.total_seconds / 3600.0, 2),
            "budget_seconds": round(self.budget_seconds, 1),
            "budget_hours": round(self.budget_seconds / 3600.0, 2),
            "within_budget": self.within_budget,
            "evaluation_images": self.mode.evaluation_images,
            "pilot_images": self.mode.pilot_images,
            "per_image_limit": self.mode.per_image_limit,
            "seeds": list(self.mode.seeds),
            "actions": list(self.actions),
            "notes": list(self.notes),
        }


def project(
    *,
    mode: ExecutionMode,
    seconds_per_image: float,
    seconds_per_cell: float,
    measured_pool_size: int,
    ablation_specs: int,
    images_already_exported: int = 0,
    overhead_seconds: float = 300.0,
    budget_seconds: float | None = None,
) -> RuntimePlan:
    """Project total runtime for ``mode`` from two measured rates."""

    target_pool = max(1, mode.evaluation_images * mode.per_image_limit)
    per_cell = scale_cell_seconds(
        seconds_per_cell, measured_pool=measured_pool_size, target_pool=target_pool
    )
    study_cells = count_study_cells(mode)
    ablation_cells = count_ablation_cells(mode, specs=ablation_specs)
    # Ablations run on the evaluation pool as well, so they carry the same
    # per-cell cost; the pilot is small enough that its cells are counted in the
    # overhead term rather than modelled separately.
    remaining_images = max(0, mode.total_images - int(images_already_exported))
    return RuntimePlan(
        mode=mode,
        seconds_per_image=float(seconds_per_image),
        seconds_per_cell=per_cell,
        measured_pool_size=int(measured_pool_size),
        study_cells=study_cells,
        ablation_cells=ablation_cells,
        export_seconds=float(seconds_per_image) * remaining_images,
        study_seconds=per_cell * study_cells,
        ablation_seconds=per_cell * ablation_cells,
        overhead_seconds=float(overhead_seconds),
        budget_seconds=float(
            budget_seconds if budget_seconds is not None else mode.runtime_budget_seconds
        ),
    )


def fit_to_budget(
    *,
    mode: ExecutionMode,
    seconds_per_image: float,
    seconds_per_cell: float,
    measured_pool_size: int,
    ablation_specs: int,
    images_already_exported: int = 0,
    overhead_seconds: float = 300.0,
    budget_seconds: float | None = None,
) -> RuntimePlan:
    """Shrink the pool — and only the pool — until the projection fits.

    Order of concessions, most defensible first:

    1. fewer evaluation images (the pilot shrinks proportionally, staying
       disjoint), because this reduces both export and per-cell cost;
    2. a smaller per-image candidate limit, which keeps image coverage — and so
       the diversity of contexts — while thinning duplicate boxes on the same
       object;
    3. drop the ablation grid, which is supporting evidence rather than the
       headline comparison.

    Seeds, strategies and severities are never reduced here: they are what the
    experiment *is*. If the budget still cannot hold them, this raises, and the
    caller is expected to choose a smaller mode explicitly.
    """

    current = mode
    actions: list[str] = []
    notes: list[str] = []
    plan = project(
        mode=current,
        seconds_per_image=seconds_per_image,
        seconds_per_cell=seconds_per_cell,
        measured_pool_size=measured_pool_size,
        ablation_specs=ablation_specs,
        images_already_exported=images_already_exported,
        overhead_seconds=overhead_seconds,
        budget_seconds=budget_seconds,
    )
    if plan.within_budget:
        plan.notes.append(
            f"Projection {plan.total_seconds / 3600:.2f} h fits the "
            f"{plan.budget_seconds / 3600:.2f} h budget; nothing was reduced."
        )
        return plan

    limit = plan.budget_seconds
    for _ in range(8):
        variable = plan.study_seconds + plan.ablation_seconds + plan.export_seconds
        allowed = limit - plan.overhead_seconds
        if allowed <= 0:
            break
        factor = min(0.95, max(0.2, (allowed / max(variable, 1e-9)) ** (1.0 / POOL_COST_EXPONENT)))
        images = max(MINIMUM_EVALUATION_IMAGES, int(current.evaluation_images * factor))
        pilot = max(0, int(current.pilot_images * factor))
        if images < current.evaluation_images:
            actions.append(
                f"evaluation images {current.evaluation_images} -> {images} "
                f"(pilot {current.pilot_images} -> {pilot})"
            )
            current = scaled(current, evaluation_images=images, pilot_images=pilot)
        elif current.per_image_limit > MINIMUM_PER_IMAGE_LIMIT:
            reduced = max(MINIMUM_PER_IMAGE_LIMIT, int(current.per_image_limit * 0.6))
            actions.append(f"per-image candidate limit {current.per_image_limit} -> {reduced}")
            current = scaled(current, per_image_limit=reduced)
        elif current.run_ablations:
            actions.append("ablation grid dropped")
            current = replace(current, run_ablations=False)
        else:
            break
        plan = project(
            mode=current,
            seconds_per_image=seconds_per_image,
            seconds_per_cell=seconds_per_cell,
            measured_pool_size=measured_pool_size,
            ablation_specs=ablation_specs,
            images_already_exported=images_already_exported,
            overhead_seconds=overhead_seconds,
            budget_seconds=budget_seconds,
        )
        if plan.within_budget:
            break

    plan.actions = actions
    plan.notes = notes
    if not plan.within_budget:
        raise RuntimeBudgetError(
            f"Even at {current.evaluation_images} evaluation images and "
            f"per-image limit {current.per_image_limit}, {mode.name} projects "
            f"{plan.total_seconds / 3600:.2f} h against a "
            f"{plan.budget_seconds / 3600:.2f} h budget. The seeds, strategies and "
            "severities are deliberately not reduced automatically: choose a "
            "smaller mode (FAST) or raise the budget explicitly."
        )
    plan.notes.append(
        f"Reduced to fit: projection {plan.total_seconds / 3600:.2f} h against a "
        f"{plan.budget_seconds / 3600:.2f} h budget. Reachable-object denominators "
        "in every metrics row describe the resized pool."
    )
    return plan


def elapsed_report(started: float, *, budget_seconds: float) -> dict[str, object]:
    """Wall-clock progress against the budget, for the notebook's live summary."""

    elapsed = time.time() - float(started)
    return {
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(elapsed / 3600.0, 2),
        "budget_hours": round(float(budget_seconds) / 3600.0, 2),
        "fraction_of_budget": round(elapsed / max(float(budget_seconds), 1e-9), 3),
    }


def summarise_timings(timings: Sequence[Timing]) -> list[Mapping[str, object]]:
    """Timing rows for ``runtime_timings.csv``."""

    return [timing.as_dict() for timing in timings]
