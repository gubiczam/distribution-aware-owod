"""One function the notebook calls, and the resume logic behind it.

``run_pipeline`` executes the whole Contribution-A experiment:

    preflight -> disjoint splits -> cached PROB inference -> candidate pool ->
    region oracle -> long-tail severities (validated) -> leakage proof ->
    pilot hyperparameter choice -> runtime plan -> main study -> ablations ->
    metrics, plots, CSVs, markdown summary, ZIP

Every stage writes its result under ``output_dir/state`` and is skipped when that
file already exists, so a disconnected Colab session resumes rather than restarts.
The stages that cost real time — detector inference and the per-severity study
matrix — are cached at a granularity small enough that a lost session costs
minutes, not hours.

Resume keys
-----------
A stage's cache file name includes a fingerprint of everything that would change
its result. Change the mode, the checkpoint, the candidate filter or the pilot's
coherence choice and the fingerprint changes, so the stale result is not reused.
This is the difference between a resumable run and a run that quietly mixes two
configurations.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from daowod import config, detector, discovery, figures, longtail, oracle, study, tables
from daowod.annotation import run_campaign
from daowod.config import ExecutionMode
from daowod.longtail import resolve_budgets
from daowod.oracle import ClassGroups
from daowod.scoring import STRATEGY_REGISTRY
from daowod.study import PreparedPool, StudyConfig, StudyOutputs

Progress = Callable[[str], None]


class RuntimeBudgetExceeded(RuntimeError):
    """The projected cost exceeds the declared budget, so the run refuses to start.

    This replaces an earlier mechanism that reacted to the same situation by
    *shrinking the evaluation pool* until the projection fit. That silently changed
    the protocol after the protocol had been declared: the reported denominators,
    and therefore the meaning of every recall in the run, depended on how loaded the
    machine happened to be. Refusing is the honest response — the caller chooses a
    smaller mode or raises the budget explicitly, and either way the choice is
    recorded in the config rather than inferred from a stopwatch.
    """


#: Empirical growth exponent of one acquisition cell in pool size. 1.0 would be
#: k-means and novelty alone; the neighbour search pushes it above that.
POOL_COST_EXPONENT = 1.15

#: Bytes per exported proposal: a 256-d float32 embedding, a 20-d posterior, a box
#: and the scalar columns, rounded up for NPZ overhead.
EXPORT_BYTES_PER_PROPOSAL = 1_200

#: Bytes per pooled proposal held in memory: the float64 embedding the acquisition
#: works on, plus the component and oracle columns.
POOL_BYTES_PER_PROPOSAL = 2_400

#: Artifacts the run promises to produce; verified before it reports success.
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "budget_curves.csv",
    "budget_curves_aggregated.csv",
    "strategy_auc.csv",
    "selected_proposals.csv",
    "component_distributions.csv",
    "long_tail_pools.csv",
    "class_frequency.csv",
    "headline_contrasts.csv",
    "cost_to_target.csv",
    "preflight.csv",
    "run_manifest.json",
    "runtime_plan.json",
    "leakage_report.json",
    "research_summary.md",
    "figure_tail_discovery_vs_budget.png",
    "figure_tail_auc_by_severity.png",
    "figure_long_tail_protocol.png",
)


class PipelineError(RuntimeError):
    """Raised when a stage cannot run and no useful result can be produced."""


@dataclass(frozen=True)
class PipelineConfig:
    """Paths, protocol constants and switches. The only thing the notebook edits."""

    mode: str = "MAIN"
    data_root: str = "/content/owod_stage"
    split_file: str = ""
    prob_repository: str = "/content/PROB"
    checkpoint: str = ""
    output_dir: str = "/content/daowod_results"
    cache_dir: str = "/content/daowod_cache"
    dataset: str = "OWDETR"
    previous_introduced_classes: int = 0
    current_introduced_classes: int = 19
    num_classes: int = 81
    objectness_temperature: float = 1.0
    batch_size: int = 2
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 0
    max_proposals_per_image: int = 100
    chunk_images: int = 250
    python_executable: str = ""
    expected_checkpoint_sha256: str = ""
    existing_export: str = ""
    require_gpu: bool = True
    auto_calibrate_severities: bool = False
    runtime_budget_seconds: float = 0.0
    target_tail_recall: float = 0.5
    iou_threshold: float = 0.5
    force: bool = False

    def __post_init__(self) -> None:
        if not self.existing_export and not self.checkpoint:
            raise PipelineError(
                "Either checkpoint (to run PROB inference) or existing_export (to "
                "reuse a previous export) must be set."
            )
        if not self.split_file and not self.existing_export:
            raise PipelineError("split_file is required when proposals are exported.")

    @property
    def annotations_dir(self) -> Path:
        return Path(self.data_root) / "Annotations"

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, **overrides: object) -> PipelineConfig:
        """Build a run from `configs/contribution_a.yaml`, registering its modes.

        The YAML is the protocol (see :mod:`daowod.config`); ``overrides`` exist for
        the paths a Colab session must supply at run time, never for sizes or seeds.
        Passing a size here would put the protocol back in the caller.
        """

        info = config.load_config(path)
        run = dict(info["run"])
        hours = float(run.pop("runtime_budget_hours", 0.0) or 0.0)
        settings: dict[str, object] = {
            "mode": info["mode"],
            "runtime_budget_seconds": hours * 3600.0,
            **run,
            **overrides,
        }
        known = {field_.name for field_ in fields(cls)}
        unknown = sorted(set(settings) - known)
        if unknown:
            raise config.ConfigError(
                f"{info['source']}: unknown run settings {unknown}. Known: {sorted(known)}"
            )
        return cls(**settings)  # type: ignore[arg-type]

    def execution_mode(self) -> ExecutionMode:
        mode = config.resolve_mode(self.mode)
        if self.runtime_budget_seconds > 0:
            mode = replace(mode, runtime_budget_seconds=float(self.runtime_budget_seconds))
        return mode

    def bridge_settings(self) -> detector.BridgeSettings:
        import sys

        return detector.BridgeSettings(
            prob_repository=self.prob_repository,
            checkpoint=self.checkpoint,
            data_root=self.data_root,
            dataset=self.dataset,
            previous_introduced_classes=self.previous_introduced_classes,
            current_introduced_classes=self.current_introduced_classes,
            num_classes=self.num_classes,
            objectness_temperature=self.objectness_temperature,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            seed=self.seed,
            max_proposals_per_image=self.max_proposals_per_image,
            python_executable=self.python_executable or sys.executable,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "data_root": self.data_root,
            "split_file": self.split_file,
            "prob_repository": self.prob_repository,
            "checkpoint": self.checkpoint,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "dataset": self.dataset,
            "previous_introduced_classes": self.previous_introduced_classes,
            "current_introduced_classes": self.current_introduced_classes,
            "num_classes": self.num_classes,
            "objectness_temperature": self.objectness_temperature,
            "batch_size": self.batch_size,
            "device": self.device,
            "seed": self.seed,
            "max_proposals_per_image": self.max_proposals_per_image,
            "chunk_images": self.chunk_images,
            "existing_export": self.existing_export,
            "require_gpu": self.require_gpu,
            "auto_calibrate_severities": self.auto_calibrate_severities,
            "runtime_budget_seconds": self.runtime_budget_seconds,
            "target_tail_recall": self.target_tail_recall,
            "iou_threshold": self.iou_threshold,
            "force": self.force,
        }

    def fingerprint(self) -> str:
        payload = dict(self.as_dict())
        for volatile in ("output_dir", "cache_dir", "force", "prob_repository"):
            payload.pop(volatile, None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]


class StateStore:
    """A directory of JSON stage results, used for resume."""

    def __init__(self, directory: str | Path, *, force: bool = False) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.force = bool(force)

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def load(self, key: str) -> object | None:
        if self.force:
            return None
        target = self.path(key)
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A half-written file from a killed session is not a resume point.
            target.unlink(missing_ok=True)
            return None

    def save(self, key: str, payload: object) -> Path:
        target = self.path(key)
        temporary = target.with_suffix(".partial")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    def cached(self, key: str, produce: Callable[[], object]) -> tuple[object, bool]:
        existing = self.load(key)
        if existing is not None:
            return existing, True
        value = produce()
        self.save(key, value)
        return value, False


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


@dataclass
class PipelineResult:
    """Everything the notebook needs to display, plus where it was written."""

    config: PipelineConfig
    mode: ExecutionMode
    output_dir: Path
    checks: list[dict[str, object]] = field(default_factory=list)
    export: dict[str, object] = field(default_factory=dict)
    pool_report: dict[str, object] = field(default_factory=dict)
    composition: dict[str, object] = field(default_factory=dict)
    severity_rows: list[dict[str, object]] = field(default_factory=list)
    severity_verdict: str = ""
    leakage: dict[str, object] = field(default_factory=dict)
    pilot: dict[str, object] = field(default_factory=dict)
    runtime_plan: dict[str, object] = field(default_factory=dict)
    outputs: StudyOutputs = field(default_factory=StudyOutputs)
    ablation_rows: list[dict[str, object]] = field(default_factory=list)
    contrasts: list[dict[str, object]] = field(default_factory=list)
    cost_rows: list[dict[str, object]] = field(default_factory=list)
    figures: list[Path] = field(default_factory=list)
    artifacts: list[dict[str, object]] = field(default_factory=list)
    archive: Path | None = None
    summary_path: Path | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def headline(self) -> list[dict[str, object]]:
        return tables.summarise_strategies(self.outputs.auc_rows)


def _log(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


# --- cost estimate: measure, report, refuse -----------------------------------


def count_study_cells(mode: ExecutionMode) -> int:
    """Cells in the main matrix: severities x strategies x seeds."""

    return len(mode.imbalance_settings) * len(mode.strategies) * len(mode.seeds)


def count_ablation_cells(mode: ExecutionMode, *, specs: int) -> int:
    """Cells in the ablation grid, which runs on one severity only."""

    return 0 if not mode.run_ablations else int(specs) * len(mode.ablation_seeds)


def scale_cell_seconds(
    seconds: float, *, measured_pool: int, target_pool: int, exponent: float = POOL_COST_EXPONENT
) -> float:
    """Predict a cell's cost at a different pool size."""

    if measured_pool < 1 or target_pool < 1:
        raise ValueError("Pool sizes must be positive.")
    return float(seconds) * (float(target_pool) / float(measured_pool)) ** float(exponent)


def measure_cell_seconds(
    *,
    prepared: PreparedPool,
    reference_embeddings: NDArray[np.float64],
    config: StudyConfig,
    strategy: str = "full",
    seed: int = 0,
) -> float:
    """Time one real campaign on the real pool.

    The gated strategy is timed because it is the most expensive — it is the only
    one needing both the neighbour search and the gate — so the estimate is an
    upper bound per cell rather than an average that under-counts the variant the
    run exists to measure.
    """

    spec = STRATEGY_REGISTRY.resolve(strategy)
    budgets = resolve_budgets(config.budgets, pool_size=prepared.size)
    started = time.perf_counter()
    run_campaign(
        pool=prepared.pool,
        spec=spec,
        reference_embeddings=np.asarray(reference_embeddings, dtype=np.float64),
        gt_class=prepared.table.gt_class,
        gt_is_unknown=prepared.table.gt_is_unknown,
        total_budget=max(budgets),
        rounds=config.rounds,
        seed=seed,
        saturation_mode=config.saturation_mode,
        saturation_strength=config.saturation_strength,
        keep_round_components=True,
    )
    return time.perf_counter() - started


@dataclass(frozen=True)
class CostEstimate:
    """Projected runtime, disk and memory, with the budget it is judged against.

    Deterministic given its inputs, and it changes nothing: if the projection does
    not fit, :meth:`enforce` raises. See :class:`RuntimeBudgetExceeded`.
    """

    mode_name: str
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
    export_disk_bytes: int
    pool_memory_bytes: int
    evaluation_images: int
    per_image_limit: int
    seeds: tuple[int, ...]

    @property
    def total_seconds(self) -> float:
        return (
            self.export_seconds + self.study_seconds + self.ablation_seconds + self.overhead_seconds
        )

    @property
    def within_budget(self) -> bool:
        return self.budget_seconds <= 0.0 or self.total_seconds <= self.budget_seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode_name,
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
            "export_disk_gb": round(self.export_disk_bytes / 1e9, 2),
            "pool_memory_gb": round(self.pool_memory_bytes / 1e9, 2),
            "evaluation_images": self.evaluation_images,
            "per_image_limit": self.per_image_limit,
            "seeds": list(self.seeds),
        }

    def report(self) -> str:
        return (
            f"projected {self.total_seconds / 3600:.2f} h "
            f"(export {self.export_seconds / 3600:.2f} h, study {self.study_seconds / 3600:.2f} h, "
            f"ablations {self.ablation_seconds / 3600:.2f} h) "
            f"against a budget of {self.budget_seconds / 3600:.2f} h; "
            f"export disk ~{self.export_disk_bytes / 1e9:.2f} GB, "
            f"pool memory ~{self.pool_memory_bytes / 1e9:.2f} GB"
        )

    def enforce(self) -> None:
        """Raise unless the projection fits the declared budget."""

        if self.within_budget:
            return
        raise RuntimeBudgetExceeded(
            f"{self.report()}. The run refuses to start rather than shrink the "
            "evaluation pool, because that would change the protocol after it was "
            "declared. Choose a smaller mode, or raise the runtime budget "
            "deliberately."
        )


def estimate_cost(
    *,
    mode: ExecutionMode,
    seconds_per_image: float,
    seconds_per_cell: float,
    measured_pool_size: int,
    ablation_specs: int,
    images_already_exported: int = 0,
    overhead_seconds: float = 0.0,
    budget_seconds: float = 0.0,
) -> CostEstimate:
    """Project the whole run from two measured rates. Changes nothing."""

    target_pool = max(mode.evaluation_images * mode.per_image_limit, 1)
    per_cell = scale_cell_seconds(
        seconds_per_cell, measured_pool=max(measured_pool_size, 1), target_pool=target_pool
    )
    study_cells = count_study_cells(mode)
    ablation_cells = count_ablation_cells(mode, specs=ablation_specs)
    images_to_export = max(mode.total_images - int(images_already_exported), 0)
    proposals = mode.total_images * mode.per_image_limit
    return CostEstimate(
        mode_name=mode.name,
        seconds_per_image=float(seconds_per_image),
        seconds_per_cell=float(seconds_per_cell),
        measured_pool_size=int(measured_pool_size),
        study_cells=study_cells,
        ablation_cells=ablation_cells,
        export_seconds=float(seconds_per_image) * images_to_export,
        study_seconds=per_cell * study_cells,
        ablation_seconds=per_cell * ablation_cells,
        overhead_seconds=float(overhead_seconds),
        budget_seconds=float(budget_seconds),
        export_disk_bytes=proposals * EXPORT_BYTES_PER_PROPOSAL,
        pool_memory_bytes=target_pool * POOL_BYTES_PER_PROPOSAL,
        evaluation_images=mode.evaluation_images,
        per_image_limit=mode.per_image_limit,
        seeds=tuple(mode.seeds),
    )


def stage_preflight(config: PipelineConfig, mode: ExecutionMode) -> list[Check]:
    """Validate the environment. Raises on any FAIL."""

    checks: list[Check] = [check_python()]
    checks += check_packages()
    needs_gpu = config.require_gpu and not config.existing_export
    checks += check_gpu(required=needs_gpu)
    if not config.existing_export:
        checks += check_prob_checkout(config.prob_repository)
        checks.append(
            check_bridge_cli(
                config.prob_repository, python_executable=config.python_executable or None
            )
        )
        checks += check_checkpoint(
            config.checkpoint,
            expected_sha256=config.expected_checkpoint_sha256 or None,
        )
        checks.append(
            check_disk(
                config.cache_dir,
                required_gb=max(
                    2.0,
                    estimate_export_gigabytes(
                        images=mode.total_images,
                        proposals_per_image=config.max_proposals_per_image,
                        dimensions=256,
                    )
                    * 1.5,
                ),
            )
        )
    else:
        checks.append(
            Check(
                name="existing_export",
                status="PASS" if Path(config.existing_export).exists() else "FAIL",
                detail=f"reusing {config.existing_export}; no detector inference will run",
            )
        )
    checks += check_dataset(
        config.data_root,
        dataset=config.dataset,
        split_file=config.split_file or None,
    )
    require_all_pass(checks)
    return checks


def stage_splits(
    config: PipelineConfig, mode: ExecutionMode, *, available_ids: Sequence[str]
) -> dict[str, list[str]]:
    """Disjoint reference / pilot / evaluation image pools."""

    counts = {
        "reference": mode.reference_images,
        "pilot": mode.pilot_images,
        "evaluation": mode.evaluation_images,
    }
    return detector.split_disjoint(available_ids, counts=counts, seed=config.seed)


def _available_ids(config: PipelineConfig) -> list[str]:
    if config.split_file:
        return read_image_ids(config.split_file)
    export = detector.load_export_file(config.existing_export)
    return sorted({str(value) for value in export["image_ids"].tolist()})


def stage_export(
    config: PipelineConfig,
    *,
    image_ids: Sequence[str],
    progress: Progress | None,
    stop_after_chunks: int | None = None,
) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
    """Proposals for ``image_ids``, from the cache, the GPU, or a supplied export."""

    if config.existing_export:
        _log(progress, f"Loading supplied export {config.existing_export}")
        export = detector.load_export_file(config.existing_export)
        wanted = {str(value) for value in image_ids}
        ids = np.asarray([str(value) for value in export["image_ids"].tolist()], dtype=object)
        keep = np.array([str(value) in wanted for value in ids.tolist()], dtype=np.bool_)
        missing = wanted - set(ids.astype(str).tolist())
        if missing:
            raise PipelineError(
                f"The supplied export is missing {len(missing)} requested image(s), "
                f"e.g. {sorted(missing)[:5]}. Export those images or restrict the "
                "split to what the file contains."
            )
        subset = {name: array[keep] for name, array in export.items()}
        return subset, {
            "source": "existing_export",
            "path": config.existing_export,
            "images": int(np.unique(ids[keep].astype(str)).size),
            "proposals": int(keep.sum()),
            "seconds_per_image": None,
        }

    result = detector.export_proposals(
        settings=config.bridge_settings(),
        image_ids=image_ids,
        cache_dir=config.cache_dir,
        chunk_images=config.chunk_images,
        progress=progress,
        stop_after_chunks=stop_after_chunks,
    )
    export = detector.load_chunks(result.paths())
    report = dict(result.as_dict())
    report["source"] = "prob_bridge"
    return export, report


def stage_probe_export_rate(
    config: PipelineConfig,
    *,
    image_ids: Sequence[str],
    probe_images: int = 25,
    progress: Progress | None = None,
) -> float | None:
    """Measure detector seconds-per-image on a small probe, before the bulk export.

    The projection has to happen *before* the expensive export, or reducing the
    image count could not save any GPU time — which is the whole point of the
    reduction. The probe writes into its own cache subdirectory so that shrinking
    the run afterwards does not invalidate a 250-image chunk of the main cache.
    """

    if config.existing_export:
        return None
    probe_ids = sorted(dict.fromkeys(str(value) for value in image_ids))[: max(probe_images, 1)]
    result = detector.export_proposals(
        settings=config.bridge_settings(),
        image_ids=probe_ids,
        cache_dir=Path(config.cache_dir) / "probe",
        chunk_images=max(probe_images, 1),
        progress=progress,
        stop_after_chunks=1,
    )
    rate = result.seconds_per_image()
    if rate is None:
        # Everything was cached, so this session has no fresh measurement. Reuse
        # the recorded rate from the probe manifest rather than inventing one.
        manifest = Path(config.cache_dir) / "probe" / "export_manifest.json"
        if manifest.exists():
            recorded = json.loads(manifest.read_text(encoding="utf-8")).get("seconds_per_image")
            if recorded:
                return float(recorded)
    return rate


def stage_pools(
    config: PipelineConfig,
    mode: ExecutionMode,
    *,
    export: Mapping[str, NDArray[np.generic]],
    splits: Mapping[str, Sequence[str]],
) -> tuple[PreparedPool, PreparedPool | None, NDArray[np.float64], ClassGroups]:
    """Evaluation pool, pilot pool, reference bank — sharing one class grouping.

    The grouping is derived from the *evaluation* pool's reachable class
    frequencies and then reused for the pilot, so head/medium/tail mean the same
    thing in both. Deriving it twice would let a pilot decision be made against a
    different definition of "tail" than the one reported.
    """

    config_study = mode.study_config()
    config_study = replace(config_study, iou_threshold=config.iou_threshold)
    evaluation = study.prepare_pool(
        export=export,
        annotations_dir=str(config.annotations_dir),
        config=config_study,
        restrict_to_images=list(splits["evaluation"]),
    )
    pilot = None
    if mode.run_pilot and splits.get("pilot"):
        pilot = study.prepare_pool(
            export=export,
            annotations_dir=str(config.annotations_dir),
            config=config_study,
            restrict_to_images=list(splits["pilot"]),
            class_groups=evaluation.class_groups,
        )
    reference_ids = {str(value) for value in splits["reference"]}
    ids = np.asarray([str(value) for value in export["image_ids"].tolist()], dtype=object)
    keep = np.array([str(value) in reference_ids for value in ids.tolist()], dtype=np.bool_)
    if not keep.any():
        raise PipelineError(
            "The reference image pool contributed no proposals, so novelty would "
            "be measured against nothing."
        )
    bank = study.reference_bank(
        {"embeddings": np.asarray(export["embeddings"])[keep]},
        limit=mode.reference_limit,
    )
    return evaluation, pilot, bank, evaluation.class_groups


def stage_severities(
    config: PipelineConfig,
    mode: ExecutionMode,
    *,
    prepared: PreparedPool,
) -> tuple[ExecutionMode, list[dict[str, object]], str]:
    """Build every severity once and refuse a run whose severities coincide."""

    if config.auto_calibrate_severities:
        name, settings, rows, attempts = longtail.choose_axis(
            prepared.table, class_groups=prepared.class_groups.groups
        )
        return (
            replace(mode, imbalance_settings=settings),
            [dict(row) for row in rows],
            f"auto-calibrated axis {name!r}; {attempts}",
        )
    rows = longtail.describe_settings(
        prepared.table,
        list(mode.imbalance_settings),
        class_groups=prepared.class_groups.groups,
    )
    verdict = longtail.validate_settings_distinct(rows)
    return mode, [dict(row) for row in rows], verdict


def stage_pilot(
    *,
    pilot: PreparedPool,
    bank: NDArray[np.float64],
    mode: ExecutionMode,
    progress: Progress | None,
) -> dict[str, object]:
    """Choose the coherence definition on the pilot pool, never on the reported one."""

    config_study = replace(
        mode.study_config(),
        # A cheap grid: the pilot's only job is to fix one configuration.
        budgets=tuple(value for value in mode.budgets if value <= max(mode.budgets) // 2)
        or (min(mode.budgets),),
        seeds=(mode.seeds[0],),
        imbalance_settings=(mode.imbalance_settings[-1],),
    )
    return study.select_hyperparameters(
        pilot=pilot,
        reference_embeddings=bank,
        config=config_study,
        seeds=(mode.seeds[0],),
        progress=progress,
    )


def stage_study(
    *,
    prepared: PreparedPool,
    bank: NDArray[np.float64],
    config_study: StudyConfig,
    state: StateStore,
    key_prefix: str,
    progress: Progress | None,
) -> StudyOutputs:
    """The main matrix, cached one severity at a time so a lost session resumes."""

    merged = StudyOutputs()
    total_seconds = 0.0
    cells = 0
    for spec in config_study.imbalance_settings:
        key = f"{key_prefix}_severity_{spec.name}"
        payload, reused = state.cached(
            key,
            lambda spec=spec: _study_payload(
                prepared=prepared,
                bank=bank,
                config_study=replace(config_study, imbalance_settings=(spec,)),
                progress=progress,
            ),
        )
        _log(
            progress,
            f"severity {spec.name}: {'reused cached results' if reused else 'completed'}",
        )
        assert isinstance(payload, dict)
        merged.strategy_rows.extend(payload["strategy_rows"])
        merged.auc_rows.extend(payload["auc_rows"])
        merged.selected_rows.extend(payload["selected_rows"])
        merged.distribution_rows.extend(payload["distribution_rows"])
        merged.outlier_rows.extend(payload["outlier_rows"])
        merged.pool_rows.extend(payload["pool_rows"])
        merged.class_frequency_rows.extend(payload["class_frequency_rows"])
        merged.anchored_rows.extend(payload.get("anchored_rows", []))
        total_seconds += float(payload["runtime"].get("seconds", 0.0))
        cells += int(payload["runtime"].get("cells", 0))
    merged.aggregated_rows = discovery.aggregate_over_seeds(merged.strategy_rows)
    merged.runtime = {
        "cells": cells,
        "seconds": round(total_seconds, 2),
        "config": config_study.as_dict(),
    }
    return merged


def _study_payload(
    *,
    prepared: PreparedPool,
    bank: NDArray[np.float64],
    config_study: StudyConfig,
    progress: Progress | None,
) -> dict[str, object]:
    outputs = study.run_study(
        prepared=prepared,
        reference_embeddings=bank,
        config=config_study,
        progress=progress,
    )
    return {
        "strategy_rows": outputs.strategy_rows,
        "auc_rows": outputs.auc_rows,
        "selected_rows": outputs.selected_rows,
        "distribution_rows": outputs.distribution_rows,
        "outlier_rows": outputs.outlier_rows,
        "pool_rows": outputs.pool_rows,
        "class_frequency_rows": outputs.class_frequency_rows,
        "anchored_rows": outputs.anchored_rows,
        "runtime": outputs.runtime,
    }


def stage_ablations(
    *,
    prepared: PreparedPool,
    bank: NDArray[np.float64],
    mode: ExecutionMode,
    config_study: StudyConfig,
    progress: Progress | None,
) -> list[dict[str, object]]:
    """The gate-form x coherence-definition grid, on the most imbalanced severity."""

    specs = study.ablation_specs()
    return study.run_ablations(
        prepared=prepared,
        reference_embeddings=bank,
        config=config_study,
        specs=specs,
        imbalance=mode.imbalance_settings[-1],
        seeds=mode.ablation_seeds,
        progress=progress,
    )


def limitations(
    *,
    mode: ExecutionMode,
    prepared: PreparedPool,
    export_report: Mapping[str, object],
    plan: Mapping[str, object],
) -> list[str]:
    """The caveats a reader needs, derived from what the run actually did."""

    notes = [
        "This measures **annotation-set quality**, not detector performance. It "
        "shows which regions a strategy buys for a fixed oracle budget; it does "
        "not retrain PROB, so no known-mAP, U-Recall, WI or A-OSE number is "
        "claimed here. The official evaluator remains the source for those.",
        "Discovery denominators are the unknown objects *reachable from the "
        "candidate pool*. Objects no proposal covers are excluded from both "
        "numerator and denominator; including them would scale every strategy "
        "down by the same constant and compress the contrast.",
        "Pseudo-classes come from k-means over decoder embeddings, so rarity is an "
        "estimate. Clustering instability is a known confound: the ablation grid "
        "includes the pseudo-label-free `radius_core` coherence definition "
        "precisely because it is immune to it.",
    ]
    tail_objects = prepared.targets.object_total("tail")
    if tail_objects < 20:
        notes.append(
            f"The tail group holds only {tail_objects} reachable objects, so tail "
            "recall moves in visible steps and small differences between "
            "strategies are not resolvable. A larger export is the only fix."
        )
    if len(mode.seeds) < 3:
        notes.append(
            f"Only {len(mode.seeds)} seed(s) were run, so the reported standard "
            "deviations are weak estimates of the acquisition's variance."
        )
    if not mode.research_grade:
        notes.append(
            f"Mode {mode.name} is a validation configuration, not the reported experiment."
        )
    if plan.get("actions"):
        notes.append(
            "The run was downscaled to fit its time budget: "
            + "; ".join(str(item) for item in plan["actions"])  # type: ignore[index]
            + "."
        )
    if str(export_report.get("source")) == "existing_export":
        notes.append(
            "Proposals were reused from a previously exported NPZ rather than "
            "inferred in this session; the detector checkpoint behind that export "
            "is recorded in its own manifest."
        )
    return notes


def run_pipeline(
    config: PipelineConfig,
    *,
    progress: Progress | None = print,
) -> PipelineResult:
    """Run every stage, resuming whatever is already on disk."""

    started = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = StateStore(output_dir / "state", force=config.force)
    mode = config.execution_mode()
    result = PipelineResult(config=config, mode=mode, output_dir=output_dir)
    stage_started = time.time()

    _log(progress, f"[1/11] Preflight ({mode.name})")
    checks = stage_preflight(config, mode)
    result.checks = rows(checks)
    tables.write_csv(output_dir / "preflight.csv", result.checks)
    result.stage_seconds["preflight"] = round(time.time() - stage_started, 1)

    _log(progress, "[2/11] Image splits")
    stage_started = time.time()
    available = _available_ids(config)
    splits = stage_splits(config, mode, available_ids=available)
    _log(
        progress,
        "  reference/pilot/evaluation images: "
        + "/".join(str(len(splits[name])) for name in ("reference", "pilot", "evaluation")),
    )
    result.stage_seconds["splits"] = round(time.time() - stage_started, 1)

    _log(progress, "[3/11] Detector rate probe and export projection")
    stage_started = time.time()
    seconds_per_image = stage_probe_export_rate(
        config, image_ids=splits["evaluation"], progress=progress
    )
    if seconds_per_image is not None:
        # Only the export term is known at this point, so it is judged against the
        # export half of the budget. Checking here means a pool that cannot even be
        # inferred in time fails before any GPU hour is committed, rather than three
        # hours in.
        export_only = estimate_cost(
            mode=mode,
            seconds_per_image=seconds_per_image,
            seconds_per_cell=0.0,
            measured_pool_size=max(mode.evaluation_images * mode.per_image_limit, 1),
            ablation_specs=0,
            budget_seconds=mode.runtime_budget_seconds * 0.5,
        )
        _log(
            progress,
            f"  {seconds_per_image:.2f}s/image; exporting {mode.total_images} images "
            f"projects {export_only.export_seconds / 3600:.2f} h; "
            f"export disk ~{export_only.export_disk_bytes / 1e9:.2f} GB",
        )
        export_only.enforce()
    state.save(f"splits_{config.fingerprint()}", dict(splits))
    result.stage_seconds["probe"] = round(time.time() - stage_started, 1)

    _log(progress, "[4/11] Proposal export (cached, resumable)")
    stage_started = time.time()
    requested = sorted(set(splits["reference"]) | set(splits["pilot"]) | set(splits["evaluation"]))
    export, export_report = stage_export(config, image_ids=requested, progress=progress)
    if export_report.get("seconds_per_image") is None and seconds_per_image is not None:
        export_report["seconds_per_image"] = seconds_per_image
        export_report["seconds_per_image_source"] = "probe"
    result.export = dict(export_report)
    tables.write_json(output_dir / "export_report.json", result.export)
    result.stage_seconds["export"] = round(time.time() - stage_started, 1)

    _log(progress, "[5/11] Candidate pool and region oracle")
    stage_started = time.time()
    evaluation, pilot, bank, groups = stage_pools(config, mode, export=export, splits=splits)
    result.pool_report = dict(evaluation.pool_report)
    result.composition = dict(evaluation.composition)
    _log(
        progress,
        f"  pool {evaluation.size} proposals; reachable unknown objects "
        f"{evaluation.targets.object_total('all')} "
        f"(head {evaluation.targets.object_total('head')}, "
        f"medium {evaluation.targets.object_total('medium')}, "
        f"tail {evaluation.targets.object_total('tail')}); "
        f"{evaluation.targets.class_total('all')} classes",
    )
    tables.write_json(
        output_dir / "pool_report.json",
        {
            "candidate_pool": result.pool_report,
            "composition": result.composition,
            "class_groups": groups.as_dict(),
            "group_counts": groups.counts(),
            "reachable_objects": {
                group: evaluation.targets.object_total(group)
                for group in ("all", "head", "medium", "tail")
            },
            "reachable_classes": {
                group: evaluation.targets.class_total(group)
                for group in ("all", "head", "medium", "tail")
            },
            "reference_bank_rows": int(bank.shape[0]),
        },
    )
    result.stage_seconds["pools"] = round(time.time() - stage_started, 1)

    _log(progress, "[6/11] Long-tail severities")
    stage_started = time.time()
    mode, severity_rows, verdict = stage_severities(config, mode, prepared=evaluation)
    result.mode = mode
    result.severity_rows = severity_rows
    result.severity_verdict = verdict
    _log(progress, f"  {verdict}")
    result.stage_seconds["severities"] = round(time.time() - stage_started, 1)

    _log(progress, "[7/11] Leakage controls")
    stage_started = time.time()
    base_study = replace(mode.study_config(), iou_threshold=config.iou_threshold)
    result.leakage = dict(
        study.leakage_check(prepared=evaluation, reference_embeddings=bank, config=base_study)
    )
    tables.write_json(output_dir / "leakage_report.json", result.leakage)
    _log(progress, f"  {result.leakage}")
    result.stage_seconds["leakage"] = round(time.time() - stage_started, 1)

    _log(progress, "[8/11] Pilot hyperparameter choice")
    stage_started = time.time()
    if pilot is not None:
        payload, reused = state.cached(
            f"pilot_{config.fingerprint()}",
            lambda: stage_pilot(pilot=pilot, bank=bank, mode=mode, progress=progress),
        )
        assert isinstance(payload, dict)
        result.pilot = payload
        _log(
            progress,
            f"  {'reused' if reused else 'chose'} coherence="
            f"{payload.get('chosen_coherence_method')} k={payload.get('chosen_neighbour_count')}",
        )
        tables.write_csv(output_dir / "pilot_ablation.csv", list(payload.get("pilot_rows", [])))
        base_study = replace(
            base_study,
            coherence_method_override=str(payload["chosen_coherence_method"]),
            neighbour_count_override=int(payload["chosen_neighbour_count"]),
        )
    else:
        _log(progress, "  skipped: this mode runs no pilot")
    result.stage_seconds["pilot"] = round(time.time() - stage_started, 1)

    _log(progress, "[9/11] Cost estimate")
    stage_started = time.time()
    seconds_per_cell = measure_cell_seconds(
        prepared=evaluation, reference_embeddings=bank, config=base_study
    )
    # The export is already paid for, so the remaining work is judged against the
    # *remaining* budget. Comparing it against the full budget would let a run that
    # spent three hours on inference still claim its four-hour matrix fits.
    remaining_budget = max(0.0, mode.runtime_budget_seconds - (time.time() - started))
    estimate = estimate_cost(
        mode=mode,
        seconds_per_image=float(export_report.get("seconds_per_image") or 0.0),
        seconds_per_cell=seconds_per_cell,
        measured_pool_size=evaluation.size,
        ablation_specs=len(study.ablation_specs()),
        images_already_exported=len(requested),
        budget_seconds=remaining_budget,
    )
    result.runtime_plan = estimate.as_dict()
    tables.write_json(output_dir / "runtime_plan.json", result.runtime_plan)
    _log(progress, f"  one cell {seconds_per_cell:.1f}s at pool {evaluation.size}")
    _log(progress, f"  {estimate.report()}")
    estimate.enforce()
    result.stage_seconds["runtime_plan"] = round(time.time() - stage_started, 1)

    _log(progress, "[10/11] Main study matrix")
    stage_started = time.time()
    study_key = f"study_{config.fingerprint()}_{_study_fingerprint(base_study, evaluation.size)}"
    result.outputs = stage_study(
        prepared=evaluation,
        bank=bank,
        config_study=base_study,
        state=state,
        key_prefix=study_key,
        progress=progress,
    )
    result.stage_seconds["study"] = round(time.time() - stage_started, 1)

    if mode.run_ablations:
        _log(progress, "  ablation grid")
        stage_started = time.time()
        payload, reused = state.cached(
            f"ablations_{config.fingerprint()}_{_study_fingerprint(base_study, evaluation.size)}",
            lambda: stage_ablations(
                prepared=evaluation,
                bank=bank,
                mode=mode,
                config_study=base_study,
                progress=progress,
            ),
        )
        assert isinstance(payload, list)
        result.ablation_rows = [dict(row) for row in payload]
        result.stage_seconds["ablations"] = round(time.time() - stage_started, 1)

    _log(progress, "[11/11] Metrics, figures, report")
    stage_started = time.time()
    result.contrasts = tables.headline_contrasts(result.outputs.auc_rows)
    arm_rows = tables.arm_comparison(result.outputs.auc_rows)
    arm_rows_tail = tables.arm_comparison(result.outputs.auc_rows, metric="tail_discovery_auc")
    counts = tables.discovery_counts(result.outputs.strategy_rows)
    tables.write_csv(output_dir / "arm_comparison_unknown.csv", arm_rows)
    tables.write_csv(output_dir / "arm_comparison_tail.csv", arm_rows_tail)
    tables.write_csv(output_dir / "discovery_counts.csv", counts)
    result.cost_rows = tables.budget_to_reach(
        result.outputs.strategy_rows, target_recall=config.target_tail_recall
    )
    written = tables.write_study_outputs(result.outputs, output_dir)
    tables.write_csv(output_dir / "headline_contrasts.csv", result.contrasts)
    tables.write_csv(output_dir / "cost_to_target.csv", result.cost_rows)
    tables.write_csv(output_dir / "strategy_summary.csv", result.headline())
    tables.write_csv(output_dir / "ablations.csv", result.ablation_rows)
    tables.write_csv(output_dir / "severity_report.csv", result.severity_rows)
    result.figures = figures.render_all(
        curve_rows=result.outputs.strategy_rows,
        auc_rows=result.outputs.auc_rows,
        distribution_rows=result.outputs.distribution_rows,
        gate_rows=result.outputs.outlier_rows,
        class_frequency_rows=result.outputs.class_frequency_rows,
        ablation_rows=result.ablation_rows,
        cost_rows=result.cost_rows,
        directory=output_dir,
    )
    total_runtime = {**result.stage_seconds, "total": round(time.time() - started, 1)}
    summary = tables.research_summary(
        mode=mode.as_dict(),
        pool_report=result.pool_report,
        composition=result.composition,
        severity_rows=result.severity_rows,
        auc_rows=result.outputs.auc_rows,
        curve_rows=result.outputs.strategy_rows,
        contrasts=result.contrasts,
        gate_rows=result.outputs.outlier_rows,
        cost_rows=result.cost_rows,
        leakage=result.leakage,
        runtime=total_runtime,
        pilot=result.pilot or None,
        ablation_rows=result.ablation_rows,
        distribution_rows=result.outputs.distribution_rows,
        arm_rows=arm_rows,
        limitations=limitations(
            mode=mode,
            prepared=evaluation,
            export_report=result.export,
            plan=result.runtime_plan,
        ),
    )
    result.summary_path = output_dir / "research_summary.md"
    result.summary_path.write_text(summary + "\n", encoding="utf-8")
    tables.write_json(
        output_dir / "run_manifest.json",
        {
            "config": config.as_dict(),
            "mode": mode.as_dict(),
            "study_config": base_study.as_dict(),
            "environment": environment_report(),
            "export": result.export,
            "severity_verdict": result.severity_verdict,
            "leakage": result.leakage,
            "runtime_plan": result.runtime_plan,
            "stage_seconds": total_runtime,
            "tables": {name: str(path) for name, path in written.items()},
            "figures": [str(path) for path in result.figures],
        },
    )
    result.artifacts = tables.verify_expected_files(output_dir, REQUIRED_ARTIFACTS)
    missing = [row["artifact"] for row in result.artifacts if row["status"] != "PASS"]
    if missing:
        raise PipelineError(f"The run finished but these artifacts are missing: {missing}")
    result.archive = tables.bundle(
        output_dir, archive=output_dir / f"daowod_contribution_a_{mode.name.lower()}.zip"
    )
    result.stage_seconds["report"] = round(time.time() - stage_started, 1)
    _log(
        progress,
        f"Done in {(time.time() - started) / 60:.1f} min. Artifacts: {output_dir}; "
        f"archive: {result.archive}",
    )
    return result


def _study_fingerprint(config_study: StudyConfig, pool_size: int) -> str:
    payload = dict(config_study.as_dict())
    payload["pool_size"] = int(pool_size)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


# =============================================================================
# Preflight: everything that must be true before a GPU hour is spent
#
# Everything that must be true before a GPU hour is spent.
#
# Each check returns a row rather than printing, so the notebook can render one
# table and the pipeline can refuse to start on a single ``FAIL``. Three statuses
# are distinguished on purpose:
#
# ``PASS``   the requirement is satisfied and was verified, not assumed.
# ``FAIL``   the run cannot produce valid results; the pipeline stops.
# ``SKIP``   the requirement does not apply to this run (no GPU needed in DEBUG).
# ``WARN``   the run can proceed but a reported number will be weaker for it.
#
# The distinction matters because the failure mode this module exists to prevent is
# a three-hour session that ends in a missing-file error, or worse, one that
# finishes and reports numbers derived from the wrong split.
#
# Merged into the pipeline because the pipeline is its only caller: preflight is
# stage 0, not a separate concern. It also carries the deterministic cost estimate
# above, which is the other thing checked before the run commits to anything.
# =============================================================================

PROB_FEATURE_MARKER = "pred_features"

#: Files the bridge contract requires inside the PROB checkout.
PROB_REQUIRED_FILES: tuple[tuple[str, str], ...] = (
    ("daowod_prob_bridge.py", "def predict("),
    ("main_open_world.py", "def get_args_parser("),
    ("models/prob_deformable_detr.py", PROB_FEATURE_MARKER),
)

#: Packages the offline study needs, with the import name where it differs.
REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("PyYAML", "yaml"),
)


class PreflightError(RuntimeError):
    """Raised when a required precondition is not met."""


@dataclass(frozen=True)
class Check:
    """One verified precondition."""

    name: str
    status: str
    detail: str
    value: object = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "check": self.name,
            "status": self.status,
            "detail": self.detail,
            "value": self.value,
        }


def _run(command: Sequence[str], *, timeout: int = 60) -> tuple[int, str]:
    try:
        result = subprocess.run(
            [str(part) for part in command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:  # pragma: no cover - env specific
        return 1, str(error)
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def check_python() -> Check:
    """Python must be 3.11 or 3.12: the package pins that range."""

    major, minor = sys.version_info[:2]
    supported = (major, minor) in ((3, 11), (3, 12))
    return Check(
        name="python_version",
        status="PASS" if supported else "FAIL",
        detail=f"{platform.python_version()} on {platform.platform()}",
        value=f"{major}.{minor}",
    )


def check_packages(packages: Sequence[tuple[str, str]] = REQUIRED_PACKAGES) -> list[Check]:
    """Every offline dependency must import and report a version."""

    checks: list[Check] = []
    for distribution, module_name in packages:
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            checks.append(
                Check(
                    name=f"package:{distribution}",
                    status="FAIL",
                    detail=f"import {module_name} failed: {error}",
                )
            )
            continue
        checks.append(
            Check(
                name=f"package:{distribution}",
                status="PASS",
                detail=f"import {module_name} ok",
                value=getattr(module, "__version__", "unknown"),
            )
        )
    return checks


def check_gpu(*, required: bool, require_t4_or_better: bool = True) -> list[Check]:
    """CUDA, a visible device, and enough memory for PROB inference.

    ``required=False`` turns every GPU check into ``SKIP`` rather than removing
    it, so a CPU-only DEBUG run still reports what it did not verify.
    """

    if not required:
        return [
            Check(
                name="gpu",
                status="SKIP",
                detail="This mode runs the offline study only; no detector inference.",
            )
        ]
    checks: list[Check] = []
    code, output = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    visible = code == 0 and output.strip() != ""
    checks.append(
        Check(
            name="nvidia_smi",
            status="PASS" if visible else "FAIL",
            detail=output.strip()[:300] if visible else "nvidia-smi unavailable or reported no GPU",
            value=output.strip().splitlines()[0] if visible else "",
        )
    )
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        checks.append(
            Check(name="torch_cuda", status="FAIL", detail=f"import torch failed: {error}")
        )
        return checks
    available = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if available else ""
    total_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if available else 0.0
    )
    checks.append(
        Check(
            name="torch_cuda",
            status="PASS" if available else "FAIL",
            detail=f"torch {torch.__version__}, cuda available={available}, device={name!r}",
            value=name,
        )
    )
    if available:
        # A T4 has 16 GB and is the reference device. Anything with at least 12 GB
        # runs the same configuration; less than that and PROB inference at the
        # protocol's resolution will not fit, which is a FAIL rather than a WARN
        # because the alternative is an out-of-memory crash mid-session.
        enough = total_gb >= 12.0
        recognised = "T4" in name.upper()
        status = "PASS" if enough else "FAIL"
        if enough and require_t4_or_better and not recognised:
            status = "WARN"
        checks.append(
            Check(
                name="gpu_memory",
                status=status,
                detail=(
                    f"{total_gb} GB on {name!r}. The runtime budget is calibrated for "
                    "a 16 GB T4; a different device changes the projection, not the "
                    "validity of the results."
                ),
                value=total_gb,
            )
        )
    return checks


def check_repository(path: str | Path, *, name: str) -> list[Check]:
    """A checkout must exist, be a git repository, and report its commit."""

    root = Path(path)
    if not root.exists():
        return [Check(name=f"{name}_checkout", status="FAIL", detail=f"missing: {root}")]
    code, head = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    commit = head.strip() if code == 0 else "unavailable"
    code, status = _run(["git", "-C", str(root), "status", "--short"])
    dirty = status.strip()
    return [
        Check(
            name=f"{name}_checkout",
            status="PASS",
            detail=f"{root} at commit {commit[:12]}",
            value=commit,
        ),
        Check(
            name=f"{name}_worktree",
            status="PASS" if not dirty else "WARN",
            detail=(
                "clean"
                if not dirty
                else f"{len(dirty.splitlines())} modified path(s); results are not "
                "reproducible from the commit alone"
            ),
            value=len(dirty.splitlines()),
        ),
    ]


def check_prob_checkout(path: str | Path) -> list[Check]:
    """The PROB checkout must expose the bridge and the decoder feature export."""

    root = Path(path)
    checks = check_repository(root, name="prob")
    if not root.exists():
        return checks
    for relative, marker in PROB_REQUIRED_FILES:
        target = root / relative
        if not target.exists():
            checks.append(
                Check(
                    name=f"prob_file:{relative}",
                    status="FAIL",
                    detail=f"missing: {target}",
                )
            )
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        present = marker in text
        checks.append(
            Check(
                name=f"prob_file:{relative}",
                status="PASS" if present else "FAIL",
                detail=(
                    f"contains {marker!r}"
                    if present
                    else (
                        f"{target} exists but does not contain {marker!r}. The "
                        "proposal export depends on this symbol; a renamed "
                        "function or a dropped model output would otherwise fail "
                        "hours into the run."
                    )
                ),
            )
        )
    return checks


def check_bridge_cli(path: str | Path, *, python_executable: str | None = None) -> Check:
    """``daowod_prob_bridge.py check`` must exit zero inside the PROB checkout."""

    root = Path(path)
    bridge = root / "daowod_prob_bridge.py"
    if not bridge.exists():
        return Check(name="prob_bridge_check", status="FAIL", detail=f"missing: {bridge}")
    interpreter = python_executable or sys.executable
    code, output = _run([interpreter, str(bridge), "check"], timeout=600)
    return Check(
        name="prob_bridge_check",
        status="PASS" if code == 0 else "FAIL",
        detail=(output.strip()[-400:] or "no output"),
        value=code,
    )


def check_dataset(
    root: str | Path,
    *,
    dataset: str = "OWDETR",
    split_file: str | Path | None = None,
    sample: int = 5,
) -> list[Check]:
    """The VOC-style tree, the split file, and a sample of real image/annotation pairs."""

    base = Path(root)
    checks: list[Check] = []
    for directory in ("Annotations", "JPEGImages", "ImageSets"):
        target = base / directory
        checks.append(
            Check(
                name=f"dataset_dir:{directory}",
                status="PASS" if target.is_dir() else "FAIL",
                detail=str(target),
                value=len(list(target.iterdir())) if target.is_dir() else 0,
            )
        )
    split_directory = base / "ImageSets" / dataset
    checks.append(
        Check(
            name=f"dataset_splits:{dataset}",
            status="PASS" if split_directory.is_dir() else "FAIL",
            detail=str(split_directory),
            value=(
                sorted(item.name for item in split_directory.glob("*.txt"))
                if split_directory.is_dir()
                else []
            ),
        )
    )
    if split_file is None:
        return checks

    path = Path(split_file)
    if not path.exists():
        checks.append(Check(name="split_file", status="FAIL", detail=f"missing: {path}"))
        return checks
    ids = read_image_ids(path)
    checks.append(
        Check(
            name="split_file",
            status="PASS" if ids else "FAIL",
            detail=f"{path} lists {len(ids)} image id(s)",
            value=len(ids),
        )
    )
    missing_images = [
        image_id
        for image_id in ids[: max(sample, 1)]
        if not (base / "JPEGImages" / f"{image_id}.jpg").exists()
    ]
    missing_annotations = [
        image_id
        for image_id in ids[: max(sample, 1)]
        if not (base / "Annotations" / f"{image_id}.xml").exists()
    ]
    checks.append(
        Check(
            name="split_assets",
            status="PASS" if not (missing_images or missing_annotations) else "FAIL",
            detail=(
                f"sampled {min(len(ids), max(sample, 1))} id(s); missing images "
                f"{missing_images[:3]}, missing annotations {missing_annotations[:3]}"
            ),
        )
    )
    if ids and not missing_annotations:
        try:
            parsed = oracle.read_voc_annotation(ids[0], base / "Annotations")
            unknown = [item.class_name for item in parsed.objects if not item.is_known]
            checks.append(
                Check(
                    name="annotation_parse",
                    status="PASS",
                    detail=(
                        f"{ids[0]}: {parsed.width}x{parsed.height}, "
                        f"{len(parsed.objects)} object(s), "
                        f"{len(unknown)} unknown at Task 1"
                    ),
                    value=len(parsed.objects),
                )
            )
        except oracle.OracleError as error:
            checks.append(Check(name="annotation_parse", status="FAIL", detail=str(error)))
    return checks


def check_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    load: bool = False,
) -> list[Check]:
    """The detector checkpoint must exist, be non-trivial, and optionally match a digest."""

    target = Path(path)
    if not target.exists():
        return [Check(name="checkpoint", status="FAIL", detail=f"missing: {target}")]
    size_mb = round(target.stat().st_size / 1024**2, 1)
    checks = [
        Check(
            name="checkpoint",
            status="PASS" if size_mb > 1.0 else "FAIL",
            detail=f"{target} ({size_mb} MB)",
            value=size_mb,
        )
    ]
    if expected_sha256:
        from daowod.dataset import file_sha256

        digest = file_sha256(target)
        checks.append(
            Check(
                name="checkpoint_sha256",
                status="PASS" if digest == expected_sha256 else "FAIL",
                detail=f"{digest} (expected {expected_sha256})",
                value=digest,
            )
        )
    if load:
        try:
            torch = importlib.import_module("torch")
            state = torch.load(str(target), map_location="cpu")
            keys = sorted(state)[:6] if isinstance(state, dict) else []
            checks.append(
                Check(
                    name="checkpoint_load",
                    status="PASS",
                    detail=f"loaded; top-level keys {keys}",
                )
            )
        except Exception as error:  # pragma: no cover - depends on torch build
            checks.append(
                Check(name="checkpoint_load", status="FAIL", detail=f"torch.load failed: {error}")
            )
    return checks


def check_disk(path: str | Path, *, required_gb: float) -> Check:
    """Free space for the proposal export, which is the largest artifact."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    free_gb = round(usage.free / 1024**3, 1)
    return Check(
        name="disk_space",
        status="PASS" if free_gb >= required_gb else "FAIL",
        detail=f"{free_gb} GB free at {target}, need about {required_gb} GB",
        value=free_gb,
    )


def estimate_export_gigabytes(*, images: int, proposals_per_image: int, dimensions: int) -> float:
    """Size of one export: embeddings dominate, at float32 on disk via ``np.savez``.

    Used by the disk check so "not enough space" is reported before the export
    rather than at the last chunk.
    """

    rows = max(int(images), 0) * max(int(proposals_per_image), 0)
    per_row_bytes = 4 * (int(dimensions) + 4 + 20 + 4)  # embeddings, box, posterior, scalars
    return round(rows * per_row_bytes / 1024**3, 2)


def read_image_ids(path: str | Path) -> list[str]:
    """Image IDs from a VOC ``ImageSets`` file, first whitespace field per line."""

    text = Path(path).read_text(encoding="utf-8")
    return [line.split()[0] for line in text.splitlines() if line.strip()]


def summarise(checks: Iterable[Check]) -> dict[str, object]:
    """Counts per status plus the failing names, for the notebook's header line."""

    rows = list(checks)
    counts: dict[str, int] = {}
    for check in rows:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "checks": len(rows),
        "counts": counts,
        "failed": [check.name for check in rows if check.status == "FAIL"],
        "warned": [check.name for check in rows if check.status == "WARN"],
    }


def require_all_pass(checks: Sequence[Check]) -> None:
    """Raise a single error naming every failed check and its detail."""

    failures = [check for check in checks if check.status == "FAIL"]
    if not failures:
        return
    lines = "\n".join(f"  - {check.name}: {check.detail}" for check in failures)
    raise PreflightError(f"{len(failures)} precondition(s) failed:\n{lines}")


def rows(checks: Sequence[Check]) -> list[Mapping[str, object]]:
    """Check rows for ``csv``."""

    return [check.as_dict() for check in checks]


def environment_report() -> dict[str, object]:
    """Free-form context recorded with every run for reproducibility."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }
