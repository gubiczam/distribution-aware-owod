"""Execution modes and the YAML that declares them.

A mode fixes every quantity that trades runtime against statistical power: how
many images are inferred, how many proposals per image survive the candidate
filter, the annotation budget grid, the round count, the seeds, the arms, and
which severity axis is used. Nothing in the pipeline asks "am I in debug?" — it
receives a mode.

The modes themselves live in ``configs/contribution_a.yaml``, not here. Sizes and
seeds *are* the protocol, so a change to them must show up as a reviewable diff in
version control rather than as an edited Python constant or notebook cell. This
module is the schema and the loader: it validates what the YAML declares and
refuses anything internally inconsistent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from daowod import longtail
from daowod.candidates import CandidatePoolSpec
from daowod.longtail import ImbalanceSpec
from daowod.study import COMPARISON_STRATEGIES, PRIMARY_STRATEGIES, StudyConfig

#: The default configuration shipped with the repository.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "contribution_a.yaml"

#: Named strategy sets a mode may request, so a YAML file cannot silently invent an
#: arm list that no result was ever reported for.
STRATEGY_SETS: dict[str, tuple[str, ...]] = {
    "primary": PRIMARY_STRATEGIES,
    "comparison": COMPARISON_STRATEGIES,
}

#: Named severity axes. `flattening` is the only one small pools can express; see
#: the measured sizes in configs/contribution_a.yaml.
SEVERITY_AXES: dict[str, tuple[ImbalanceSpec, ...]] = {
    "default": longtail.DEFAULT_IMBALANCE_SETTINGS,
    "flattening": longtail.FLATTENING_IMBALANCE_SETTINGS,
}


class ModeError(ValueError):
    """Raised for an unknown or internally inconsistent execution mode."""


@dataclass(frozen=True)
class ExecutionMode:
    """One complete, self-consistent experiment size."""

    name: str
    description: str
    evaluation_images: int
    pilot_images: int
    reference_images: int
    per_image_limit: int
    budgets: tuple[int, ...]
    rounds: int
    seeds: tuple[int, ...]
    strategies: tuple[str, ...]
    imbalance_settings: tuple[ImbalanceSpec, ...]
    run_pilot: bool
    run_ablations: bool
    ablation_seeds: tuple[int, ...]
    reference_limit: int
    runtime_budget_seconds: float
    research_grade: bool

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.upper():
            raise ModeError(f"A mode name must be non-empty and upper-case, got {self.name!r}.")
        if min(self.evaluation_images, self.reference_images) < 1:
            raise ModeError(f"{self.name}: image counts must be positive.")
        if self.pilot_images < 0:
            raise ModeError(f"{self.name}: pilot_images must be non-negative.")
        if self.run_pilot and self.pilot_images < 1:
            raise ModeError(f"{self.name}: a pilot needs pilot_images >= 1.")
        if not self.budgets or min(self.budgets) < 1:
            raise ModeError(f"{self.name}: budgets must be positive.")
        if not self.seeds:
            raise ModeError(f"{self.name}: at least one seed is required.")
        if len(self.imbalance_settings) < 2:
            raise ModeError(
                f"{self.name}: at least two severities are required for a long-tail contrast."
            )

    @property
    def total_images(self) -> int:
        """Images the detector must be run over — what the GPU time buys."""

        return self.evaluation_images + self.pilot_images + self.reference_images

    def study_config(self) -> StudyConfig:
        """The :class:`~daowod.study.StudyConfig` this mode implies."""

        return StudyConfig(
            budgets=tuple(self.budgets),
            rounds=self.rounds,
            seeds=tuple(self.seeds),
            strategies=tuple(self.strategies),
            imbalance_settings=tuple(self.imbalance_settings),
            candidate_spec=CandidatePoolSpec(per_image_limit=self.per_image_limit),
            reference_limit=self.reference_limit,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "evaluation_images": self.evaluation_images,
            "pilot_images": self.pilot_images,
            "reference_images": self.reference_images,
            "total_images": self.total_images,
            "per_image_limit": self.per_image_limit,
            "budgets": list(self.budgets),
            "rounds": self.rounds,
            "seeds": list(self.seeds),
            "strategies": list(self.strategies),
            "imbalance_settings": [spec.as_dict() for spec in self.imbalance_settings],
            "run_pilot": self.run_pilot,
            "run_ablations": self.run_ablations,
            "ablation_seeds": list(self.ablation_seeds),
            "reference_limit": self.reference_limit,
            "runtime_budget_seconds": self.runtime_budget_seconds,
            "research_grade": self.research_grade,
        }


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


#: Modes loaded from YAML, keyed by name. Populated by :func:`load_modes`, and by
#: :func:`register` for a size a test or a one-off analysis needs.
MODES: dict[str, ExecutionMode] = {}


def _require(section: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in section:
        raise ConfigError(f"mode {name!r}: missing required key {key!r}.")
    return section[key]


def mode_from_mapping(name: str, payload: Mapping[str, Any]) -> ExecutionMode:
    """Build one validated mode from its YAML mapping."""

    strategies = str(_require(payload, "strategies", name))
    if strategies not in STRATEGY_SETS:
        raise ConfigError(
            f"mode {name!r}: unknown strategy set {strategies!r}. "
            f"Supported: {sorted(STRATEGY_SETS)}"
        )
    axis = str(_require(payload, "severity_axis", name))
    if axis not in SEVERITY_AXES:
        raise ConfigError(
            f"mode {name!r}: unknown severity_axis {axis!r}. Supported: {sorted(SEVERITY_AXES)}"
        )
    return ExecutionMode(
        name=name.upper(),
        description=" ".join(str(payload.get("description", "")).split()),
        evaluation_images=int(_require(payload, "evaluation_images", name)),
        pilot_images=int(_require(payload, "pilot_images", name)),
        reference_images=int(_require(payload, "reference_images", name)),
        per_image_limit=int(_require(payload, "per_image_limit", name)),
        budgets=tuple(int(value) for value in _require(payload, "budgets", name)),
        rounds=int(_require(payload, "rounds", name)),
        seeds=tuple(int(value) for value in _require(payload, "seeds", name)),
        strategies=STRATEGY_SETS[strategies],
        imbalance_settings=SEVERITY_AXES[axis],
        run_pilot=bool(_require(payload, "run_pilot", name)),
        run_ablations=bool(_require(payload, "run_ablations", name)),
        ablation_seeds=tuple(int(value) for value in payload.get("ablation_seeds", (0,))),
        reference_limit=int(_require(payload, "reference_limit", name)),
        runtime_budget_seconds=float(payload.get("runtime_budget_hours", 0.0)) * 3600.0,
        research_grade=bool(_require(payload, "research_grade", name)),
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a configuration file, validate every mode, and register them all.

    Returns the ``run`` section plus the resolved default mode name, so a caller
    can build a :class:`~daowod.pipeline.PipelineConfig` without re-reading YAML.
    """

    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not source.exists():
        raise ConfigError(f"Configuration file not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"{source}: top level must be a mapping.")

    declared = payload.get("modes") or {}
    if not declared:
        raise ConfigError(f"{source}: no modes declared.")
    for name, section in declared.items():
        if not isinstance(section, Mapping):
            raise ConfigError(f"{source}: mode {name!r} must be a mapping.")
        register(mode_from_mapping(str(name), section), replace_existing=True)

    default = str(payload.get("mode", "MAIN")).strip().upper()
    if default not in MODES:
        raise ConfigError(f"{source}: mode: {default!r} is not declared. Declared: {sorted(MODES)}")
    run = payload.get("run") or {}
    if not isinstance(run, Mapping):
        raise ConfigError(f"{source}: `run` must be a mapping.")
    return {"source": str(source), "mode": default, "run": dict(run)}


def load_modes(path: str | Path | None = None) -> dict[str, ExecutionMode]:
    """Register every mode declared in a configuration file."""

    load_config(path)
    return dict(MODES)


def normalise_mode_name(name: str) -> str:
    """Canonical mode key: upper-case, underscores and hyphens removed.

    So ``MAIN_REVEALED``, ``main-revealed`` and ``MAINREVEALED`` are one mode. The
    separator is cosmetic and a run must not silently fall through to a *different*
    protocol because a caller typed it differently.
    """

    return str(name).strip().upper().replace("_", "").replace("-", "")


def resolve_mode(name: str) -> ExecutionMode:
    """Look up a mode by name, ignoring case and separators; loads the default file once."""

    if not MODES:
        load_config()
    key = normalise_mode_name(name)
    lookup = {normalise_mode_name(declared): declared for declared in MODES}
    if key not in lookup:
        raise ModeError(f"Unknown execution mode {name!r}. Supported: {sorted(MODES)}")
    return MODES[lookup[key]]


def register(mode: ExecutionMode, *, replace_existing: bool = False) -> ExecutionMode:
    """Add a size to the registry so `mode:` and `PipelineConfig(mode=...)` accept it.

    Overwriting an existing mode is refused unless asked for explicitly: a run that
    reports MAIN must mean the MAIN that configs/contribution_a.yaml declares.
    """

    key = mode.name.strip().upper()
    if key in MODES and not replace_existing:
        raise ModeError(
            f"Mode {key!r} already exists. Pass replace_existing=True to override it, "
            "or choose another name."
        )
    MODES[key] = mode
    return mode
