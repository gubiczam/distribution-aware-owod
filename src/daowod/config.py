"""The single configuration path: YAML -> validated config -> scorer -> report.

Before the audit the live campaign was configured by Python constants in a
notebook cell and ``configs/experiment.yaml`` was inert. Everything now resolves
through :func:`load_config`, and strategy names resolve through the one registry
in :mod:`daowod.scoring`, so the validator, scorer, CLI and reports cannot
disagree about what a strategy means.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from daowod.acquisition import AcquisitionWeights
from daowod.scoring import (
    IMAGE_AGGREGATIONS,
    STRATEGY_REGISTRY,
    StrategyError,
    StrategySpec,
)


class ConfigError(ValueError):
    """Raised for a malformed or self-inconsistent experiment configuration."""


@dataclass(frozen=True)
class ActiveLearningConfig:
    rounds: int = 1
    initial_images: int = 20
    budget_per_round: int = 10
    seeds: tuple[int, ...] = (0,)
    #: Legacy field: the single-strategy entry point still accepts it.
    strategy: str = "v2:full"
    budget: int = 10

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ConfigError("rounds must be positive.")
        if min(self.budget, self.initial_images, self.budget_per_round) < 1:
            raise ConfigError("Active-learning values must be positive.")
        if not self.seeds:
            raise ConfigError("At least one seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigError(f"Duplicate seeds: {self.seeds}")
        STRATEGY_REGISTRY.resolve(self.strategy)


@dataclass(frozen=True)
class AcquisitionConfig:
    """Which strategies to run, and the parameters shared by all of them.

    ``strategies`` are registry names; a name that exists in both semantics
    versions must be qualified (``v1:full`` / ``v2:full``). Parameter overrides
    below are applied on top of each resolved spec, so one config can sweep
    ``coherence_method`` across every strategy without redefining them.
    """

    strategies: tuple[str, ...] = ("v2:random", "v2:full")
    uncertainty_method: str | None = None
    rarity_method: str | None = None
    coherence_method: str | None = None
    normalisation: str | None = None
    pseudo_label_source: str | None = None
    cluster_count: int | None = None
    neighbour_count: int | None = None
    image_aggregation: str | None = None
    top_k: int | None = None
    coherence_exponent: float | None = None
    singleton_coherence: float | None = None
    minimum_cluster_size: int | None = None
    #: Retained so legacy configs keep loading; used only by v1 specs.
    weights: AcquisitionWeights = field(default_factory=AcquisitionWeights)

    def __post_init__(self) -> None:
        if not self.strategies:
            raise ConfigError("At least one acquisition strategy is required.")
        if len(set(self.strategies)) != len(self.strategies):
            raise ConfigError(f"Duplicate strategies: {self.strategies}")
        if self.image_aggregation is not None and self.image_aggregation not in IMAGE_AGGREGATIONS:
            raise ConfigError(
                f"Unknown image_aggregation {self.image_aggregation!r}. "
                f"Supported: {list(IMAGE_AGGREGATIONS)}"
            )
        for spec in self.resolved_specs():
            if spec.deprecated:
                # Loud but not fatal: reproducing published numbers is legitimate.
                print(
                    f"NOTE: strategy {spec.name!r} (v{spec.semantics_version}) is deprecated: {spec.deprecated}"
                )

    def overrides(self) -> dict[str, object]:
        candidates = {
            "uncertainty_method": self.uncertainty_method,
            "rarity_method": self.rarity_method,
            "coherence_method": self.coherence_method,
            "normalisation": self.normalisation,
            "pseudo_label_source": self.pseudo_label_source,
            "cluster_count": self.cluster_count,
            "neighbour_count": self.neighbour_count,
            "image_aggregation": self.image_aggregation,
            "top_k": self.top_k,
            "coherence_exponent": self.coherence_exponent,
            "singleton_coherence": self.singleton_coherence,
            "minimum_cluster_size": self.minimum_cluster_size,
        }
        return {key: value for key, value in candidates.items() if value is not None}

    def resolved_specs(self) -> tuple[StrategySpec, ...]:
        """The declarative specs this configuration actually runs."""

        overrides = self.overrides()
        specs = []
        for name in self.strategies:
            spec = STRATEGY_REGISTRY.resolve(name)
            if not overrides:
                specs.append(spec)
                continue
            applicable = dict(overrides)
            if spec.semantics_version == 1:
                # Overriding a v1 spec would destroy the reproducibility it
                # exists for. Only the coherence exponent is a declared v1 knob.
                applicable = {
                    key: value for key, value in applicable.items() if key == "coherence_exponent"
                }
            if spec.random_selection:
                applicable = {
                    key: value
                    for key, value in applicable.items()
                    if key in ("image_aggregation", "top_k")
                }
            try:
                specs.append(StrategySpec(**{**spec.as_dict(), **applicable}))
            except StrategyError as error:
                raise ConfigError(f"{name}: {error}") from error
        return tuple(specs)

    def spec_for(self, name: str) -> StrategySpec:
        for spec, requested in zip(self.resolved_specs(), self.strategies, strict=True):
            if requested == name or spec.name == name:
                return spec
        raise ConfigError(f"Strategy {name!r} is not part of this configuration.")


@dataclass(frozen=True)
class LongTailConfig:
    enabled: bool = True
    imbalance_ratio: float = 50.0

    def __post_init__(self) -> None:
        if self.imbalance_ratio < 1:
            raise ConfigError("imbalance_ratio must be >= 1.")


@dataclass(frozen=True)
class DatasetConfig:
    image_set_path: str
    annotations_dir: str
    unknown_classes: tuple[str, ...]
    long_tail: LongTailConfig
    known_classes: tuple[str, ...] = ()
    #: Overrides the class_stats.csv group mapping; normally left unset.
    class_groups_path: str | None = None
    known_class_groups_path: str | None = None

    def __post_init__(self) -> None:
        if not self.unknown_classes:
            raise ConfigError("unknown_classes must not be empty.")
        if len(set(self.unknown_classes)) != len(self.unknown_classes):
            raise ConfigError("unknown_classes must not contain duplicates.")
        overlap = set(self.unknown_classes) & set(self.known_classes)
        if overlap:
            raise ConfigError(f"Classes cannot be both known and unknown: {sorted(overlap)}")


@dataclass(frozen=True)
class ProbConfig:
    repository_path: str
    initial_checkpoint: str | None
    train_command: str
    predict_command: str
    evaluate_command: str
    timeout_seconds: int = 86400

    def __post_init__(self) -> None:
        if self.timeout_seconds < 1:
            raise ConfigError("timeout_seconds must be positive.")
        for name, template in (
            ("predict_command", self.predict_command),
            ("evaluate_command", self.evaluate_command),
        ):
            if "{" not in template:
                raise ConfigError(f"{name} has no substitution placeholders.")


@dataclass(frozen=True)
class EvaluationConfig:
    """How grouped long-tail metrics are computed from the detections artifact."""

    grouped_metrics: bool = True
    iou_threshold: float = 0.5
    unknown_prediction_name: str = "unknown"
    #: Fail the round if the bridge produced no detections artifact.
    require_detections: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ConfigError("iou_threshold must lie in (0, 1].")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    active_learning: ActiveLearningConfig
    acquisition: AcquisitionConfig
    dataset: DatasetConfig
    prob: ProbConfig
    output_dir: str
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    source_path: str | None = None
    #: Provenance for deprecated YAML keys, e.g.
    #: ``{"acquisition.uncertainty_mode": "ambiguity -> legacy_prob_score"}``.
    #: Kept so a manifest records what the file said, not only what it resolved to.
    legacy_aliases: Mapping[str, str] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable digest of everything that affects the numbers."""

        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "active_learning": {
                "rounds": self.active_learning.rounds,
                "initial_images": self.active_learning.initial_images,
                "budget_per_round": self.active_learning.budget_per_round,
                "seeds": list(self.active_learning.seeds),
                "strategy": self.active_learning.strategy,
                "budget": self.active_learning.budget,
            },
            "acquisition": {
                "strategies": list(self.acquisition.strategies),
                "overrides": self.acquisition.overrides(),
                "resolved": [spec.as_dict() for spec in self.acquisition.resolved_specs()],
            },
            "dataset": {
                "image_set_path": self.dataset.image_set_path,
                "annotations_dir": self.dataset.annotations_dir,
                "unknown_classes": list(self.dataset.unknown_classes),
                "known_classes": list(self.dataset.known_classes),
                "class_groups_path": self.dataset.class_groups_path,
                "known_class_groups_path": self.dataset.known_class_groups_path,
                "long_tail": {
                    "enabled": self.dataset.long_tail.enabled,
                    "imbalance_ratio": self.dataset.long_tail.imbalance_ratio,
                },
            },
            "prob": {
                "repository_path": self.prob.repository_path,
                "initial_checkpoint": self.prob.initial_checkpoint,
                "train_command": self.prob.train_command,
                "predict_command": self.prob.predict_command,
                "evaluate_command": self.prob.evaluate_command,
                "timeout_seconds": self.prob.timeout_seconds,
            },
            "evaluation": {
                "grouped_metrics": self.evaluation.grouped_metrics,
                "iou_threshold": self.evaluation.iou_threshold,
                "unknown_prediction_name": self.evaluation.unknown_prediction_name,
                "require_detections": self.evaluation.require_detections,
            },
            "output_dir": self.output_dir,
            "source_path": self.source_path,
            "legacy_aliases": dict(self.legacy_aliases),
        }


def _section(data: Mapping[str, Any], name: str, *, required: bool = True) -> dict[str, Any]:
    value = data.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid configuration section: {name}")
    return dict(value)


def _optional(data: Mapping[str, Any], key: str, cast: Any) -> Any:
    value = data.get(key)
    return None if value is None else cast(value)


def _string_tuple(values: Sequence[Any] | None) -> tuple[str, ...]:
    return tuple(str(value) for value in (values or ()))


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""

    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping.")

    active = _section(raw, "active_learning")
    acquisition = _section(raw, "acquisition")
    dataset = _section(raw, "dataset")
    long_tail = _section(dataset, "long_tail", required=False)
    prob = _section(raw, "prob")
    evaluation = _section(raw, "evaluation", required=False)
    weights_raw = acquisition.get("weights") or {}

    weights = AcquisitionWeights(
        uncertainty=float(weights_raw.get("uncertainty", 0.3)),
        novelty=float(weights_raw.get("novelty", 0.2)),
        rarity=float(weights_raw.get("rarity", 0.5)),
        coherence_power=float(weights_raw.get("coherence_power", 1.0)),
        rarity_power=float(weights_raw.get("rarity_power", 1.0)),
    )

    # 'uncertainty_mode' was the pre-audit name and meant the legacy transform.
    legacy_aliases: dict[str, str] = {}
    legacy_mode = acquisition.get("uncertainty_mode")
    uncertainty_method = acquisition.get("uncertainty_method")
    if uncertainty_method is None and legacy_mode is not None:
        uncertainty_method = {
            "ambiguity": "legacy_prob_score",
            "entropy": "entropy",
            "margin": "margin",
        }.get(str(legacy_mode))
        if uncertainty_method is None:
            raise ConfigError(
                f"Unknown legacy uncertainty_mode {legacy_mode!r}; use "
                "uncertainty_method with one of entropy, margin, one_minus_max, "
                "legacy_prob_score."
            )
        legacy_aliases["acquisition.uncertainty_mode"] = (
            f"{legacy_mode} -> uncertainty_method={uncertainty_method}"
        )
        print(
            f"NOTE: acquisition.uncertainty_mode={legacy_mode!r} is deprecated; "
            f"interpreted as uncertainty_method={uncertainty_method!r}."
        )

    coherence_exponent = acquisition.get("coherence_exponent")
    if coherence_exponent is None and "coherence_power" in weights_raw:
        coherence_exponent = float(weights_raw["coherence_power"])
        legacy_aliases["acquisition.weights.coherence_power"] = (
            f"{coherence_exponent} -> coherence_exponent"
        )

    try:
        config = ExperimentConfig(
            name=str(raw.get("name", "contribution-a")),
            active_learning=ActiveLearningConfig(
                rounds=int(active.get("rounds", 1)),
                initial_images=int(active.get("initial_images", 20)),
                budget_per_round=int(active.get("budget_per_round", 10)),
                seeds=tuple(int(seed) for seed in active.get("seeds", [0])),
                strategy=str(active.get("strategy", "v2:full")),
                budget=int(active.get("budget", 10)),
            ),
            acquisition=AcquisitionConfig(
                strategies=_string_tuple(acquisition.get("strategies")) or ("v2:random", "v2:full"),
                uncertainty_method=(
                    str(uncertainty_method) if uncertainty_method is not None else None
                ),
                rarity_method=_optional(acquisition, "rarity_method", str),
                coherence_method=_optional(acquisition, "coherence_method", str),
                normalisation=_optional(acquisition, "normalisation", str),
                pseudo_label_source=_optional(acquisition, "pseudo_label_source", str),
                cluster_count=_optional(acquisition, "cluster_count", int),
                neighbour_count=_optional(acquisition, "neighbour_count", int),
                image_aggregation=_optional(acquisition, "image_aggregation", str),
                top_k=_optional(acquisition, "top_k", int),
                coherence_exponent=(
                    float(coherence_exponent) if coherence_exponent is not None else None
                ),
                singleton_coherence=_optional(acquisition, "singleton_coherence", float),
                minimum_cluster_size=_optional(acquisition, "minimum_cluster_size", int),
                weights=weights,
            ),
            dataset=DatasetConfig(
                image_set_path=str(dataset["image_set_path"]),
                annotations_dir=str(dataset["annotations_dir"]),
                unknown_classes=_string_tuple(dataset["unknown_classes"]),
                known_classes=_string_tuple(dataset.get("known_classes")),
                class_groups_path=_optional(dataset, "class_groups_path", str),
                known_class_groups_path=_optional(dataset, "known_class_groups_path", str),
                long_tail=LongTailConfig(
                    enabled=bool(long_tail.get("enabled", True)),
                    imbalance_ratio=float(long_tail.get("imbalance_ratio", 50.0)),
                ),
            ),
            prob=ProbConfig(
                repository_path=str(prob["repository_path"]),
                initial_checkpoint=prob.get("initial_checkpoint"),
                train_command=str(prob["train_command"]),
                predict_command=str(prob["predict_command"]),
                evaluate_command=str(prob["evaluate_command"]),
                timeout_seconds=int(prob.get("timeout_seconds", 86400)),
            ),
            evaluation=EvaluationConfig(
                grouped_metrics=bool(evaluation.get("grouped_metrics", True)),
                iou_threshold=float(evaluation.get("iou_threshold", 0.5)),
                unknown_prediction_name=str(evaluation.get("unknown_prediction_name", "unknown")),
                require_detections=bool(evaluation.get("require_detections", True)),
            ),
            output_dir=str(raw.get("output_dir", "outputs/contribution-a")),
            source_path=str(source),
            legacy_aliases=legacy_aliases,
        )
    except KeyError as error:
        raise ConfigError(f"Missing required configuration key: {error}") from error
    except StrategyError as error:
        raise ConfigError(str(error)) from error
    return config
