"""The single configuration path: YAML -> validated config -> scorer -> report.

Before the audit the live campaign was configured by Python constants in a
notebook cell and ``configs/experiment.yaml`` was inert. Everything now resolves
through :func:`load_config`, and strategy names resolve through the one registry
in :mod:`daowod.scoring`, so the validator, scorer, CLI and reports cannot
disagree about what a strategy means.
"""

import hashlib
import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from daowod.acquisition import AcquisitionWeights
from daowod.dataset import file_sha256, read_image_ids, read_voc_classes
from daowod.groups import ClassGroups
from daowod.scoring import (
    IMAGE_AGGREGATIONS,
    STRATEGY_REGISTRY,
    StrategyError,
    StrategySpec,
)


class ConfigError(ValueError):
    """Raised for a malformed or self-inconsistent experiment configuration."""


def _flag_values(command: str) -> dict[str, str | bool]:
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise ConfigError(f"Command cannot be parsed with shlex: {command!r}") from error
    values: dict[str, str | bool] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
            index += 1
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            values[token] = tokens[index + 1]
            index += 2
        else:
            values[token] = True
            index += 1
    return values


@dataclass(frozen=True)
class ProtocolConfig:
    """The one task/protocol object that every stage must agree with."""

    dataset_protocol: str
    data_root: str
    previous_introduced_classes: int
    current_introduced_classes: int
    num_classes: int
    objectness_temperature: float
    candidate_pool_split: str
    reference_split: str
    evaluation_split: str
    evaluation_split_sha256: str
    initial_labelled_split: str | None
    checkpoint: str
    checkpoint_sha256: str
    image_aggregation: str
    top_k: int
    uncertainty_method: str
    clustering_method: str
    acquisition_budget: int
    active_learning_rounds: int
    training_schedule: str
    evaluation_settings: str
    pool_policy: str = "stage1_exact"
    reference_policy: str = "fixed_stage1_representation_bank"
    long_tail_transformation: str = "none"
    train_split: str = "runtime_selected_ids"
    class_group_mapping: str = "stage1_candidate_frequency_thirds"
    clustering_seed_policy: str = "derive_seed('pool', model_seed, round_index)"
    cuda_determinism: str = "recorded_not_forced"
    allow_candidate_evaluation_overlap: bool = False
    allow_labelled_evaluation_overlap: bool = False

    def __post_init__(self) -> None:
        if self.dataset_protocol not in {"OWDETR", "TOWOD", "VOC2007"}:
            raise ConfigError(f"Unknown dataset_protocol: {self.dataset_protocol!r}")
        if self.previous_introduced_classes < 0 or self.current_introduced_classes < 1:
            raise ConfigError("Introduced class counts must be non-negative/positive.")
        if self.num_classes < self.current_introduced_classes:
            raise ConfigError("num_classes must cover the introduced classes.")
        if self.objectness_temperature <= 0:
            raise ConfigError("objectness_temperature must be positive.")
        if self.top_k < 1:
            raise ConfigError("protocol.top_k must be positive.")
        if self.acquisition_budget < 1 or self.active_learning_rounds < 1:
            raise ConfigError("protocol acquisition budget and rounds must be positive.")
        required_paths = {
            "data_root": self.data_root,
            "candidate_pool_split": self.candidate_pool_split,
            "reference_split": self.reference_split,
            "checkpoint": self.checkpoint,
        }
        for name, value in required_paths.items():
            if not value:
                raise ConfigError(f"protocol.{name} is required.")
        if not self.evaluation_split:
            raise ConfigError("protocol.evaluation_split is required.")
        if not self.evaluation_split_sha256:
            raise ConfigError("protocol.evaluation_split_sha256 is required.")
        for name, value in (
            ("image_aggregation", self.image_aggregation),
            ("uncertainty_method", self.uncertainty_method),
            ("clustering_method", self.clustering_method),
            ("training_schedule", self.training_schedule),
            ("evaluation_settings", self.evaluation_settings),
        ):
            if not value:
                raise ConfigError(f"protocol.{name} is required.")

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_protocol": self.dataset_protocol,
            "data_root": self.data_root,
            "previous_introduced_classes": self.previous_introduced_classes,
            "current_introduced_classes": self.current_introduced_classes,
            "num_classes": self.num_classes,
            "objectness_temperature": self.objectness_temperature,
            "train_split": self.train_split,
            "candidate_pool_split": self.candidate_pool_split,
            "reference_split": self.reference_split,
            "evaluation_split": self.evaluation_split,
            "evaluation_split_sha256": self.evaluation_split_sha256,
            "initial_labelled_split": self.initial_labelled_split,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "image_aggregation": self.image_aggregation,
            "top_k": self.top_k,
            "uncertainty_method": self.uncertainty_method,
            "clustering_method": self.clustering_method,
            "acquisition_budget": self.acquisition_budget,
            "active_learning_rounds": self.active_learning_rounds,
            "training_schedule": self.training_schedule,
            "evaluation_settings": self.evaluation_settings,
            "pool_policy": self.pool_policy,
            "reference_policy": self.reference_policy,
            "long_tail_transformation": self.long_tail_transformation,
            "class_group_mapping": self.class_group_mapping,
            "clustering_seed_policy": self.clustering_seed_policy,
            "cuda_determinism": self.cuda_determinism,
            "allow_candidate_evaluation_overlap": self.allow_candidate_evaluation_overlap,
            "allow_labelled_evaluation_overlap": self.allow_labelled_evaluation_overlap,
        }


def validate_command_parity(protocol: ProtocolConfig, prob: "ProbConfig") -> dict[str, object]:
    """Fail if train/predict/evaluate would run different OWOD task settings."""

    commands = {
        "train": prob.train_command,
        "predict": prob.predict_command,
        "evaluate": prob.evaluate_command,
    }
    parsed = {stage: _flag_values(command) for stage, command in commands.items()}
    expected_common = {
        "--data-root": protocol.data_root,
        "--dataset": protocol.dataset_protocol,
        "--prev-introduced-classes": str(protocol.previous_introduced_classes),
        "--current-introduced-classes": str(protocol.current_introduced_classes),
        "--num-classes": str(protocol.num_classes),
        "--objectness-temperature": f"{protocol.objectness_temperature:g}",
    }
    required_placeholders = {
        "train": {
            "--labelled-ids": "{labelled_ids}",
            "--previous-checkpoint": "{previous_checkpoint}",
            "--output-checkpoint": "{checkpoint}",
            "--output-dir": "{output_dir}",
        },
        "predict": {
            "--image-ids": "{image_ids}",
            "--checkpoint": "{checkpoint}",
            "--output": "{proposals}",
        },
        "evaluate": {
            "--checkpoint": "{checkpoint}",
            "--output": "{metrics}",
            "--output-dir": "{output_dir}",
            "--test-set": protocol.evaluation_split,
        },
    }
    errors: list[str] = []
    for stage, flags in parsed.items():
        for key, expected in expected_common.items():
            observed = flags.get(key)
            if observed is None:
                errors.append(f"{stage}: missing {key}={expected!r}")
            elif str(observed) != expected:
                errors.append(f"{stage}: {key}={observed!r} != protocol {expected!r}")
        for key, expected in required_placeholders[stage].items():
            observed = flags.get(key)
            if observed is None:
                errors.append(f"{stage}: missing required argument {key}")
            elif str(observed) != expected:
                errors.append(f"{stage}: {key}={observed!r} != expected {expected!r}")
    if errors:
        raise ConfigError("Command/protocol parity failed:\n- " + "\n- ".join(errors))
    return {
        "status": "ok",
        "commands": commands,
        "parsed_flags": parsed,
        "protocol": protocol.as_dict(),
    }


def validate_resolved_command_parity(
    protocol: ProtocolConfig, commands: Mapping[str, str]
) -> dict[str, object]:
    """Validate the concrete train/predict/evaluate commands for a round."""

    expected_common = {
        "--data-root": protocol.data_root,
        "--dataset": protocol.dataset_protocol,
        "--prev-introduced-classes": str(protocol.previous_introduced_classes),
        "--current-introduced-classes": str(protocol.current_introduced_classes),
        "--num-classes": str(protocol.num_classes),
        "--objectness-temperature": f"{protocol.objectness_temperature:g}",
    }
    required_args = {
        "train": ("--labelled-ids", "--previous-checkpoint", "--output-checkpoint", "--output-dir"),
        "candidate_predict": ("--image-ids", "--checkpoint", "--output"),
        "reference_predict": ("--image-ids", "--checkpoint", "--output"),
        "evaluate": ("--checkpoint", "--output", "--output-dir", "--test-set"),
    }
    parsed = {stage: _flag_values(command) for stage, command in commands.items()}
    errors: list[str] = []
    for stage, command in commands.items():
        if "{" in command or "}" in command:
            errors.append(f"{stage}: unresolved placeholder remains in command")
    for stage, keys in required_args.items():
        flags = parsed.get(stage)
        if flags is None:
            errors.append(f"missing resolved {stage} command")
            continue
        for key, expected in expected_common.items():
            observed = flags.get(key)
            if observed is None:
                errors.append(f"{stage}: missing {key}={expected!r}")
            elif str(observed) != expected:
                errors.append(f"{stage}: {key}={observed!r} != protocol {expected!r}")
        for key in keys:
            observed = flags.get(key)
            if observed in (None, ""):
                errors.append(f"{stage}: missing resolved argument {key}")
        if stage == "evaluate" and str(flags.get("--test-set")) != protocol.evaluation_split:
            errors.append(
                f"{stage}: --test-set={flags.get('--test-set')!r} "
                f"!= protocol {protocol.evaluation_split!r}"
            )
    if errors:
        raise ConfigError("Resolved command/protocol parity failed:\n- " + "\n- ".join(errors))
    return {"status": "ok", "parsed_flags": parsed}


CLASS_ALIASES = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
    "couch": "sofa",
    "dining table": "diningtable",
    "potted plant": "pottedplant",
    "tv": "tvmonitor",
}


def _image_exists(directory: str | Path, image_id: str) -> bool:
    root = Path(directory)
    return any((root / f"{image_id}{suffix}").exists() for suffix in (".jpg", ".jpeg", ".png"))


def validate_evaluation_assets(
    protocol: ProtocolConfig,
    dataset: "DatasetConfig",
    evaluation: "EvaluationConfig",
) -> dict[str, Any]:
    """Validate the frozen evaluation split before any campaign can run."""

    split_path = (
        Path(protocol.data_root)
        / "ImageSets"
        / protocol.dataset_protocol
        / f"{protocol.evaluation_split}.txt"
    )
    if not split_path.exists():
        raise ConfigError(f"Missing evaluation split: {split_path}")
    observed_digest = file_sha256(split_path)
    if observed_digest != protocol.evaluation_split_sha256:
        raise ConfigError(
            "Evaluation split digest mismatch: "
            f"{observed_digest} != protocol {protocol.evaluation_split_sha256}"
        )

    evaluation_ids = read_image_ids(split_path)
    candidate_ids = read_image_ids(protocol.candidate_pool_split)
    reference_ids = read_image_ids(protocol.reference_split)
    labelled_ids = (
        read_image_ids(protocol.initial_labelled_split) if protocol.initial_labelled_split else []
    )
    annotations_dir = Path(dataset.annotations_dir)
    images_dir = Path(protocol.data_root) / "JPEGImages"
    missing_annotations = [
        image_id
        for image_id in evaluation_ids
        if not (annotations_dir / f"{image_id}.xml").exists()
    ]
    missing_images = [
        image_id for image_id in evaluation_ids if not _image_exists(images_dir, image_id)
    ]
    if missing_annotations:
        raise ConfigError(
            f"Evaluation split has {len(missing_annotations)} missing annotation(s): "
            f"{missing_annotations[:20]}"
        )
    if missing_images:
        raise ConfigError(
            f"Evaluation split has {len(missing_images)} missing image(s): {missing_images[:20]}"
        )

    candidate_overlap = sorted(set(candidate_ids) & set(evaluation_ids))
    labelled_overlap = sorted(set(labelled_ids) & set(evaluation_ids))
    reference_overlap = sorted(set(reference_ids) & set(evaluation_ids))
    if candidate_overlap and not protocol.allow_candidate_evaluation_overlap:
        raise ConfigError(
            "Evaluation split overlaps the acquisition candidate pool: "
            f"{len(candidate_overlap)} image(s), first IDs {candidate_overlap[:20]}"
        )
    if labelled_overlap and not protocol.allow_labelled_evaluation_overlap:
        raise ConfigError(
            "Evaluation split overlaps the initial labelled training split: "
            f"{len(labelled_overlap)} image(s), first IDs {labelled_overlap[:20]}"
        )

    known_classes = set(dataset.known_classes)
    unknown_classes = set(dataset.unknown_classes)
    groups = (
        ClassGroups.from_class_stats_csv(dataset.class_groups_path)
        if dataset.class_groups_path
        else None
    )
    support: dict[str, Any] = {
        "image_count": len(evaluation_ids),
        "known_objects": 0,
        "unknown_objects": 0,
        "head_objects": 0,
        "medium_objects": 0,
        "tail_objects": 0,
    }
    for image_id in evaluation_ids:
        for raw_name in read_voc_classes(image_id, annotations_dir):
            class_name = CLASS_ALIASES.get(raw_name, raw_name)
            if class_name in known_classes:
                support["known_objects"] += 1
            if class_name in unknown_classes:
                support["unknown_objects"] += 1
                if groups is not None:
                    group = groups.groups.get(class_name)
                    if group in ("head", "medium", "tail"):
                        support[f"{group}_objects"] += 1

    if support["known_objects"] == 0 or support["unknown_objects"] == 0:
        raise ConfigError(f"Evaluation split has invalid known/unknown support: {support}")
    if evaluation.grouped_metrics and groups is not None:
        missing_groups = [
            group for group in ("head", "medium", "tail") if support[f"{group}_objects"] == 0
        ]
        if missing_groups:
            raise ConfigError(
                f"Evaluation split has zero support for grouped metric(s): {missing_groups}"
            )
    return {
        "status": "ok",
        "split_path": str(split_path),
        "split_sha256": observed_digest,
        "image_count": len(evaluation_ids),
        "candidate_evaluation_overlap_count": len(candidate_overlap),
        "candidate_evaluation_overlap_first20": candidate_overlap[:20],
        "reference_evaluation_overlap_count": len(reference_overlap),
        "reference_evaluation_overlap_first20": reference_overlap[:20],
        "labelled_evaluation_overlap_count": len(labelled_overlap),
        "labelled_evaluation_overlap_first20": labelled_overlap[:20],
        "support": support,
    }


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
        if self.initial_images < 0 or min(self.budget, self.budget_per_round) < 1:
            raise ConfigError("Active-learning values must be non-negative/positive.")
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
            ("train_command", self.train_command),
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
    protocol: ProtocolConfig | None = None
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
            "protocol": self.protocol.as_dict() if self.protocol else None,
            "command_parity": (
                validate_command_parity(self.protocol, self.prob) if self.protocol else None
            ),
            "evaluation_preflight": (
                validate_evaluation_assets(self.protocol, self.dataset, self.evaluation)
                if self.protocol
                else None
            ),
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
    protocol_raw = _section(raw, "protocol", required=False)
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
        protocol = (
            ProtocolConfig(
                dataset_protocol=str(protocol_raw["dataset_protocol"]),
                data_root=str(protocol_raw["data_root"]),
                previous_introduced_classes=int(protocol_raw["previous_introduced_classes"]),
                current_introduced_classes=int(protocol_raw["current_introduced_classes"]),
                num_classes=int(protocol_raw["num_classes"]),
                objectness_temperature=float(protocol_raw["objectness_temperature"]),
                train_split=str(protocol_raw.get("train_split", "runtime_selected_ids")),
                candidate_pool_split=str(protocol_raw["candidate_pool_split"]),
                reference_split=str(protocol_raw["reference_split"]),
                evaluation_split=str(protocol_raw["evaluation_split"]),
                evaluation_split_sha256=str(protocol_raw["evaluation_split_sha256"]),
                initial_labelled_split=(
                    str(protocol_raw["initial_labelled_split"])
                    if protocol_raw.get("initial_labelled_split") is not None
                    else None
                ),
                checkpoint=str(protocol_raw["checkpoint"]),
                checkpoint_sha256=str(protocol_raw["checkpoint_sha256"]),
                image_aggregation=str(protocol_raw["image_aggregation"]),
                top_k=int(protocol_raw["top_k"]),
                uncertainty_method=str(protocol_raw["uncertainty_method"]),
                clustering_method=str(protocol_raw["clustering_method"]),
                acquisition_budget=int(protocol_raw["acquisition_budget"]),
                active_learning_rounds=int(protocol_raw["active_learning_rounds"]),
                training_schedule=str(protocol_raw["training_schedule"]),
                evaluation_settings=str(protocol_raw["evaluation_settings"]),
                pool_policy=str(protocol_raw.get("pool_policy", "stage1_exact")),
                reference_policy=str(
                    protocol_raw.get("reference_policy", "fixed_stage1_representation_bank")
                ),
                long_tail_transformation=str(protocol_raw.get("long_tail_transformation", "none")),
                class_group_mapping=str(
                    protocol_raw.get("class_group_mapping", "stage1_candidate_frequency_thirds")
                ),
                clustering_seed_policy=str(
                    protocol_raw.get(
                        "clustering_seed_policy", "derive_seed('pool', model_seed, round_index)"
                    )
                ),
                cuda_determinism=str(protocol_raw.get("cuda_determinism", "recorded_not_forced")),
                allow_candidate_evaluation_overlap=bool(
                    protocol_raw.get("allow_candidate_evaluation_overlap", False)
                ),
                allow_labelled_evaluation_overlap=bool(
                    protocol_raw.get("allow_labelled_evaluation_overlap", False)
                ),
            )
            if protocol_raw
            else None
        )
        prob_config = ProbConfig(
            repository_path=str(prob["repository_path"]),
            initial_checkpoint=prob.get("initial_checkpoint"),
            train_command=str(prob["train_command"]),
            predict_command=str(prob["predict_command"]),
            evaluate_command=str(prob["evaluate_command"]),
            timeout_seconds=int(prob.get("timeout_seconds", 86400)),
        )
        if protocol:
            validate_command_parity(protocol, prob_config)
            if (
                prob_config.initial_checkpoint
                and str(prob_config.initial_checkpoint) != protocol.checkpoint
            ):
                raise ConfigError("prob.initial_checkpoint must equal protocol.checkpoint.")
            if dataset["image_set_path"] != protocol.candidate_pool_split:
                raise ConfigError(
                    "dataset.image_set_path must equal protocol.candidate_pool_split."
                )
            if long_tail.get("enabled", True) and protocol.long_tail_transformation == "none":
                raise ConfigError(
                    "dataset.long_tail.enabled must be false when protocol says none."
                )
            if int(active.get("rounds", 1)) != protocol.active_learning_rounds:
                raise ConfigError(
                    "active_learning.rounds must equal protocol.active_learning_rounds."
                )
            if int(active.get("budget_per_round", 10)) != protocol.acquisition_budget:
                raise ConfigError(
                    "active_learning.budget_per_round must equal protocol.acquisition_budget."
                )
            if (
                str(acquisition.get("image_aggregation", "top_k_mean"))
                != protocol.image_aggregation
            ):
                raise ConfigError(
                    "acquisition.image_aggregation must equal protocol.image_aggregation."
                )
            if int(acquisition.get("top_k", 3)) != protocol.top_k:
                raise ConfigError("acquisition.top_k must equal protocol.top_k.")
        evaluation_config = EvaluationConfig(
            grouped_metrics=bool(evaluation.get("grouped_metrics", True)),
            iou_threshold=float(evaluation.get("iou_threshold", 0.5)),
            unknown_prediction_name=str(evaluation.get("unknown_prediction_name", "unknown")),
            require_detections=bool(evaluation.get("require_detections", True)),
        )
        dataset_config = DatasetConfig(
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
        )
        if protocol:
            validate_evaluation_assets(protocol, dataset_config, evaluation_config)
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
            dataset=dataset_config,
            prob=prob_config,
            protocol=protocol,
            evaluation=evaluation_config,
            output_dir=str(raw.get("output_dir", "outputs/contribution-a")),
            source_path=str(source),
            legacy_aliases=legacy_aliases,
        )
    except KeyError as error:
        raise ConfigError(f"Missing required configuration key: {error}") from error
    except StrategyError as error:
        raise ConfigError(str(error)) from error
    return config
