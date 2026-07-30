"""YAML configuration for contribution-A experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from daowod.acquisition import AcquisitionWeights


@dataclass(frozen=True)
class ActiveLearningConfig:
    rounds: int = 1
    strategy: str = "full"
    budget: int = 10
    initial_images: int = 20
    budget_per_round: int = 10
    seeds: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be positive.")
        allowed = {
            "random",
            "uncertainty",
            "uncertainty_novelty",
            "rarity",
            "rarity_no_coherence",
            "rarity_coherence",
            "ungated_full",
            "full",
        }
        if self.strategy not in allowed:
            raise ValueError("Unsupported acquisition strategy.")
        if min(self.budget, self.initial_images, self.budget_per_round) < 1:
            raise ValueError("Active-learning values must be positive.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")


@dataclass(frozen=True)
class AcquisitionConfig:
    strategies: tuple[str, ...] = ("random", "uncertainty", "full")
    uncertainty_mode: str = "ambiguity"
    pseudo_label_source: str = "cluster"
    cluster_count: int = 20
    neighbour_count: int = 5
    top_k: int = 3
    weights: AcquisitionWeights = field(default_factory=AcquisitionWeights)

    def __post_init__(self) -> None:
        allowed = {
            "random",
            "uncertainty",
            "uncertainty_novelty",
            "rarity",
            "rarity_no_coherence",
            "rarity_coherence",
            "ungated_full",
            "full",
        }
        invalid = set(self.strategies) - allowed
        if invalid:
            raise ValueError(f"Unknown strategies: {sorted(invalid)}")
        if self.uncertainty_mode not in {"ambiguity", "entropy", "margin"}:
            raise ValueError("Unknown uncertainty mode.")
        if self.pseudo_label_source not in {"cluster", "predicted"}:
            raise ValueError("Unknown pseudo-label source.")
        if min(self.cluster_count, self.neighbour_count, self.top_k) < 1:
            raise ValueError("Acquisition integer values must be positive.")


@dataclass(frozen=True)
class LongTailConfig:
    enabled: bool = True
    imbalance_ratio: float = 50.0

    def __post_init__(self) -> None:
        if self.imbalance_ratio < 1:
            raise ValueError("imbalance_ratio must be >= 1.")


@dataclass(frozen=True)
class DatasetConfig:
    image_set_path: str
    annotations_dir: str
    unknown_classes: tuple[str, ...]
    long_tail: LongTailConfig

    def __post_init__(self) -> None:
        if not self.unknown_classes:
            raise ValueError("unknown_classes must not be empty.")


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
            raise ValueError("timeout_seconds must be positive.")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    active_learning: ActiveLearningConfig
    acquisition: AcquisitionConfig
    dataset: DatasetConfig
    prob: ProbConfig
    output_dir: str


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid configuration section: {name}")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")

    active = _section(raw, "active_learning")
    acquisition = _section(raw, "acquisition")
    dataset = _section(raw, "dataset")
    long_tail = _section(dataset, "long_tail")
    prob = _section(raw, "prob")
    weights_raw = acquisition.get("weights", {})

    weights = AcquisitionWeights(
        uncertainty=float(weights_raw.get("uncertainty", 0.3)),
        novelty=float(weights_raw.get("novelty", 0.2)),
        rarity=float(weights_raw.get("rarity", 0.5)),
        coherence_power=float(weights_raw.get("coherence_power", 1.0)),
        rarity_power=float(weights_raw.get("rarity_power", 1.0)),
    )

    return ExperimentConfig(
        name=str(raw.get("name", "contribution-a")),
        active_learning=ActiveLearningConfig(
            rounds=int(active.get("rounds", 1)),
            strategy=str(active.get("strategy", "full")),
            budget=int(active.get("budget", 10)),
            initial_images=int(active.get("initial_images", 20)),
            budget_per_round=int(active.get("budget_per_round", 10)),
            seeds=tuple(int(seed) for seed in active.get("seeds", [0])),
        ),
        acquisition=AcquisitionConfig(
            strategies=tuple(acquisition.get("strategies", ["random", "full"])),
            uncertainty_mode=str(acquisition.get("uncertainty_mode", "ambiguity")),
            pseudo_label_source=str(acquisition.get("pseudo_label_source", "cluster")),
            cluster_count=int(acquisition.get("cluster_count", 20)),
            neighbour_count=int(acquisition.get("neighbour_count", 5)),
            top_k=int(acquisition.get("top_k", 3)),
            weights=weights,
        ),
        dataset=DatasetConfig(
            image_set_path=str(dataset["image_set_path"]),
            annotations_dir=str(dataset["annotations_dir"]),
            unknown_classes=tuple(str(name) for name in dataset["unknown_classes"]),
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
        output_dir=str(raw.get("output_dir", "outputs/contribution-a")),
    )
