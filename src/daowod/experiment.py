"""Multi-seed, multi-strategy contribution-A experiment runner."""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from daowod.acquisition import (
    AcquisitionResult,
    score_proposals,
    select_images,
)
from daowod.config import ExperimentConfig
from daowod.dataset import DatasetState, build_long_tail_pool, read_image_ids
from daowod.metrics import grouped_unknown_recall, load_detection_json
from daowod.prob_adapter import ProbAdapter


@dataclass(frozen=True)
class ExperimentResult:
    """Generated metrics, selections and output location."""

    metrics: list[dict[str, object]]
    selections: list[dict[str, object]]
    output_dir: Path


class ActiveLearningExperiment:
    """Run all configured strategies with identical seeds and budgets."""

    def __init__(self, config: ExperimentConfig, detector: ProbAdapter) -> None:
        self.config = config
        self.detector = detector

    def run(self) -> ExperimentResult:
        """Run the complete contribution-A experiment campaign."""

        output_root = Path(self.config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        base_image_ids = read_image_ids(self.config.dataset.image_set_path)
        metrics_rows: list[dict[str, object]] = []
        selection_rows: list[dict[str, object]] = []

        for seed in self.config.active_learning.seeds:
            image_ids, class_groups = self._prepare_pool(
                base_image_ids,
                seed=seed,
                output_root=output_root,
            )
            for strategy in self.config.acquisition.strategies:
                metrics, selections = self._run_single(
                    image_ids,
                    class_groups=class_groups,
                    seed=seed,
                    strategy=strategy,
                    output_root=output_root,
                )
                metrics_rows.extend(metrics)
                selection_rows.extend(selections)

        self._write_csv(output_root / "metrics.csv", metrics_rows)
        (output_root / "selections.json").write_text(
            json.dumps(selection_rows, indent=2),
            encoding="utf-8",
        )
        return ExperimentResult(metrics_rows, selection_rows, output_root)

    def _prepare_pool(
        self,
        image_ids: Sequence[str],
        *,
        seed: int,
        output_root: Path,
    ) -> tuple[list[str], dict[str, str]]:
        long_tail = self.config.dataset.long_tail
        if not long_tail.enabled:
            return list(image_ids), {name: "medium" for name in self.config.dataset.unknown_classes}

        return build_long_tail_pool(
            image_ids,
            annotations_dir=self.config.dataset.annotations_dir,
            unknown_classes=self.config.dataset.unknown_classes,
            tail_max=long_tail.tail_max,
            head_min=long_tail.head_min,
            head_retention=long_tail.head_retention,
            medium_retention=long_tail.medium_retention,
            tail_retention=long_tail.tail_retention,
            seed=seed,
            manifest_path=output_root / f"long_tail_seed{seed}.json",
        )

    def _run_single(
        self,
        image_ids: Sequence[str],
        *,
        class_groups: dict[str, str],
        seed: int,
        strategy: str,
        output_root: Path,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        run_dir = output_root / f"seed_{seed}" / strategy
        run_dir.mkdir(parents=True, exist_ok=True)
        state = DatasetState.initialise(
            image_ids,
            initial_images=self.config.active_learning.initial_images,
            seed=seed,
        )
        checkpoint = self.config.prob.initial_checkpoint
        metrics_rows: list[dict[str, object]] = []
        selection_rows: list[dict[str, object]] = []
        rng = np.random.default_rng(seed)

        for round_index in range(self.config.active_learning.rounds):
            round_dir = run_dir / f"round_{round_index + 1}"
            checkpoint = self.detector.train(
                state.labelled_ids,
                previous_checkpoint=checkpoint,
                run_dir=round_dir,
                round_index=round_index,
                seed=seed,
            )
            metrics = self.detector.evaluate(
                checkpoint=checkpoint,
                output_path=round_dir / "metrics.json",
            )

            detections_path = metrics.get("detections_path")
            if detections_path:
                ground_truth, detections = load_detection_json(str(detections_path))
                metrics.update(
                    grouped_unknown_recall(
                        ground_truth,
                        detections,
                        unknown_classes=self.config.dataset.unknown_classes,
                        class_groups=class_groups,
                    )
                )

            metrics_rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "round": round_index + 1,
                    "labelled_images": len(state.labelled_ids),
                    **metrics,
                }
            )

            is_last = round_index == self.config.active_learning.rounds - 1
            if is_last or not state.pool_ids:
                break

            budget = min(
                self.config.active_learning.budget_per_round,
                len(state.pool_ids),
            )
            if strategy == "random":
                selected = [
                    str(value)
                    for value in rng.choice(
                        state.pool_ids,
                        size=budget,
                        replace=False,
                    ).tolist()
                ]
            else:
                candidates = self.detector.predict(
                    state.pool_ids,
                    checkpoint=checkpoint,
                    output_path=round_dir / "candidate_proposals.npz",
                )
                references = self.detector.predict(
                    state.labelled_ids,
                    checkpoint=checkpoint,
                    output_path=round_dir / "reference_proposals.npz",
                )
                acquisition = score_proposals(
                    strategy=strategy,
                    uncertainty_mode=self.config.acquisition.uncertainty_mode,
                    pseudo_label_source=self.config.acquisition.pseudo_label_source,
                    confidence=candidates.confidence,
                    posterior=candidates.posterior,
                    embeddings=candidates.embeddings,
                    reference_embeddings=references.embeddings,
                    predicted_labels=candidates.predicted_labels,
                    cluster_count=self.config.acquisition.cluster_count,
                    neighbour_count=self.config.acquisition.neighbour_count,
                    seed=seed + round_index,
                    weights=self.config.acquisition.weights,
                )
                selected = [
                    str(value)
                    for value in select_images(
                        candidates.image_ids,
                        acquisition.scores,
                        budget=budget,
                        top_k=self.config.acquisition.top_k,
                    )
                ]
                self._write_diagnostics(
                    round_dir / "proposal_diagnostics.csv",
                    candidates.image_ids,
                    acquisition,
                )

            state.reveal(selected)
            selection_rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "after_round": round_index + 1,
                    "selected_image_ids": selected,
                }
            )

        return metrics_rows, selection_rows

    @staticmethod
    def _write_diagnostics(
        path: Path,
        image_ids: np.ndarray,
        result: AcquisitionResult,
    ) -> None:
        fields = [
            "image_id",
            "uncertainty",
            "novelty",
            "pseudo_label",
            "rarity",
            "coherence",
            "score",
        ]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for index, image_id in enumerate(image_ids):
                writer.writerow(
                    {
                        "image_id": image_id,
                        "uncertainty": result.uncertainty[index],
                        "novelty": result.novelty[index],
                        "pseudo_label": result.pseudo_labels[index],
                        "rarity": result.rarity[index],
                        "coherence": result.coherence[index],
                        "score": result.scores[index],
                    }
                )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
