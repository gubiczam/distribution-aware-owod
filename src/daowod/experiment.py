"""Multi-seed, multi-strategy contribution-A experiment runner."""

import csv
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from daowod.acquisition import (
    AcquisitionResult,
    aggregate_image_scores,
    score_proposals,
    select_images,
)
from daowod.config import AcquisitionConfig, ExperimentConfig
from daowod.dataset import DatasetState, build_long_tail_pool, file_sha256, read_image_ids
from daowod.metrics import grouped_unknown_recall, load_detection_json
from daowod.prob_adapter import ProbAdapter


@dataclass(frozen=True)
class ExperimentResult:
    """Generated metrics, selections and output location."""

    metrics: list[dict[str, object]]
    selections: list[dict[str, object]]
    output_dir: Path


def _unique_ids(name: str, values: Sequence[str]) -> list[str]:
    ids = [str(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} must contain unique image IDs.")
    return ids


def _write_ids(path: Path, image_ids: Sequence[str]) -> None:
    text = "\n".join(image_ids)
    if image_ids:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_score_tables(
    proposal_path: Path,
    image_path: Path,
    image_scores: dict[object, float],
    proposal_image_ids: Sequence[object],
    result: AcquisitionResult,
    *,
    rarity_bonus: np.ndarray,
) -> None:
    with proposal_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "uncertainty",
                "novelty",
                "rarity",
                "coherence",
                "rarity_bonus",
                "score",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for index, image_id in enumerate(proposal_image_ids):
            writer.writerow(
                {
                    "image_id": str(image_id),
                    "uncertainty": float(result.uncertainty[index]),
                    "novelty": float(result.novelty[index]),
                    "rarity": float(result.rarity[index]),
                    "coherence": float(result.coherence[index]),
                    "rarity_bonus": float(rarity_bonus[index]),
                    "score": float(result.scores[index]),
                }
            )

    with image_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_id", "score"],
            lineterminator="\n",
        )
        writer.writeheader()
        for image_id, score in sorted(
            image_scores.items(), key=lambda item: (-item[1], str(item[0]))
        ):
            writer.writerow({"image_id": str(image_id), "score": float(score)})


def run_active_round(
    *,
    adapter: ProbAdapter,
    checkpoint: str | Path,
    candidate_ids: Sequence[str],
    reference_ids: Sequence[str],
    labelled_ids: Sequence[str],
    output_dir: str | Path,
    strategy: str,
    budget: int,
    acquisition_config: AcquisitionConfig,
    seed: int,
    round_index: int = 0,
) -> dict[str, object]:
    """Run one detector-backed active-learning round."""

    if strategy not in {"random", "rarity_no_coherence", "full"}:
        raise ValueError("strategy must be 'random', 'rarity_no_coherence', or 'full'.")
    if budget < 1:
        raise ValueError("budget must be positive.")

    candidates = _unique_ids("candidate_ids", candidate_ids)
    references = _unique_ids("reference_ids", reference_ids)
    labelled = _unique_ids("labelled_ids", labelled_ids)
    if budget > len(candidates):
        raise ValueError("budget must not exceed candidate image count.")
    if strategy != "random" and not references:
        raise ValueError("reference_ids must be non-empty for proposal scoring.")
    overlap = set(candidates) & set(references)
    if overlap:
        raise ValueError(f"candidate_ids and reference_ids must not overlap: {sorted(overlap)}")
    labelled_overlap = set(candidates) & set(labelled)
    if labelled_overlap:
        raise ValueError(
            f"candidate_ids and labelled_ids must not overlap: {sorted(labelled_overlap)}"
        )

    round_dir = Path(output_dir)
    manifest_path = round_dir / "round_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("completed") is True:
            raise ValueError(f"Completed round would be overwritten: {round_dir}")
    round_dir.mkdir(parents=True, exist_ok=True)

    candidate_proposals_path = round_dir / "candidate_proposals.npz"
    reference_proposals_path = round_dir / "reference_proposals.npz"
    proposal_scores_path = round_dir / "proposal_scores.csv"
    image_scores_path = round_dir / "image_scores.csv"
    selected_path = round_dir / "selected_ids.txt"
    labelled_path = round_dir / "labelled_ids.txt"
    remaining_path = round_dir / "remaining_pool_ids.txt"
    checkpoint_path = round_dir / "checkpoint.pth"
    metrics_path = round_dir / "metrics.json"

    manifest: dict[str, object] = {
        "protocol_version": 1,
        "round_index": round_index,
        "strategy": strategy,
        "seed": seed,
        "budget": budget,
        "input_checkpoint": str(checkpoint),
        "output_checkpoint": checkpoint_path.name,
        "candidate_count_before": len(candidates),
        "labelled_count_before": len(labelled),
        "reference_count": len(references),
        "acquisition_parameters": {
            "uncertainty_mode": acquisition_config.uncertainty_mode,
            "pseudo_label_source": acquisition_config.pseudo_label_source,
            "cluster_count": acquisition_config.cluster_count,
            "neighbour_count": acquisition_config.neighbour_count,
            "weights": vars(acquisition_config.weights),
        },
        "image_aggregation": {
            "method": "mean_top_k_proposal_scores",
            "top_k": acquisition_config.top_k,
        },
        "completed": False,
    }
    _write_json(manifest_path, manifest)

    try:
        candidate_batch = adapter.predict(
            candidates,
            checkpoint=checkpoint,
            output_path=candidate_proposals_path,
        )
    except Exception as exc:
        raise RuntimeError("candidate proposal export failed") from exc

    selected: list[str]
    if strategy == "random":
        shuffled = list(candidates)
        random.Random(f"{seed}:{round_index}").shuffle(shuffled)
        selected = shuffled[:budget]
    else:
        try:
            reference_batch = adapter.predict(
                references,
                checkpoint=checkpoint,
                output_path=reference_proposals_path,
            )
        except Exception as exc:
            raise RuntimeError("reference proposal export failed") from exc

        try:
            scoring_strategy = "ungated_full" if strategy == "rarity_no_coherence" else "full"
            acquisition = score_proposals(
                strategy=scoring_strategy,
                uncertainty_mode=acquisition_config.uncertainty_mode,
                pseudo_label_source=acquisition_config.pseudo_label_source,
                confidence=candidate_batch.confidence,
                posterior=candidate_batch.posterior,
                embeddings=candidate_batch.embeddings,
                reference_embeddings=reference_batch.embeddings,
                predicted_labels=candidate_batch.predicted_labels,
                cluster_count=acquisition_config.cluster_count,
                neighbour_count=acquisition_config.neighbour_count,
                seed=seed + round_index,
                weights=acquisition_config.weights,
            )
            image_scores = aggregate_image_scores(
                candidate_batch.image_ids,
                acquisition.scores,
                top_k=acquisition_config.top_k,
            )
            rarity_bonus = (
                acquisition.rarity
                if strategy == "rarity_no_coherence"
                else acquisition.rarity
                * np.power(acquisition.coherence, acquisition_config.weights.coherence_power)
            )
        except Exception as exc:
            raise RuntimeError("proposal scoring failed") from exc

        _write_score_tables(
            proposal_scores_path,
            image_scores_path,
            image_scores,
            candidate_batch.image_ids,
            acquisition,
            rarity_bonus=rarity_bonus,
        )
        selected = [
            str(image_id)
            for image_id in select_images(
                candidate_batch.image_ids,
                acquisition.scores,
                budget=budget,
                top_k=acquisition_config.top_k,
            )
        ]

    selected_set = set(selected)
    cumulative_labelled = [*labelled, *selected]
    remaining = [image_id for image_id in candidates if image_id not in selected_set]
    _write_ids(selected_path, selected)
    _write_ids(labelled_path, cumulative_labelled)
    _write_ids(remaining_path, remaining)

    try:
        trained_checkpoint = adapter.train(
            cumulative_labelled,
            previous_checkpoint=checkpoint,
            run_dir=round_dir,
            round_index=round_index,
            seed=seed,
        )
    except Exception as exc:
        raise RuntimeError("training failed") from exc
    if Path(trained_checkpoint) != checkpoint_path:
        raise ValueError(f"Unexpected checkpoint path: {trained_checkpoint}")

    try:
        metrics = adapter.evaluate(checkpoint=checkpoint_path, output_path=metrics_path)
    except Exception as exc:
        raise RuntimeError("evaluation failed") from exc

    manifest.update(
        {
            "candidate_count_after": len(remaining),
            "labelled_count_after": len(cumulative_labelled),
            "selected_ids_sha256": file_sha256(selected_path),
            "remaining_pool_sha256": file_sha256(remaining_path),
            "labelled_ids_sha256": file_sha256(labelled_path),
            "candidate_proposals_sha256": file_sha256(candidate_proposals_path),
            "reference_proposals_sha256": (
                file_sha256(reference_proposals_path) if reference_proposals_path.exists() else None
            ),
            "metrics": metrics,
            "completed": True,
        }
    )
    _write_json(manifest_path, manifest)

    return {
        "selected_image_ids": selected,
        "remaining_candidate_ids": remaining,
        "labelled_ids": cumulative_labelled,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "round_manifest_path": manifest_path,
    }


class ActiveLearningExperiment:
    """Run all configured strategies with identical seeds and budgets."""

    def __init__(self, config: ExperimentConfig, detector: ProbAdapter) -> None:
        self.config = config
        self.detector = detector

    def run(self) -> ExperimentResult:
        """Run the complete contribution-A experiment campaign."""

        output_root = Path(self.config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        base_image_ids = (
            []
            if self.config.dataset.long_tail.enabled
            else read_image_ids(self.config.dataset.image_set_path)
        )
        metrics_rows: list[dict[str, object]] = []
        selection_rows: list[dict[str, object]] = []

        for seed in self.config.active_learning.seeds:
            pool = self._prepare_pool(
                base_image_ids,
                seed=seed,
                output_root=output_root,
            )
            for strategy in self.config.acquisition.strategies:
                metrics, selections = self._run_single(
                    pool["selected_image_ids"],
                    class_groups=pool["class_groups"],
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
    ) -> dict[str, object]:
        long_tail = self.config.dataset.long_tail
        if not long_tail.enabled:
            return {
                "selected_image_ids": list(image_ids),
                "class_groups": {name: "medium" for name in self.config.dataset.unknown_classes},
                "manifest_path": None,
                "class_stats_path": None,
                "pool_split_path": None,
            }

        return build_long_tail_pool(
            annotation_dir=self.config.dataset.annotations_dir,
            source_split=self.config.dataset.image_set_path,
            task_class_names=self.config.dataset.unknown_classes,
            output_dir=output_root / f"long_tail_seed_{seed}",
            imbalance_ratio=long_tail.imbalance_ratio,
            seed=seed,
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
