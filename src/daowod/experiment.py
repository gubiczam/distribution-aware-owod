"""The one active-learning loop.

Before the audit there were two: ``run_active_round`` (used by the live campaign,
restricted to three strategies) and ``ActiveLearningExperiment._run_single``
(unused, untested, and the only caller of the grouped long-tail metrics). They
had already diverged in seeding, artifacts and validation. There is now a single
round implementation, driven by a :class:`~daowod.scoring.StrategySpec`, and a
single campaign loop built on top of it.

What each round writes, beyond the checkpoint and metrics:

``round_manifest.json``   protocol, strategy spec, digests, counts, metrics
``proposal_scores.csv``   per-proposal record (:mod:`daowod.diagnostics`)
``image_scores.csv``      per-image aggregate scores
``component_diagnostics.json``  rarity / coherence / gate diagnostics
``grouped_metrics.json``  head / medium / tail recall and AP, when available
"""

import csv
import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daowod import diagnostics as diag
from daowod.config import (
    EvaluationConfig,
    ExperimentConfig,
    ProtocolConfig,
    validate_resolved_command_parity,
)
from daowod.dataset import (
    DatasetState,
    build_long_tail_pool,
    file_sha256,
    read_image_ids,
    read_voc_classes,
)
from daowod.groups import ClassGroups
from daowod.metrics import (
    grouped_detection_metrics,
    load_detection_json,
    require_consistent_category_space,
    validate_grouped_metrics,
)
from daowod.prob_adapter import ProbAdapter
from daowod.scoring import ScoringResult, StrategySpec, score_pool, select_images

PROTOCOL_VERSION = 2
CLASS_ALIASES = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
    "couch": "sofa",
    "dining table": "diningtable",
    "potted plant": "pottedplant",
    "tv": "tvmonitor",
}

RESUME_COMPLETED_ENV = "DAOWOD_RESUME_COMPLETED"


def derive_seed(*parts: object) -> int:
    """A reproducible 32-bit seed from any tuple of identifying parts.

    ``seed + round_index`` — the pre-audit derivation — collided across pairs:
    (seed 0, round 1) and (seed 1, round 0) produced the same clustering.
    """

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def derive_pool_seed(seed: int, round_index: int) -> int:
    """The clustering seed, which must NOT depend on the strategy.

    Pseudo-labelling partitions the *pool*; it is not part of a strategy's
    definition. Including the strategy name here was measured to inject a
    confound: on a 3,960-proposal pool, two strategies scoring identical
    proposals received KMeans partitions agreeing on only 88 % of pairwise
    co-memberships, and 0.280 of an apparent 0.462 selection difference between
    ``v2:full`` and ``v2:full_no_coherence`` came from the differing partition
    rather than from the coherence gate. Every strategy scoring the same pool now
    shares one clustering, so a measured difference is attributable to the score.
    """

    return derive_seed("pool", seed, round_index)


@dataclass(frozen=True)
class RoundResult:
    """Everything one active-learning round produced."""

    selected_image_ids: list[str]
    remaining_candidate_ids: list[str]
    labelled_ids: list[str]
    checkpoint_path: Path
    metrics_path: Path
    round_manifest_path: Path
    metrics: dict[str, Any]
    grouped_metrics: dict[str, Any] = field(default_factory=dict)
    scoring: ScoringResult | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)


def _load_completed_round(round_dir: Path) -> RoundResult:
    """Load and validate one completed round for opt-in campaign resume."""

    manifest_path = round_dir / "round_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("completed") is not True:
        raise ValueError(f"Round manifest is not completed: {manifest_path}")

    required = {
        "selected_ids": round_dir / "selected_ids.txt",
        "labelled_ids": round_dir / "labelled_ids.txt",
        "remaining_pool_ids": round_dir / "remaining_pool_ids.txt",
        "metrics": round_dir / "metrics.json",
        "checkpoint": round_dir / "checkpoint.pth",
    }
    missing = [
        name for name, path in required.items() if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise ValueError(f"Completed round is missing required artifacts {missing}: {round_dir}")

    metrics = json.loads(required["metrics"].read_text(encoding="utf-8"))
    grouped_path = round_dir / "grouped_metrics.json"
    grouped = json.loads(grouped_path.read_text(encoding="utf-8")) if grouped_path.exists() else {}
    artifacts = {
        name: path
        for name, path in (
            ("candidate_proposals", round_dir / "candidate_proposals.npz"),
            ("reference_proposals", round_dir / "reference_proposals.npz"),
            ("proposal_scores", round_dir / "proposal_scores.csv"),
            ("image_scores", round_dir / "image_scores.csv"),
            ("component_diagnostics", round_dir / "component_diagnostics.json"),
            ("grouped_metrics", grouped_path),
        )
        if path.exists()
    }
    return RoundResult(
        selected_image_ids=read_image_ids(required["selected_ids"]),
        remaining_candidate_ids=read_image_ids(required["remaining_pool_ids"]),
        labelled_ids=read_image_ids(required["labelled_ids"]),
        checkpoint_path=required["checkpoint"],
        metrics_path=required["metrics"],
        round_manifest_path=manifest_path,
        metrics=dict(metrics),
        grouped_metrics=dict(grouped),
        artifacts=artifacts,
    )


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


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _id_digest(values: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in values)
    if values:
        payload += "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _support_counts(
    image_ids: Sequence[str],
    *,
    annotations_dir: str,
    unknown_classes: Sequence[str],
    known_classes: Sequence[str],
    class_groups: ClassGroups | None,
) -> dict[str, Any]:
    unknown_set = set(unknown_classes)
    known_set = set(known_classes)
    counts = {
        "image_count": len(image_ids),
        "known_objects": 0,
        "unknown_objects": 0,
        "head_objects": 0,
        "medium_objects": 0,
        "tail_objects": 0,
        "by_class": {},
    }
    by_class: dict[str, int] = {}
    for image_id in image_ids:
        for raw_name in read_voc_classes(image_id, annotations_dir):
            class_name = CLASS_ALIASES.get(raw_name, raw_name)
            by_class[class_name] = by_class.get(class_name, 0) + 1
            if class_name in known_set:
                counts["known_objects"] += 1
            if class_name in unknown_set:
                counts["unknown_objects"] += 1
                if class_groups is not None:
                    group = class_groups.groups.get(class_name)
                    if group in ("head", "medium", "tail"):
                        counts[f"{group}_objects"] += 1
    counts["by_class"] = by_class
    return counts


def _write_image_scores(path: Path, image_scores: Mapping[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "score"], lineterminator="\n")
        writer.writeheader()
        for image_id, score in sorted(
            image_scores.items(), key=lambda item: (-item[1], str(item[0]))
        ):
            writer.writerow({"image_id": str(image_id), "score": float(score)})


def compute_grouped_metrics(
    metrics: Mapping[str, Any],
    *,
    class_groups: ClassGroups,
    unknown_classes: Sequence[str],
    known_classes: Sequence[str] = (),
    known_class_groups: ClassGroups | None = None,
    evaluation: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Decompose the official metrics into head / medium / tail groups.

    Returns an empty mapping when the evaluator wrote no detections artifact, so
    a caller can decide whether that is fatal (it is, by default).
    """

    settings = evaluation or EvaluationConfig()
    detections_path = metrics.get("detections_path")
    if not detections_path:
        return {}
    ground_truth, detections = load_detection_json(str(detections_path))
    category_report = require_consistent_category_space(
        ground_truth,
        detections,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        unknown_prediction_name=settings.unknown_prediction_name,
    )
    grouped = grouped_detection_metrics(
        ground_truth,
        detections,
        unknown_classes=unknown_classes,
        class_groups=class_groups,
        known_classes=known_classes,
        known_class_groups=known_class_groups,
        unknown_prediction_name=settings.unknown_prediction_name,
        iou_threshold=settings.iou_threshold,
    )
    validate_grouped_metrics(grouped)
    grouped["category_space"] = category_report
    grouped["detections_path"] = str(detections_path)
    return grouped


def run_active_round(
    *,
    adapter: ProbAdapter,
    checkpoint: str | Path,
    candidate_ids: Sequence[str],
    reference_ids: Sequence[str],
    labelled_ids: Sequence[str],
    evaluation_ids: Sequence[str] = (),
    output_dir: str | Path,
    spec: StrategySpec,
    budget: int,
    seed: int,
    round_index: int = 0,
    run_id: str = "",
    class_groups: ClassGroups | None = None,
    unknown_classes: Sequence[str] = (),
    known_classes: Sequence[str] = (),
    known_class_groups: ClassGroups | None = None,
    annotations_dir: str = "",
    evaluation: EvaluationConfig | None = None,
    protocol: ProtocolConfig | None = None,
    export_proposals_for_random: bool = False,
) -> RoundResult:
    """Run one detector-backed active-learning round with any strategy.

    Behaviour change from the pre-audit version, recorded in
    ``docs/migration_strategies.md``: a random strategy no longer exports
    proposals for the whole pool by default. The old code ran a full PROB forward
    pass and discarded it, which was a quarter of the pilot campaign's inference
    budget. Set ``export_proposals_for_random`` to restore the old artifacts.
    """

    settings = evaluation or EvaluationConfig()
    if budget < 1:
        raise ValueError("budget must be positive.")

    candidates = _unique_ids("candidate_ids", candidate_ids)
    references = _unique_ids("reference_ids", reference_ids)
    labelled = _unique_ids("labelled_ids", labelled_ids)
    if budget > len(candidates):
        raise ValueError("budget must not exceed candidate image count.")
    if not spec.random_selection and not references:
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
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("completed") is True:
            raise ValueError(f"Completed round would be overwritten: {round_dir}")
    round_dir.mkdir(parents=True, exist_ok=True)

    candidate_proposals_path = round_dir / "candidate_proposals.npz"
    reference_proposals_path = round_dir / "reference_proposals.npz"
    proposal_scores_path = round_dir / "proposal_scores.csv"
    image_scores_path = round_dir / "image_scores.csv"
    diagnostics_path = round_dir / "component_diagnostics.json"
    grouped_metrics_path = round_dir / "grouped_metrics.json"
    selected_path = round_dir / "selected_ids.txt"
    reference_path = round_dir / "reference_ids.txt"
    labelled_before_path = round_dir / "labelled_ids_before_selection.txt"
    labelled_path = round_dir / "labelled_ids.txt"
    training_path = round_dir / "training_ids.txt"
    remaining_path = round_dir / "remaining_pool_ids.txt"
    candidates_before_path = round_dir / "candidate_ids_before_selection.txt"
    evaluation_path = round_dir / "evaluation_ids.txt"
    checkpoint_path = round_dir / "checkpoint.pth"
    metrics_path = round_dir / "metrics.json"

    pool_seed = derive_pool_seed(seed, round_index)
    identifier = run_id or f"{spec.name}-s{seed}-r{round_index}"

    support_counts = {}
    if annotations_dir:
        support_counts = {
            "candidate_pool_before": _support_counts(
                candidates,
                annotations_dir=annotations_dir,
                unknown_classes=unknown_classes,
                known_classes=known_classes,
                class_groups=class_groups,
            ),
            "reference": _support_counts(
                references,
                annotations_dir=annotations_dir,
                unknown_classes=unknown_classes,
                known_classes=known_classes,
                class_groups=class_groups,
            ),
            "labelled_before": _support_counts(
                labelled,
                annotations_dir=annotations_dir,
                unknown_classes=unknown_classes,
                known_classes=known_classes,
                class_groups=class_groups,
            ),
            "evaluation": _support_counts(
                evaluation_ids,
                annotations_dir=annotations_dir,
                unknown_classes=unknown_classes,
                known_classes=known_classes,
                class_groups=class_groups,
            ),
        }
        if evaluation_ids and settings.grouped_metrics and class_groups is not None:
            missing_groups = [
                group
                for group in ("head", "medium", "tail")
                if support_counts["evaluation"][f"{group}_objects"] == 0
            ]
            if missing_groups:
                raise RuntimeError(
                    "Evaluation split has zero grouped-metric support for "
                    f"{missing_groups}; refusing to train."
                )

    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": identifier,
        "round_index": round_index,
        "strategy": spec.name,
        "semantics_version": spec.semantics_version,
        "strategy_spec": spec.as_dict(),
        "seed": seed,
        "pool_seed": pool_seed,
        "budget": budget,
        "input_checkpoint": str(checkpoint),
        "output_checkpoint": checkpoint_path.name,
        "candidate_count_before": len(candidates),
        "labelled_count_before": len(labelled),
        "reference_count": len(references),
        "evaluation_count": len(evaluation_ids),
        "image_aggregation": {
            "method": spec.image_aggregation,
            "top_k": spec.top_k,
        },
        "id_list_sha256_before": {
            "candidate_ids": _id_digest(candidates),
            "reference_ids": _id_digest(references),
            "labelled_ids": _id_digest(labelled),
            "evaluation_ids": _id_digest(evaluation_ids),
        },
        "overlap_policy": {
            "candidate_reference_overlap": sorted(set(candidates) & set(references))[:20],
            "candidate_labelled_overlap": sorted(set(candidates) & set(labelled))[:20],
            "candidate_evaluation_overlap": sorted(set(candidates) & set(evaluation_ids))[:20],
            "labelled_evaluation_overlap": sorted(set(labelled) & set(evaluation_ids))[:20],
        },
        "support_counts": support_counts,
        "completed": False,
    }
    _write_json(manifest_path, manifest)

    scoring: ScoringResult | None = None
    selected: list[str]

    if spec.random_selection:
        if export_proposals_for_random:
            try:
                adapter.predict(
                    candidates, checkpoint=checkpoint, output_path=candidate_proposals_path
                )
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise RuntimeError("candidate proposal export failed") from exc
        else:
            manifest["candidate_proposals_skipped"] = (
                "a random strategy needs no proposal scores; the forward pass was "
                "skipped to save inference budget"
            )
        shuffled = list(candidates)
        random.Random(f"{seed}:{round_index}:{spec.name}").shuffle(shuffled)
        selected = shuffled[:budget]
    else:
        try:
            candidate_batch = adapter.predict(
                candidates, checkpoint=checkpoint, output_path=candidate_proposals_path
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError("candidate proposal export failed") from exc
        try:
            reference_batch = adapter.predict(
                references, checkpoint=checkpoint, output_path=reference_proposals_path
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError("reference proposal export failed") from exc

        try:
            scoring = score_pool(
                spec=spec,
                image_ids=candidate_batch.image_ids,
                embeddings=candidate_batch.embeddings,
                reference_embeddings=reference_batch.embeddings,
                confidence=candidate_batch.confidence,
                posterior=candidate_batch.posterior,
                predicted_labels=candidate_batch.predicted_labels,
                seed=pool_seed,
                compute_all_components=True,
            )
            selected = select_images(scoring.image_scores, budget=budget)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError("proposal scoring failed") from exc

        rows = diag.proposal_table(
            scoring,
            run_id=identifier,
            seed=seed,
            round_index=round_index,
            selected_image_ids=selected,
            posterior=candidate_batch.posterior,
            confidence=candidate_batch.confidence,
            predicted_labels=candidate_batch.predicted_labels,
        )
        # Acquisition-time artifacts must never carry ground truth.
        diag.assert_no_ground_truth(rows)
        diag.write_rows(proposal_scores_path, rows)
        _write_image_scores(image_scores_path, scoring.image_scores)
        _write_json(
            diagnostics_path,
            diag.component_diagnostics(scoring, budget=budget),
        )

    selected_set = set(selected)
    cumulative_labelled = [*labelled, *selected]
    remaining = [image_id for image_id in candidates if image_id not in selected_set]
    _write_ids(candidates_before_path, candidates)
    _write_ids(reference_path, references)
    _write_ids(labelled_before_path, labelled)
    _write_ids(selected_path, selected)
    _write_ids(labelled_path, cumulative_labelled)
    _write_ids(training_path, cumulative_labelled)
    _write_ids(remaining_path, remaining)
    _write_ids(evaluation_path, evaluation_ids)
    if not set(selected).issubset(candidates):
        raise ValueError("Selected IDs must be a subset of the candidate pool.")
    if set(selected) & set(labelled):
        raise ValueError("Selected IDs were already labelled.")
    if len(cumulative_labelled) != len(labelled) + len(selected):
        raise ValueError("Labelled set did not grow exactly by the selected budget.")

    resolved_commands = {}
    if hasattr(adapter, "resolved_train_command"):
        resolved_commands["train"] = adapter.resolved_train_command(
            labelled_ids=labelled_path,
            previous_checkpoint=checkpoint,
            checkpoint=checkpoint_path,
            output_dir=round_dir,
            round_index=round_index,
            seed=seed,
        )
    if hasattr(adapter, "resolved_predict_command"):
        resolved_commands["candidate_predict"] = adapter.resolved_predict_command(
            image_ids=candidate_proposals_path.with_suffix(".ids.txt"),
            checkpoint=checkpoint,
            proposals=candidate_proposals_path,
        )
        resolved_commands["reference_predict"] = adapter.resolved_predict_command(
            image_ids=reference_proposals_path.with_suffix(".ids.txt"),
            checkpoint=checkpoint,
            proposals=reference_proposals_path,
        )
    if hasattr(adapter, "resolved_evaluate_command"):
        resolved_commands["evaluate"] = adapter.resolved_evaluate_command(
            checkpoint=checkpoint_path,
            metrics=metrics_path,
        )
    resolved_command_parity = (
        validate_resolved_command_parity(protocol, resolved_commands)
        if protocol is not None
        else None
    )

    try:
        trained_checkpoint = adapter.train(
            cumulative_labelled,
            previous_checkpoint=checkpoint,
            run_dir=round_dir,
            round_index=round_index,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RuntimeError("training failed") from exc
    if Path(trained_checkpoint) != checkpoint_path:
        raise ValueError(f"Unexpected checkpoint path: {trained_checkpoint}")

    try:
        metrics = adapter.evaluate(checkpoint=checkpoint_path, output_path=metrics_path)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RuntimeError("evaluation failed") from exc

    grouped: dict[str, Any] = {}
    if settings.grouped_metrics and class_groups is not None:
        grouped = compute_grouped_metrics(
            metrics,
            class_groups=class_groups,
            unknown_classes=unknown_classes,
            known_classes=known_classes,
            known_class_groups=known_class_groups,
            evaluation=settings,
        )
        if not grouped and settings.require_detections:
            raise RuntimeError(
                "The evaluator wrote no 'detections_path', so head/medium/tail "
                "metrics cannot be computed. Use a bridge that supports "
                "'evaluate --detections-output', or set "
                "evaluation.require_detections=false to proceed without grouped "
                "metrics."
            )
        if grouped:
            _write_json(grouped_metrics_path, grouped)

    manifest.update(
        {
            "candidate_count_after": len(remaining),
            "labelled_count_after": len(cumulative_labelled),
            "selected_ids_sha256": file_sha256(selected_path),
            "candidate_ids_before_sha256": file_sha256(candidates_before_path),
            "reference_ids_sha256": file_sha256(reference_path),
            "labelled_ids_before_sha256": file_sha256(labelled_before_path),
            "remaining_pool_sha256": file_sha256(remaining_path),
            "labelled_ids_sha256": file_sha256(labelled_path),
            "training_ids_sha256": file_sha256(training_path),
            "evaluation_ids_sha256": file_sha256(evaluation_path),
            "resolved_commands": resolved_commands,
            "resolved_command_parity": resolved_command_parity,
            "candidate_proposals_sha256": (
                file_sha256(candidate_proposals_path) if candidate_proposals_path.exists() else None
            ),
            "reference_proposals_sha256": (
                file_sha256(reference_proposals_path) if reference_proposals_path.exists() else None
            ),
            "metrics": metrics,
            "grouped_metrics": grouped or None,
            "scoring_diagnostics": dict(scoring.diagnostics) if scoring else None,
            "completed": True,
        }
    )
    _write_json(manifest_path, manifest)

    artifacts = {
        name: path
        for name, path in (
            ("candidate_proposals", candidate_proposals_path),
            ("reference_proposals", reference_proposals_path),
            ("proposal_scores", proposal_scores_path),
            ("image_scores", image_scores_path),
            ("component_diagnostics", diagnostics_path),
            ("grouped_metrics", grouped_metrics_path),
        )
        if path.exists()
    }

    return RoundResult(
        selected_image_ids=selected,
        remaining_candidate_ids=remaining,
        labelled_ids=cumulative_labelled,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        round_manifest_path=manifest_path,
        metrics=dict(metrics),
        grouped_metrics=grouped,
        scoring=scoring,
        artifacts=artifacts,
    )


class ActiveLearningCampaign:
    """Run every configured strategy over every seed through one round loop."""

    def __init__(self, config: ExperimentConfig, detector: ProbAdapter) -> None:
        self.config = config
        self.detector = detector

    def run(self) -> ExperimentResult:
        output_root = Path(self.config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_root / "experiment_config.json", self.config.as_dict())

        metrics_rows: list[dict[str, object]] = []
        selection_rows: list[dict[str, object]] = []
        specs = self.config.acquisition.resolved_specs()

        for seed in self.config.active_learning.seeds:
            pool = self._prepare_pool(seed=seed, output_root=output_root)
            class_groups = self._class_groups(pool)
            known_class_groups = self._known_class_groups()
            for spec in specs:
                rows, selections = self._run_strategy(
                    spec,
                    pool_ids=pool["selected_image_ids"],
                    class_groups=class_groups,
                    known_class_groups=known_class_groups,
                    seed=seed,
                    output_root=output_root,
                )
                metrics_rows.extend(rows)
                selection_rows.extend(selections)

        if metrics_rows:
            diag.write_rows(output_root / "metrics.csv", metrics_rows)
        (output_root / "selections.json").write_text(
            json.dumps(selection_rows, indent=2, default=str), encoding="utf-8"
        )
        return ExperimentResult(metrics_rows, selection_rows, output_root)

    def _prepare_pool(self, *, seed: int, output_root: Path) -> dict[str, Any]:
        dataset = self.config.dataset
        if self.config.protocol and self.config.protocol.pool_policy == "stage1_exact":
            return {
                "selected_image_ids": read_image_ids(self.config.protocol.candidate_pool_split),
                "class_stats_path": dataset.class_groups_path,
                "pool_split_path": self.config.protocol.candidate_pool_split,
                "pool_policy": self.config.protocol.pool_policy,
            }
        if not dataset.long_tail.enabled:
            return {
                "selected_image_ids": read_image_ids(dataset.image_set_path),
                "class_stats_path": None,
            }
        return build_long_tail_pool(
            annotation_dir=dataset.annotations_dir,
            source_split=dataset.image_set_path,
            task_class_names=dataset.unknown_classes,
            output_dir=output_root / f"long_tail_seed_{seed}",
            imbalance_ratio=dataset.long_tail.imbalance_ratio,
            seed=seed,
        )

    def _class_groups(self, pool: Mapping[str, Any]) -> ClassGroups | None:
        configured = self.config.dataset.class_groups_path
        if configured:
            return ClassGroups.from_class_stats_csv(configured)
        stats_path = pool.get("class_stats_path")
        if stats_path:
            return ClassGroups.from_class_stats_csv(stats_path)
        return None

    def _known_class_groups(self) -> ClassGroups | None:
        """Optional frequency grouping for the *known* classes.

        Without it the known side of the grouped metrics is reported as "not
        defined" rather than silently omitted, because the long-tail protocol
        only assigns groups to the task (unknown) classes.
        """

        configured = self.config.dataset.known_class_groups_path
        return ClassGroups.from_class_stats_csv(configured) if configured else None

    def _run_strategy(
        self,
        spec: StrategySpec,
        *,
        pool_ids: Sequence[str],
        class_groups: ClassGroups | None,
        known_class_groups: ClassGroups | None,
        seed: int,
        output_root: Path,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        settings = self.config.active_learning
        run_dir = output_root / f"seed_{seed}" / f"v{spec.semantics_version}_{spec.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = DatasetState.initialise(pool_ids, initial_images=settings.initial_images, seed=seed)
        checkpoint: str | Path | None = self.config.prob.initial_checkpoint
        if checkpoint is None:
            raise ValueError("prob.initial_checkpoint is required to run a campaign.")
        fixed_reference_policies = {
            "fixed_stage1_representation_bank",
            "fixed_stage1b_representation_bank",
        }
        reference_bank_ids = (
            read_image_ids(self.config.protocol.reference_split)
            if self.config.protocol
            and self.config.protocol.reference_policy in fixed_reference_policies
            else None
        )
        evaluation_ids = (
            read_image_ids(
                Path(self.config.protocol.data_root)
                / "ImageSets"
                / self.config.protocol.dataset_protocol
                / f"{self.config.protocol.evaluation_split}.txt"
            )
            if self.config.protocol
            else []
        )

        metrics_rows: list[dict[str, object]] = []
        selection_rows: list[dict[str, object]] = []
        candidates = list(state.pool_ids)
        labelled = list(state.labelled_ids)
        resume_completed = os.environ.get(RESUME_COMPLETED_ENV) == "1"

        for round_index in range(1, settings.rounds + 1):
            budget = min(settings.budget_per_round, len(candidates))
            if budget < 1:
                break
            round_dir = run_dir / f"round_{round_index:02d}"
            if resume_completed and (round_dir / "round_manifest.json").exists():
                result = _load_completed_round(round_dir)
                metrics_rows.append(
                    {
                        "strategy": spec.name,
                        "semantics_version": spec.semantics_version,
                        "seed": seed,
                        "round": round_index,
                        "cumulative_budget": round_index * settings.budget_per_round,
                        "labelled_images": len(result.labelled_ids),
                        **{
                            key: value
                            for key, value in result.metrics.items()
                            if isinstance(value, int | float | str) or value is None
                        },
                        **{
                            key: value
                            for key, value in result.grouped_metrics.items()
                            if isinstance(value, int | float | str) or value is None
                        },
                    }
                )
                selection_rows.append(
                    {
                        "strategy": spec.name,
                        "semantics_version": spec.semantics_version,
                        "seed": seed,
                        "after_round": round_index,
                        "selected_image_ids": result.selected_image_ids,
                    }
                )
                candidates = list(result.remaining_candidate_ids)
                labelled = list(result.labelled_ids)
                checkpoint = result.checkpoint_path
                continue
            result = run_active_round(
                adapter=self.detector,
                checkpoint=checkpoint,
                candidate_ids=candidates,
                reference_ids=reference_bank_ids or labelled,
                labelled_ids=labelled,
                evaluation_ids=evaluation_ids,
                output_dir=round_dir,
                spec=spec,
                budget=budget,
                seed=seed,
                round_index=round_index,
                run_id=f"{self.config.name}-{spec.name}-s{seed}-r{round_index}",
                class_groups=class_groups,
                unknown_classes=self.config.dataset.unknown_classes,
                known_classes=self.config.dataset.known_classes,
                known_class_groups=known_class_groups,
                annotations_dir=self.config.dataset.annotations_dir,
                evaluation=self.config.evaluation,
                protocol=self.config.protocol,
            )
            metrics_rows.append(
                {
                    "strategy": spec.name,
                    "semantics_version": spec.semantics_version,
                    "seed": seed,
                    "round": round_index,
                    "cumulative_budget": round_index * settings.budget_per_round,
                    "labelled_images": len(labelled) + len(result.selected_image_ids),
                    **{
                        key: value
                        for key, value in result.metrics.items()
                        if isinstance(value, int | float | str) or value is None
                    },
                    **{
                        key: value
                        for key, value in result.grouped_metrics.items()
                        if isinstance(value, int | float | str) or value is None
                    },
                }
            )
            selection_rows.append(
                {
                    "strategy": spec.name,
                    "semantics_version": spec.semantics_version,
                    "seed": seed,
                    "after_round": round_index,
                    "selected_image_ids": result.selected_image_ids,
                }
            )
            checkpoint = result.checkpoint_path
            labelled = [*labelled, *result.selected_image_ids]
            candidates = result.remaining_candidate_ids

        return metrics_rows, selection_rows
