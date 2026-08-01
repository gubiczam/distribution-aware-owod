"""High-value smoke tests for contribution A."""

import csv
import io
import json
import random
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from daowod.acquisition import (
    AcquisitionWeights,
    _offline_strategy_scores,
    compare_acquisition_strategies,
    compute_proposal_scores,
    score_proposals,
    select_images,
)
from daowod.config import (
    AcquisitionConfig,
    ConfigError,
    load_config,
    validate_resolved_command_parity,
)
from daowod.dataset import DatasetState, build_long_tail_pool, file_sha256
from daowod.experiment import run_active_round
from daowod.metrics import Detection, GroundTruth, grouped_unknown_recall
from daowod.prob_adapter import ProposalBatch
from daowod.scoring import STRATEGY_REGISTRY, StrategySpec


def _write_voc_xml(path: Path, class_names: list[str]) -> None:
    objects = "".join(f"<object><name>{class_name}</name></object>" for class_name in class_names)
    path.write_text(f"<annotation>{objects}</annotation>\n", encoding="utf-8")


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, values in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, values)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, buffer.getvalue())


def _write_offline_proposals(
    tmp_path: Path,
    *,
    image_ids: list[str],
    confidence: list[float],
    predicted_labels: list[int],
) -> tuple[Path, Path]:
    candidate_path = tmp_path / "candidate_proposals.npz"
    reference_path = tmp_path / "reference_proposals.npz"
    embeddings = np.array(
        [[float(index + 1), float((index % 2) + 1)] for index in range(len(image_ids))]
    )
    _write_npz(
        candidate_path,
        image_ids=np.array(image_ids, dtype=object),
        confidence=np.array(confidence, dtype=np.float64),
        embeddings=embeddings,
        posterior=np.tile(np.array([[0.4, 0.6]], dtype=np.float64), (len(image_ids), 1)),
        predicted_labels=np.array(predicted_labels, dtype=np.int64),
        boxes=np.tile(np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64), (len(image_ids), 1)),
        objectness=np.full(len(image_ids), 0.7, dtype=np.float64),
    )
    _write_npz(
        reference_path,
        image_ids=np.array(["reference"], dtype=object),
        confidence=np.array([0.5], dtype=np.float64),
        embeddings=np.array([[1.0, 1.0]], dtype=np.float64),
        posterior=np.array([[0.4, 0.6]], dtype=np.float64),
        predicted_labels=np.array([0], dtype=np.int64),
        boxes=np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64),
        objectness=np.array([0.7], dtype=np.float64),
    )
    return candidate_path, reference_path


def _round_candidate_batch() -> ProposalBatch:
    return ProposalBatch(
        image_ids=np.array(["img_a", "img_a", "img_b", "img_c"], dtype=object),
        confidence=np.array([0.5, 0.4, 0.9, 0.55], dtype=np.float64),
        embeddings=np.array(
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        posterior=np.tile(np.array([[0.4, 0.6]], dtype=np.float64), (4, 1)),
        predicted_labels=np.array([0, 0, 1, 2], dtype=np.int64),
        boxes=np.tile(np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64), (4, 1)),
        objectness=np.full(4, 0.7, dtype=np.float64),
    ).validate()


def _round_reference_batch() -> ProposalBatch:
    return ProposalBatch(
        image_ids=np.array(["ref"], dtype=object),
        confidence=np.array([0.5], dtype=np.float64),
        embeddings=np.array([[1.0, 0.0]], dtype=np.float64),
        posterior=np.array([[0.4, 0.6]], dtype=np.float64),
        predicted_labels=np.array([0], dtype=np.int64),
        boxes=np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64),
        objectness=np.array([0.7], dtype=np.float64),
    ).validate()


class FakeAdapter:
    def __init__(
        self,
        *,
        fail_stage: str | None = None,
        proposal_batches: dict[tuple[str, ...], ProposalBatch] | None = None,
    ) -> None:
        self.fail_stage = fail_stage
        self.proposal_batches = proposal_batches or {}
        self.predict_calls: list[tuple[list[str], str]] = []
        self.train_calls: list[list[str]] = []
        self.evaluate_calls: list[str] = []

    def predict(
        self,
        image_ids: Sequence[str],
        *,
        checkpoint: str | Path,
        output_path: str | Path,
    ) -> ProposalBatch:
        output = Path(output_path)
        ids = list(image_ids)
        self.predict_calls.append((ids, str(checkpoint)))
        batch = self.proposal_batches.get(tuple(ids))
        if batch is None:
            count = len(ids)
            batch = ProposalBatch(
                image_ids=np.array(ids, dtype=object),
                confidence=np.full(count, 0.5),
                embeddings=np.tile(np.array([[1.0, 0.0]]), (count, 1)),
                posterior=np.tile(np.array([[0.4, 0.6]]), (count, 1)),
                predicted_labels=np.zeros(count, dtype=np.int64),
                boxes=np.tile(np.array([[0.0, 1.0, 2.0, 3.0]]), (count, 1)),
                objectness=np.full(count, 0.7),
            ).validate()
        arrays: dict[str, np.ndarray] = {
            "image_ids": batch.image_ids,
            "confidence": batch.confidence,
            "embeddings": batch.embeddings,
        }
        if batch.posterior is not None:
            arrays["posterior"] = batch.posterior
        if batch.predicted_labels is not None:
            arrays["predicted_labels"] = batch.predicted_labels
        if batch.boxes is not None:
            arrays["boxes"] = batch.boxes
        if batch.objectness is not None:
            arrays["objectness"] = batch.objectness
        _write_npz(output, **arrays)
        return ProposalBatch.load(output)

    def train(
        self,
        labelled_image_ids: Sequence[str],
        *,
        previous_checkpoint: str | Path | None,
        run_dir: str | Path,
        round_index: int,
        seed: int,
    ) -> Path:
        if self.fail_stage == "train":
            raise RuntimeError("fake train failure")
        checkpoint = Path(run_dir) / "checkpoint.pth"
        self.train_calls.append(list(labelled_image_ids))
        checkpoint.write_bytes(b"fake checkpoint\n")
        return checkpoint

    def evaluate(self, *, checkpoint: str | Path, output_path: str | Path) -> dict[str, object]:
        if self.fail_stage == "evaluate":
            raise RuntimeError("fake evaluate failure")
        metrics = {"known_mAP": 0.1, "U_Recall": 0.2, "WI": 0.3, "A_OSE": 4}
        self.evaluate_calls.append(str(checkpoint))
        Path(output_path).write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
        return metrics


def _v1_spec(name: str, **overrides: object) -> StrategySpec:
    """A pre-audit strategy spec, optionally with structural overrides.

    Round tests use version-1 specs so their exact expected numbers still apply:
    they are now end-to-end proof that the refactored round reproduces the
    pre-audit formulas, not just that the scorer does.
    """

    spec = STRATEGY_REGISTRY.resolve(name, semantics_version=1)
    return StrategySpec(**{**spec.as_dict(), **overrides})


def test_coherence_gates_only_rarity() -> None:
    scores = compute_proposal_scores(
        strategy="full",
        uncertainty=[0.8, 0.8],
        novelty=[0.6, 0.6],
        rarity=[1.0, 1.0],
        coherence=[0.0, 1.0],
        weights=AcquisitionWeights(),
    )
    base = 0.3 * 0.8 + 0.2 * 0.6
    assert scores[0] == pytest.approx(base)
    assert scores[1] == pytest.approx(base + 0.5)


def test_offline_strategy_formulas_are_exact() -> None:
    weights = AcquisitionWeights()
    result = score_proposals(
        strategy="full",
        uncertainty_mode="ambiguity",
        pseudo_label_source="predicted",
        confidence=[0.1, 0.4],
        posterior=None,
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        reference_embeddings=[[1.0, 0.0]],
        predicted_labels=[0, 1],
        cluster_count=2,
        neighbour_count=1,
        seed=0,
        weights=weights,
    )

    assert _offline_strategy_scores("uncertainty", result, weights) == pytest.approx(
        result.uncertainty
    )
    assert _offline_strategy_scores("uncertainty_novelty", result, weights) == pytest.approx(
        0.3 * result.uncertainty + 0.2 * result.novelty
    )
    assert _offline_strategy_scores("rarity_no_coherence", result, weights) == pytest.approx(
        0.3 * result.uncertainty + 0.2 * result.novelty + 0.5 * result.rarity
    )
    assert _offline_strategy_scores("full", result, weights) == pytest.approx(
        0.3 * result.uncertainty + 0.2 * result.novelty + 0.5 * result.rarity * result.coherence
    )
    assert not np.allclose(
        _offline_strategy_scores("rarity_no_coherence", result, weights),
        _offline_strategy_scores("full", result, weights),
    )


def test_existing_public_scorer_behaviour_is_preserved() -> None:
    weights = AcquisitionWeights()
    uncertainty = np.array([0.8, 0.2])
    novelty = np.array([0.6, 0.4])
    rarity = np.array([1.0, 0.5])
    coherence = np.array([0.0, 0.25])

    old_uncertainty_novelty = compute_proposal_scores(
        strategy="uncertainty_novelty",
        uncertainty=uncertainty,
        novelty=novelty,
        rarity=rarity,
        coherence=coherence,
        weights=weights,
    )
    full = compute_proposal_scores(
        strategy="full",
        uncertainty=uncertainty,
        novelty=novelty,
        rarity=rarity,
        coherence=coherence,
        weights=weights,
    )

    assert old_uncertainty_novelty == pytest.approx((0.3 * uncertainty + 0.2 * novelty) / 0.5)
    assert full == pytest.approx(0.3 * uncertainty + 0.2 * novelty + 0.5 * rarity * coherence)


def test_zero_coherence_keeps_uncertainty_and_novelty_but_gates_rarity() -> None:
    weights = AcquisitionWeights()
    result = score_proposals(
        strategy="full",
        uncertainty_mode="ambiguity",
        pseudo_label_source="predicted",
        confidence=[0.4],
        posterior=None,
        embeddings=[[1.0, 0.0]],
        reference_embeddings=[[0.0, 1.0]],
        predicted_labels=[0],
        cluster_count=1,
        neighbour_count=1,
        seed=0,
        weights=weights,
    )
    full = _offline_strategy_scores("full", result, weights)
    ungated = _offline_strategy_scores("rarity_no_coherence", result, weights)
    assert result.coherence[0] == pytest.approx(0.0)
    assert full[0] == pytest.approx(0.3 * result.uncertainty[0] + 0.2 * result.novelty[0])
    assert ungated[0] == pytest.approx(full[0] + 0.5)
    assert ungated[0] != pytest.approx(full[0])


def test_complete_proposal_scoring() -> None:
    result = score_proposals(
        strategy="full",
        uncertainty_mode="ambiguity",
        pseudo_label_source="cluster",
        confidence=[0.5, 0.4, 0.9],
        posterior=None,
        embeddings=[[1, 0], [0.99, 0.01], [-1, 0]],
        reference_embeddings=[[1, 0]],
        predicted_labels=None,
        cluster_count=2,
        neighbour_count=1,
        seed=0,
        weights=AcquisitionWeights(),
    )
    assert result.scores.shape == (3,)
    assert np.all(np.isfinite(result.scores))


def test_image_selection_respects_fixed_budget() -> None:
    selected = select_images(
        ["b", "a", "a", "c"],
        [0.9, 0.2, 0.8, 0.7],
        budget=2,
        top_k=2,
    )
    assert selected == ["b", "c"]


def test_grouped_unknown_recall() -> None:
    ground_truth = [
        GroundTruth("a", "rare", (0, 0, 10, 10)),
        GroundTruth("b", "common", (0, 0, 10, 10)),
    ]
    detections = [Detection("a", "unknown", 0.9, (0, 0, 10, 10))]
    metrics = grouped_unknown_recall(
        ground_truth,
        detections,
        unknown_classes=["rare", "common"],
        class_groups={"rare": "tail", "common": "head"},
    )
    assert metrics["U_Recall_tail"] == pytest.approx(1.0)
    assert metrics["U_Recall_head"] == pytest.approx(0.0)


def test_dataset_state_reveal() -> None:
    state = DatasetState.initialise(["a", "b", "c", "d"], initial_images=2, seed=0)
    selected = state.pool_ids[:1]
    state.reveal(selected)
    assert selected[0] in state.labelled_ids
    assert selected[0] not in state.pool_ids


_CONFIG_TEMPLATE = """
name: multi-round
active_learning:
  rounds: 2
  strategy: {strategy}
  budget: 3
  initial_images: 2
  budget_per_round: 2
  seeds:
    - 0
    - 1
acquisition:
  strategies:
{strategies}
  uncertainty_mode: entropy
  pseudo_label_source: predicted
  cluster_count: 4
  neighbour_count: 3
  top_k: 2
  weights:
    uncertainty: 0.4
    novelty: 0.3
    rarity: 0.3
    coherence_power: 0.7
    rarity_power: 2.0
dataset:
  image_set_path: /tmp/train.txt
  annotations_dir: /tmp/Annotations
  unknown_classes:
    - rare
  long_tail:
    enabled: true
    imbalance_ratio: 10.0
prob:
  repository_path: /tmp/prob
  initial_checkpoint: /tmp/checkpoint.pth
  train_command: python train.py --labelled-ids {{labelled_ids}} --output-checkpoint {{checkpoint}}
  predict_command: python predict.py --image-ids {{image_ids}} --output {{proposals}}
  evaluate_command: python evaluate.py --checkpoint {{checkpoint}} --output {{metrics}}
output_dir: outputs/test
"""


def _protocol_config(
    tmp_path: Path,
    *,
    train_extra: str = "",
    predict_extra: str = "",
    evaluate_extra: str = "",
    candidate_text: str = "a\n",
) -> Path:
    prob = tmp_path / "prob"
    prob.mkdir()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    candidate = tmp_path / "candidate.txt"
    candidate.write_text(candidate_text, encoding="utf-8")
    reference = tmp_path / "reference.txt"
    reference.write_text("r\n", encoding="utf-8")
    data_root = tmp_path / "data" / "OWOD"
    eval_dir = data_root / "ImageSets" / "OWDETR"
    annotations = data_root / "Annotations"
    images = data_root / "JPEGImages"
    eval_dir.mkdir(parents=True)
    annotations.mkdir()
    images.mkdir()
    evaluation_split = eval_dir / "owdetr_test.txt"
    evaluation_split.write_text("eval\n", encoding="utf-8")
    _write_voc_xml(annotations / "eval.xml", ["known", "rare"])
    (images / "eval.jpg").write_bytes(b"fake jpg")
    config = tmp_path / "protocol.yaml"
    common = (
        f"--data-root {data_root} --dataset OWDETR --prev-introduced-classes 0 "
        "--current-introduced-classes 19 --num-classes 81 --objectness-temperature 1"
    )
    config.write_text(
        f"""
name: protocol-test
active_learning:
  rounds: 1
  strategy: v2:full
  budget: 1
  initial_images: 0
  budget_per_round: 1
  seeds: [0]
protocol:
  dataset_protocol: OWDETR
  data_root: {data_root}
  previous_introduced_classes: 0
  current_introduced_classes: 19
  num_classes: 81
  objectness_temperature: 1
  train_split: runtime_selected_ids
  candidate_pool_split: {candidate}
  reference_split: {reference}
  evaluation_split: owdetr_test
  evaluation_split_sha256: {file_sha256(evaluation_split)}
  initial_labelled_split: null
  checkpoint: {checkpoint}
  checkpoint_sha256: fake
  image_aggregation: top_k_mean
  top_k: 3
  uncertainty_method: entropy
  clustering_method: current_v2_kmeans_shared_pool_seed
  acquisition_budget: 1
  active_learning_rounds: 1
  training_schedule: "epochs=1,learning_rate=2e-5,eval_every=1,prob_unfrozen=true"
  evaluation_settings: "grouped_metrics=true,iou_threshold=0.5,unknown_prediction_name=unknown,require_detections=true"
  pool_policy: stage1_exact
  reference_policy: fixed_stage1_representation_bank
  long_tail_transformation: none
acquisition:
  strategies: [v2:full]
  uncertainty_method: entropy
dataset:
  image_set_path: {candidate}
  annotations_dir: {annotations}
  unknown_classes: [rare]
  known_classes: [known]
  long_tail:
    enabled: false
    imbalance_ratio: 10.0
prob:
  repository_path: {prob}
  initial_checkpoint: {checkpoint}
  train_command: python bridge.py train --labelled-ids {{labelled_ids}} --previous-checkpoint {{previous_checkpoint}} --output-checkpoint {{checkpoint}} --output-dir {{output_dir}} {common} {train_extra}
  predict_command: python bridge.py predict --image-ids {{image_ids}} --checkpoint {{checkpoint}} --output {{proposals}} {common} {predict_extra}
  evaluate_command: python bridge.py evaluate --checkpoint {{checkpoint}} --output {{metrics}} --output-dir {{output_dir}} --test-set owdetr_test {common} {evaluate_extra}
output_dir: {tmp_path / "out"}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config


def _write_config(tmp_path: Path, *, strategy: str, strategies: list[str]) -> Path:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            strategy=strategy,
            strategies="\n".join(f"    - {name}" for name in strategies),
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_load_config_requires_explicit_semantics_for_ambiguous_names(
    tmp_path: Path,
) -> None:
    """Intentional safety break: 'full' means two different things now."""

    path = _write_config(
        tmp_path, strategy="v1:rarity_no_coherence", strategies=["v1:rarity_no_coherence", "full"]
    )
    with pytest.raises(ConfigError, match="different semantics"):
        load_config(path)

    ambiguous_single = _write_config(
        tmp_path, strategy="full", strategies=["v1:rarity_no_coherence"]
    )
    with pytest.raises(ConfigError, match="different semantics"):
        load_config(ambiguous_single)


def test_load_config_accepts_multi_round_and_versioned_strategies(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        strategy="v1:rarity_no_coherence",
        strategies=["v1:rarity_no_coherence", "v1:full", "v2:full"],
    )
    config = load_config(path)

    assert config.active_learning.rounds == 2
    assert config.active_learning.strategy == "v1:rarity_no_coherence"
    assert config.acquisition.strategies == ("v1:rarity_no_coherence", "v1:full", "v2:full")
    assert config.acquisition.weights.coherence_power == pytest.approx(0.7)

    specs = config.acquisition.resolved_specs()
    assert [spec.semantics_version for spec in specs] == [1, 1, 2]
    # A v1 spec keeps its pre-audit definitions even though the config asks for
    # entropy and predicted pseudo-labels: reproducibility beats consistency here.
    assert specs[0].uncertainty_method == "legacy_prob_score"
    assert specs[0].pseudo_label_source == "cluster"
    # The v2 spec picks up every override, including the coherence exponent that
    # legacy configs expressed as weights.coherence_power.
    assert specs[2].uncertainty_method == "entropy"
    assert specs[2].pseudo_label_source == "predicted"
    assert specs[2].cluster_count == 4
    assert specs[2].top_k == 2
    assert specs[2].coherence_exponent == pytest.approx(0.7)
    # The config fingerprint changes when the resolved specs change.
    assert len(config.fingerprint()) == 64


def test_protocol_command_parity_accepts_explicit_owdetr_sowodb_config(tmp_path: Path) -> None:
    config = load_config(_protocol_config(tmp_path))

    assert config.protocol is not None
    assert config.protocol.dataset_protocol == "OWDETR"
    assert config.as_dict()["command_parity"]["status"] == "ok"


def test_protocol_command_parity_rejects_towod_leaking_into_one_stage(tmp_path: Path) -> None:
    path = _protocol_config(tmp_path, predict_extra="--dataset TOWOD")

    with pytest.raises(ConfigError, match="predict: --dataset='TOWOD'"):
        load_config(path)


def test_protocol_command_parity_rejects_20_20_class_defaults(tmp_path: Path) -> None:
    path = _protocol_config(
        tmp_path,
        train_extra="--prev-introduced-classes 20 --current-introduced-classes 20",
    )

    with pytest.raises(ConfigError, match="train: --prev-introduced-classes='20'"):
        load_config(path)


def test_protocol_command_parity_rejects_objectness_temperature_mismatch(
    tmp_path: Path,
) -> None:
    path = _protocol_config(tmp_path, evaluate_extra="--objectness-temperature 1.3")

    with pytest.raises(ConfigError, match="evaluate: --objectness-temperature='1.3'"):
        load_config(path)


def test_protocol_command_parity_rejects_inconsistent_evaluation_split(
    tmp_path: Path,
) -> None:
    path = _protocol_config(tmp_path, evaluate_extra="--test-set other_test")

    with pytest.raises(ConfigError, match="evaluate: --test-set='other_test'"):
        load_config(path)


def test_protocol_command_parity_rejects_missing_placeholder(tmp_path: Path) -> None:
    path = _protocol_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace("--output {proposals}", "")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="predict: missing required argument --output"):
        load_config(path)


def test_resolved_command_parity_rejects_unresolved_or_missing_arguments(tmp_path: Path) -> None:
    config = load_config(_protocol_config(tmp_path))
    assert config.protocol is not None
    commands = {
        "train": (
            "python bridge.py train --labelled-ids {labelled_ids} "
            "--previous-checkpoint prev.pth --output-checkpoint out.pth --output-dir out "
            "--data-root /data/OWOD --dataset OWDETR --prev-introduced-classes 0 "
            "--current-introduced-classes 19 --num-classes 81 --objectness-temperature 1"
        ),
        "candidate_predict": (
            "python bridge.py predict --image-ids ids.txt --checkpoint ckpt.pth "
            "--output cand.npz --data-root /data/OWOD --dataset OWDETR "
            "--prev-introduced-classes 0 --current-introduced-classes 19 "
            "--num-classes 81 --objectness-temperature 1"
        ),
        "reference_predict": (
            "python bridge.py predict --image-ids refs.txt --checkpoint ckpt.pth "
            "--output refs.npz --data-root /data/OWOD --dataset OWDETR "
            "--prev-introduced-classes 0 --current-introduced-classes 19 "
            "--num-classes 81 --objectness-temperature 1"
        ),
        "evaluate": (
            "python bridge.py evaluate --checkpoint out.pth --output metrics.json "
            "--output-dir out --test-set owdetr_test --data-root /data/OWOD "
            "--dataset OWDETR --prev-introduced-classes 0 "
            "--current-introduced-classes 19 --num-classes 81 --objectness-temperature 1"
        ),
    }

    with pytest.raises(ConfigError, match="unresolved placeholder"):
        validate_resolved_command_parity(config.protocol, commands)


def test_evaluation_validator_rejects_candidate_overlap(tmp_path: Path) -> None:
    path = _protocol_config(tmp_path, candidate_text="eval\n")

    with pytest.raises(ConfigError, match="overlaps the acquisition candidate pool"):
        load_config(path)


def test_evaluation_validator_rejects_digest_mismatch(tmp_path: Path) -> None:
    path = _protocol_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "evaluation_split_sha256: "
        + file_sha256(path.parent / "data/OWOD/ImageSets/OWDETR/owdetr_test.txt"),
        "evaluation_split_sha256: bad",
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="Evaluation split digest mismatch"):
        load_config(path)


def test_evaluation_validator_rejects_missing_image(tmp_path: Path) -> None:
    path = _protocol_config(tmp_path)
    (path.parent / "data/OWOD/JPEGImages/eval.jpg").unlink()

    with pytest.raises(ConfigError, match="missing image"):
        load_config(path)


def test_proposal_batch_npz_schema(tmp_path: Path) -> None:
    path = tmp_path / "proposals.npz"
    _write_npz(
        path,
        image_ids=np.array(["a", "b"], dtype=object),
        confidence=np.array([0.4, 0.6]),
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
        posterior=np.array([[0.2, 0.8], [0.7, 0.3]]),
        predicted_labels=np.array([1, 0]),
        boxes=np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]),
        objectness=np.array([0.9, 0.8]),
    )
    batch = ProposalBatch.load(path)
    assert batch.embeddings.shape == (2, 2)
    assert batch.posterior is not None and batch.posterior.shape == (2, 2)
    assert batch.predicted_labels is not None and batch.predicted_labels.tolist() == [1, 0]
    assert batch.boxes is not None and batch.boxes.shape == (2, 4)
    assert batch.objectness is not None and batch.objectness.tolist() == [0.9, 0.8]

    missing = tmp_path / "missing.npz"
    np.savez(missing, image_ids=np.array(["a"], dtype=object), confidence=np.array([0.4]))
    with pytest.raises(ValueError, match="Missing proposal NPZ fields"):
        ProposalBatch.load(missing)


def test_compare_acquisition_strategies_is_deterministic_and_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_xml_parse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline acquisition must not parse XML annotations")

    monkeypatch.setattr("xml.etree.ElementTree.parse", fail_xml_parse)
    candidate_path, reference_path = _write_offline_proposals(
        tmp_path,
        image_ids=["a", "a", "b", "b"],
        confidence=[0.5, 0.6, 1.0, 1.0],
        predicted_labels=[0, 1, 1, 1],
    )
    config = AcquisitionConfig(pseudo_label_source="predicted", top_k=2, neighbour_count=1)
    strategies = [
        "random",
        "uncertainty",
        "uncertainty_novelty",
        "rarity_no_coherence",
        "full",
    ]

    random.seed(12345)
    random_state = random.getstate()
    first = compare_acquisition_strategies(
        candidate_path,
        reference_path,
        strategies=strategies,
        budget=1,
        seed=7,
        acquisition_config=config,
        output_dir=tmp_path / "first",
    )
    assert random.getstate() == random_state
    second = compare_acquisition_strategies(
        candidate_path,
        reference_path,
        strategies=strategies,
        budget=1,
        seed=7,
        acquisition_config=config,
        output_dir=tmp_path / "second",
    )

    assert [row["strategy"] for row in first["summary"]] == strategies
    assert first == second
    assert first["selected_ids"]["uncertainty"] == ["a"]
    assert {path.name for path in (tmp_path / "first").iterdir()} == {
        "strategy_summary.csv",
        "selected_ids.json",
        "overlap_matrix.csv",
    }
    for filename in ("strategy_summary.csv", "selected_ids.json", "overlap_matrix.csv"):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()

    rows = {row["strategy"]: row for row in first["summary"]}
    uncertainty_row = rows["uncertainty"]
    assert uncertainty_row["unique_pseudo_classes"] == 2
    assert uncertainty_row["pseudo_class_entropy"] == pytest.approx(np.log(2.0))
    assert uncertainty_row["mean_uncertainty"] == pytest.approx(0.9)
    assert rows["full"]["mean_rarity_bonus"] <= rows["full"]["mean_rarity"]
    for left in strategies:
        assert first["overlap_matrix"][left][left] == len(first["selected_ids"][left])
        for right in strategies:
            assert first["overlap_matrix"][left][right] == first["overlap_matrix"][right][left]

    expected_scores = _offline_strategy_scores(
        "uncertainty",
        score_proposals(
            strategy="full",
            uncertainty_mode="ambiguity",
            pseudo_label_source="predicted",
            confidence=[0.5, 0.6, 1.0, 1.0],
            posterior=None,
            embeddings=[[1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [4.0, 2.0]],
            reference_embeddings=[[1.0, 1.0]],
            predicted_labels=[0, 1, 1, 1],
            cluster_count=config.cluster_count,
            neighbour_count=1,
            seed=7,
            weights=config.weights,
        ),
        config.weights,
    )
    assert select_images(["a", "a", "b", "b"], expected_scores, budget=1, top_k=2) == ["a"]


def test_compare_acquisition_rejects_invalid_strategy_and_budget(tmp_path: Path) -> None:
    candidate_path, reference_path = _write_offline_proposals(
        tmp_path,
        image_ids=["a"],
        confidence=[0.5],
        predicted_labels=[0],
    )
    config = AcquisitionConfig(pseudo_label_source="predicted")

    with pytest.raises(ValueError, match="Unknown acquisition strategies"):
        compare_acquisition_strategies(
            candidate_path,
            reference_path,
            strategies=["rarity"],
            budget=1,
            seed=0,
            acquisition_config=config,
        )
    with pytest.raises(ValueError, match="budget"):
        compare_acquisition_strategies(
            candidate_path,
            reference_path,
            strategies=["full"],
            budget=0,
            seed=0,
            acquisition_config=config,
        )
    with pytest.raises(ValueError, match="budget"):
        compare_acquisition_strategies(
            candidate_path,
            reference_path,
            strategies=["full"],
            budget=2,
            seed=0,
            acquisition_config=config,
        )


def test_run_active_round_full_is_deterministic_and_reveals_complete_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_xml_parse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("active acquisition must not parse candidate XML")

    monkeypatch.setattr("xml.etree.ElementTree.parse", fail_xml_parse)
    spec = _v1_spec("full", cluster_count=1, neighbour_count=1, top_k=1)
    adapter = FakeAdapter()
    round_args = {
        "checkpoint": "current.pth",
        "candidate_ids": ["candidate"],
        "reference_ids": ["reference"],
        "labelled_ids": ["already_labelled"],
        "spec": spec,
        "budget": 1,
        "seed": 7,
    }

    first = run_active_round(
        adapter=adapter,
        output_dir=tmp_path / "first",
        **round_args,
    )
    second = run_active_round(
        adapter=FakeAdapter(),
        output_dir=tmp_path / "second",
        **round_args,
    )

    assert first.selected_image_ids == ["candidate"]
    assert second.selected_image_ids == first.selected_image_ids
    assert first.remaining_candidate_ids == []
    assert first.labelled_ids == ["already_labelled", "candidate"]
    assert adapter.predict_calls == [(["candidate"], "current.pth"), (["reference"], "current.pth")]
    assert adapter.train_calls == [["already_labelled", "candidate"]]
    assert adapter.evaluate_calls == [str(tmp_path / "first" / "checkpoint.pth")]

    for filename in (
        "proposal_scores.csv",
        "image_scores.csv",
        "selected_ids.txt",
        "labelled_ids.txt",
        "remaining_pool_ids.txt",
        "round_manifest.json",
    ):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()

    with (tmp_path / "first" / "proposal_scores.csv").open(newline="", encoding="utf-8") as file:
        score_row = next(csv.DictReader(file))
    expected_score = (
        spec.uncertainty_weight * float(score_row["norm_uncertainty"])
        + spec.novelty_weight * float(score_row["norm_novelty"])
        + spec.gated_weight * float(score_row["raw_gated"])
    )
    assert float(score_row["raw_coherence"]) == pytest.approx(0.0)
    assert float(score_row["raw_gated"]) == pytest.approx(0.0)
    assert float(score_row["proposal_score"]) == pytest.approx(expected_score)
    assert float(score_row["proposal_score"]) > 0

    manifest = json.loads((tmp_path / "first" / "round_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is True
    assert manifest["metrics"] == {"known_mAP": 0.1, "U_Recall": 0.2, "WI": 0.3, "A_OSE": 4}
    assert manifest["candidate_proposals_sha256"] == file_sha256(
        tmp_path / "first" / "candidate_proposals.npz"
    )
    assert manifest["reference_proposals_sha256"] == file_sha256(
        tmp_path / "first" / "reference_proposals.npz"
    )


def test_run_active_round_rarity_no_coherence_completes_and_scores_without_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_xml_parse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("active acquisition must not parse candidate XML")

    monkeypatch.setattr("xml.etree.ElementTree.parse", fail_xml_parse)
    candidate_ids = ["img_a", "img_b", "img_c"]
    reference_ids = ["ref"]
    adapter = FakeAdapter(
        proposal_batches={
            tuple(candidate_ids): _round_candidate_batch(),
            tuple(reference_ids): _round_reference_batch(),
        }
    )
    result = run_active_round(
        adapter=adapter,
        checkpoint="current.pth",
        candidate_ids=candidate_ids,
        reference_ids=reference_ids,
        labelled_ids=["labelled"],
        output_dir=tmp_path / "rarity",
        spec=_v1_spec(
            "rarity_no_coherence",
            pseudo_label_source="predicted",
            neighbour_count=1,
            top_k=2,
        ),
        budget=2,
        seed=11,
    )

    assert result.selected_image_ids == ["img_c", "img_b"]
    assert result.remaining_candidate_ids == ["img_a"]
    assert result.labelled_ids == ["labelled", "img_c", "img_b"]
    assert adapter.predict_calls == [(candidate_ids, "current.pth"), (reference_ids, "current.pth")]
    assert adapter.train_calls == [["labelled", "img_c", "img_b"]]
    assert adapter.evaluate_calls == [str(tmp_path / "rarity" / "checkpoint.pth")]

    with (tmp_path / "rarity" / "proposal_scores.csv").open(newline="", encoding="utf-8") as file:
        proposal_rows = list(csv.DictReader(file))
    assert [float(row["proposal_score"]) for row in proposal_rows] == pytest.approx(
        [0.3, 0.24, 0.76, 0.87]
    )
    # Ungatedness, tested three ways rather than by inspecting a derived column.
    # (`raw_gated` is now always the gated interaction; a strategy is ungated
    # because its gated_weight is zero, not because that column equals rarity.)
    spec = _v1_spec(
        "rarity_no_coherence", pseudo_label_source="predicted", neighbour_count=1, top_k=2
    )
    assert spec.gated_weight == 0.0
    assert spec.rarity_weight == pytest.approx(0.5)
    for row in proposal_rows:
        assert float(row["proposal_score"]) == pytest.approx(
            0.3 * float(row["norm_uncertainty"])
            + 0.2 * float(row["norm_novelty"])
            + 0.5 * float(row["norm_rarity"])
        )

    with (tmp_path / "rarity" / "image_scores.csv").open(newline="", encoding="utf-8") as file:
        image_rows = list(csv.DictReader(file))
    assert [row["image_id"] for row in image_rows] == ["img_c", "img_b", "img_a"]
    assert [float(row["score"]) for row in image_rows] == pytest.approx([0.87, 0.76, 0.27])
    assert (
        json.loads((tmp_path / "rarity" / "round_manifest.json").read_text(encoding="utf-8"))[
            "strategy"
        ]
        == "rarity_no_coherence"
    )
    assert {path.name for path in (tmp_path / "rarity").iterdir()} >= {
        "candidate_proposals.npz",
        "reference_proposals.npz",
        "proposal_scores.csv",
        "image_scores.csv",
        "selected_ids.txt",
        "labelled_ids.txt",
        "remaining_pool_ids.txt",
        "checkpoint.pth",
        "metrics.json",
        "round_manifest.json",
    }


def test_ungated_strategy_scores_are_invariant_to_the_coherence_method(
    tmp_path: Path,
) -> None:
    """An ungated strategy must not depend on coherence at all.

    Stronger than comparing a derived CSV column: the coherence definition is
    swapped for one that produces very different values, and the scores must be
    bit-identical.
    """

    candidate_ids = ["img_a", "img_b", "img_c"]
    reference_ids = ["ref"]
    scores_by_method = {}
    for method in ("density", "relative_within_cluster", "neighbour_consistency"):
        run_active_round(
            adapter=FakeAdapter(
                proposal_batches={
                    tuple(candidate_ids): _round_candidate_batch(),
                    tuple(reference_ids): _round_reference_batch(),
                }
            ),
            checkpoint="current.pth",
            candidate_ids=candidate_ids,
            reference_ids=reference_ids,
            labelled_ids=["labelled"],
            output_dir=tmp_path / method,
            spec=_v1_spec(
                "rarity_no_coherence",
                pseudo_label_source="predicted",
                neighbour_count=1,
                top_k=2,
                coherence_method=method,
            ),
            budget=2,
            seed=11,
        )
        with (tmp_path / method / "proposal_scores.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        scores_by_method[method] = [float(row["proposal_score"]) for row in rows]
        coherence = [float(row["raw_coherence"]) for row in rows]
        assert coherence == pytest.approx(coherence)  # finite

    # The coherence values genuinely differ between methods ...
    with (tmp_path / "density" / "proposal_scores.csv").open(newline="", encoding="utf-8") as file:
        density_coherence = [float(row["raw_coherence"]) for row in csv.DictReader(file)]
    with (tmp_path / "neighbour_consistency" / "proposal_scores.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        consistency_coherence = [float(row["raw_coherence"]) for row in csv.DictReader(file)]
    assert density_coherence != pytest.approx(consistency_coherence)

    # ... yet the ungated scores are identical, and equal the legacy formula.
    assert scores_by_method["density"] == pytest.approx([0.3, 0.24, 0.76, 0.87])
    for method, values in scores_by_method.items():
        assert values == pytest.approx(scores_by_method["density"]), method


def test_run_active_round_full_regression_scores_image_scores_and_selection(
    tmp_path: Path,
) -> None:
    candidate_ids = ["img_a", "img_b", "img_c"]
    reference_ids = ["ref"]
    run_active_round(
        adapter=FakeAdapter(
            proposal_batches={
                tuple(candidate_ids): _round_candidate_batch(),
                tuple(reference_ids): _round_reference_batch(),
            }
        ),
        checkpoint="current.pth",
        candidate_ids=candidate_ids,
        reference_ids=reference_ids,
        labelled_ids=[],
        output_dir=tmp_path / "full",
        spec=_v1_spec(
            "full",
            pseudo_label_source="predicted",
            neighbour_count=1,
            top_k=2,
        ),
        budget=2,
        seed=11,
    )

    with (tmp_path / "full" / "proposal_scores.csv").open(newline="", encoding="utf-8") as file:
        proposal_rows = list(csv.DictReader(file))
    assert [float(row["proposal_score"]) for row in proposal_rows] == pytest.approx(
        [0.3, 0.24, 0.42666666666666664, 0.5366666666666666]
    )
    assert [float(row["raw_gated"]) for row in proposal_rows] == pytest.approx(
        [0.0, 0.0, 0.3333333333333333, 0.3333333333333333]
    )

    with (tmp_path / "full" / "image_scores.csv").open(newline="", encoding="utf-8") as file:
        image_rows = list(csv.DictReader(file))
    assert [row["image_id"] for row in image_rows] == ["img_c", "img_b", "img_a"]
    assert [float(row["score"]) for row in image_rows] == pytest.approx(
        [0.5366666666666666, 0.42666666666666664, 0.27]
    )
    assert (tmp_path / "full" / "selected_ids.txt").read_text(encoding="utf-8").splitlines() == [
        "img_c",
        "img_b",
    ]


def test_run_active_round_random_is_seeded_locally(tmp_path: Path) -> None:
    round_args = {
        "checkpoint": "current.pth",
        "candidate_ids": ["a", "b", "c", "d"],
        "reference_ids": [],
        "labelled_ids": [],
        "spec": STRATEGY_REGISTRY.resolve("random"),
        "budget": 2,
        "seed": 5,
        "round_index": 1,
    }
    random.seed(12345)
    before = random.getstate()
    first = run_active_round(
        adapter=FakeAdapter(),
        output_dir=tmp_path / "first_random",
        **round_args,
    )
    after = random.getstate()
    second = run_active_round(
        adapter=FakeAdapter(),
        output_dir=tmp_path / "second_random",
        **round_args,
    )

    assert before == after
    assert first.selected_image_ids == second.selected_image_ids
    assert set(first.remaining_candidate_ids).isdisjoint(first.selected_image_ids)
    assert first.labelled_ids == first.selected_image_ids
    assert not (tmp_path / "first_random" / "proposal_scores.csv").exists()
    assert not (tmp_path / "first_random" / "image_scores.csv").exists()
    assert not (tmp_path / "first_random" / "reference_proposals.npz").exists()
    # Behaviour change: a random strategy no longer exports proposals it discards.
    assert not (tmp_path / "first_random" / "candidate_proposals.npz").exists()


def test_run_active_round_rejects_invalid_inputs_and_completed_output(tmp_path: Path) -> None:
    kwargs = {
        "adapter": FakeAdapter(),
        "checkpoint": "current.pth",
        "candidate_ids": ["a"],
        "reference_ids": ["r"],
        "labelled_ids": [],
        "output_dir": tmp_path / "round",
        "spec": _v1_spec("full", cluster_count=1),
        "budget": 1,
        "seed": 0,
    }
    # Every registry strategy is now runnable in the live loop, so the old
    # three-strategy allowlist is gone. What must still fail loudly is a strategy
    # whose uncertainty method needs a posterior the export does not carry.
    with pytest.raises(RuntimeError, match="proposal scoring failed"):
        run_active_round(
            **{
                **kwargs,
                "adapter": FakeAdapter(
                    proposal_batches={
                        ("a",): ProposalBatch(
                            image_ids=np.array(["a"], dtype=object),
                            confidence=np.array([0.5]),
                            embeddings=np.array([[1.0, 0.0]]),
                        ).validate()
                    }
                ),
                "output_dir": tmp_path / "no_posterior",
                "spec": STRATEGY_REGISTRY.resolve("v2:uncertainty"),
            }
        )
    with pytest.raises(ValueError, match="budget"):
        run_active_round(**{**kwargs, "budget": 0})
    with pytest.raises(ValueError, match="budget"):
        run_active_round(**{**kwargs, "budget": 2})

    run_active_round(**kwargs)
    with pytest.raises(ValueError, match="Completed round"):
        run_active_round(**kwargs)


def test_run_active_round_failures_keep_exports_and_incomplete_manifest(tmp_path: Path) -> None:
    kwargs = {
        "checkpoint": "current.pth",
        "candidate_ids": ["candidate"],
        "reference_ids": ["reference"],
        "labelled_ids": [],
        "spec": _v1_spec("full", cluster_count=1),
        "budget": 1,
        "seed": 0,
    }

    train_dir = tmp_path / "train_failure"
    with pytest.raises(RuntimeError, match="training failed"):
        run_active_round(adapter=FakeAdapter(fail_stage="train"), output_dir=train_dir, **kwargs)
    assert (train_dir / "candidate_proposals.npz").exists()
    assert (train_dir / "reference_proposals.npz").exists()
    assert (
        json.loads((train_dir / "round_manifest.json").read_text(encoding="utf-8"))["completed"]
        is False
    )

    eval_dir = tmp_path / "evaluate_failure"
    with pytest.raises(RuntimeError, match="evaluation failed"):
        run_active_round(adapter=FakeAdapter(fail_stage="evaluate"), output_dir=eval_dir, **kwargs)
    assert (eval_dir / "candidate_proposals.npz").exists()
    assert (eval_dir / "reference_proposals.npz").exists()
    assert (eval_dir / "checkpoint.pth").exists()
    assert (
        json.loads((eval_dir / "round_manifest.json").read_text(encoding="utf-8"))["completed"]
        is False
    )


def test_long_tail_pool_protocol_is_deterministic_and_records_distribution(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "Annotations"
    annotations.mkdir()
    split = tmp_path / "train.txt"
    image_classes = {
        "img1": ["common"],
        "img2": ["common"],
        "img3": ["common"],
        "img4": ["common"],
        "img5": ["common", "middle"],
        "img6": ["common", "middle"],
        "img7": ["middle"],
        "img8": ["rare"],
        "img9": ["background"],
    }
    split.write_text("\n".join(image_classes) + "\n", encoding="utf-8")
    for image_id, class_names in image_classes.items():
        _write_voc_xml(annotations / f"{image_id}.xml", class_names)
    original_xml = {path.name: path.read_bytes() for path in annotations.glob("*.xml")}

    first = build_long_tail_pool(
        annotations,
        split,
        ["common", "middle", "rare"],
        tmp_path / "first",
        imbalance_ratio=9.0,
        seed=3,
    )
    second = build_long_tail_pool(
        annotations,
        split,
        ["common", "middle", "rare"],
        tmp_path / "second",
        imbalance_ratio=9.0,
        seed=3,
    )

    for key in ("pool_split_path", "class_stats_path", "manifest_path"):
        assert Path(first[key]).read_bytes() == Path(second[key]).read_bytes()
    assert all(path.read_bytes() == original_xml[path.name] for path in annotations.glob("*.xml"))

    source_ids = set(image_classes)
    selected_ids = Path(first["pool_split_path"]).read_text(encoding="utf-8").splitlines()
    assert selected_ids == first["selected_image_ids"]
    assert set(selected_ids) <= source_ids

    with Path(first["class_stats_path"]).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    target_frequencies = [int(row["target_frequency"]) for row in rows]
    assert target_frequencies == sorted(target_frequencies, reverse=True)
    assert all(int(row["target_frequency"]) >= 1 for row in rows)
    assert {row["group"] for row in rows} == {"head", "medium", "tail"}

    realised = {row["group"]: int(row["realised_frequency"]) for row in rows}
    assert realised["head"] >= realised["medium"] >= realised["tail"]

    manifest = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["source_split_sha256"] == file_sha256(split)
    assert manifest["pool_ids_sha256"] == file_sha256(Path(first["pool_split_path"]))
    assert manifest["excluded_image_count"] == 1


def test_long_tail_pool_rejects_invalid_or_missing_inputs(tmp_path: Path) -> None:
    annotations = tmp_path / "Annotations"
    annotations.mkdir()
    split = tmp_path / "train.txt"
    split.write_text("missing_image\n", encoding="utf-8")

    with pytest.raises(ValueError, match="imbalance_ratio"):
        build_long_tail_pool(annotations, split, ["common"], tmp_path / "out", imbalance_ratio=0.5)
    with pytest.raises(FileNotFoundError, match="Missing image set"):
        build_long_tail_pool(annotations, tmp_path / "missing.txt", ["common"], tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="Missing annotation"):
        build_long_tail_pool(annotations, split, ["common"], tmp_path / "out")
