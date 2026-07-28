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
from daowod.config import AcquisitionConfig
from daowod.dataset import DatasetState, build_long_tail_pool, file_sha256
from daowod.experiment import run_active_round
from daowod.metrics import Detection, GroundTruth, grouped_unknown_recall
from daowod.prob_adapter import ProposalBatch


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


class FakeAdapter:
    def __init__(self, *, fail_stage: str | None = None) -> None:
        self.fail_stage = fail_stage
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
        self.predict_calls.append((list(image_ids), str(checkpoint)))
        count = len(image_ids)
        _write_npz(
            output,
            image_ids=np.array(image_ids, dtype=object),
            confidence=np.full(count, 0.5),
            embeddings=np.tile(np.array([[1.0, 0.0]]), (count, 1)),
            posterior=np.tile(np.array([[0.4, 0.6]]), (count, 1)),
            predicted_labels=np.zeros(count, dtype=np.int64),
            boxes=np.tile(np.array([[0.0, 1.0, 2.0, 3.0]]), (count, 1)),
            objectness=np.full(count, 0.7),
        )
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
    acquisition = AcquisitionConfig(cluster_count=1, neighbour_count=1, top_k=1)
    adapter = FakeAdapter()
    round_args = {
        "checkpoint": "current.pth",
        "candidate_ids": ["candidate"],
        "reference_ids": ["reference"],
        "labelled_ids": ["already_labelled"],
        "strategy": "full",
        "budget": 1,
        "acquisition_config": acquisition,
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

    assert first["selected_image_ids"] == ["candidate"]
    assert second["selected_image_ids"] == first["selected_image_ids"]
    assert first["remaining_candidate_ids"] == []
    assert first["labelled_ids"] == ["already_labelled", "candidate"]
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
        acquisition.weights.uncertainty * float(score_row["uncertainty"])
        + acquisition.weights.novelty * float(score_row["novelty"])
        + acquisition.weights.rarity * float(score_row["rarity_bonus"])
    )
    assert float(score_row["coherence"]) == pytest.approx(0.0)
    assert float(score_row["rarity_bonus"]) == pytest.approx(0.0)
    assert float(score_row["score"]) == pytest.approx(expected_score)
    assert float(score_row["score"]) > 0

    manifest = json.loads((tmp_path / "first" / "round_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is True
    assert manifest["metrics"] == {"known_mAP": 0.1, "U_Recall": 0.2, "WI": 0.3, "A_OSE": 4}
    assert manifest["candidate_proposals_sha256"] == file_sha256(
        tmp_path / "first" / "candidate_proposals.npz"
    )
    assert manifest["reference_proposals_sha256"] == file_sha256(
        tmp_path / "first" / "reference_proposals.npz"
    )


def test_run_active_round_random_is_seeded_locally(tmp_path: Path) -> None:
    round_args = {
        "checkpoint": "current.pth",
        "candidate_ids": ["a", "b", "c", "d"],
        "reference_ids": [],
        "labelled_ids": [],
        "strategy": "random",
        "budget": 2,
        "acquisition_config": AcquisitionConfig(),
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
    assert first["selected_image_ids"] == second["selected_image_ids"]
    assert set(first["remaining_candidate_ids"]).isdisjoint(first["selected_image_ids"])
    assert first["labelled_ids"] == first["selected_image_ids"]
    assert not (tmp_path / "first_random" / "proposal_scores.csv").exists()
    assert not (tmp_path / "first_random" / "image_scores.csv").exists()
    assert not (tmp_path / "first_random" / "reference_proposals.npz").exists()


def test_run_active_round_rejects_invalid_inputs_and_completed_output(tmp_path: Path) -> None:
    kwargs = {
        "adapter": FakeAdapter(),
        "checkpoint": "current.pth",
        "candidate_ids": ["a"],
        "reference_ids": ["r"],
        "labelled_ids": [],
        "output_dir": tmp_path / "round",
        "strategy": "full",
        "budget": 1,
        "acquisition_config": AcquisitionConfig(cluster_count=1),
        "seed": 0,
    }
    with pytest.raises(ValueError, match="strategy"):
        run_active_round(**{**kwargs, "strategy": "uncertainty"})
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
        "strategy": "full",
        "budget": 1,
        "acquisition_config": AcquisitionConfig(cluster_count=1),
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
