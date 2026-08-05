"""High-value smoke tests for contribution A."""

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from daowod.dataset import DatasetState, build_long_tail_pool, file_sha256
from daowod.prob_adapter import ProposalBatch

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1B_CANDIDATE_SPLIT = Path("data/protocol/stage1b/stage1b_candidate_500.txt")
STAGE1B_REFERENCE_SPLIT = Path("data/protocol/stage1b/stage1b_reference_3500.txt")
STAGE1B_CANDIDATE_SHA256 = "70fa185514dcbbba8397781d85275362c888e6ea0c4d6c1325ad6c82fa18aac6"
STAGE1B_REFERENCE_SHA256 = "25a1b33614bcb77c8ef9b238ab878950b62861d0fc048fc58574c7fd0c6df762"


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
