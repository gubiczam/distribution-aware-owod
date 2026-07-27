"""High-value smoke tests for contribution A."""

from pathlib import Path

import numpy as np
import pytest

from daowod.acquisition import (
    AcquisitionWeights,
    compute_proposal_scores,
    score_proposals,
    select_images,
)
from daowod.dataset import DatasetState
from daowod.metrics import Detection, GroundTruth, grouped_unknown_recall
from daowod.prob_adapter import ProposalBatch


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
    assert selected == ["b", "a"]


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
    np.savez(
        path,
        image_ids=np.array(["a", "b"], dtype=object),
        confidence=np.array([0.4, 0.6]),
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    batch = ProposalBatch.load(path)
    assert batch.embeddings.shape == (2, 2)

    missing = tmp_path / "missing.npz"
    np.savez(missing, image_ids=np.array(["a"], dtype=object), confidence=np.array([0.4]))
    with pytest.raises(ValueError, match="Missing proposal NPZ fields"):
        ProposalBatch.load(missing)
