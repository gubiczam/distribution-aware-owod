"""Distribution-aware active-learning acquisition for contribution A."""

import csv
import json
import random
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from daowod.prob_adapter import ProposalBatch

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Strategy = Literal[
    "uncertainty",
    "uncertainty_novelty",
    "rarity",
    "rarity_coherence",
    "ungated_full",
    "full",
]
UncertaintyMode = Literal["ambiguity", "entropy", "margin"]
PseudoLabelSource = Literal["cluster", "predicted"]


@dataclass(frozen=True)
class AcquisitionWeights:
    uncertainty: float = 0.3
    novelty: float = 0.2
    rarity: float = 0.5
    coherence_power: float = 1.0
    rarity_power: float = 1.0

    def __post_init__(self) -> None:
        values = (self.uncertainty, self.novelty, self.rarity)
        if any(value < 0 for value in values):
            raise ValueError("Acquisition weights must be non-negative.")
        if sum(values) == 0:
            raise ValueError("At least one acquisition weight must be positive.")
        if self.coherence_power < 0 or self.rarity_power <= 0:
            raise ValueError("Invalid coherence or rarity power.")


@dataclass(frozen=True)
class AcquisitionResult:
    uncertainty: FloatArray
    novelty: FloatArray
    pseudo_labels: IntArray
    rarity: FloatArray
    coherence: FloatArray
    scores: FloatArray


def _vector(name: str, values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")
    return result


def _matrix(name: str, values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result


def _normalise_rows(values: ArrayLike) -> FloatArray:
    matrix = _matrix("embeddings", values)
    if matrix.shape[0] == 0:
        return matrix.copy()
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def _minmax(values: ArrayLike) -> FloatArray:
    result = _vector("values", values)
    if result.size == 0:
        return result.copy()
    low, high = float(result.min()), float(result.max())
    if high - low < 1e-12:
        return np.ones_like(result)
    return (result - low) / (high - low)


def compute_uncertainty(
    *,
    confidence: ArrayLike | None = None,
    posterior: ArrayLike | None = None,
    mode: UncertaintyMode = "ambiguity",
) -> FloatArray:
    """Compute uncertainty from confidence or class posteriors."""

    if mode == "ambiguity":
        if confidence is None:
            raise ValueError("ambiguity requires confidence values.")
        values = _vector("confidence", confidence)
        if np.any((values < 0) | (values > 1)):
            raise ValueError("confidence must be in [0, 1].")
        return 1.0 - np.abs(2.0 * values - 1.0)

    if posterior is None:
        raise ValueError(f"{mode} requires posterior probabilities.")
    probabilities = _matrix("posterior", posterior)
    if probabilities.shape[1] < 2 or np.any(probabilities < 0):
        raise ValueError("posterior must contain at least two non-negative classes.")
    mass = probabilities.sum(axis=1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("Every posterior row must have positive mass.")
    probabilities = probabilities / mass

    if mode == "entropy":
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0)
        return -terms.sum(axis=1) / np.log(probabilities.shape[1])
    if mode == "margin":
        ordered = np.sort(probabilities, axis=1)
        return 1.0 - (ordered[:, -1] - ordered[:, -2])
    raise ValueError(f"Unknown uncertainty mode: {mode}")


def compute_novelty(candidate_embeddings: ArrayLike, reference_embeddings: ArrayLike) -> FloatArray:
    """Distance from the nearest labelled reference proposal."""

    candidates = _normalise_rows(candidate_embeddings)
    references = _normalise_rows(reference_embeddings)
    if candidates.shape[1] != references.shape[1]:
        raise ValueError("Candidate and reference dimensions must match.")
    if candidates.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if references.shape[0] == 0:
        return np.ones(candidates.shape[0], dtype=np.float64)
    return _minmax(1.0 - (candidates @ references.T).max(axis=1))


def assign_pseudo_labels(
    embeddings: ArrayLike,
    *,
    source: PseudoLabelSource,
    cluster_count: int,
    seed: int,
    predicted_labels: ArrayLike | None = None,
) -> IntArray:
    """Estimate unknown classes before oracle annotation."""

    vectors = _matrix("embeddings", embeddings)
    count = vectors.shape[0]
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if source == "predicted":
        if predicted_labels is None:
            raise ValueError("Predicted pseudo-labels are unavailable.")
        labels = np.asarray(predicted_labels, dtype=np.int64)
        if labels.shape != (count,):
            raise ValueError("predicted_labels must match proposal count.")
        return labels
    if source != "cluster" or cluster_count < 1:
        raise ValueError("Invalid pseudo-label configuration.")
    return (
        KMeans(
            n_clusters=min(cluster_count, count),
            random_state=seed,
            n_init="auto",
        )
        .fit_predict(_normalise_rows(vectors))
        .astype(np.int64)
    )


def compute_rarity(pseudo_labels: ArrayLike, *, rarity_power: float = 1.0) -> FloatArray:
    """Estimate rarity as inverse pseudo-class frequency."""

    if rarity_power <= 0:
        raise ValueError("rarity_power must be positive.")
    labels = np.asarray(pseudo_labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("pseudo_labels must be one-dimensional.")
    if labels.size == 0:
        return np.empty(0, dtype=np.float64)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return _minmax(np.power(counts[inverse].astype(np.float64), -rarity_power))


def compute_coherence(embeddings: ArrayLike, *, neighbour_count: int = 5) -> FloatArray:
    """Local support measured from the k-th nearest-neighbour distance."""

    vectors = _normalise_rows(embeddings)
    count = vectors.shape[0]
    if neighbour_count < 1:
        raise ValueError("neighbour_count must be positive.")
    if count == 0:
        return np.empty(0, dtype=np.float64)
    if count == 1:
        return np.zeros(1, dtype=np.float64)
    k = min(neighbour_count, count - 1)
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(vectors).kneighbors(vectors)
    kth_distance = distances[:, k]
    scale = max(float(np.median(kth_distance)), 1e-12)
    return np.clip(1.0 / (1.0 + kth_distance / scale), 0.0, 1.0)


def compute_proposal_scores(
    *,
    strategy: Strategy,
    uncertainty: ArrayLike,
    novelty: ArrayLike,
    rarity: ArrayLike,
    coherence: ArrayLike,
    weights: AcquisitionWeights,
) -> FloatArray:
    """Apply score = alpha*u + beta*n + gamma*r*coherence**p."""

    u, n, r, c = (
        _vector("uncertainty", uncertainty),
        _vector("novelty", novelty),
        _vector("rarity", rarity),
        _vector("coherence", coherence),
    )
    if not (u.shape == n.shape == r.shape == c.shape):
        raise ValueError("All acquisition components must have equal shape.")
    if strategy == "uncertainty":
        return u
    if strategy == "uncertainty_novelty":
        denominator = weights.uncertainty + weights.novelty
        if denominator == 0:
            raise ValueError("Uncertainty or novelty weight must be positive.")
        return (weights.uncertainty * u + weights.novelty * n) / denominator
    if strategy == "rarity":
        return r

    gated_rarity = r * np.power(c, weights.coherence_power)
    if strategy == "rarity_coherence":
        return gated_rarity
    if strategy == "ungated_full":
        return weights.uncertainty * u + weights.novelty * n + weights.rarity * r
    if strategy == "full":
        return weights.uncertainty * u + weights.novelty * n + weights.rarity * gated_rarity
    raise ValueError(f"Unknown strategy: {strategy}")


def score_proposals(
    *,
    strategy: Strategy,
    uncertainty_mode: UncertaintyMode,
    pseudo_label_source: PseudoLabelSource,
    confidence: ArrayLike,
    posterior: ArrayLike | None,
    embeddings: ArrayLike,
    reference_embeddings: ArrayLike,
    predicted_labels: ArrayLike | None,
    cluster_count: int,
    neighbour_count: int,
    seed: int,
    weights: AcquisitionWeights,
) -> AcquisitionResult:
    """Run the complete proposal-scoring pipeline."""

    uncertainty = compute_uncertainty(
        confidence=confidence,
        posterior=posterior,
        mode=uncertainty_mode,
    )
    novelty = compute_novelty(embeddings, reference_embeddings)
    pseudo_labels = assign_pseudo_labels(
        embeddings,
        source=pseudo_label_source,
        cluster_count=cluster_count,
        seed=seed,
        predicted_labels=predicted_labels,
    )
    rarity = compute_rarity(pseudo_labels, rarity_power=weights.rarity_power)
    coherence = compute_coherence(embeddings, neighbour_count=neighbour_count)
    scores = compute_proposal_scores(
        strategy=strategy,
        uncertainty=uncertainty,
        novelty=novelty,
        rarity=rarity,
        coherence=coherence,
        weights=weights,
    )
    return AcquisitionResult(uncertainty, novelty, pseudo_labels, rarity, coherence, scores)


def aggregate_image_scores(
    image_ids: ArrayLike,
    proposal_scores: ArrayLike,
    *,
    top_k: int = 3,
) -> dict[Hashable, float]:
    """Average the top-k proposal scores of each image."""

    ids = np.asarray(image_ids, dtype=object)
    scores = _vector("proposal_scores", proposal_scores)
    if ids.ndim != 1 or ids.shape[0] != scores.shape[0] or top_k < 1:
        raise ValueError("Invalid image aggregation inputs.")
    result: dict[Hashable, float] = {}
    for image_id in dict.fromkeys(ids.tolist()):
        result[image_id] = float(np.sort(scores[ids == image_id])[-top_k:].mean())
    return result


def select_images(
    image_ids: ArrayLike,
    proposal_scores: ArrayLike,
    *,
    budget: int,
    top_k: int = 3,
) -> list[Hashable]:
    """Select highest-scoring images under a fixed annotation budget."""

    if budget < 1:
        raise ValueError("budget must be positive.")
    image_scores = aggregate_image_scores(image_ids, proposal_scores, top_k=top_k)
    return sorted(
        image_scores,
        key=lambda image_id: (-image_scores[image_id], str(image_id)),
    )[:budget]


def _offline_strategy_scores(
    strategy: str,
    result: AcquisitionResult,
    weights: AcquisitionWeights,
) -> FloatArray:
    if strategy == "uncertainty":
        return result.uncertainty
    if strategy == "uncertainty_novelty":
        return weights.uncertainty * result.uncertainty + weights.novelty * result.novelty
    if strategy == "rarity_no_coherence":
        return (
            weights.uncertainty * result.uncertainty
            + weights.novelty * result.novelty
            + weights.rarity * result.rarity
        )
    if strategy == "full":
        return result.scores
    raise ValueError(f"Unknown acquisition strategy: {strategy}")


def _top_k_indices(
    image_ids: ArrayLike,
    proposal_scores: ArrayLike,
    selected_image_ids: list[Hashable],
    *,
    top_k: int,
) -> IntArray:
    ids = np.asarray(image_ids, dtype=object)
    scores = _vector("proposal_scores", proposal_scores)
    selected: list[int] = []
    for image_id in selected_image_ids:
        indices = np.flatnonzero(ids == image_id)
        ordered = sorted(indices.tolist(), key=lambda index: (-scores[index], index))
        selected.extend(ordered[:top_k])
    return np.asarray(selected, dtype=np.int64)


def compare_acquisition_strategies(
    candidate_proposals: str | Path | ProposalBatch,
    reference_proposals: str | Path | ProposalBatch,
    *,
    strategies: tuple[str, ...] | list[str],
    budget: int,
    seed: int,
    acquisition_config: object,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Compare offline acquisition strategies from exported proposal NPZ files."""

    allowed = {
        "random",
        "uncertainty",
        "uncertainty_novelty",
        "rarity_no_coherence",
        "full",
    }
    strategy_names = tuple(str(strategy) for strategy in strategies)
    invalid = set(strategy_names) - allowed
    if invalid:
        raise ValueError(f"Unknown acquisition strategies: {sorted(invalid)}")
    if not strategy_names:
        raise ValueError("At least one acquisition strategy is required.")
    if budget < 1:
        raise ValueError("budget must be positive.")

    candidates = (
        candidate_proposals
        if isinstance(candidate_proposals, ProposalBatch)
        else ProposalBatch.load(candidate_proposals)
    )
    references = (
        reference_proposals
        if isinstance(reference_proposals, ProposalBatch)
        else ProposalBatch.load(reference_proposals)
    )
    image_ids = candidates.image_ids
    unique_image_ids = list(dict.fromkeys(image_ids.tolist()))
    if budget > len(unique_image_ids):
        raise ValueError("budget must not exceed candidate image count.")

    weights = getattr(acquisition_config, "weights", AcquisitionWeights())
    if not isinstance(weights, AcquisitionWeights):
        raise ValueError("acquisition_config.weights must be an AcquisitionWeights instance.")
    top_k = int(getattr(acquisition_config, "top_k", 3))
    base = score_proposals(
        strategy="full",
        uncertainty_mode=str(getattr(acquisition_config, "uncertainty_mode", "ambiguity")),
        pseudo_label_source=str(getattr(acquisition_config, "pseudo_label_source", "cluster")),
        confidence=candidates.confidence,
        posterior=candidates.posterior,
        embeddings=candidates.embeddings,
        reference_embeddings=references.embeddings,
        predicted_labels=candidates.predicted_labels,
        cluster_count=int(getattr(acquisition_config, "cluster_count", 20)),
        neighbour_count=int(getattr(acquisition_config, "neighbour_count", 5)),
        seed=seed,
        weights=weights,
    )

    summary_fields = [
        "strategy",
        "selected_image_count",
        "mean_uncertainty",
        "mean_novelty",
        "mean_rarity",
        "mean_coherence",
        "mean_rarity_bonus",
        "unique_pseudo_classes",
        "pseudo_class_entropy",
        "low_coherence_fraction",
        "overlap_with_full",
    ]
    summary: list[dict[str, object]] = []
    selected_ids: dict[str, list[str]] = {}
    selected_sets: dict[str, set[str]] = {}
    low_coherence_threshold = float(np.quantile(base.coherence, 0.1))

    for strategy in strategy_names:
        if strategy == "random":
            selected = unique_image_ids.copy()
            random.Random(seed).shuffle(selected)
            selected = [str(image_id) for image_id in selected[:budget]]
            # Random has no proposal score; use uncertainty-ranked proposals for diagnostics.
            diagnostic_scores = base.uncertainty
            rarity_bonus = np.zeros_like(base.rarity)
        else:
            diagnostic_scores = _offline_strategy_scores(strategy, base, weights)
            selected = [
                str(image_id)
                for image_id in select_images(
                    image_ids,
                    diagnostic_scores,
                    budget=budget,
                    top_k=top_k,
                )
            ]
            rarity_bonus = (
                base.rarity * np.power(base.coherence, weights.coherence_power)
                if strategy == "full"
                else base.rarity
                if strategy == "rarity_no_coherence"
                else np.zeros_like(base.rarity)
            )

        selected_ids[strategy] = selected
        selected_sets[strategy] = set(selected)
        selected_indices = _top_k_indices(
            image_ids,
            diagnostic_scores,
            selected,
            top_k=top_k,
        )
        selected_labels = base.pseudo_labels[selected_indices]
        _, counts = np.unique(selected_labels, return_counts=True)
        probabilities = counts / counts.sum() if counts.size else np.empty(0, dtype=np.float64)
        selected_coherence = base.coherence[selected_indices]

        def mean(values: ArrayLike) -> float:
            array = np.asarray(values)
            return float(array.mean()) if array.size else 0.0

        summary.append(
            {
                "strategy": strategy,
                "selected_image_count": len(selected),
                "mean_uncertainty": mean(base.uncertainty[selected_indices]),
                "mean_novelty": mean(base.novelty[selected_indices]),
                "mean_rarity": mean(base.rarity[selected_indices]),
                "mean_coherence": mean(selected_coherence),
                "mean_rarity_bonus": mean(rarity_bonus[selected_indices]),
                "unique_pseudo_classes": int(counts.size),
                "pseudo_class_entropy": float(-(probabilities * np.log(probabilities)).sum())
                if probabilities.size
                else 0.0,
                "low_coherence_fraction": mean(selected_coherence <= low_coherence_threshold),
                "overlap_with_full": 0,
            }
        )

    full_set = selected_sets.get("full", set())
    for row in summary:
        row["overlap_with_full"] = len(selected_sets[str(row["strategy"])] & full_set)

    overlap_matrix = {
        left: {right: len(selected_sets[left] & selected_sets[right]) for right in strategy_names}
        for left in strategy_names
    }

    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "strategy_summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=summary_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary)
        (directory / "selected_ids.json").write_text(
            json.dumps(selected_ids, indent=2) + "\n",
            encoding="utf-8",
        )
        with (directory / "overlap_matrix.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["strategy", *strategy_names],
                lineterminator="\n",
            )
            writer.writeheader()
            for strategy in strategy_names:
                writer.writerow({"strategy": strategy, **overlap_matrix[strategy]})

    return {
        "summary": summary,
        "selected_ids": selected_ids,
        "overlap_matrix": overlap_matrix,
    }
