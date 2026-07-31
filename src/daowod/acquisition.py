"""Backward-compatibility shim for the pre-audit acquisition API.

Every function here delegates to :mod:`daowod.components`,
:mod:`daowod.normalisation` and :mod:`daowod.scoring`. There is no second
implementation of any formula: the weighted sums all route through
:func:`daowod.scoring.combine_components`, and the component maths lives in
:mod:`daowod.components`. What this module still owns is the *version-1
conventions* — which component is min-maxed, which is not, and which legacy entry
point weight-normalises — so previously published numbers stay reproducible.

New code should use :func:`daowod.scoring.score_pool` with a
:class:`~daowod.scoring.StrategySpec`. See ``docs/migration_strategies.md``.

One historical quirk is preserved deliberately: ``compute_proposal_scores`` and
``_offline_strategy_scores`` disagree about ``uncertainty_novelty``. The online
formula divides by ``alpha + beta``; the offline one does not. That drift is what
the audit found, and both are kept under their own names rather than silently
unified, because published offline comparisons used the un-normalised form.
"""

import csv
import json
import random
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from daowod import components
from daowod.normalisation import normalise
from daowod.prob_adapter import ProposalBatch
from daowod.scoring import StrategySpec, combine_components
from daowod.scoring import aggregate_image_scores as _canonical_aggregate

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

#: Version-1 normalisation conventions: uncertainty and coherence raw, novelty
#: and rarity min-maxed. Recorded once, used by every legacy entry point.
LEGACY_COMPONENT_NORMALISATION: dict[str, str] = {
    "uncertainty": "none",
    "novelty": "minmax",
    "rarity": "minmax",
    "coherence": "none",
    "gated": "none",
}
_UNCERTAINTY_MODE_TO_METHOD: dict[str, str] = {
    "ambiguity": "legacy_prob_score",
    "entropy": "entropy",
    "margin": "margin",
}


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
    return components.as_vector(name, values)


def _matrix(name: str, values: ArrayLike) -> FloatArray:
    return components.as_matrix(name, values)


def _normalise_rows(values: ArrayLike) -> FloatArray:
    return components.normalise_rows(values)


def _minmax(values: ArrayLike) -> FloatArray:
    return normalise(components.as_vector("values", values), "minmax")


def _legacy_spec(
    name: str,
    weights: AcquisitionWeights,
    *,
    uncertainty_mode: str = "ambiguity",
    pseudo_label_source: str = "cluster",
    cluster_count: int = 20,
    neighbour_count: int = 5,
    top_k: int = 3,
    weight_normalise_uncertainty_novelty: bool = True,
) -> StrategySpec:
    """Build the version-1 spec for one legacy strategy name."""

    method = _UNCERTAINTY_MODE_TO_METHOD.get(uncertainty_mode)
    if method is None:
        raise ValueError(f"Unknown uncertainty mode: {uncertainty_mode}")

    shared: dict[str, object] = {
        "semantics_version": 1,
        "uncertainty_method": method,
        "rarity_method": "inverse_frequency",
        "rarity_power": weights.rarity_power,
        "coherence_method": "density",
        "coherence_exponent": weights.coherence_power,
        "normalisation": "minmax",
        "component_normalisation": LEGACY_COMPONENT_NORMALISATION,
        "pseudo_label_source": pseudo_label_source,
        "cluster_count": cluster_count,
        "neighbour_count": neighbour_count,
        "image_aggregation": "top_k_mean",
        "top_k": top_k,
    }
    if name == "random":
        return StrategySpec(name="random", semantics_version=1, random_selection=True)
    if name == "uncertainty":
        return StrategySpec(name=name, uncertainty_weight=1.0, **shared)
    if name == "uncertainty_novelty":
        if weights.uncertainty + weights.novelty == 0:
            raise ValueError("Uncertainty or novelty weight must be positive.")
        return StrategySpec(
            name=name,
            uncertainty_weight=weights.uncertainty,
            novelty_weight=weights.novelty,
            weight_normalise=weight_normalise_uncertainty_novelty,
            **shared,
        )
    if name == "rarity":
        return StrategySpec(name=name, rarity_weight=1.0, **shared)
    if name == "rarity_coherence":
        return StrategySpec(name=name, gated_weight=1.0, **shared)
    if name in ("ungated_full", "rarity_no_coherence"):
        return StrategySpec(
            name=name,
            uncertainty_weight=weights.uncertainty,
            novelty_weight=weights.novelty,
            rarity_weight=weights.rarity,
            **shared,
        )
    if name == "full":
        return StrategySpec(
            name=name,
            uncertainty_weight=weights.uncertainty,
            novelty_weight=weights.novelty,
            gated_weight=weights.rarity,
            **shared,
        )
    raise ValueError(f"Unknown strategy: {name}")


def compute_uncertainty(
    *,
    confidence: ArrayLike | None = None,
    posterior: ArrayLike | None = None,
    mode: UncertaintyMode = "ambiguity",
) -> FloatArray:
    """Compute uncertainty from confidence or class posteriors (version 1).

    ``mode='ambiguity'`` is ``1 - |2c - 1|``, which the audit showed is a monotone
    rescaling of the PROB unknown score. New code should use
    :func:`daowod.components.compute_uncertainty` with ``method='entropy'``.
    """

    method = _UNCERTAINTY_MODE_TO_METHOD.get(mode)
    if method is None:
        raise ValueError(f"Unknown uncertainty mode: {mode}")
    if method == "legacy_prob_score" and confidence is None:
        raise ValueError("ambiguity requires confidence values.")
    if method != "legacy_prob_score" and posterior is None:
        raise ValueError(f"{mode} requires posterior probabilities.")
    return components.compute_uncertainty(method=method, posterior=posterior, confidence=confidence)


def compute_novelty(candidate_embeddings: ArrayLike, reference_embeddings: ArrayLike) -> FloatArray:
    """Min-maxed distance from the nearest labelled reference proposal."""

    raw = components.compute_novelty(candidate_embeddings, reference_embeddings)
    if raw.size == 0:
        return raw
    references = components.normalise_rows(reference_embeddings)
    if references.shape[0] == 0:
        return raw
    return normalise(raw, "minmax")


def assign_pseudo_labels(
    embeddings: ArrayLike,
    *,
    source: PseudoLabelSource,
    cluster_count: int,
    seed: int,
    predicted_labels: ArrayLike | None = None,
) -> IntArray:
    """Estimate unknown classes before oracle annotation."""

    if source == "predicted" and predicted_labels is None:
        raise ValueError("Predicted pseudo-labels are unavailable.")
    if source not in components.PSEUDO_LABEL_SOURCES or (source == "cluster" and cluster_count < 1):
        raise ValueError("Invalid pseudo-label configuration.")
    return components.assign_pseudo_labels(
        embeddings,
        source=source,
        cluster_count=cluster_count,
        seed=seed,
        predicted_labels=predicted_labels,
    )


def compute_rarity(pseudo_labels: ArrayLike, *, rarity_power: float = 1.0) -> FloatArray:
    """Min-maxed inverse pseudo-class frequency (version 1)."""

    raw = components.compute_rarity(
        pseudo_labels, method="inverse_frequency", rarity_power=rarity_power
    )
    return raw if raw.size == 0 else normalise(raw, "minmax")


def compute_coherence(embeddings: ArrayLike, *, neighbour_count: int = 5) -> FloatArray:
    """Absolute local density from the k-th nearest-neighbour distance (version 1)."""

    return components.compute_coherence(
        embeddings, method="density", neighbour_count=neighbour_count
    ).coherence


def compute_proposal_scores(
    *,
    strategy: Strategy,
    uncertainty: ArrayLike,
    novelty: ArrayLike,
    rarity: ArrayLike,
    coherence: ArrayLike,
    weights: AcquisitionWeights,
) -> FloatArray:
    """Apply score = alpha*u + beta*n + gamma*r*coherence**p (version 1)."""

    values = {
        "uncertainty": _vector("uncertainty", uncertainty),
        "novelty": _vector("novelty", novelty),
        "rarity": _vector("rarity", rarity),
        "coherence": _vector("coherence", coherence),
    }
    if len({array.shape for array in values.values()}) > 1:
        raise ValueError("All acquisition components must have equal shape.")
    values["gated"] = values["rarity"] * np.power(values["coherence"], weights.coherence_power)
    return combine_components(_legacy_spec(strategy, weights), values)


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
    """Run the complete version-1 proposal-scoring pipeline."""

    uncertainty = compute_uncertainty(
        confidence=confidence, posterior=posterior, mode=uncertainty_mode
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
    """Average the top-k proposal scores of each image, in first-appearance order."""

    ids = np.asarray(image_ids, dtype=object)
    scores = _vector("proposal_scores", proposal_scores)
    if ids.ndim != 1 or ids.shape[0] != scores.shape[0] or top_k < 1:
        raise ValueError("Invalid image aggregation inputs.")
    canonical = _canonical_aggregate(ids, scores, method="top_k_mean", top_k=top_k)
    # The pre-audit function returned first-appearance order; preserved because
    # callers iterate the mapping directly.
    return {image_id: canonical[str(image_id)] for image_id in dict.fromkeys(ids.tolist())}


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
    """Offline strategy formulas, including the historical un-normalised form.

    ``uncertainty_novelty`` here is ``alpha*u + beta*n`` with no division by
    ``alpha + beta``, unlike :func:`compute_proposal_scores`. That divergence is
    the pre-audit behaviour and is preserved so published offline comparisons
    remain reproducible.
    """

    if strategy not in (
        "uncertainty",
        "uncertainty_novelty",
        "rarity_no_coherence",
        "full",
    ):
        raise ValueError(f"Unknown acquisition strategy: {strategy}")
    spec = _legacy_spec(
        strategy,
        weights,
        weight_normalise_uncertainty_novelty=False,
    )
    return combine_components(
        spec,
        {
            "uncertainty": result.uncertainty,
            "novelty": result.novelty,
            "rarity": result.rarity,
            "coherence": result.coherence,
            "gated": result.rarity * np.power(result.coherence, weights.coherence_power),
        },
    )


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
    """Compare version-1 offline acquisition strategies from proposal NPZ files.

    Superseded by :mod:`daowod.offline`, which supports every registry strategy
    and multiple seeds. Retained because published pilot comparisons used it.
    """

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

    weights = getattr(acquisition_config, "weights", None) or AcquisitionWeights()
    if not isinstance(weights, AcquisitionWeights):
        raise ValueError("acquisition_config.weights must be an AcquisitionWeights instance.")

    # A configuration field left unset must fall back to the version-1 default,
    # never be stringified into "None".
    def setting(name: str, default: object) -> object:
        value = getattr(acquisition_config, name, None)
        return default if value is None else value

    top_k = int(setting("top_k", 3))
    uncertainty_mode = str(setting("uncertainty_mode", "ambiguity"))
    pseudo_label_source = str(setting("pseudo_label_source", "cluster"))
    cluster_count = int(setting("cluster_count", 20))
    neighbour_count = int(setting("neighbour_count", 5))

    base = score_proposals(
        strategy="full",
        uncertainty_mode=uncertainty_mode,  # type: ignore[arg-type]
        pseudo_label_source=pseudo_label_source,  # type: ignore[arg-type]
        confidence=candidates.confidence,
        posterior=candidates.posterior,
        embeddings=candidates.embeddings,
        reference_embeddings=references.embeddings,
        predicted_labels=candidates.predicted_labels,
        cluster_count=cluster_count,
        neighbour_count=neighbour_count,
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


__all__ = [
    "AcquisitionResult",
    "AcquisitionWeights",
    "PseudoLabelSource",
    "Strategy",
    "UncertaintyMode",
    "aggregate_image_scores",
    "assign_pseudo_labels",
    "compare_acquisition_strategies",
    "compute_coherence",
    "compute_novelty",
    "compute_proposal_scores",
    "compute_rarity",
    "compute_uncertainty",
    "score_proposals",
    "select_images",
]
