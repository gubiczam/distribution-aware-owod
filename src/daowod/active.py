"""Proposal-level, multi-round offline active-annotation simulation.

Why this is not :mod:`daowod.experiment`
----------------------------------------
``experiment.py`` spends an *image* budget and calls PROB to retrain between
rounds. Contribution A's score ``s(x)`` is defined on a candidate *region*, and
the quantity the plan wants on the x-axis is annotation cost. One annotated
region is one unit of oracle work, so the budget here counts proposals and the
loop never touches a GPU: a single cached PROB export supports every strategy,
seed and ablation, which is what makes 3 seeds x 5 strategies x 2 severities
affordable inside a Colab session.

What the oracle may and may not tell the acquisition
---------------------------------------------------
After a round, the oracle reveals the true class of the proposals *that were
annotated*. That is not leakage — it is the definition of active learning, and
the plan explicitly asks for it ("fedd fel a kiválasztott proposal valódi
osztályát; frissítsd a megfigyelt pszeudoeloszlást"). Leakage would be letting
ground truth influence the score of a proposal that has **not** been annotated.
Two guards enforce the distinction:

* :func:`score_round` receives PROB arrays only; the oracle table is passed to
  :func:`reveal` afterwards, in a separate call;
* :func:`AcquisitionState.saturation_weights` derives its multiplier from
  *revealed* labels and cluster identity alone, and
  :func:`daowod.discovery.assert_selection_is_ground_truth_free` re-derives every
  strategy's ranking from the recorded component values to confirm the score
  actually used contains no oracle term.

Feedback applied between rounds
-------------------------------
1. **Feature bank growth.** Annotated proposals join the novelty reference bank,
   so a region resembling something already annotated stops looking novel. This
   is the mechanism that stops all strategies re-buying the same object.
2. **Discovery saturation.** A cluster (and, when revealed, a class) that has
   already consumed annotations is down-weighted, so ``rarity`` means "rare *and
   not yet bought*" rather than "rare". Applied identically to every
   rarity-using strategy, so no strategy gains an unfair update rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from daowod.normalisation import normalise
from daowod.revealed import RevealedBank, anchored_rarity, support
from daowod.scoring import ScoringResult, StrategySpec, score_pool

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ObjectArray = NDArray[np.object_]

SaturationMode = Literal["none", "cluster", "cluster_and_class"]
SATURATION_MODES: tuple[str, ...] = ("none", "cluster", "cluster_and_class")


class AcquisitionError(ValueError):
    """Raised when an acquisition protocol is inconsistent."""


@dataclass(frozen=True)
class ProposalPool:
    """The cached PROB export for one candidate pool. Contains no ground truth."""

    proposal_ids: ObjectArray
    image_ids: ObjectArray
    embeddings: FloatArray
    posterior: FloatArray
    confidence: FloatArray
    objectness: FloatArray
    predicted_labels: IntArray
    boxes_cxcywh: FloatArray

    def __post_init__(self) -> None:
        count = self.proposal_ids.shape[0]
        for name, array in (
            ("image_ids", self.image_ids),
            ("confidence", self.confidence),
            ("objectness", self.objectness),
            ("predicted_labels", self.predicted_labels),
        ):
            if array.shape != (count,):
                raise AcquisitionError(f"{name} must be a vector of length {count}.")
        for name, array in (
            ("embeddings", self.embeddings),
            ("posterior", self.posterior),
            ("boxes_cxcywh", self.boxes_cxcywh),
        ):
            if array.ndim != 2 or array.shape[0] != count:
                raise AcquisitionError(f"{name} must have {count} rows.")

    @property
    def size(self) -> int:
        return int(self.proposal_ids.shape[0])

    def subset(self, mask_or_indices: ArrayLike) -> ProposalPool:
        """A pool restricted to a mask or index array, preserving order."""

        selector = np.asarray(mask_or_indices)
        indices = (
            np.flatnonzero(selector) if selector.dtype == np.bool_ else selector.astype(np.int64)
        )
        return ProposalPool(
            proposal_ids=self.proposal_ids[indices],
            image_ids=self.image_ids[indices],
            embeddings=self.embeddings[indices],
            posterior=self.posterior[indices],
            confidence=self.confidence[indices],
            objectness=self.objectness[indices],
            predicted_labels=self.predicted_labels[indices],
            boxes_cxcywh=self.boxes_cxcywh[indices],
        )


@dataclass
class AcquisitionState:
    """Everything the loop carries between rounds.

    ``annotated`` is a mask over the pool. ``reference_embeddings`` starts as the
    fixed representation bank and grows with annotated proposals.
    """

    annotated: BoolArray
    reference_embeddings: FloatArray
    annotations_per_cluster: dict[int, int] = field(default_factory=dict)
    annotations_per_revealed_class: dict[str, int] = field(default_factory=dict)
    revealed_classes: set[str] = field(default_factory=set)
    rounds_completed: int = 0
    bank: RevealedBank = field(default_factory=RevealedBank)
    """Embeddings of annotated regions, split by oracle verdict.

    Filled only by :func:`reveal`, and only at annotated positions. This is what
    the label-anchored estimators in :mod:`daowod.revealed` read; they never see a
    ground-truth array themselves.
    """

    @property
    def annotated_count(self) -> int:
        return int(self.annotated.sum())

    def saturation_weights(
        self,
        pseudo_labels: ArrayLike,
        *,
        mode: SaturationMode,
        strength: float,
    ) -> FloatArray:
        """Down-weight regions whose neighbourhood already consumed annotations.

        ``1 / (1 + strength * n)`` where ``n`` is the number of annotations
        already spent in the proposal's own pseudo-cluster. With ``strength = 0``
        or ``mode = "none"`` this is exactly 1 everywhere, so the un-saturated
        behaviour remains available for ablation.

        Only cluster identity and *already revealed* counts enter; the ground
        truth of an unannotated proposal never does.
        """

        labels = np.asarray(pseudo_labels, dtype=np.int64)
        if mode == "none" or strength <= 0.0:
            return np.ones(labels.shape[0], dtype=np.float64)
        counts = np.array(
            [self.annotations_per_cluster.get(int(label), 0) for label in labels.tolist()],
            dtype=np.float64,
        )
        return 1.0 / (1.0 + float(strength) * counts)


def initial_state(*, pool_size: int, reference_embeddings: ArrayLike) -> AcquisitionState:
    """A fresh state with nothing annotated and the fixed reference bank in place."""

    references = np.asarray(reference_embeddings, dtype=np.float64)
    if references.ndim != 2:
        raise AcquisitionError("reference_embeddings must be a 2-D array.")
    return AcquisitionState(
        annotated=np.zeros(int(pool_size), dtype=np.bool_),
        reference_embeddings=references,
    )


@dataclass(frozen=True)
class RoundSelection:
    """One round's selection, plus the component values behind it."""

    round_index: int
    selected_indices: IntArray
    scores: FloatArray
    components: Mapping[str, FloatArray]
    pseudo_labels: IntArray
    cluster_sizes: IntArray
    isolated: BoolArray
    saturation: FloatArray
    diagnostics: Mapping[str, object]
    anchored: Mapping[str, object] = field(default_factory=dict)
    """What the label-anchored estimator had, per round; empty when unused.

    Kept per round because "the bank was still cold" and "the anchored term was
    uninformative" are different findings, and a report that cannot separate them
    would credit or blame the wrong thing.
    """


@dataclass(frozen=True)
class RoundScores:
    """One round's scoring output, aligned to the *full* pool where noted."""

    scores: FloatArray
    """Score per pool position; ``-inf`` at already-annotated positions."""

    result: ScoringResult
    """The raw scoring result over the available candidates only."""

    components: Mapping[str, FloatArray]
    """Normalised components over the available candidates, post-saturation."""

    saturation: FloatArray
    """Saturation multiplier over the available candidates."""

    available: IntArray
    """Pool positions the scores in :attr:`components` refer to."""

    anchored: Mapping[str, object] = field(default_factory=dict)
    """What the label-anchored estimator had to work with, empty when unused.

    Recorded per round because "the anchored term was cold" and "the anchored term
    was uninformative" are different findings and must not be confused in the
    report.
    """


def score_round(
    *,
    pool: ProposalPool,
    spec: StrategySpec,
    state: AcquisitionState,
    seed: int,
    saturation_mode: SaturationMode = "cluster",
    saturation_strength: float = 1.0,
    verify_components: bool = True,
) -> RoundScores:
    """Score the not-yet-annotated proposals. Ground-truth free by construction.

    Already-annotated positions score ``-inf`` so they can never be bought twice.
    The saturation multiplier scales the *distribution* part of the score (rarity
    and the gated interaction), not uncertainty or novelty, because saturation is
    a statement about how much of a class we already own.

    ``verify_components`` re-derives the returned scores from the returned
    components on every round. It is on by default because the cost is a few array
    operations against the value of never reporting a ranking that the recorded
    components cannot explain.
    """

    if saturation_mode not in SATURATION_MODES:
        raise AcquisitionError(
            f"Unknown saturation mode {saturation_mode!r}. Supported: {list(SATURATION_MODES)}"
        )
    available = np.flatnonzero(~state.annotated)
    if available.size == 0:
        raise AcquisitionError("Every proposal is already annotated.")
    candidates = pool.subset(available)

    result = score_pool(
        spec=spec,
        image_ids=candidates.image_ids,
        embeddings=candidates.embeddings,
        reference_embeddings=state.reference_embeddings,
        confidence=candidates.confidence,
        posterior=candidates.posterior,
        predicted_labels=candidates.predicted_labels,
        objectness=candidates.objectness,
        boxes_cxcywh=candidates.boxes_cxcywh,
        seed=seed,
        compute_all_components=True,
    )

    saturation = state.saturation_weights(
        result.pseudo_labels, mode=saturation_mode, strength=saturation_strength
    )
    weights = spec.weights()
    normalised = dict(result.normalised)
    anchored: dict[str, object] = {}

    if spec.distribution_estimator == "revealed":
        # The one variable this experiment changes: where the distribution-aware
        # term's rarity and coherence come from. Before any unknown has been
        # revealed both fall back to the unsupervised values, so a cold round is
        # bit-identical to the baseline and the contrast is attributable to the
        # labels rather than to a different cold-start policy.
        raw_support, support_report = support(
            candidates.embeddings,
            state.bank,
            neighbours=spec.support_neighbours,
            fallback=result.raw["coherence"],
        )
        raw_rarity, rarity_report = anchored_rarity(
            candidates.embeddings, state.bank, fallback=result.raw["rarity"]
        )
        normalised["coherence"] = normalise(raw_support, spec.normalisation_for("coherence"))
        normalised["rarity"] = normalise(raw_rarity, spec.normalisation_for("rarity"))
        # The gate keeps its multiplicative form: normalised rarity times the
        # support raised to the coherence exponent, renormalised, exactly as the
        # unsupervised path does.
        normalised["gated"] = normalise(
            normalised["rarity"] * np.power(raw_support, spec.coherence_exponent),
            spec.normalisation_for("gated"),
        )
        anchored = {"support": dict(support_report), "rarity": dict(rarity_report)}

    if saturation_mode != "none" and (weights["rarity"] > 0 or weights["gated"] > 0):
        adjusted = dict(normalised)
        # Re-normalise after saturation so the weights keep their intended
        # relative influence; scaling in place would shrink the term's range and
        # silently reduce gamma.
        for name in ("rarity", "gated"):
            if weights[name] > 0:
                adjusted[name] = normalise(
                    normalised[name] * saturation, spec.normalisation_for(name)
                )
        from daowod.scoring import combine_components

        candidate_scores = combine_components(spec, adjusted)
        components_used: Mapping[str, FloatArray] = adjusted
    elif spec.distribution_estimator == "revealed":
        from daowod.scoring import combine_components

        candidate_scores = combine_components(spec, normalised)
        components_used = normalised
    else:
        candidate_scores = result.scores
        components_used = dict(result.normalised)

    if verify_components:
        # Every round of every cell, not once per run: the components recorded for
        # the diagnostics must be exactly the ones that produced the ranking used
        # to spend the budget. If a term ever entered the score without being
        # recorded — an oracle term above all — this identity breaks and the
        # campaign stops instead of publishing a number nothing explains.
        from daowod.discovery import assert_selection_is_ground_truth_free

        assert_selection_is_ground_truth_free(
            scores=candidate_scores,
            components=components_used,
            spec_weights=weights,
        )

    scores = np.full(pool.size, -np.inf, dtype=np.float64)
    scores[available] = candidate_scores
    return RoundScores(
        scores=scores,
        result=result,
        components=components_used,
        saturation=saturation,
        available=available,
        anchored=anchored,
    )


def select_batch(
    scores: ArrayLike,
    *,
    batch_size: int,
    proposal_ids: ArrayLike,
    random_selection: bool = False,
    seed: int = 0,
) -> IntArray:
    """Top-``batch_size`` available proposals, deterministically tie-broken.

    Ties are broken by proposal ID, not array position, so two strategies that
    produce identical scores also produce identical selections regardless of how
    the pool was assembled. ``random_selection`` samples uniformly from whatever
    is still available, which is the ``v2:random`` control.
    """

    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray([str(value) for value in np.asarray(proposal_ids, dtype=object)], dtype=object)
    if ids.shape != values.shape:
        raise AcquisitionError("proposal_ids must be parallel to scores.")
    available = np.flatnonzero(np.isfinite(values))
    if available.size == 0:
        raise AcquisitionError("No available proposal to select.")
    take = int(min(batch_size, available.size))
    if random_selection:
        generator = np.random.default_rng(seed)
        chosen = generator.choice(available.size, size=take, replace=False)
        return np.sort(available[np.asarray(sorted(chosen.tolist()), dtype=np.int64)])
    order = np.lexsort((ids[available], -values[available]))
    return np.sort(available[order[:take]])


def reveal(
    state: AcquisitionState,
    *,
    selected: ArrayLike,
    pool: ProposalPool,
    pseudo_labels_full: ArrayLike,
    gt_class: ArrayLike,
    gt_is_unknown: ArrayLike,
    saturation_mode: SaturationMode = "cluster",
) -> AcquisitionState:
    """Apply the oracle's answer for the annotated proposals only.

    ``gt_class`` / ``gt_is_unknown`` are indexed by pool position but are read
    **exclusively** at ``selected`` positions. Any implementation that touched
    unselected positions would be leakage; the test suite asserts that the
    resulting state is identical when the unselected entries are scrambled.
    """

    indices = np.asarray(selected, dtype=np.int64)
    labels = np.asarray(pseudo_labels_full, dtype=np.int64)
    classes = np.asarray(gt_class, dtype=object)
    unknown_flags = np.asarray(gt_is_unknown, dtype=np.bool_)

    state.annotated[indices] = True
    for position in indices.tolist():
        cluster = int(labels[position])
        state.annotations_per_cluster[cluster] = state.annotations_per_cluster.get(cluster, 0) + 1
        if saturation_mode == "cluster_and_class" and unknown_flags[position]:
            name = str(classes[position])
            if name:
                state.annotations_per_revealed_class[name] = (
                    state.annotations_per_revealed_class.get(name, 0) + 1
                )
        if unknown_flags[position]:
            name = str(classes[position])
            if name:
                state.revealed_classes.add(name)
        # The label-anchored bank is filled here and only here: this loop runs over
        # ``selected`` alone, so an unannotated proposal's ground truth can never
        # reach the estimators that read the bank.
        state.bank.add(
            pool.embeddings[position],
            is_unknown=bool(unknown_flags[position]),
            class_name=str(classes[position]) if unknown_flags[position] else "",
        )
    state.reference_embeddings = np.concatenate(
        [state.reference_embeddings, pool.embeddings[indices]], axis=0
    )
    state.rounds_completed += 1
    return state


@dataclass(frozen=True)
class CampaignResult:
    """The full annotation trajectory for one (strategy, seed, severity) cell."""

    strategy: str
    seed: int
    imbalance_setting: str
    selection_order: IntArray
    round_boundaries: tuple[int, ...]
    rounds: tuple[RoundSelection, ...]
    saturation_mode: str

    def prefix(self, budget: int) -> IntArray:
        """The first ``budget`` annotated proposals, in acquisition order.

        Budget curves are prefixes of one trajectory rather than independent
        runs, so a strategy cannot look better at budget 100 by having made
        different choices at budget 50 — exactly the monotonicity a cumulative
        annotation cost implies.
        """

        if budget < 0:
            raise AcquisitionError("budget must be non-negative.")
        return self.selection_order[: int(budget)]


def run_campaign(
    *,
    pool: ProposalPool,
    spec: StrategySpec,
    reference_embeddings: ArrayLike,
    gt_class: ArrayLike,
    gt_is_unknown: ArrayLike,
    total_budget: int,
    rounds: int,
    seed: int,
    imbalance_setting: str = "natural",
    saturation_mode: SaturationMode = "cluster",
    saturation_strength: float = 1.0,
    keep_round_components: bool = True,
) -> CampaignResult:
    """Run one iterative annotation campaign to ``total_budget`` proposals.

    The budget is split into ``rounds`` equal batches (the remainder goes to the
    earliest rounds). Every round rescores the surviving pool, so the pseudo-class
    structure, rarity and coherence are re-estimated against a feature bank that
    has grown with the previous round's annotations — the iterative variant the
    plan prefers over a single ranking.
    """

    if total_budget < 1:
        raise AcquisitionError("total_budget must be positive.")
    if rounds < 1:
        raise AcquisitionError("rounds must be positive.")
    if total_budget > pool.size:
        raise AcquisitionError(f"total_budget {total_budget} exceeds the pool size {pool.size}.")

    base, remainder = divmod(int(total_budget), int(rounds))
    batch_sizes = [base + (1 if index < remainder else 0) for index in range(int(rounds))]
    batch_sizes = [size for size in batch_sizes if size > 0]

    state = initial_state(pool_size=pool.size, reference_embeddings=reference_embeddings)
    order: list[int] = []
    boundaries: list[int] = []
    collected: list[RoundSelection] = []

    for round_index, batch_size in enumerate(batch_sizes):
        round_seed = int(seed) * 1000 + round_index
        scored = score_round(
            pool=pool,
            spec=spec,
            state=state,
            seed=round_seed,
            saturation_mode=saturation_mode,
            saturation_strength=saturation_strength,
        )
        scores, result, saturation = scored.scores, scored.result, scored.saturation
        selected = select_batch(
            scores,
            batch_size=batch_size,
            proposal_ids=pool.proposal_ids,
            random_selection=spec.random_selection,
            seed=round_seed,
        )
        # Rank the batch internally so the acquisition *order* inside a round is
        # also score-driven; budget curves read prefixes of this order.
        internal = selected[np.argsort(-scores[selected], kind="stable")]

        available = scored.available
        labels_full = np.full(pool.size, -1, dtype=np.int64)
        labels_full[available] = result.pseudo_labels
        saturation_full = np.ones(pool.size, dtype=np.float64)
        saturation_full[available] = saturation

        if keep_round_components:
            component_view = {
                name: _scatter(values, available, pool.size)
                for name, values in scored.components.items()
            }
            collected.append(
                RoundSelection(
                    round_index=round_index,
                    selected_indices=internal,
                    scores=scores,
                    components=component_view,
                    pseudo_labels=labels_full,
                    cluster_sizes=_scatter_int(result.cluster_sizes, available, pool.size),
                    isolated=_scatter_bool(result.isolated, available, pool.size),
                    saturation=saturation_full,
                    diagnostics=dict(result.diagnostics),
                    anchored=dict(scored.anchored),
                )
            )

        order.extend(int(value) for value in internal.tolist())
        boundaries.append(len(order))
        state = reveal(
            state,
            selected=internal,
            pool=pool,
            pseudo_labels_full=labels_full,
            gt_class=gt_class,
            gt_is_unknown=gt_is_unknown,
            saturation_mode=saturation_mode,
        )

    return CampaignResult(
        strategy=spec.name,
        seed=int(seed),
        imbalance_setting=str(imbalance_setting),
        selection_order=np.asarray(order, dtype=np.int64),
        round_boundaries=tuple(boundaries),
        rounds=tuple(collected),
        saturation_mode=str(saturation_mode),
    )


def _scatter(values: FloatArray, positions: IntArray, size: int) -> FloatArray:
    full = np.full(size, np.nan, dtype=np.float64)
    full[positions] = values
    return full


def _scatter_int(values: IntArray, positions: IntArray, size: int) -> IntArray:
    full = np.full(size, -1, dtype=np.int64)
    full[positions] = values
    return full


def _scatter_bool(values: BoolArray, positions: IntArray, size: int) -> BoolArray:
    full = np.zeros(size, dtype=np.bool_)
    full[positions] = values
    return full


def resolve_round_count(*, total_budget: int, rounds: int, minimum_batch: int = 5) -> int:
    """Largest round count up to ``rounds`` that keeps batches usable.

    A campaign with more rounds than budget would produce empty batches; a batch
    of one makes the per-round rarity update meaningless. Reported so the
    notebook can state the round count it actually used.
    """

    if total_budget < 1:
        raise AcquisitionError("total_budget must be positive.")
    usable = max(1, min(int(rounds), int(total_budget) // max(int(minimum_batch), 1)))
    return usable


def selection_frame_rows(
    result: CampaignResult,
    *,
    pool: ProposalPool,
    budgets: Sequence[int],
) -> list[dict[str, object]]:
    """Rows for ``selected_proposals.csv``: which region each budget bought."""

    rows: list[dict[str, object]] = []
    largest = max(int(value) for value in budgets) if budgets else 0
    boundaries = list(result.round_boundaries)
    for rank, index in enumerate(result.selection_order[:largest].tolist()):
        round_index = next(
            (position for position, bound in enumerate(boundaries) if rank < bound),
            len(boundaries) - 1,
        )
        rows.append(
            {
                "strategy": result.strategy,
                "seed": result.seed,
                "imbalance_setting": result.imbalance_setting,
                "acquisition_rank": rank + 1,
                "round_index": round_index,
                "pool_index": int(index),
                "proposal_id": str(pool.proposal_ids[index]),
                "image_id": str(pool.image_ids[index]),
                "objectness": float(pool.objectness[index]),
                "unknown_score": float(pool.confidence[index]),
                "predicted_label": int(pool.predicted_labels[index]),
            }
        )
    return rows
