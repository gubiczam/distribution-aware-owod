"""The one canonical acquisition scorer and the one strategy registry.

Everything — the annotation study, the component audit, the representation
experiment, ablations and tests — scores proposals through :func:`score_pool`. An
audit once found this formula duplicated in four places and already drifted
(``uncertainty_novelty`` was weight-normalised in one copy and not in another);
representing strategies as declarative specs removes the possibility.

Canonical score, the proposal's equation (1)
--------------------------------------------
    S = w_u * U_hat + w_n * N_hat + w_r * R_hat + w_g * G_hat + w_c * C_hat

``G_hat`` is the normalised gated interaction ``normalise(R_hat * coherence**p)``
— the AND relation between rarity and local coherence that the proposal calls
indispensable. ``w_c`` extends the specified form so that a pure ``coherence``
baseline is expressible; with ``w_c = 0`` the score reduces exactly to equation (1).

Current science: posterior entropy for uncertainty, rank normalisation, and
scale-free within-cluster coherence. There is one semantics, so strategy names are
plain — ``full``, not ``full``. See :class:`StrategyRegistry`.
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from daowod import components
from daowod.normalisation import NORMALISATION_METHODS, normalise

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ObjectArray = NDArray[np.object_]

COMPONENT_NAMES: tuple[str, ...] = (
    "uncertainty",
    "novelty",
    "rarity",
    "coherence",
    "gated",
)

#: Where the distribution-aware term's rarity and coherence come from.
#:
#: ``cluster``  the original, fully unsupervised path: k-means pseudo-classes for
#:              rarity and pool density for coherence. This is the published
#:              baseline and stays the default, so no existing strategy changes.
#: ``revealed`` label-anchored: rarity from the nearest *revealed* class and
#:              coherence from similarity to regions the oracle has already
#:              confirmed as unknown objects. Falls back to ``cluster`` values
#:              until labels exist, so cold rounds are identical.
DistributionEstimator = Literal["cluster", "revealed"]
DISTRIBUTION_ESTIMATORS: tuple[str, ...] = ("cluster", "revealed")

ImageAggregation = Literal["top_k_mean", "max", "mean", "noisy_or"]
IMAGE_AGGREGATIONS: tuple[str, ...] = ("top_k_mean", "max", "mean", "noisy_or")


class StrategyError(ValueError):
    """Raised for an unknown, ambiguous or internally invalid strategy."""


@dataclass(frozen=True)
class StrategySpec:
    """A complete, declarative description of one acquisition strategy."""

    name: str
    uncertainty_weight: float = 0.0
    novelty_weight: float = 0.0
    rarity_weight: float = 0.0
    gated_weight: float = 0.0
    coherence_weight: float = 0.0
    coherence_exponent: float = 1.0
    uncertainty_method: str = "entropy"
    rarity_method: str = "log_inverse_frequency"
    rarity_power: float = 1.0
    coherence_method: str = "relative_within_cluster"
    normalisation: str = "rank"
    component_normalisation: Mapping[str, str] = field(default_factory=dict)
    pseudo_label_source: str = "cluster"
    cluster_count: int = 20
    neighbour_count: int = 5
    singleton_coherence: float = 0.0
    minimum_cluster_size: int = 3
    isolation_quantile: float = 0.9
    radius_quantile: float = 0.1
    minimum_samples: int = 4
    distribution_estimator: str = "cluster"
    support_neighbours: int = 5
    image_aggregation: str = "top_k_mean"
    top_k: int = 3
    random_selection: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise StrategyError("A strategy must have a name.")
        weights = self.weights()
        if any(value < 0 for value in weights.values()):
            raise StrategyError(f"{self.name}: weights must be non-negative.")
        if not self.random_selection and sum(weights.values()) <= 0:
            raise StrategyError(
                f"{self.name}: at least one component weight must be positive "
                "for a scored strategy."
            )
        if self.random_selection and sum(weights.values()) > 0:
            raise StrategyError(f"{self.name}: a random strategy must not carry component weights.")
        if self.coherence_exponent < 0:
            raise StrategyError(f"{self.name}: coherence_exponent must be >= 0.")
        if self.normalisation not in NORMALISATION_METHODS:
            raise StrategyError(f"{self.name}: unknown normalisation {self.normalisation!r}.")
        for component, method in self.component_normalisation.items():
            if component not in COMPONENT_NAMES:
                raise StrategyError(
                    f"{self.name}: component_normalisation names "
                    f"{component!r}, which is not one of {list(COMPONENT_NAMES)}."
                )
            if method not in NORMALISATION_METHODS:
                raise StrategyError(
                    f"{self.name}: unknown normalisation {method!r} for {component}."
                )
        if self.uncertainty_method not in components.UNCERTAINTY_METHODS:
            raise StrategyError(
                f"{self.name}: unknown uncertainty method {self.uncertainty_method!r}."
            )
        if self.rarity_method not in components.RARITY_METHODS:
            raise StrategyError(f"{self.name}: unknown rarity method {self.rarity_method!r}.")
        if self.coherence_method not in components.COHERENCE_METHODS:
            raise StrategyError(f"{self.name}: unknown coherence method {self.coherence_method!r}.")
        if self.pseudo_label_source not in components.PSEUDO_LABEL_SOURCES:
            raise StrategyError(
                f"{self.name}: unknown pseudo-label source {self.pseudo_label_source!r}."
            )
        if self.image_aggregation not in IMAGE_AGGREGATIONS:
            raise StrategyError(
                f"{self.name}: unknown image aggregation "
                f"{self.image_aggregation!r}. Supported: {list(IMAGE_AGGREGATIONS)}"
            )
        if min(self.cluster_count, self.neighbour_count, self.top_k) < 1:
            raise StrategyError(f"{self.name}: integer parameters must be positive.")
        if not 0.0 < self.radius_quantile < 1.0:
            raise StrategyError(f"{self.name}: radius_quantile must lie strictly in (0, 1).")
        if self.minimum_samples < 2:
            raise StrategyError(f"{self.name}: minimum_samples must be >= 2.")
        if self.distribution_estimator not in DISTRIBUTION_ESTIMATORS:
            raise StrategyError(
                f"{self.name}: unknown distribution_estimator "
                f"{self.distribution_estimator!r}. Supported: {list(DISTRIBUTION_ESTIMATORS)}"
            )
        if self.support_neighbours < 1:
            raise StrategyError(f"{self.name}: support_neighbours must be positive.")

    def weights(self) -> dict[str, float]:
        return {
            "uncertainty": float(self.uncertainty_weight),
            "novelty": float(self.novelty_weight),
            "rarity": float(self.rarity_weight),
            "gated": float(self.gated_weight),
            "coherence": float(self.coherence_weight),
        }

    def active_components(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.weights().items() if value > 0)

    def normalisation_for(self, component: str) -> str:
        return str(self.component_normalisation.get(component, self.normalisation))

    def needs_posterior(self) -> bool:
        return self.uncertainty_weight > 0 and self.uncertainty_method != "legacy_prob_score"

    def behaviour_key(self) -> tuple[object, ...]:
        """Everything that affects the numbers, and nothing that does not.

        Used to decide whether two same-named specs in different semantics
        versions are genuinely the same strategy. Prose fields are excluded: a
        reworded description must not make a name look ambiguous.
        """

        return (
            self.uncertainty_weight,
            self.novelty_weight,
            self.rarity_weight,
            self.gated_weight,
            self.coherence_weight,
            self.coherence_exponent,
            self.uncertainty_method,
            self.rarity_method,
            self.rarity_power,
            self.coherence_method,
            self.normalisation,
            tuple(sorted(self.component_normalisation.items())),
            self.pseudo_label_source,
            self.cluster_count,
            self.neighbour_count,
            self.singleton_coherence,
            self.minimum_cluster_size,
            self.isolation_quantile,
            self.radius_quantile,
            self.minimum_samples,
            self.distribution_estimator,
            self.support_neighbours,
            self.image_aggregation,
            self.top_k,
            self.random_selection,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "uncertainty_weight": self.uncertainty_weight,
            "novelty_weight": self.novelty_weight,
            "rarity_weight": self.rarity_weight,
            "gated_weight": self.gated_weight,
            "coherence_weight": self.coherence_weight,
            "coherence_exponent": self.coherence_exponent,
            "uncertainty_method": self.uncertainty_method,
            "rarity_method": self.rarity_method,
            "rarity_power": self.rarity_power,
            "coherence_method": self.coherence_method,
            "normalisation": self.normalisation,
            "component_normalisation": dict(self.component_normalisation),
            "pseudo_label_source": self.pseudo_label_source,
            "cluster_count": self.cluster_count,
            "neighbour_count": self.neighbour_count,
            "singleton_coherence": self.singleton_coherence,
            "minimum_cluster_size": self.minimum_cluster_size,
            "isolation_quantile": self.isolation_quantile,
            "radius_quantile": self.radius_quantile,
            "minimum_samples": self.minimum_samples,
            "distribution_estimator": self.distribution_estimator,
            "support_neighbours": self.support_neighbours,
            "image_aggregation": self.image_aggregation,
            "top_k": self.top_k,
            "random_selection": self.random_selection,
            "description": self.description,
        }


# --- the strategy registry ---------------------------------------------------
#: The reported weights for equation (1): alpha, beta, gamma.
ALPHA, BETA, GAMMA = 0.3, 0.2, 0.5

_SPECS: tuple[StrategySpec, ...] = (
    StrategySpec(
        name="random",
        random_selection=True,
        description="Uniform random image selection.",
    ),
    StrategySpec(
        name="uncertainty",
        uncertainty_weight=1.0,
        description="Normalised predictive entropy of the exported posterior.",
    ),
    StrategySpec(
        name="novelty",
        novelty_weight=1.0,
        description="Cosine distance to the nearest labelled reference proposal.",
    ),
    StrategySpec(
        name="rarity",
        rarity_weight=1.0,
        description="Rank-normalised log inverse pseudo-class frequency.",
    ),
    StrategySpec(
        name="coherence",
        coherence_weight=1.0,
        description="Scale-free within-cluster coherence, on its own.",
    ),
    StrategySpec(
        name="uncertainty_novelty",
        uncertainty_weight=ALPHA,
        novelty_weight=BETA,
        description="Entropy + novelty.",
    ),
    StrategySpec(
        name="uncertainty_rarity",
        uncertainty_weight=ALPHA,
        rarity_weight=GAMMA,
        description="Entropy + rarity, no coherence gate.",
    ),
    StrategySpec(
        name="uncertainty_coherence",
        uncertainty_weight=ALPHA,
        coherence_weight=GAMMA,
        description="Entropy + coherence, no rarity.",
    ),
    StrategySpec(
        name="rarity_coherence",
        gated_weight=1.0,
        description="The gated interaction rarity * coherence**p on its own.",
    ),
    StrategySpec(
        name="rarity_plus_coherence",
        rarity_weight=0.5,
        coherence_weight=0.5,
        description="Additive rarity + coherence, for contrast with the gate.",
    ),
    StrategySpec(
        name="full",
        uncertainty_weight=ALPHA,
        novelty_weight=BETA,
        gated_weight=GAMMA,
        description="Contribution A: entropy + novelty + gated rarity.",
    ),
    StrategySpec(
        name="full_no_coherence",
        uncertainty_weight=ALPHA,
        novelty_weight=BETA,
        rarity_weight=GAMMA,
        description="Full with the coherence gate removed (rarity ungated).",
    ),
    StrategySpec(
        name="full_no_rarity",
        uncertainty_weight=ALPHA,
        novelty_weight=BETA,
        coherence_weight=GAMMA,
        description="Full with rarity removed, coherence kept additively.",
    ),
    StrategySpec(
        name="full_no_uncertainty",
        novelty_weight=BETA,
        gated_weight=GAMMA,
        description="Full with the uncertainty term removed.",
    ),
    StrategySpec(
        name="full_no_novelty",
        uncertainty_weight=ALPHA,
        gated_weight=GAMMA,
        description="Full with the novelty term removed; matches the written "
        "proposal S = U + gamma * w(c) * coh.",
    ),
    StrategySpec(
        name="proposal_formula",
        uncertainty_weight=1.0,
        rarity_weight=GAMMA,
        gated_weight=GAMMA,
        description="The formula exactly as written in the research proposal: "
        "S = U + lambda*D + gamma*w(c)*coh, i.e. both an ungated and a gated "
        "distribution term.",
    ),
    # --- informativeness prior ------------------------------------------------
    # The audit's control arm. Measured on the real Task-1 pool, objectness x box
    # scale reaches ROC-AUC 0.777 for unknown-versus-background where every
    # semantic component sits near 0.48, and a static sort by it finds 85 unknown
    # objects in a 2 000-region budget against 16 for full. A distribution-aware
    # method that does not beat this is not earning its complexity, so it is an arm
    # of the comparison rather than a remark in the discussion.
    StrategySpec(
        name="objectness_area_prior",
        uncertainty_weight=1.0,
        uncertainty_method="objectness_area_prior",
        description="Informativeness prior only: objectness x box scale. The "
        "free-heuristic control every semantic strategy is read against.",
    ),
    StrategySpec(
        name="prior_full",
        uncertainty_weight=ALPHA,
        novelty_weight=BETA,
        gated_weight=GAMMA,
        uncertainty_method="objectness_area_prior",
        description="Contribution A's composition with the informativeness prior in "
        "the U slot: does distribution-awareness add anything on top of a prior "
        "that already works?",
    ),
    StrategySpec(
        name="prior_revealed_full",
        uncertainty_weight=ALPHA,
        novelty_weight=BETA,
        gated_weight=GAMMA,
        uncertainty_method="objectness_area_prior",
        distribution_estimator="revealed",
        description="The informativeness prior plus the label-anchored "
        "distribution term: the strongest combination the audit motivates.",
    ),
    # --- label-anchored distribution estimation ------------------------------
    # Same formula, same weights, same gate form as full. The single changed
    # variable is where rarity and coherence come from: the regions the oracle has
    # already confirmed, rather than k-means over a pool that is 75 % background.
    # Motivated by the measured 0.35 ROC-AUC gap between what the decoder features
    # support (supervised probe 0.837) and what the unsupervised estimators extract
    # (0.44-0.49); see daowod.revealed.
    StrategySpec(
        name="revealed_full",
        uncertainty_weight=ALPHA,
        novelty_weight=BETA,
        gated_weight=GAMMA,
        distribution_estimator="revealed",
        description="Contribution A with a label-anchored distribution term: "
        "entropy + novelty + gated (revealed-class rarity x revealed-unknown "
        "support).",
    ),
    StrategySpec(
        name="revealed_no_gate",
        uncertainty_weight=ALPHA,
        novelty_weight=BETA,
        rarity_weight=GAMMA,
        distribution_estimator="revealed",
        description="Label-anchored rarity without the support gate, isolating "
        "what the gate's form contributes once rarity is anchored.",
    ),
    StrategySpec(
        name="revealed_support_only",
        uncertainty_weight=ALPHA,
        novelty_weight=BETA,
        coherence_weight=GAMMA,
        distribution_estimator="revealed",
        description="Revealed-unknown support with no rarity term, isolating how "
        "much of any gain is 'resembles a confirmed unknown' rather than "
        "'belongs to a rare class'.",
    ),
)


class StrategyRegistry:
    """The single authoritative source of strategy definitions.

    One flat namespace, because equation (1) is one formula. An earlier version
    carried two semantics versions side by side and required a ``v1:``/``v2:``
    prefix to disambiguate; the pre-audit definitions it preserved reproduced
    numbers from a pool later disqualified as a subset of the evaluation split, so
    there was nothing publishable to reproduce. They are in git history at tag
    ``pre-refactor-snapshot``, which is where a superseded definition belongs.
    """

    def __init__(self, specs: Sequence[StrategySpec]) -> None:
        self._specs: dict[str, StrategySpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise StrategyError(f"Duplicate strategy {spec.name!r}.")
            self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def resolve(self, name: str) -> StrategySpec:
        """Look up a strategy by its plain name."""

        try:
            return self._specs[str(name)]
        except KeyError:
            raise StrategyError(f"Unknown strategy {name!r}. Available: {self.names()}") from None


STRATEGY_REGISTRY = StrategyRegistry(_SPECS)

#: The full ablation matrix required for the thesis, in report order.
REQUIRED_STRATEGIES: tuple[str, ...] = (
    "random",
    "uncertainty",
    "novelty",
    "rarity",
    "coherence",
    "uncertainty_novelty",
    "uncertainty_rarity",
    "uncertainty_coherence",
    "rarity_coherence",
    "rarity_plus_coherence",
    "full",
    "full_no_coherence",
    "full_no_rarity",
    "full_no_uncertainty",
    "full_no_novelty",
    "proposal_formula",
    "revealed_full",
    "objectness_area_prior",
    "prior_full",
    "prior_revealed_full",
    "revealed_no_gate",
    "revealed_support_only",
)


@dataclass(frozen=True)
class ScoringResult:
    """Everything one scoring pass produced, raw and normalised."""

    spec: StrategySpec
    image_ids: ObjectArray
    proposal_index: IntArray
    pseudo_labels: IntArray
    cluster_sizes: IntArray
    isolated: BoolArray
    kth_distance: FloatArray
    raw: Mapping[str, FloatArray]
    normalised: Mapping[str, FloatArray]
    scores: FloatArray
    image_scores: dict[Hashable, float]
    diagnostics: Mapping[str, object]

    @property
    def proposal_count(self) -> int:
        return int(self.scores.size)

    def selected_proposal_mask(self, selected_image_ids: Sequence[Hashable]) -> BoolArray:
        """Proposals inside selected images that drove the image's score."""

        chosen = {str(value) for value in selected_image_ids}
        mask = np.zeros(self.proposal_count, dtype=np.bool_)
        if not chosen:
            return mask
        in_image = np.array(
            [str(value) in chosen for value in self.image_ids.tolist()], dtype=np.bool_
        )
        if self.spec.image_aggregation in ("mean", "noisy_or"):
            return in_image
        limit = 1 if self.spec.image_aggregation == "max" else self.spec.top_k
        for image_id in chosen:
            indices = np.flatnonzero(
                np.array(
                    [str(value) == image_id for value in self.image_ids.tolist()],
                    dtype=np.bool_,
                )
            )
            if indices.size == 0:
                continue
            order = indices[np.argsort(-self.scores[indices], kind="stable")]
            mask[order[:limit]] = True
        return mask


def combine_components(
    spec: StrategySpec, components_by_name: Mapping[str, ArrayLike]
) -> FloatArray:
    """The weighted sum. The only place a strategy's arithmetic happens.

    ``score_pool`` and the legacy compatibility shim in
    :mod:`daowod.acquisition` both route through here, so no second copy of the
    formula can drift away from this one.
    """

    weights = spec.weights()
    arrays = {
        name: np.asarray(components_by_name[name], dtype=np.float64)
        for name in weights
        if name in components_by_name
    }
    missing = [name for name, weight in weights.items() if weight > 0 and name not in arrays]
    if missing:
        raise StrategyError(f"{spec.name}: missing weighted components {missing}.")
    lengths = {array.shape for array in arrays.values()}
    if len(lengths) > 1:
        raise ValueError("All acquisition components must have equal shape.")
    size = next(iter(lengths))[0] if lengths else 0
    scores = np.zeros(size, dtype=np.float64)
    for name, weight in weights.items():
        if weight > 0:
            scores = scores + weight * arrays[name]
    return scores


def aggregate_image_scores(
    image_ids: ArrayLike,
    proposal_scores: ArrayLike,
    *,
    method: ImageAggregation = "top_k_mean",
    top_k: int = 3,
) -> dict[Hashable, float]:
    """Reduce proposal scores to one score per image in O(N log N).

    The pre-audit implementation compared ``ids == image_id`` once per image,
    which is quadratic. This groups once with ``np.unique`` and a single
    ``lexsort``.
    """

    ids = np.asarray(image_ids, dtype=object)
    scores = components.as_vector("proposal_scores", proposal_scores)
    if ids.ndim != 1 or ids.shape[0] != scores.shape[0]:
        raise ValueError("image_ids and proposal_scores must be parallel vectors.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if method not in IMAGE_AGGREGATIONS:
        raise ValueError(
            f"Unknown image aggregation {method!r}. Supported: {list(IMAGE_AGGREGATIONS)}"
        )
    if ids.size == 0:
        return {}

    keys = np.asarray([str(value) for value in ids.tolist()], dtype=object)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    group_count = unique_keys.shape[0]

    if method == "noisy_or":
        # 1 - prod(1 - s); computed in log space for stability.
        clipped = np.clip(scores, 0.0, 1.0 - 1e-12)
        log_complement = np.bincount(inverse, weights=np.log1p(-clipped), minlength=group_count)
        values = 1.0 - np.exp(log_complement)
        return {str(key): float(value) for key, value in zip(unique_keys, values, strict=True)}

    if method == "mean":
        totals = np.bincount(inverse, weights=scores, minlength=group_count)
        counts = np.bincount(inverse, minlength=group_count)
        values = totals / np.maximum(counts, 1)
        return {str(key): float(value) for key, value in zip(unique_keys, values, strict=True)}

    limit = 1 if method == "max" else top_k
    order = np.lexsort((-scores, inverse))
    sorted_groups = inverse[order]
    sorted_scores = scores[order]
    starts = np.searchsorted(sorted_groups, np.arange(group_count), side="left")
    within = np.arange(sorted_groups.size) - starts[sorted_groups]
    keep = within < limit
    totals = np.bincount(sorted_groups[keep], weights=sorted_scores[keep], minlength=group_count)
    counts = np.bincount(sorted_groups[keep], minlength=group_count)
    values = totals / np.maximum(counts, 1)
    return {str(key): float(value) for key, value in zip(unique_keys, values, strict=True)}


def select_images(image_scores: Mapping[Hashable, float], *, budget: int) -> list[str]:
    """Highest-scoring images under a fixed budget, ties broken by ID."""

    if budget < 1:
        raise ValueError("budget must be positive.")
    if budget > len(image_scores):
        raise ValueError(f"budget {budget} exceeds the {len(image_scores)} scorable images.")
    return [
        str(image_id)
        for image_id in sorted(image_scores, key=lambda key: (-image_scores[key], str(key)))[
            :budget
        ]
    ]


def score_pool(
    *,
    spec: StrategySpec,
    image_ids: ArrayLike,
    embeddings: ArrayLike,
    reference_embeddings: ArrayLike,
    confidence: ArrayLike | None = None,
    posterior: ArrayLike | None = None,
    predicted_labels: ArrayLike | None = None,
    objectness: ArrayLike | None = None,
    boxes_cxcywh: ArrayLike | None = None,
    seed: int = 0,
    compute_all_components: bool = False,
) -> ScoringResult:
    """Score every candidate proposal with one strategy. The only scorer.

    ``compute_all_components`` evaluates every component even where its weight is
    zero. Diagnostics need that (a component's distribution is interesting even
    when a strategy ignores it); the live loop leaves it off so a strategy never
    pays for what it does not use.
    """

    ids = np.asarray(image_ids, dtype=object)
    vectors = components.as_matrix("embeddings", embeddings)
    count = vectors.shape[0]
    if ids.ndim != 1 or ids.shape[0] != count:
        raise ValueError("image_ids must be parallel to embeddings.")
    if spec.uncertainty_method == "objectness_area_prior" and (
        objectness is None or boxes_cxcywh is None
    ):
        raise ValueError(
            f"Strategy {spec.name!r} uses the objectness/box-scale informativeness "
            "prior, which needs the exported objectness and boxes."
        )
    if spec.needs_posterior() and posterior is None:
        raise ValueError(
            f"Strategy {spec.name!r} uses uncertainty method "
            f"{spec.uncertainty_method!r}, which requires the exported posterior."
        )

    pseudo_labels = components.assign_pseudo_labels(
        vectors,
        source=spec.pseudo_label_source,
        cluster_count=spec.cluster_count,
        seed=seed,
        predicted_labels=predicted_labels,
    )
    sizes = components.cluster_sizes(pseudo_labels)

    weights = spec.weights()
    raw: dict[str, FloatArray] = {}

    if weights["uncertainty"] > 0 or compute_all_components:
        raw["uncertainty"] = components.compute_uncertainty(
            method=spec.uncertainty_method,
            posterior=posterior,
            confidence=confidence,
            objectness=objectness,
            boxes_cxcywh=boxes_cxcywh,
        )
    else:
        raw["uncertainty"] = np.zeros(count, dtype=np.float64)

    if weights["novelty"] > 0 or compute_all_components:
        raw["novelty"] = components.compute_novelty(vectors, reference_embeddings)
    else:
        raw["novelty"] = np.zeros(count, dtype=np.float64)

    needs_rarity = weights["rarity"] > 0 or weights["gated"] > 0 or compute_all_components
    raw["rarity"] = (
        components.compute_rarity(
            pseudo_labels, method=spec.rarity_method, rarity_power=spec.rarity_power
        )
        if needs_rarity
        else np.zeros(count, dtype=np.float64)
    )

    needs_coherence = weights["coherence"] > 0 or weights["gated"] > 0 or compute_all_components
    if needs_coherence:
        coherence_result = components.compute_coherence(
            vectors,
            method=spec.coherence_method,
            pseudo_labels=pseudo_labels,
            neighbour_count=spec.neighbour_count,
            singleton_coherence=spec.singleton_coherence,
            minimum_cluster_size=spec.minimum_cluster_size,
            isolation_quantile=spec.isolation_quantile,
            radius_quantile=spec.radius_quantile,
            minimum_samples=spec.minimum_samples,
        )
    else:
        coherence_result = components.CoherenceResult(
            coherence=np.zeros(count, dtype=np.float64),
            isolated=np.zeros(count, dtype=np.bool_),
            kth_distance=np.zeros(count, dtype=np.float64),
            method=spec.coherence_method,
            details={"skipped": "no coherence weight"},
        )
    raw["coherence"] = coherence_result.coherence

    normalised: dict[str, FloatArray] = {
        name: normalise(raw[name], spec.normalisation_for(name))
        for name in ("uncertainty", "novelty", "rarity", "coherence")
    }

    # The gate acts on the *normalised* rarity, then the interaction itself is
    # normalised, so w_g is comparable with the other weights.
    gated_raw = normalised["rarity"] * np.power(raw["coherence"], spec.coherence_exponent)
    raw["gated"] = gated_raw
    normalised["gated"] = normalise(gated_raw, spec.normalisation_for("gated"))

    scores = combine_components(spec, normalised)

    image_scores = aggregate_image_scores(
        ids, scores, method=spec.image_aggregation, top_k=spec.top_k
    )

    diagnostics: dict[str, object] = {
        "strategy": spec.name,
        "proposals": count,
        "images": len(image_scores),
        "pseudo_classes": int(np.unique(pseudo_labels).size),
        "singleton_pseudo_classes": int((sizes == 1).sum()),
        "isolated_proposals": int(coherence_result.isolated.sum()),
        "coherence_method": coherence_result.method,
        "coherence_details": dict(coherence_result.details),
        "normalisation": {name: spec.normalisation_for(name) for name in COMPONENT_NAMES},
        "active_components": list(spec.active_components()),
        "seed": seed,
    }

    return ScoringResult(
        spec=spec,
        image_ids=ids,
        proposal_index=np.arange(count, dtype=np.int64),
        pseudo_labels=pseudo_labels,
        cluster_sizes=sizes,
        isolated=coherence_result.isolated,
        kth_distance=coherence_result.kth_distance,
        raw=raw,
        normalised=normalised,
        scores=scores,
        image_scores=image_scores,
        diagnostics=diagnostics,
    )
