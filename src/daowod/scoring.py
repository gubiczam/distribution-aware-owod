"""The one canonical acquisition scorer and the one strategy registry.

Everything — the live campaign, offline analysis, simulations, notebook
diagnostics, ablations and tests — scores proposals through :func:`score_pool`.
The audit found the scoring formula duplicated in four places and already
drifted (``uncertainty_novelty`` was weight-normalised online and un-normalised
offline); representing strategies as declarative specs removes the possibility.

Canonical score
---------------
    S = w_u * U_hat + w_n * N_hat + w_r * R_hat + w_g * G_hat + w_c * C_hat

``G_hat`` is the normalised gated interaction ``normalise(R_hat * coherence**p)``.
``w_c`` extends the specified form so that a pure ``coherence`` baseline is
expressible; with ``w_c = 0`` the score reduces exactly to the specified one.

Semantics versions
------------------
Version 2 is the current science: posterior entropy, rank normalisation,
scale-free coherence. Version 1 reproduces the pre-audit behaviour bit for bit
and exists so published numbers remain reproducible. A name that exists in both
versions must be requested with an explicit ``semantics_version`` — nothing is
silently reinterpreted.
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

ImageAggregation = Literal["top_k_mean", "max", "mean", "noisy_or"]
IMAGE_AGGREGATIONS: tuple[str, ...] = ("top_k_mean", "max", "mean", "noisy_or")


class StrategyError(ValueError):
    """Raised for an unknown, ambiguous or internally invalid strategy."""


@dataclass(frozen=True)
class StrategySpec:
    """A complete, declarative description of one acquisition strategy."""

    name: str
    semantics_version: int = 2
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
    image_aggregation: str = "top_k_mean"
    top_k: int = 3
    weight_normalise: bool = False
    random_selection: bool = False
    deprecated: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise StrategyError("A strategy must have a name.")
        if self.semantics_version not in (1, 2):
            raise StrategyError(
                f"{self.name}: semantics_version must be 1 or 2, got {self.semantics_version}."
            )
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
            self.image_aggregation,
            self.top_k,
            self.weight_normalise,
            self.random_selection,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "semantics_version": self.semantics_version,
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
            "image_aggregation": self.image_aggregation,
            "top_k": self.top_k,
            "weight_normalise": self.weight_normalise,
            "random_selection": self.random_selection,
            "deprecated": self.deprecated,
            "description": self.description,
        }


# --- version 1: exact pre-audit semantics, preserved for reproducibility -----
# uncertainty and coherence were unnormalised, novelty and rarity min-maxed.
_LEGACY_NORMALISATION: dict[str, str] = {
    "uncertainty": "none",
    "novelty": "minmax",
    "rarity": "minmax",
    "coherence": "none",
    "gated": "none",
}
_LEGACY_COMMON: dict[str, object] = {
    "semantics_version": 1,
    "uncertainty_method": "legacy_prob_score",
    "rarity_method": "inverse_frequency",
    "coherence_method": "density",
    "normalisation": "minmax",
    "component_normalisation": _LEGACY_NORMALISATION,
    "pseudo_label_source": "cluster",
    "cluster_count": 20,
    "neighbour_count": 5,
    "image_aggregation": "top_k_mean",
    "top_k": 3,
}
_LEGACY_ALPHA, _LEGACY_BETA, _LEGACY_GAMMA = 0.3, 0.2, 0.5

_VERSION_1: tuple[StrategySpec, ...] = (
    StrategySpec(
        name="random",
        semantics_version=1,
        random_selection=True,
        description="Uniform random image selection (identical in both versions).",
    ),
    StrategySpec(
        name="uncertainty",
        uncertainty_weight=1.0,
        deprecated="1 - |2c - 1| is a monotone rescaling of the PROB unknown "
        "score (audit S2); use semantics_version 2 'uncertainty'.",
        description="Legacy: raw ambiguity of the unknown score.",
        **_LEGACY_COMMON,
    ),
    StrategySpec(
        name="uncertainty_novelty",
        uncertainty_weight=_LEGACY_ALPHA,
        novelty_weight=_LEGACY_BETA,
        weight_normalise=True,
        deprecated="Weight normalisation differed between the online and "
        "offline implementations before the audit.",
        description="Legacy: (alpha*u + beta*n) / (alpha + beta).",
        **_LEGACY_COMMON,
    ),
    StrategySpec(
        name="rarity",
        rarity_weight=1.0,
        deprecated="count**-1 after min-max is a near-binary singleton indicator (audit S4).",
        description="Legacy: min-maxed inverse pseudo-class frequency.",
        **_LEGACY_COMMON,
    ),
    StrategySpec(
        name="rarity_coherence",
        gated_weight=1.0,
        coherence_exponent=1.0,
        deprecated="Absolute-density coherence is frequency-confounded (audit S5).",
        description="Legacy: rarity * density_coherence.",
        **_LEGACY_COMMON,
    ),
    StrategySpec(
        name="ungated_full",
        uncertainty_weight=_LEGACY_ALPHA,
        novelty_weight=_LEGACY_BETA,
        rarity_weight=_LEGACY_GAMMA,
        description="Legacy: alpha*u + beta*n + gamma*r, no coherence gate.",
        **_LEGACY_COMMON,
    ),
    StrategySpec(
        name="rarity_no_coherence",
        uncertainty_weight=_LEGACY_ALPHA,
        novelty_weight=_LEGACY_BETA,
        rarity_weight=_LEGACY_GAMMA,
        description="Legacy alias of ungated_full used by the pilot campaign.",
        **_LEGACY_COMMON,
    ),
    StrategySpec(
        name="full",
        uncertainty_weight=_LEGACY_ALPHA,
        novelty_weight=_LEGACY_BETA,
        gated_weight=_LEGACY_GAMMA,
        coherence_exponent=1.0,
        description="Legacy: alpha*u + beta*n + gamma*r*coherence**1.",
        **_LEGACY_COMMON,
    ),
    StrategySpec(
        name="full_p1",
        uncertainty_weight=_LEGACY_ALPHA,
        novelty_weight=_LEGACY_BETA,
        gated_weight=_LEGACY_GAMMA,
        coherence_exponent=1.0,
        description="Pilot variant full_p1: legacy full with p = 1.0.",
        **_LEGACY_COMMON,
    ),
    StrategySpec(
        name="full_p05",
        uncertainty_weight=_LEGACY_ALPHA,
        novelty_weight=_LEGACY_BETA,
        gated_weight=_LEGACY_GAMMA,
        coherence_exponent=0.5,
        description="Pilot variant full_p05: legacy full with p = 0.5.",
        **_LEGACY_COMMON,
    ),
)

# --- version 2: post-audit semantics ----------------------------------------
_V2_ALPHA, _V2_BETA, _V2_GAMMA = 0.3, 0.2, 0.5

_VERSION_2: tuple[StrategySpec, ...] = (
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
        uncertainty_weight=_V2_ALPHA,
        novelty_weight=_V2_BETA,
        description="Entropy + novelty.",
    ),
    StrategySpec(
        name="uncertainty_rarity",
        uncertainty_weight=_V2_ALPHA,
        rarity_weight=_V2_GAMMA,
        description="Entropy + rarity, no coherence gate.",
    ),
    StrategySpec(
        name="uncertainty_coherence",
        uncertainty_weight=_V2_ALPHA,
        coherence_weight=_V2_GAMMA,
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
        uncertainty_weight=_V2_ALPHA,
        novelty_weight=_V2_BETA,
        gated_weight=_V2_GAMMA,
        description="Contribution A: entropy + novelty + gated rarity.",
    ),
    StrategySpec(
        name="full_no_coherence",
        uncertainty_weight=_V2_ALPHA,
        novelty_weight=_V2_BETA,
        rarity_weight=_V2_GAMMA,
        description="Full with the coherence gate removed (rarity ungated).",
    ),
    StrategySpec(
        name="full_no_rarity",
        uncertainty_weight=_V2_ALPHA,
        novelty_weight=_V2_BETA,
        coherence_weight=_V2_GAMMA,
        description="Full with rarity removed, coherence kept additively.",
    ),
    StrategySpec(
        name="full_no_uncertainty",
        novelty_weight=_V2_BETA,
        gated_weight=_V2_GAMMA,
        description="Full with the uncertainty term removed.",
    ),
    StrategySpec(
        name="full_no_novelty",
        uncertainty_weight=_V2_ALPHA,
        gated_weight=_V2_GAMMA,
        description="Full with the novelty term removed; matches the written "
        "proposal S = U + gamma * w(c) * coh.",
    ),
    StrategySpec(
        name="proposal_formula",
        uncertainty_weight=1.0,
        rarity_weight=_V2_GAMMA,
        gated_weight=_V2_GAMMA,
        description="The formula exactly as written in the research proposal: "
        "S = U + lambda*D + gamma*w(c)*coh, i.e. both an ungated and a gated "
        "distribution term.",
    ),
)


class StrategyRegistry:
    """The single authoritative source of strategy definitions."""

    def __init__(self, specs: Sequence[StrategySpec]) -> None:
        self._by_version: dict[int, dict[str, StrategySpec]] = {1: {}, 2: {}}
        for spec in specs:
            bucket = self._by_version[spec.semantics_version]
            if spec.name in bucket:
                raise StrategyError(
                    f"Duplicate strategy {spec.name!r} in semantics version "
                    f"{spec.semantics_version}."
                )
            bucket[spec.name] = spec

    def names(self, semantics_version: int | None = None) -> list[str]:
        if semantics_version is None:
            return sorted({name for bucket in self._by_version.values() for name in bucket})
        return sorted(self._by_version[semantics_version])

    def qualified_names(self) -> list[str]:
        return sorted(
            f"v{version}:{name}" for version, bucket in self._by_version.items() for name in bucket
        )

    def resolve(self, name: str, *, semantics_version: int | None = None) -> StrategySpec:
        """Look up a strategy, refusing to guess which semantics were meant.

        ``"v1:full"`` / ``"v2:full"`` prefixes are accepted so a config can pin
        the version inline.
        """

        if ":" in name:
            prefix, _, bare = name.partition(":")
            if prefix not in ("v1", "v2"):
                raise StrategyError(
                    f"Unknown semantics prefix {prefix!r} in {name!r}; use 'v1:' or 'v2:'."
                )
            requested = int(prefix[1:])
            if semantics_version is not None and semantics_version != requested:
                raise StrategyError(
                    f"{name!r} pins semantics version {requested} but "
                    f"semantics_version={semantics_version} was requested."
                )
            return self.resolve(bare, semantics_version=requested)

        if semantics_version is not None:
            bucket = self._by_version.get(semantics_version)
            if bucket is None:
                raise StrategyError(f"Unknown semantics version: {semantics_version}.")
            if name not in bucket:
                raise StrategyError(
                    f"Unknown strategy {name!r} in semantics version "
                    f"{semantics_version}. Available: {sorted(bucket)}"
                )
            return bucket[name]

        in_versions = [version for version, bucket in self._by_version.items() if name in bucket]
        if not in_versions:
            raise StrategyError(f"Unknown strategy {name!r}. Available: {self.qualified_names()}")
        if len(in_versions) > 1:
            keys = {self._by_version[version][name].behaviour_key() for version in in_versions}
            if len(keys) == 1:
                # Behaviourally identical in every version (e.g. 'random').
                return self._by_version[max(in_versions)][name]
            raise StrategyError(
                f"Strategy {name!r} exists with different semantics in versions "
                f"{sorted(in_versions)}. Request it explicitly, e.g. "
                f"'v1:{name}' to reproduce published results or 'v2:{name}' for "
                "the current definition. See docs/migration_strategies.md."
            )
        return self._by_version[in_versions[0]][name]


STRATEGY_REGISTRY = StrategyRegistry((*_VERSION_1, *_VERSION_2))

#: The full ablation matrix required for the thesis, in report order.
REQUIRED_STRATEGIES: tuple[str, ...] = (
    "v2:random",
    "v2:uncertainty",
    "v2:novelty",
    "v2:rarity",
    "v2:coherence",
    "v2:uncertainty_novelty",
    "v2:uncertainty_rarity",
    "v2:uncertainty_coherence",
    "v2:rarity_coherence",
    "v2:rarity_plus_coherence",
    "v2:full",
    "v2:full_no_coherence",
    "v2:full_no_rarity",
    "v2:full_no_uncertainty",
    "v2:full_no_novelty",
    "v2:proposal_formula",
    "v1:rarity_no_coherence",
    "v1:full_p05",
    "v1:full_p1",
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
    if spec.weight_normalise:
        total = sum(value for value in weights.values() if value > 0)
        if total <= 0:
            raise StrategyError(f"{spec.name}: cannot weight-normalise zero weights.")
        scores = scores / total
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
            method=spec.uncertainty_method, posterior=posterior, confidence=confidence
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
        "semantics_version": spec.semantics_version,
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
