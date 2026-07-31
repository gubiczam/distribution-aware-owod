"""Acquisition components: uncertainty, novelty, pseudo-labels, rarity, coherence.

This is the single place each component is defined. Design decisions recorded
here follow ``docs/decisions.md``:

* **Uncertainty** (Decision 1) defaults to normalised predictive entropy over the
  exported posterior. The legacy ``1 - |2c - 1|`` transform of the PROB unknown
  score is retained under the explicit name ``legacy_prob_score`` because the
  audit showed it is a strictly monotone rescaling of the score used to select
  which proposals exist at all (Spearman +1.000).
* **Rarity** (Decision 3) represents relative sparsity of a proposal's
  pseudo-class.
* **Coherence** (Decision 3) must answer "is this proposal part of a locally
  consistent structure?" and must *not* be an absolute density proxy that
  penalises tail classes for being small. ``relative_within_cluster`` and
  ``neighbour_consistency`` are scale-free; ``density`` is the legacy absolute
  measure, kept for comparison.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

UncertaintyMethod = Literal["entropy", "margin", "one_minus_max", "legacy_prob_score"]
UNCERTAINTY_METHODS: tuple[str, ...] = (
    "entropy",
    "margin",
    "one_minus_max",
    "legacy_prob_score",
)

RarityMethod = Literal["log_inverse_frequency", "inverse_frequency", "negative_count"]
RARITY_METHODS: tuple[str, ...] = (
    "log_inverse_frequency",
    "inverse_frequency",
    "negative_count",
)

CoherenceMethod = Literal["relative_within_cluster", "neighbour_consistency", "density"]
COHERENCE_METHODS: tuple[str, ...] = (
    "relative_within_cluster",
    "neighbour_consistency",
    "density",
)

PseudoLabelSource = Literal["cluster", "predicted"]
PSEUDO_LABEL_SOURCES: tuple[str, ...] = ("cluster", "predicted")

ENTROPY_EPSILON = 1e-12


def as_vector(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")
    return array


def as_matrix(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def normalise_rows(values: ArrayLike) -> FloatArray:
    matrix = as_matrix("embeddings", values)
    if matrix.shape[0] == 0:
        return matrix.copy()
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def _probabilities(posterior: ArrayLike) -> FloatArray:
    probabilities = as_matrix("posterior", posterior)
    if probabilities.shape[1] < 2 or np.any(probabilities < 0):
        raise ValueError("posterior must contain at least two non-negative classes.")
    mass = probabilities.sum(axis=1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("Every posterior row must have positive mass.")
    return probabilities / mass


def compute_uncertainty(
    *,
    method: UncertaintyMethod = "entropy",
    posterior: ArrayLike | None = None,
    confidence: ArrayLike | None = None,
) -> FloatArray:
    """Proposal uncertainty on an approximately [0, 1] scale.

    ``entropy``           -sum p log(p + eps) / log(K)
    ``margin``            1 - (p_top1 - p_top2)
    ``one_minus_max``     1 - p_top1
    ``legacy_prob_score`` 1 - |2c - 1| over the PROB unknown score (deprecated)
    """

    if method == "legacy_prob_score":
        if confidence is None:
            raise ValueError("legacy_prob_score requires confidence values.")
        values = as_vector("confidence", confidence)
        if np.any((values < 0) | (values > 1)):
            raise ValueError("confidence must be in [0, 1].")
        return 1.0 - np.abs(2.0 * values - 1.0)

    if method not in UNCERTAINTY_METHODS:
        raise ValueError(
            f"Unknown uncertainty method: {method!r}. Supported: {list(UNCERTAINTY_METHODS)}"
        )
    if posterior is None:
        raise ValueError(
            f"Uncertainty method {method!r} requires the exported posterior. "
            "Re-export proposals with a bridge that writes 'posterior', or set "
            "uncertainty_method='legacy_prob_score' to use the deprecated "
            "confidence transform."
        )
    probabilities = _probabilities(posterior)
    if method == "entropy":
        terms = probabilities * np.log(probabilities + ENTROPY_EPSILON)
        return np.clip(-terms.sum(axis=1) / np.log(probabilities.shape[1]), 0.0, 1.0)
    if method == "one_minus_max":
        return 1.0 - probabilities.max(axis=1)
    ordered = np.sort(probabilities, axis=1)
    return 1.0 - (ordered[:, -1] - ordered[:, -2])


def compute_novelty(candidate_embeddings: ArrayLike, reference_embeddings: ArrayLike) -> FloatArray:
    """Raw cosine distance from the nearest labelled reference proposal.

    Unlike the legacy ``daowod.acquisition.compute_novelty`` this returns the
    *raw* distance; normalisation is the scorer's responsibility so that every
    component is treated identically (S6).
    """

    candidates = normalise_rows(candidate_embeddings)
    references = normalise_rows(reference_embeddings)
    if candidates.shape[1] != references.shape[1]:
        raise ValueError("Candidate and reference dimensions must match.")
    if candidates.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if references.shape[0] == 0:
        return np.ones(candidates.shape[0], dtype=np.float64)
    return 1.0 - (candidates @ references.T).max(axis=1)


def assign_pseudo_labels(
    embeddings: ArrayLike,
    *,
    source: PseudoLabelSource = "cluster",
    cluster_count: int = 20,
    seed: int = 0,
    predicted_labels: ArrayLike | None = None,
) -> IntArray:
    """Estimate a proposal's class before any oracle annotation."""

    vectors = as_matrix("embeddings", embeddings)
    count = vectors.shape[0]
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if source == "predicted":
        if predicted_labels is None:
            raise ValueError("pseudo_label_source='predicted' requires exported predicted_labels.")
        labels = np.asarray(predicted_labels, dtype=np.int64)
        if labels.shape != (count,):
            raise ValueError("predicted_labels must match the proposal count.")
        return labels
    if source != "cluster":
        raise ValueError(
            f"Unknown pseudo-label source: {source!r}. Supported: {list(PSEUDO_LABEL_SOURCES)}"
        )
    if cluster_count < 1:
        raise ValueError("cluster_count must be positive.")
    return (
        KMeans(
            n_clusters=min(cluster_count, count),
            random_state=seed,
            n_init="auto",
        )
        .fit_predict(normalise_rows(vectors))
        .astype(np.int64)
    )


def cluster_sizes(pseudo_labels: ArrayLike) -> IntArray:
    """Size of each proposal's own pseudo-class."""

    labels = np.asarray(pseudo_labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("pseudo_labels must be one-dimensional.")
    if labels.size == 0:
        return np.empty(0, dtype=np.int64)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return counts[inverse].astype(np.int64)


def compute_rarity(
    pseudo_labels: ArrayLike,
    *,
    method: RarityMethod = "log_inverse_frequency",
    rarity_power: float = 1.0,
) -> FloatArray:
    """Raw sparsity of a proposal's pseudo-class. Higher means rarer.

    All three methods are strictly decreasing in the pseudo-class size, so under
    rank normalisation they are equivalent. They differ only when a
    scale-sensitive normaliser (``minmax``, ``none``) is selected — which is
    exactly the regime where S4's concentration near zero appears.
    """

    if rarity_power <= 0:
        raise ValueError("rarity_power must be positive.")
    sizes = cluster_sizes(pseudo_labels)
    if sizes.size == 0:
        return np.empty(0, dtype=np.float64)
    counts = sizes.astype(np.float64)
    if method == "inverse_frequency":
        return np.power(counts, -rarity_power)
    if method == "negative_count":
        return -counts
    if method == "log_inverse_frequency":
        return -np.log(counts / float(counts.size)) * rarity_power
    raise ValueError(f"Unknown rarity method: {method!r}. Supported: {list(RARITY_METHODS)}")


def _kth_neighbour_distance(vectors: FloatArray, neighbour_count: int) -> FloatArray:
    count = vectors.shape[0]
    k = min(neighbour_count, count - 1)
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(vectors).kneighbors(vectors)
    return distances[:, k].astype(np.float64)


@dataclass(frozen=True)
class CoherenceResult:
    """Coherence values plus the bookkeeping the diagnostics need."""

    coherence: FloatArray
    isolated: BoolArray
    kth_distance: FloatArray
    method: str
    details: Mapping[str, object]


def compute_coherence(
    embeddings: ArrayLike,
    *,
    method: CoherenceMethod = "relative_within_cluster",
    pseudo_labels: ArrayLike | None = None,
    neighbour_count: int = 5,
    singleton_coherence: float = 0.0,
    minimum_cluster_size: int = 3,
    isolation_quantile: float = 0.9,
) -> CoherenceResult:
    """Local structural support for each proposal.

    ``relative_within_cluster`` measures the k-th nearest-neighbour distance
    *inside the proposal's own pseudo-class*, divided by that class's own median.
    Because the scale is per-cluster, a tight three-member cluster and a tight
    three-hundred-member cluster both centre on 0.5 — this is what removes the
    direct class-frequency confound in S5.

    ``neighbour_consistency`` measures how many of the k global nearest
    neighbours share the proposal's pseudo-class, divided by the largest number
    attainable given that class's size, so small classes are not penalised for
    being small.

    ``density`` is the legacy absolute measure ``1 / (1 + d_k / median(d_k))``.

    Clusters of size one have no internal structure at all; they receive
    ``singleton_coherence`` (0.0 by default, i.e. "an isolated proposal is not
    coherent") and are flagged ``isolated``. Clusters below
    ``minimum_cluster_size`` cannot support their own scale estimate, so they
    borrow the pooled median and are reported in ``details``.
    """

    vectors = normalise_rows(embeddings)
    count = vectors.shape[0]
    if neighbour_count < 1:
        raise ValueError("neighbour_count must be positive.")
    if not 0.0 < isolation_quantile < 1.0:
        raise ValueError("isolation_quantile must lie strictly in (0, 1).")
    if not 0.0 <= singleton_coherence <= 1.0:
        raise ValueError("singleton_coherence must lie in [0, 1].")
    if count == 0:
        empty_f = np.empty(0, dtype=np.float64)
        return CoherenceResult(empty_f, np.empty(0, dtype=np.bool_), empty_f, method, {})
    if count == 1:
        return CoherenceResult(
            np.array([singleton_coherence], dtype=np.float64),
            np.array([True]),
            np.zeros(1, dtype=np.float64),
            method,
            {"reason": "a single proposal has no neighbourhood"},
        )

    global_kth = _kth_neighbour_distance(vectors, neighbour_count)
    isolation_threshold = float(np.quantile(global_kth, isolation_quantile))

    if method == "density":
        scale = max(float(np.median(global_kth)), 1e-12)
        coherence = np.clip(1.0 / (1.0 + global_kth / scale), 0.0, 1.0)
        return CoherenceResult(
            coherence,
            global_kth > isolation_threshold,
            global_kth,
            method,
            {
                "scale": scale,
                "note": "legacy absolute density; frequency-confounded (S5)",
                "isolation_threshold": isolation_threshold,
            },
        )

    if pseudo_labels is None:
        raise ValueError(f"Coherence method {method!r} requires pseudo_labels.")
    labels = np.asarray(pseudo_labels, dtype=np.int64)
    if labels.shape != (count,):
        raise ValueError("pseudo_labels must match the proposal count.")
    sizes = cluster_sizes(labels)
    singletons = sizes == 1

    if method == "neighbour_consistency":
        k = min(neighbour_count, count - 1)
        _, indices = NearestNeighbors(n_neighbors=k + 1).fit(vectors).kneighbors(vectors)
        neighbours = labels[indices[:, 1:]]
        agreement = (neighbours == labels[:, None]).sum(axis=1).astype(np.float64)
        attainable = np.minimum(k, np.maximum(sizes - 1, 0)).astype(np.float64)
        coherence = np.where(
            attainable > 0, agreement / np.maximum(attainable, 1.0), singleton_coherence
        )
        coherence = np.clip(coherence, 0.0, 1.0)
        coherence[singletons] = singleton_coherence
        return CoherenceResult(
            coherence,
            singletons | (global_kth > isolation_threshold),
            global_kth,
            method,
            {
                "k": k,
                "singleton_clusters": int(singletons.sum()),
                "isolation_threshold": isolation_threshold,
            },
        )

    if method != "relative_within_cluster":
        raise ValueError(
            f"Unknown coherence method: {method!r}. Supported: {list(COHERENCE_METHODS)}"
        )

    within = np.zeros(count, dtype=np.float64)
    small_clusters: list[int] = []
    for label in np.unique(labels):
        member = np.flatnonzero(labels == label)
        if member.size < 2:
            continue
        within[member] = _kth_neighbour_distance(vectors[member], neighbour_count)
        if member.size < minimum_cluster_size:
            small_clusters.append(int(label))

    measurable = ~singletons
    pooled = within[measurable]
    pooled_scale = max(float(np.median(pooled)) if pooled.size else 0.0, 1e-12)

    coherence = np.full(count, singleton_coherence, dtype=np.float64)
    borrowed = 0
    for label in np.unique(labels):
        member = np.flatnonzero(labels == label)
        if member.size < 2:
            continue
        if member.size >= minimum_cluster_size:
            scale = max(float(np.median(within[member])), 1e-12)
        else:
            scale = pooled_scale
            borrowed += int(member.size)
        coherence[member] = 1.0 / (1.0 + within[member] / scale)
    coherence = np.clip(coherence, 0.0, 1.0)
    coherence[singletons] = singleton_coherence

    return CoherenceResult(
        coherence,
        singletons | (global_kth > isolation_threshold),
        global_kth,
        method,
        {
            "pooled_scale": pooled_scale,
            "singleton_clusters": int(singletons.sum()),
            "clusters_borrowing_pooled_scale": sorted(small_clusters),
            "proposals_borrowing_pooled_scale": borrowed,
            "minimum_cluster_size": minimum_cluster_size,
            "isolation_threshold": isolation_threshold,
        },
    )
