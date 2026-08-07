"""Acquisition components: uncertainty, novelty, pseudo-labels, rarity, coherence.

This is the single place each component is defined. Design decisions recorded
here follow ``docs/decisions.md``:

* **Uncertainty** (Decision 1) is normalised predictive entropy over the exported
  posterior. A pre-audit ``1 - |2c - 1|`` transform of the PROB unknown score was
  removed with the v1 semantics: the audit showed it is a strictly monotone
  rescaling of the score that selects which proposals exist at all
  (Spearman +1.000), so it cannot add information to the pool it defines.
* **Rarity** (Decision 3) represents relative sparsity of a proposal's
  pseudo-class.
* **Coherence** (Decision 3) must answer "is this proposal part of a locally
  consistent structure?" and must *not* be an absolute density proxy that
  penalises tail classes for being small. ``relative_within_cluster`` and
  ``neighbour_consistency`` are scale-free; ``density`` is the legacy absolute
  measure, kept for comparison.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

UncertaintyMethod = Literal[
    "entropy",
    "margin",
    "one_minus_max",
    "objectness_weighted_entropy",
    "objectness_area_prior",
]
UNCERTAINTY_METHODS: tuple[str, ...] = (
    "entropy",
    "margin",
    "one_minus_max",
    "objectness_weighted_entropy",
    "objectness_area_prior",
)

RarityMethod = Literal["log_inverse_frequency", "inverse_frequency", "negative_count"]
RARITY_METHODS: tuple[str, ...] = (
    "log_inverse_frequency",
    "inverse_frequency",
    "negative_count",
)

CoherenceMethod = Literal[
    "relative_within_cluster",
    "neighbour_consistency",
    "density",
    "radius_core",
]
COHERENCE_METHODS: tuple[str, ...] = (
    "relative_within_cluster",
    "neighbour_consistency",
    "density",
    "radius_core",
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
    objectness: ArrayLike | None = None,
    boxes_cxcywh: ArrayLike | None = None,
) -> FloatArray:
    """Proposal informativeness on an approximately [0, 1] scale.

    ``entropy``                     -sum p log(p + eps) / log(K)
    ``margin``                      1 - (p_top1 - p_top2)
    ``one_minus_max``               1 - p_top1
    ``objectness_weighted_entropy`` sqrt(entropy * unknown score)
    ``objectness_area_prior``       objectness * sqrt(box area)

    ``objectness_area_prior`` is not an uncertainty at all; it occupies the same
    ``U(x)`` slot as an *informativeness prior*, and it is here because the audit
    measured it to be the strongest signal available without labels. On the real
    2 400-image Task-1 pool its ROC-AUC for "sits on an unknown object versus
    background" is 0.777, against 0.624 for PROB's own unknown score, 0.483 for
    posterior entropy and 0.481 for the coherence term. Sorting the pool by it
    finds 85 unknown objects inside a 2 000-region budget where the full
    distribution-aware strategy finds 16. Any semantic acquisition term has to be
    read against that number, so the term is available as a first-class arm rather
    than only as a footnote in the analysis.

    Why box scale carries so much: PROB emits one proposal per decoder query and
    the background queries settle on small, low-scale boxes, so scale separates
    "region that contains something" from "patch of texture" more reliably than any
    of the model's semantic heads do at Task 1.

    A caution measured on a PROB-calibrated pool and recorded in
    ``docs/decisions.md``: PROB's posterior is
    ``objectness * sigmoid(class logits)``, renormalised. A *confident* unknown
    detection therefore has a peaked posterior and **low** entropy, while a
    background query has a diffuse posterior and **high** entropy. Plain
    ``entropy`` consequently anti-correlates with a proposal actually sitting on
    an object (point-biserial -0.39 on the calibrated pool, versus +0.78 for the
    unknown score). ``objectness_weighted_entropy`` is the geometric mean of the
    two, so a proposal must be both object-like and ambiguous to score highly.
    Which of these to prefer is an empirical question for the real pool; the
    diagnostics report the correlation with objectness for every method.
    """

    if method == "objectness_weighted_entropy":
        if confidence is None:
            raise ValueError(
                "objectness_weighted_entropy requires the unknown score "
                "(confidence) as well as the posterior."
            )
        entropy = compute_uncertainty(method="entropy", posterior=posterior)
        score = as_vector("confidence", confidence)
        if entropy.shape != score.shape:
            raise ValueError("confidence must match the posterior row count.")
        # Both factors are rank-normalised before combining. A geometric mean of
        # the *raw* values was measured to be dominated by the unknown score
        # (Spearman 0.9997 with it), which reproduces exactly the S2 defect this
        # method exists to avoid: the score spans orders of magnitude while
        # entropy sits in a narrow band, so the raw product carries no entropy
        # information. On ranks the two factors have equal influence.
        return np.sqrt(normalise(entropy, "rank") * normalise(score, "rank"))

    if method == "objectness_area_prior":
        if objectness is None or boxes_cxcywh is None:
            raise ValueError(
                "objectness_area_prior requires both objectness and boxes_cxcywh; "
                "re-export proposals with a bridge that writes 'objectness' and "
                "'boxes', or choose a posterior-only uncertainty method."
            )
        scores = as_vector("objectness", objectness)
        boxes = as_matrix("boxes_cxcywh", boxes_cxcywh)
        if boxes.shape[1] != 4:
            raise ValueError("boxes_cxcywh must have four columns.")
        if boxes.shape[0] != scores.shape[0]:
            raise ValueError("objectness must be parallel to boxes_cxcywh.")
        # sqrt of the normalised area, so the term is a linear box *scale* rather
        # than an area; areas span four orders of magnitude on real proposals and
        # would make the product a pure area ranking.
        scale = np.sqrt(np.clip(boxes[:, 2] * boxes[:, 3], 0.0, 1.0))
        return np.sqrt(normalise(scores, "rank") * normalise(scale, "rank"))

    if method not in UNCERTAINTY_METHODS:
        raise ValueError(
            f"Unknown uncertainty method: {method!r}. Supported: {list(UNCERTAINTY_METHODS)}"
        )
    if posterior is None:
        raise ValueError(
            f"Uncertainty method {method!r} requires the exported posterior. "
            "Re-export proposals with a bridge that writes 'posterior', or choose "
            "'objectness_area_prior', which needs only objectness and boxes."
        )
    probabilities = _probabilities(posterior)
    if method == "entropy":
        terms = probabilities * np.log(probabilities + ENTROPY_EPSILON)
        return np.clip(-terms.sum(axis=1) / np.log(probabilities.shape[1]), 0.0, 1.0)
    if method == "one_minus_max":
        return 1.0 - probabilities.max(axis=1)
    ordered = np.sort(probabilities, axis=1)
    return 1.0 - (ordered[:, -1] - ordered[:, -2])


#: Similarity-matrix elements held in memory at once by :func:`compute_novelty`.
#: 16 M float64 elements is 128 MB, which fits a Colab CPU runtime alongside a
#: 400 k-proposal export. The unchunked expression ``candidates @ references.T``
#: allocates ``N * R`` elements: at N = 70 000 and R = 20 000 that is 11.2 GB and
#: the process is killed, so the chunk bound is a correctness requirement at real
#: pool sizes, not a micro-optimisation.
NOVELTY_CHUNK_ELEMENTS = 16_000_000


def compute_novelty(
    candidate_embeddings: ArrayLike,
    reference_embeddings: ArrayLike,
    *,
    chunk_elements: int = NOVELTY_CHUNK_ELEMENTS,
) -> FloatArray:
    """Raw cosine distance from the nearest labelled reference proposal.

    Unlike the legacy ``daowod.acquisition.compute_novelty`` this returns the
    *raw* distance; normalisation is the scorer's responsibility so that every
    component is treated identically (S6).

    The nearest-reference search is blocked over candidate rows and only the
    running maximum similarity is kept, so peak memory is bounded by
    ``chunk_elements`` regardless of pool or bank size. ``max`` over a partitioned
    axis is exact, so blocking is mathematically equivalent to the unblocked form;
    the two can still differ in the last floating-point digit, because BLAS sums a
    tall block and a single row in different orders. That is irrelevant to
    selection — every component is rank-normalised — and it does not affect
    determinism, since a given call always takes the same path.
    """

    candidates = normalise_rows(candidate_embeddings)
    references = normalise_rows(reference_embeddings)
    if candidates.shape[1] != references.shape[1]:
        raise ValueError("Candidate and reference dimensions must match.")
    if candidates.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if references.shape[0] == 0:
        return np.ones(candidates.shape[0], dtype=np.float64)
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive.")

    rows_per_chunk = max(1, int(chunk_elements) // max(references.shape[0], 1))
    best = np.empty(candidates.shape[0], dtype=np.float64)
    for start in range(0, candidates.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, candidates.shape[0])
        best[start:stop] = (candidates[start:stop] @ references.T).max(axis=1)
    return 1.0 - best


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
    radius_quantile: float = 0.1,
    minimum_samples: int = 4,
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

    ``radius_core`` is the plan's second candidate definition: the number of
    neighbours inside an epsilon ball, scaled by DBSCAN's ``min_samples``, so a
    core point scores 1.0 and a noise point ~0. It needs no pseudo-labels, which
    makes it the one coherence measure immune to clustering instability — the
    defect ``docs/scientific_validation.md`` identifies as dominating the gate's
    signal (clustering noise 0.317 versus gate signal 0.294).

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

    if method == "radius_core":
        # DBSCAN's core-point criterion, made continuous. eps is a quantile of the
        # observed k-th neighbour distances rather than an absolute number,
        # because a fixed eps in a 256-d cosine space is meaningless across pools
        # and would silently become "everything is isolated" on a different
        # checkpoint. A proposal reaching min_samples neighbours inside eps scores
        # 1.0 (a DBSCAN core point); a lone proposal scores ~0.
        radius = float(np.quantile(global_kth, radius_quantile))
        radius = max(radius, 1e-12)
        counts = np.asarray(
            NearestNeighbors(radius=radius)
            .fit(vectors)
            .radius_neighbors(vectors, return_distance=False),
            dtype=object,
        )
        # radius_neighbors includes the point itself; subtract it.
        neighbours = np.array([max(len(item) - 1, 0) for item in counts], dtype=np.float64)
        required = float(max(minimum_samples - 1, 1))
        coherence = np.clip(neighbours / required, 0.0, 1.0)
        return CoherenceResult(
            coherence,
            neighbours < required,
            global_kth,
            method,
            {
                "radius": radius,
                "radius_quantile": radius_quantile,
                "minimum_samples": minimum_samples,
                "core_points": int((neighbours >= required).sum()),
                "mean_neighbours_in_radius": float(neighbours.mean()),
                "isolation_rule": "fewer than minimum_samples-1 neighbours within eps",
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


# =============================================================================
# Component normalisation
#
# Component normalisation for the canonical acquisition score.
#
# The audit (S6) found that ``novelty`` and ``rarity`` were min-max normalised
# while ``uncertainty`` and ``coherence`` were not, so the nominal weights
# ``0.3 : 0.2 : 0.5`` did not describe the components' actual influence. Every
# component now passes through one declared normaliser.
#
# Rank normalisation is the default. It is invariant to any strictly monotone
# transform of a component, which is what makes it the right answer to S4: the
# concentration of ``count**-1`` near zero is a property of the *transform*, not of
# the ordering, so ranking removes it without discarding information.
#
# It lives beside the components it normalises rather than in scoring.py:
# scoring imports this module, so a merge there would be circular.
# =============================================================================

FloatArray = NDArray[np.float64]

NormalisationMethod = Literal["rank", "minmax", "zscore_sigmoid", "ecdf", "none"]
NORMALISATION_METHODS: tuple[str, ...] = (
    "rank",
    "minmax",
    "zscore_sigmoid",
    "ecdf",
    "none",
)


def _as_normalisation_vector(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Components must be one-dimensional.")
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError("Components must be finite.")
    return array


def average_ranks(values: ArrayLike) -> FloatArray:
    """Zero-based average ranks; tied values share one rank, deterministically.

    Determinism matters because selection is compared across strategies and
    seeds: two equal component values must never be ordered by array position.
    """

    array = _as_normalisation_vector(values)
    if array.size == 0:
        return array.copy()
    _, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    group_rank = starts + (counts - 1) / 2.0
    return group_rank[inverse].astype(np.float64)


def normalise(values: ArrayLike, method: NormalisationMethod = "rank") -> FloatArray:
    """Map a raw component onto a comparable scale.

    ``rank``           average rank / (N - 1); constant input -> 0.5
    ``ecdf``           average 1-based rank / N, in (0, 1]
    ``minmax``         legacy behaviour, constant input -> 1.0
    ``zscore_sigmoid`` logistic of the z-score; zero variance -> 0.5
    ``none``           pass through unchanged
    """

    array = _as_normalisation_vector(values)
    if method == "none":
        return array.copy()
    if array.size == 0:
        return array.copy()

    if method == "rank":
        ranks = average_ranks(array)
        if array.size == 1:
            return np.full(1, 0.5, dtype=np.float64)
        return ranks / float(array.size - 1)
    if method == "ecdf":
        return (average_ranks(array) + 1.0) / float(array.size)
    if method == "minmax":
        low, high = float(array.min()), float(array.max())
        if high - low < 1e-12:
            return np.ones_like(array)
        return (array - low) / (high - low)
    if method == "zscore_sigmoid":
        deviation = float(array.std())
        if deviation < 1e-12:
            return np.full_like(array, 0.5)
        centred = (array - float(array.mean())) / deviation
        return 1.0 / (1.0 + np.exp(-centred))
    raise ValueError(
        f"Unknown normalisation method: {method!r}. Supported: {list(NORMALISATION_METHODS)}"
    )


# =============================================================================
# Label-anchored estimators: rarity and support from revealed labels
#
# Label-anchored distribution estimation: rarity and support from revealed labels.
#
# Why this module exists
# ----------------------
# The first full run of Contribution A measured, on the real 2 400-image S-OWODB
# Task-1 pool, that the *unsupervised* estimators of the distribution-aware term
# carry essentially no information about whether a proposal sits on an unknown
# object (ROC-AUC against background):
#
# ===============================================  =====
# signal                                            AUC
# ===============================================  =====
# k-means pseudo-class rarity                      0.485
# coherence, relative-within-cluster               0.481
# coherence, radius-core                           0.445
# gated rarity x coherence                         0.489
# k-NN density                                     0.385
# local outlier factor                             0.521
# shared-nearest-neighbour density                 0.447
# mutual-k-NN coherence                            0.498
# neighbourhood mean objectness                    0.564
# ===============================================  =====
#
# while a *supervised* linear probe on the same 256-d decoder embeddings reaches
# 0.837. The information is in the representation; no unsupervised local-structure
# statistic tested extracts it. The diagnosed reason is that background dominates
# local structure: a background proposal's ten nearest neighbours are 89 %
# background, whereas a tail-class proposal's are 1.5 % its own class. Density
# therefore ranks background highest, and a gate built on density promotes coherent
# background.
#
# Active learning supplies the missing supervision for free. The plan already asks
# for it — "fedd fel a kiválasztott proposal valódi osztályát; frissítsd a
# megfigyelt pszeudoeloszlást" — and the previous implementation used revealed
# labels only to *down-weight* saturated clusters, never to locate the unknown
# region of feature space. Measured sample complexity of doing so (held-out AUC for
# unknown vs background, mean over 8 draws):
#
# =====================  ===================  ==========
# revealed unknowns      similarity-anchored  probe
# =====================  ===================  ==========
# 5                      0.671                0.710
# 10                     0.678                0.738
# 20                     0.694                0.755
# 40                     0.686                0.771
# 160                    0.689                0.814
# =====================  ===================  ==========
#
# Five revealed unknowns already beat every unsupervised alternative by ~0.19 AUC.
#
# What this module does and does not claim
# ----------------------------------------
# It re-estimates the two distribution-aware components:
#
# * :func:`support` replaces cluster coherence — "does this region resemble regions
#   the oracle has already confirmed to be unknown objects?" rather than "is this
#   region in a dense part of the pool?";
# * :func:`anchored_rarity` replaces pseudo-class rarity — inverse frequency of the
#   nearest *revealed* class rather than of a k-means cluster.
#
# It does **not** claim to solve per-class tail estimation. With ~20 unknown classes
# reachable and a realistic budget revealing 10-40 unknown regions, most revealed
# classes have one or two examples, so the per-class frequency estimate stays weak.
# The prediction registered before running the campaign was: unknown discovery and
# annotation precision improve; tail-versus-head selectivity does not.
#
# Leakage contract
# ----------------
# Every array here comes from :class:`~daowod.annotation.RevealedBank`, which is filled
# inside :func:`daowod.annotation.reveal` — the one function that is allowed to read the
# oracle, and only at positions that have already been annotated. Nothing in this
# module receives a ground-truth array, and the acquisition score is still
# re-derived from its recorded components on every round.
#
# These are component estimators, so they live with the components. Only the
# SOURCE of rarity and coherence changes; the score's shape is untouched.
# =============================================================================

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Neighbours used by :func:`support`. Small because the bank is small: with ten
#: revealed unknowns, a k of 32 would average over the whole bank and destroy the
#: locality the term is supposed to measure.
DEFAULT_SUPPORT_NEIGHBOURS = 5

#: Value handed to the gate before any unknown has been revealed. 1.0 makes the
#: gate inactive, so a cold round reduces exactly to ungated rarity instead of
#: silently zeroing the distribution term.
COLD_START_SUPPORT = 1.0


class RevealedError(ValueError):
    """Raised when a label-anchored estimator is asked for the impossible."""


@dataclass
class RevealedBank:
    """Embeddings and oracle verdicts of the regions annotated so far.

    Split by verdict rather than stored as one bank because the two roles differ:
    the unknown embeddings anchor the support term, and the negatives (background
    and known-class regions) are what a discriminative variant would need. Class
    names are kept per unknown embedding so the rarity term can count them.
    """

    unknown_embeddings: list[FloatArray] = field(default_factory=list)
    unknown_classes: list[str] = field(default_factory=list)
    negative_embeddings: list[FloatArray] = field(default_factory=list)

    def add(self, embedding: ArrayLike, *, is_unknown: bool, class_name: str = "") -> None:
        vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if is_unknown:
            self.unknown_embeddings.append(vector)
            self.unknown_classes.append(str(class_name))
        else:
            self.negative_embeddings.append(vector)

    @property
    def unknown_count(self) -> int:
        return len(self.unknown_embeddings)

    @property
    def negative_count(self) -> int:
        return len(self.negative_embeddings)

    @property
    def revealed_class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in self.unknown_classes:
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def unknown_matrix(self) -> FloatArray:
        if not self.unknown_embeddings:
            return np.zeros((0, 0), dtype=np.float64)
        return np.vstack(self.unknown_embeddings)

    def negative_matrix(self) -> FloatArray:
        if not self.negative_embeddings:
            return np.zeros((0, 0), dtype=np.float64)
        return np.vstack(self.negative_embeddings)

    def report(self) -> dict[str, object]:
        counts = self.revealed_class_counts
        return {
            "revealed_unknown_regions": self.unknown_count,
            "revealed_negative_regions": self.negative_count,
            "revealed_unknown_classes": len(counts),
            "revealed_class_counts": dict(sorted(counts.items())),
            "singleton_revealed_classes": sum(1 for value in counts.values() if value == 1),
        }


def _unit(matrix: ArrayLike) -> FloatArray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise RevealedError("Embeddings must be a 2-D array.")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def support(
    embeddings: ArrayLike,
    bank: RevealedBank,
    *,
    neighbours: int = DEFAULT_SUPPORT_NEIGHBOURS,
    fallback: ArrayLike | None = None,
) -> tuple[FloatArray, dict[str, object]]:
    """Similarity to the nearest confirmed unknown regions, in [0, 1].

    Mean cosine similarity to the ``neighbours`` most similar revealed-unknown
    embeddings, mapped from [-1, 1] to [0, 1]. The mean over the *top* few rather
    than over the whole bank is deliberate: an unknown class the oracle has
    confirmed once should support regions resembling *it*, not be diluted by every
    other confirmed class.

    Before any unknown has been revealed there is nothing to anchor on, and
    ``fallback`` — the unsupervised coherence for the same proposals — is returned
    unchanged. That choice matters for the experiment's validity: it makes a cold
    round *bit-identical* to the baseline, so every difference in the final result
    is attributable to the labels the campaign bought rather than to a different
    cold-start policy. With no fallback supplied the neutral
    :data:`COLD_START_SUPPORT` is used, which leaves the gate inactive.

    Returns the values and a report; the report is what lets a reader see that a
    round was cold rather than that the term was uninformative.
    """

    candidates = _unit(embeddings)
    if bank.unknown_count == 0:
        cold = (
            np.full(candidates.shape[0], float(COLD_START_SUPPORT), dtype=np.float64)
            if fallback is None
            else np.asarray(fallback, dtype=np.float64).copy()
        )
        if cold.shape != (candidates.shape[0],):
            raise RevealedError("fallback must be parallel to the candidate embeddings.")
        return cold, {
            "cold_start": True,
            "revealed_unknown_regions": 0,
            "neighbours_used": 0,
            "source": (
                "unsupervised coherence fallback" if fallback is not None else "neutral constant"
            ),
        }
    anchors = _unit(bank.unknown_matrix())
    if anchors.shape[1] != candidates.shape[1]:
        raise RevealedError("Revealed embeddings and candidates have different dimensions.")
    take = int(min(max(neighbours, 1), anchors.shape[0]))
    # Blocked so peak memory stays bounded by the block, not by pool x bank.
    values = np.empty(candidates.shape[0], dtype=np.float64)
    rows_per_block = max(1, 8_000_000 // max(anchors.shape[0], 1))
    for start in range(0, candidates.shape[0], rows_per_block):
        stop = min(start + rows_per_block, candidates.shape[0])
        similarity = candidates[start:stop] @ anchors.T
        if take >= similarity.shape[1]:
            values[start:stop] = similarity.mean(axis=1)
        else:
            partition = np.partition(similarity, -take, axis=1)[:, -take:]
            values[start:stop] = partition.mean(axis=1)
    return (
        np.clip((values + 1.0) / 2.0, 0.0, 1.0),
        {
            "cold_start": False,
            "revealed_unknown_regions": bank.unknown_count,
            "neighbours_used": take,
            "mean_support": float(np.mean((values + 1.0) / 2.0)),
        },
    )


def anchored_rarity(
    embeddings: ArrayLike,
    bank: RevealedBank,
    *,
    fallback: ArrayLike,
    minimum_classes: int = 2,
) -> tuple[FloatArray, dict[str, object]]:
    """Inverse frequency of the nearest revealed class.

    Each candidate is assigned to the revealed unknown class whose *nearest*
    revealed example it most resembles (nearest-neighbour rather than centroid,
    because a class with one confirmed example has no meaningful centroid), and
    its rarity is ``-log(count / total)`` of that class among the revealed
    regions — the same functional form the unsupervised term uses, so only the
    *source* of the class assignment differs.

    Until ``minimum_classes`` distinct classes have been revealed there is no
    distribution to be aware of, and the ``fallback`` (the unsupervised
    pseudo-class rarity) is returned unchanged. This keeps the cold rounds
    identical to the baseline, so the experiment isolates one variable: how the
    distribution is estimated once labels exist.
    """

    default = np.asarray(fallback, dtype=np.float64)
    counts = bank.revealed_class_counts
    if len(counts) < int(minimum_classes):
        return default.copy(), {
            "cold_start": True,
            "revealed_unknown_classes": len(counts),
            "source": "unsupervised pseudo-class fallback",
        }

    candidates = _unit(embeddings)
    anchors = _unit(bank.unknown_matrix())
    names = np.asarray(bank.unknown_classes, dtype=object)
    total = float(sum(counts.values()))
    per_anchor_rarity = np.array(
        [-np.log(max(counts.get(str(name), 1), 1) / total) for name in names.tolist()],
        dtype=np.float64,
    )
    values = np.empty(candidates.shape[0], dtype=np.float64)
    assigned = np.empty(candidates.shape[0], dtype=np.int64)
    rows_per_block = max(1, 8_000_000 // max(anchors.shape[0], 1))
    for start in range(0, candidates.shape[0], rows_per_block):
        stop = min(start + rows_per_block, candidates.shape[0])
        similarity = candidates[start:stop] @ anchors.T
        nearest = np.argmax(similarity, axis=1)
        assigned[start:stop] = nearest
        values[start:stop] = per_anchor_rarity[nearest]
    _, assigned_counts = np.unique(names[assigned], return_counts=True)
    return values, {
        "cold_start": False,
        "revealed_unknown_classes": len(counts),
        "source": "nearest revealed class",
        "assigned_classes": int(assigned_counts.size),
        "largest_assigned_share": float(assigned_counts.max() / candidates.shape[0]),
    }


def diagnostics(
    bank: RevealedBank,
    *,
    support_values: ArrayLike | None = None,
) -> dict[str, object]:
    """Per-round record of what the anchored estimator had to work with."""

    report = bank.report()
    if support_values is not None:
        values = np.asarray(support_values, dtype=np.float64)
        report["support_mean"] = float(values.mean())
        report["support_p90"] = float(np.quantile(values, 0.9))
    return report


def sample_complexity_rows(
    *,
    measurements: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Rows for ``revealed_sample_complexity.csv``.

    Kept here rather than in the analysis script so the measured curve that
    motivates this module travels with the code that implements it.
    """

    return [dict(row) for row in measurements]
