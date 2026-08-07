"""Geometry of a feature space, measured against the oracle's strata.

Representation Experiment E4 compares feature spaces, so it needs statistics that
describe a space rather than a strategy. Everything here takes an embedding matrix
and the oracle strata and returns numbers that are comparable across spaces of
different dimensionality.

The decisive statistic, and a correction to it
---------------------------------------------
The coherence gate needs one thing to be true: a rare unknown object's
neighbourhood must be more class-consistent than a background region's. In PROB's
decoder space it is not — measured tail same-label fraction 0.015 against
background's 0.888, a ratio of 0.017.

That ratio, which the earlier audit quoted, is **confounded**. A class with ``c``
members cannot supply more than ``min(c - 1, k)`` same-class neighbours, so the tail
group — whose classes hold 1 to 6 proposals — has a mean *ceiling* of 0.235 at
k = 10, while background's is 1.0. Part of the apparent gap is the class frequency
that defines the tail, not the geometry under test.

:func:`purity_summary` therefore reports both the raw ratio and a
ceiling-normalised one, and the verdict uses the normalised version.
:func:`same_class_rank` adds a statistic with no ceiling at all — the rank of a
point's nearest same-class sibling — and the *head* unknown group provides a third,
caveat-free check, because its classes hold 12 to 65 proposals and so have a ceiling
of exactly 1.0.

Why these particular metrics
----------------------------
Silhouette and Davies-Bouldin summarise whether *labelled* groups form compact,
separated clusters; they are the standard answers to "is this space organised by
this label set". Nearest-neighbour precision and mutual-neighbour consistency
describe the *local* structure a k-NN coherence term actually reads. Local density
per stratum says which stratum a density measure will rank highest — the mechanism
that broke the gate. Background-versus-tail overlap quantifies how much of the tail
sits inside the background mass. All are computed on cosine geometry over
L2-normalised rows so that a 2048-d space and a 20-d space are comparable.

Cost control
------------
Silhouette and Davies-Bouldin are O(n^2) and O(nk); on 48 000 points the former is
prohibitive, so it is computed on a deterministic stratified subsample and the
sample size is reported. Nothing is silently approximated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import davies_bouldin_score, roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from daowod.audit import Strata

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

#: Strata every table reports, in the order a reader compares them.
STRATA_ORDER: tuple[str, ...] = ("tail", "medium", "head", "known", "background")

#: Points used for the O(n^2) silhouette. 4 000 keeps it under a few seconds while
#: still containing every stratum at its natural proportion.
SILHOUETTE_SAMPLE = 4_000


class GeometryError(ValueError):
    """Raised when a geometry statistic cannot be computed from the given inputs."""


def unit_rows(embeddings: ArrayLike) -> FloatArray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise GeometryError("Embeddings must be a 2-D matrix.")
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class Neighbourhoods:
    """Cached k-NN structure for one space, so every metric reuses one search."""

    distances: FloatArray
    indices: IntArray
    neighbours: int

    @classmethod
    def build(cls, embeddings: ArrayLike, *, neighbours: int = 10) -> Neighbourhoods:
        vectors = unit_rows(embeddings)
        if vectors.shape[0] <= neighbours + 1:
            raise GeometryError(
                f"{vectors.shape[0]} points is too few for a {neighbours}-neighbour search."
            )
        distances, indices = (
            NearestNeighbors(n_neighbors=neighbours + 1).fit(vectors).kneighbors(vectors)
        )
        return cls(distances=distances[:, 1:], indices=indices[:, 1:], neighbours=neighbours)


def _stratum_masks(strata: Strata) -> dict[str, BoolArray]:
    return {
        "tail": strata.is_tail,
        "medium": strata.is_medium,
        "head": strata.is_head,
        "known": strata.is_known,
        "background": strata.is_background,
    }


def purity_ceiling(labels: NDArray[np.object_], *, neighbours: int) -> FloatArray:
    """The largest same-label fraction each point could possibly achieve.

    A point whose label has ``c`` members in the pool has at most ``min(c - 1, k)``
    same-label neighbours available, so its ceiling is ``min(c - 1, k) / k``. This
    matters more than it looks. Measured on the real evaluation pool at k = 10, the
    tail group's mean ceiling is **0.235** — three tail classes have a single
    proposal, so their ceiling is exactly zero — while background's is 1.0. A raw
    comparison of tail against background purity therefore mixes the geometry under
    test with the class frequencies that *define* the tail, and overstates the gap by
    roughly four times.

    Reporting the ceiling turns the headline into a fair question: of the same-class
    neighbours that exist at all, how many does this space place nearby?
    """

    values, counts = np.unique(labels, return_counts=True)
    lookup = dict(zip(values.tolist(), counts.tolist(), strict=True))
    k = float(max(neighbours, 1))
    return np.array(
        [min(float(lookup[label] - 1), k) / k for label in labels.tolist()],
        dtype=np.float64,
    )


def same_class_rank(
    embeddings: ArrayLike,
    strata: Strata,
    *,
    representation: str,
    limit: int = 3_000,
    seed: int = 0,
) -> dict[str, object]:
    """Rank of a point's nearest same-class sibling — a ceiling-free statistic.

    Purity is bounded by class size; a rank is not. For each unknown proposal whose
    class has at least two members, this reports where its closest sibling sits in
    the full similarity ordering over the whole pool. A space that groups the class
    answers 1 or 2; a space that does not answers in the thousands. Singleton classes
    have no sibling to find and are excluded, with the count reported.
    """

    vectors = unit_rows(embeddings)
    labels = strata.class_name
    generator = np.random.default_rng(seed)
    unknown_counts: dict[str, int] = {}
    for label in labels[strata.is_unknown].tolist():
        unknown_counts[label] = unknown_counts.get(label, 0) + 1

    rows: dict[str, object] = {"representation": representation, "pool": int(vectors.shape[0])}
    for stratum, mask in (
        ("tail", strata.is_tail),
        ("medium", strata.is_medium),
        ("head", strata.is_head),
        ("unknown", strata.is_unknown),
    ):
        present = np.flatnonzero(mask)
        if present.size == 0:
            continue
        eligible = np.array(
            [index for index in present.tolist() if unknown_counts.get(labels[index], 0) >= 2],
            dtype=np.int64,
        )
        rows[f"{stratum}_singletons_excluded"] = int(present.size - eligible.size)
        if eligible.size == 0:
            rows[f"{stratum}_median_sibling_rank"] = float("nan")
            continue
        if eligible.size > limit:
            eligible = np.sort(generator.choice(eligible, size=limit, replace=False))
        ranks: list[float] = []
        for index in eligible.tolist():
            similarity = vectors @ vectors[index]
            similarity[index] = -np.inf
            order = np.argsort(-similarity, kind="stable")
            siblings = np.flatnonzero(labels[order] == labels[index])
            if siblings.size:
                ranks.append(float(siblings[0] + 1))
        if not ranks:
            rows[f"{stratum}_median_sibling_rank"] = float("nan")
            continue
        array = np.asarray(ranks, dtype=np.float64)
        rows[f"{stratum}_evaluated"] = int(array.size)
        rows[f"{stratum}_median_sibling_rank"] = float(np.median(array))
        rows[f"{stratum}_sibling_within_10"] = float(np.mean(array <= 10))
        rows[f"{stratum}_sibling_within_100"] = float(np.mean(array <= 100))
    return rows


def purity_summary(
    neighbourhoods: Neighbourhoods,
    strata: Strata,
    *,
    representation: str,
) -> dict[str, object]:
    """Same-label, unknown and background neighbour fractions per stratum.

    The headline of E4. ``tail_purity_advantage`` is the ratio the coherence gate's
    premise reduces to; a value below 1 means a density-based coherence term will
    rank background above the tail in this space, no matter how the term is defined.

    Both a raw and a **ceiling-normalised** version are reported. The raw ratio is
    what the earlier audit quoted; the normalised one divides each stratum's observed
    purity by the largest value its class sizes permit (:func:`purity_ceiling`), and
    it is the fair comparison. The *head* unknown group is the cleanest evidence of
    all, because its classes hold 12 to 65 proposals and therefore have a ceiling of
    exactly 1.0 — no frequency artefact can explain a low value there.
    """

    labels = strata.label_for_purity()
    neighbours = neighbourhoods.indices
    same = np.array(
        [float(np.mean(labels[row] == labels[position])) for position, row in enumerate(neighbours)]
    )
    neighbour_unknown = strata.is_unknown[neighbours].mean(axis=1)
    neighbour_background = strata.is_background[neighbours].mean(axis=1)

    ceiling = purity_ceiling(labels, neighbours=neighbourhoods.neighbours)
    row: dict[str, object] = {
        "representation": representation,
        "neighbours": neighbourhoods.neighbours,
    }
    for name, mask in _stratum_masks(strata).items():
        row[f"{name}_n"] = int(mask.sum())
        row[f"{name}_same_label"] = float(same[mask].mean()) if mask.any() else float("nan")
        row[f"{name}_purity_ceiling"] = float(ceiling[mask].mean()) if mask.any() else float("nan")
        achievable = mask & (ceiling > 0)
        row[f"{name}_same_label_normalised"] = (
            float((same[achievable] / ceiling[achievable]).mean())
            if achievable.any()
            else float("nan")
        )
        row[f"{name}_neighbour_unknown"] = (
            float(neighbour_unknown[mask].mean()) if mask.any() else float("nan")
        )
        row[f"{name}_neighbour_background"] = (
            float(neighbour_background[mask].mean()) if mask.any() else float("nan")
        )
    tail = float(row["tail_same_label"])
    background = float(row["background_same_label"])
    row["tail_purity_advantage"] = (
        float(tail / background) if np.isfinite(background) and background > 0 else float("nan")
    )
    unknown_same = (
        float(same[strata.is_unknown].mean()) if strata.is_unknown.any() else float("nan")
    )
    row["unknown_same_label"] = unknown_same
    row["unknown_purity_advantage"] = (
        float(unknown_same / background)
        if np.isfinite(background) and background > 0
        else float("nan")
    )

    normalised_tail = float(row["tail_same_label_normalised"])
    normalised_background = float(row["background_same_label_normalised"])
    row["tail_purity_advantage_normalised"] = (
        float(normalised_tail / normalised_background)
        if np.isfinite(normalised_background) and normalised_background > 0
        else float("nan")
    )
    achievable_unknown = strata.is_unknown & (ceiling > 0)
    row["unknown_same_label_normalised"] = (
        float((same[achievable_unknown] / ceiling[achievable_unknown]).mean())
        if achievable_unknown.any()
        else float("nan")
    )
    row["unknown_purity_advantage_normalised"] = (
        float(row["unknown_same_label_normalised"] / normalised_background)
        if np.isfinite(normalised_background) and normalised_background > 0
        else float("nan")
    )
    # The verdict uses the *normalised* ratio, because that is the fair comparison.
    row["coherence_premise_holds"] = bool(
        np.isfinite(row["tail_purity_advantage_normalised"])
        and row["tail_purity_advantage_normalised"] > 1.0
    )
    row["coherence_premise_holds_raw"] = bool(
        np.isfinite(row["tail_purity_advantage"]) and row["tail_purity_advantage"] > 1.0
    )
    return row


def density_summary(
    neighbourhoods: Neighbourhoods,
    strata: Strata,
    *,
    representation: str,
) -> list[dict[str, object]]:
    """Local density per stratum: which stratum a density term will prefer.

    Density is reported as the k-th neighbour distance (smaller is denser) and as
    its rank among all points, because the acquisition score rank-normalises every
    component and the rank is therefore what the weights actually see.
    """

    kth = neighbourhoods.distances[:, -1]
    order = np.argsort(np.argsort(kth))
    percentile = order / max(kth.shape[0] - 1, 1)
    rows: list[dict[str, object]] = []
    for name, mask in _stratum_masks(strata).items():
        if not mask.any():
            continue
        rows.append(
            {
                "representation": representation,
                "stratum": name,
                "n": int(mask.sum()),
                "kth_neighbour_distance_median": float(np.median(kth[mask])),
                "kth_neighbour_distance_p10": float(np.quantile(kth[mask], 0.1)),
                "kth_neighbour_distance_p90": float(np.quantile(kth[mask], 0.9)),
                "density_rank_percentile_median": float(np.median(percentile[mask])),
                "mean_neighbour_distance": float(neighbourhoods.distances[mask].mean()),
            }
        )
    return rows


def nearest_neighbour_precision(
    neighbourhoods: Neighbourhoods,
    strata: Strata,
    *,
    representation: str,
) -> dict[str, object]:
    """1-NN and k-NN label agreement, overall and for the unknown strata.

    A 1-NN precision near the base rate means the space carries no usable local
    label information at all — the condition under which every coherence variant
    tested in the earlier audit landed at chance.
    """

    labels = strata.label_for_purity()
    first = neighbourhoods.indices[:, 0]
    agree_first = labels[first] == labels
    agree_k = np.array(
        [
            float(np.mean(labels[row] == labels[position]))
            for position, row in enumerate(neighbourhoods.indices)
        ]
    )
    row: dict[str, object] = {
        "representation": representation,
        "one_nn_precision_all": float(agree_first.mean()),
        "knn_precision_all": float(agree_k.mean()),
    }
    for name, mask in _stratum_masks(strata).items():
        if not mask.any():
            continue
        row[f"one_nn_precision_{name}"] = float(agree_first[mask].mean())
    # For the unknown strata, "same label" means the same unknown *class*, so this is
    # the fraction of rare regions whose nearest neighbour is a sibling of its class.
    if strata.is_unknown.any():
        row["one_nn_precision_unknown"] = float(agree_first[strata.is_unknown].mean())
    return row


def mutual_neighbour_consistency(
    neighbourhoods: Neighbourhoods,
    strata: Strata,
    *,
    representation: str,
) -> dict[str, object]:
    """Fraction of a point's neighbours that return the favour.

    Mutuality is the cheapest available proxy for "this neighbourhood is a real
    structure rather than an artefact of one point sitting on the edge of a dense
    mass", which is precisely the distinction the gate wants. Reported per stratum
    because a mass of background can be mutually consistent while carrying no class
    information.
    """

    indices = neighbourhoods.indices
    sets = [set(row.tolist()) for row in indices]
    mutual = np.array(
        [
            float(np.mean([1.0 if position in sets[int(other)] else 0.0 for other in row.tolist()]))
            for position, row in enumerate(indices)
        ]
    )
    row: dict[str, object] = {
        "representation": representation,
        "mutual_neighbour_fraction_all": float(mutual.mean()),
    }
    for name, mask in _stratum_masks(strata).items():
        if mask.any():
            row[f"mutual_neighbour_fraction_{name}"] = float(mutual[mask].mean())
    return row


def cluster_quality(
    embeddings: ArrayLike,
    strata: Strata,
    *,
    representation: str,
    sample: int = SILHOUETTE_SAMPLE,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Silhouette and Davies-Bouldin for two label sets.

    ``strata`` asks "is the space organised into object/background/known groups";
    ``unknown_class`` asks the sharper question the gate depends on — "are the
    individual unknown categories compact and separated". The second is computed on
    unknown proposals only, so its sample size is small and reported.
    """

    vectors = unit_rows(embeddings)
    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    stratum_label = np.array(
        [
            "tail"
            if tail
            else "medium"
            if medium
            else "head"
            if head
            else "known"
            if known
            else "background"
            for tail, medium, head, known in zip(
                strata.is_tail, strata.is_medium, strata.is_head, strata.is_known, strict=True
            )
        ],
        dtype=object,
    )
    for label_name, labels, mask in (
        ("strata", stratum_label, np.ones(vectors.shape[0], dtype=bool)),
        ("unknown_class", strata.class_name, strata.is_unknown),
    ):
        selected = np.flatnonzero(mask)
        if selected.size < 10:
            continue
        used = selected
        if used.size > sample:
            used = np.sort(generator.choice(used, size=sample, replace=False))
        subset_labels = np.asarray(labels)[used]
        distinct = {value for value in subset_labels.tolist()}
        if len(distinct) < 2:
            continue
        # Classes with a single member make silhouette undefined for that class and
        # Davies-Bouldin unstable; dropping them is reported rather than hidden.
        counts = {value: int((subset_labels == value).sum()) for value in distinct}
        keep = np.array([counts[value] >= 2 for value in subset_labels.tolist()])
        if keep.sum() < 10 or len({v for v, c in counts.items() if c >= 2}) < 2:
            continue
        matrix = vectors[used][keep]
        final_labels = subset_labels[keep]
        rows.append(
            {
                "representation": representation,
                "label_set": label_name,
                "points": int(matrix.shape[0]),
                "groups": int(len({value for value in final_labels.tolist()})),
                "dropped_singleton_groups": int(
                    len(distinct) - len({v for v, c in counts.items() if c >= 2})
                ),
                "silhouette": float(silhouette_score(matrix, final_labels, metric="cosine")),
                "davies_bouldin": float(davies_bouldin_score(matrix, final_labels)),
                "subsampled": bool(selected.size > sample),
            }
        )
    return rows


def compactness_separation(
    embeddings: ArrayLike,
    strata: Strata,
    *,
    representation: str,
    max_per_class: int = 60,
    seed: int = 0,
) -> dict[str, object]:
    """Within-class versus between-class cosine distance for the unknown classes.

    The ratio is the dimension-free version of "do the unknown categories form
    clusters": below 1 means a class's own members are closer to it than other
    classes are, above 1 means the class labels carry no metric structure. Computed
    on unknown proposals only, capped per class so one frequent class cannot
    dominate the average.
    """

    vectors = unit_rows(embeddings)
    generator = np.random.default_rng(seed)
    unknown = np.flatnonzero(strata.is_unknown)
    if unknown.size < 4:
        return {"representation": representation, "available": False}
    names = strata.class_name[unknown]
    chosen: list[int] = []
    for value in sorted({name for name in names.tolist()}):
        members = unknown[names == value]
        if members.size > max_per_class:
            members = generator.choice(members, size=max_per_class, replace=False)
        chosen.extend(int(item) for item in members.tolist())
    chosen_array = np.sort(np.asarray(chosen, dtype=np.int64))
    matrix = vectors[chosen_array]
    labels = strata.class_name[chosen_array]
    similarity = matrix @ matrix.T
    distance = 1.0 - similarity
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    off_diagonal = ~np.eye(distance.shape[0], dtype=bool)
    within = distance[same]
    between = distance[off_diagonal & ~same]
    if within.size == 0 or between.size == 0:
        return {"representation": representation, "available": False}
    return {
        "representation": representation,
        "available": True,
        "points": int(matrix.shape[0]),
        "classes": int(len({value for value in labels.tolist()})),
        "within_class_distance_mean": float(within.mean()),
        "between_class_distance_mean": float(between.mean()),
        "compactness_ratio": float(within.mean() / max(between.mean(), 1e-12)),
        "structure_present": bool(within.mean() < between.mean()),
    }


def background_tail_overlap(
    embeddings: ArrayLike,
    strata: Strata,
    *,
    representation: str,
    seed: int = 0,
) -> dict[str, object]:
    """How separable tail regions are from background *in this space*.

    Two complementary numbers. ``nearest_tail_auc`` asks whether proximity to other
    tail regions distinguishes a tail region from background — a purely local,
    unsupervised-style statistic, computed leave-one-out so a point is never its own
    evidence. ``centroid_auc`` asks the same of a single global direction. A space
    where the local number is at chance but the global one is high is a space in
    which the class information exists but not *locally* — which is the diagnosis the
    earlier audit reached for PROB's decoder space.
    """

    vectors = unit_rows(embeddings)
    tail = np.flatnonzero(strata.is_tail)
    background = np.flatnonzero(strata.is_background)
    if tail.size < 4 or background.size < 4:
        return {"representation": representation, "available": False}

    generator = np.random.default_rng(seed)
    sampled_background = (
        generator.choice(background, size=min(background.size, 20_000), replace=False)
        if background.size > 20_000
        else background
    )
    evaluated = np.sort(np.concatenate([tail, sampled_background]))
    is_tail = strata.is_tail[evaluated]

    anchors = vectors[tail]
    similarity = vectors[evaluated] @ anchors.T
    # Leave-one-out: a tail region must not count itself as evidence that it is
    # near a tail region, or the statistic would be trivially perfect.
    for position, index in enumerate(evaluated.tolist()):
        matches = np.flatnonzero(tail == index)
        if matches.size:
            similarity[position, matches[0]] = -np.inf
    local = np.max(similarity, axis=1)

    centroid = anchors.mean(axis=0)
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    global_score = vectors[evaluated] @ centroid

    return {
        "representation": representation,
        "available": True,
        "tail": int(tail.size),
        "background_evaluated": int(sampled_background.size),
        "nearest_tail_auc": float(roc_auc_score(is_tail.astype(int), local)),
        "centroid_auc": float(roc_auc_score(is_tail.astype(int), global_score)),
    }


def component_distributions(
    embeddings: ArrayLike,
    strata: Strata,
    *,
    representation: str,
    cluster_count: int = 20,
    neighbour_count: int = 3,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Coherence and rarity, computed in this space, summarised per stratum.

    The acquisition terms themselves, so a reader can see directly whether changing
    the space changes what the gate would do. Uses the repository's canonical
    component functions — not a re-implementation — so these are the values a
    campaign in this space would actually score with.
    """

    from daowod import components

    matrix = np.asarray(embeddings, dtype=np.float64)
    pseudo = components.assign_pseudo_labels(matrix, cluster_count=cluster_count, seed=seed)
    rarity = components.compute_rarity(pseudo)
    coherence = components.compute_coherence(
        matrix,
        method="relative_within_cluster",
        pseudo_labels=pseudo,
        neighbour_count=neighbour_count,
    )
    rows: list[dict[str, object]] = []
    for label, values in (("coherence", coherence.coherence), ("rarity", rarity)):
        for name, mask in _stratum_masks(strata).items():
            if not mask.any():
                continue
            selected = values[mask]
            rows.append(
                {
                    "representation": representation,
                    "component": label,
                    "stratum": name,
                    "n": int(mask.sum()),
                    "median": float(np.median(selected)),
                    "mean": float(selected.mean()),
                    "p10": float(np.quantile(selected, 0.1)),
                    "p90": float(np.quantile(selected, 0.9)),
                }
            )
    return rows


def pseudo_class_alignment(
    embeddings: ArrayLike,
    strata: Strata,
    *,
    representation: str,
    cluster_count: int = 20,
    seed: int = 0,
) -> dict[str, object]:
    """Does clustering *this* space recover the unknown classes and their frequency?

    Reuses :func:`daowod.audit.pseudo_class_quality` so the numbers are directly
    comparable with the frozen baseline's (ARI 0.007, rarity-versus-true-rarity
    Spearman 0.116).
    """

    from daowod import audit, components

    matrix = np.asarray(embeddings, dtype=np.float64)
    pseudo = components.assign_pseudo_labels(matrix, cluster_count=cluster_count, seed=seed)
    rarity = components.compute_rarity(pseudo)
    report = audit.pseudo_class_quality(pseudo, rarity, strata)
    report["representation"] = representation
    report["cluster_count"] = int(cluster_count)
    return report


def evaluate_representation(
    embeddings: ArrayLike,
    strata: Strata,
    *,
    representation: str,
    neighbours: int = 10,
    seed: int = 0,
) -> dict[str, object]:
    """Every Phase-3 statistic for one space, as a bundle of row lists."""

    structure = Neighbourhoods.build(embeddings, neighbours=neighbours)
    return {
        "purity": purity_summary(structure, strata, representation=representation),
        "density": density_summary(structure, strata, representation=representation),
        "nn_precision": nearest_neighbour_precision(
            structure, strata, representation=representation
        ),
        "mutual": mutual_neighbour_consistency(structure, strata, representation=representation),
        "clusters": cluster_quality(embeddings, strata, representation=representation, seed=seed),
        "compactness": compactness_separation(
            embeddings, strata, representation=representation, seed=seed
        ),
        "overlap": background_tail_overlap(
            embeddings, strata, representation=representation, seed=seed
        ),
        "components": component_distributions(
            embeddings, strata, representation=representation, seed=seed
        ),
        "sibling_rank": same_class_rank(
            embeddings, strata, representation=representation, seed=seed
        ),
        "pseudo_classes": pseudo_class_alignment(
            embeddings, strata, representation=representation, seed=seed
        ),
    }


def headline_table(purity_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """The one table that answers E4, ordered by the decisive statistic.

    Ordered by the *normalised* advantage, since the raw one is confounded by class
    frequency, and carrying both so a reader can see the difference.
    """

    rows = [
        {
            "representation": row["representation"],
            "tail_same_label": row["tail_same_label"],
            "tail_purity_ceiling": row["tail_purity_ceiling"],
            "tail_same_label_normalised": row["tail_same_label_normalised"],
            "head_same_label": row["head_same_label"],
            "head_purity_ceiling": row["head_purity_ceiling"],
            "background_same_label": row["background_same_label"],
            "tail_purity_advantage": row["tail_purity_advantage"],
            "tail_purity_advantage_normalised": row["tail_purity_advantage_normalised"],
            "unknown_purity_advantage_normalised": row["unknown_purity_advantage_normalised"],
            "coherence_premise_holds": row["coherence_premise_holds"],
        }
        for row in purity_rows
    ]
    return sorted(
        rows,
        key=lambda row: (
            -(
                float(row["tail_purity_advantage_normalised"])
                if np.isfinite(float(row["tail_purity_advantage_normalised"]))
                else -np.inf
            )
        ),
    )
