"""Label-anchored distribution estimation: rarity and support from revealed labels.

Why this module exists
----------------------
The first full run of Contribution A measured, on the real 2 400-image S-OWODB
Task-1 pool, that the *unsupervised* estimators of the distribution-aware term
carry essentially no information about whether a proposal sits on an unknown
object (ROC-AUC against background):

===============================================  =====
signal                                            AUC
===============================================  =====
k-means pseudo-class rarity                      0.485
coherence, relative-within-cluster               0.481
coherence, radius-core                           0.445
gated rarity x coherence                         0.489
k-NN density                                     0.385
local outlier factor                             0.521
shared-nearest-neighbour density                 0.447
mutual-k-NN coherence                            0.498
neighbourhood mean objectness                    0.564
===============================================  =====

while a *supervised* linear probe on the same 256-d decoder embeddings reaches
0.837. The information is in the representation; no unsupervised local-structure
statistic tested extracts it. The diagnosed reason is that background dominates
local structure: a background proposal's ten nearest neighbours are 89 %
background, whereas a tail-class proposal's are 1.5 % its own class. Density
therefore ranks background highest, and a gate built on density promotes coherent
background.

Active learning supplies the missing supervision for free. The plan already asks
for it — "fedd fel a kiválasztott proposal valódi osztályát; frissítsd a
megfigyelt pszeudoeloszlást" — and the previous implementation used revealed
labels only to *down-weight* saturated clusters, never to locate the unknown
region of feature space. Measured sample complexity of doing so (held-out AUC for
unknown vs background, mean over 8 draws):

=====================  ===================  ==========
revealed unknowns      similarity-anchored  probe
=====================  ===================  ==========
5                      0.671                0.710
10                     0.678                0.738
20                     0.694                0.755
40                     0.686                0.771
160                    0.689                0.814
=====================  ===================  ==========

Five revealed unknowns already beat every unsupervised alternative by ~0.19 AUC.

What this module does and does not claim
----------------------------------------
It re-estimates the two distribution-aware components:

* :func:`support` replaces cluster coherence — "does this region resemble regions
  the oracle has already confirmed to be unknown objects?" rather than "is this
  region in a dense part of the pool?";
* :func:`anchored_rarity` replaces pseudo-class rarity — inverse frequency of the
  nearest *revealed* class rather than of a k-means cluster.

It does **not** claim to solve per-class tail estimation. With ~20 unknown classes
reachable and a realistic budget revealing 10-40 unknown regions, most revealed
classes have one or two examples, so the per-class frequency estimate stays weak.
The prediction registered before running the campaign was: unknown discovery and
annotation precision improve; tail-versus-head selectivity does not.

Leakage contract
----------------
Every array here comes from :class:`~daowod.active.RevealedBank`, which is filled
inside :func:`daowod.active.reveal` — the one function that is allowed to read the
oracle, and only at positions that have already been annotated. Nothing in this
module receives a ground-truth array, and the acquisition score is still
re-derived from its recorded components on every round.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
