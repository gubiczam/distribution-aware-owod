"""Component normalisation for the canonical acquisition score.

The audit (S6) found that ``novelty`` and ``rarity`` were min-max normalised
while ``uncertainty`` and ``coherence`` were not, so the nominal weights
``0.3 : 0.2 : 0.5`` did not describe the components' actual influence. Every
component now passes through one declared normaliser.

Rank normalisation is the default. It is invariant to any strictly monotone
transform of a component, which is what makes it the right answer to S4: the
concentration of ``count**-1`` near zero is a property of the *transform*, not of
the ordering, so ranking removes it without discarding information.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

NormalisationMethod = Literal["rank", "minmax", "zscore_sigmoid", "ecdf", "none"]
NORMALISATION_METHODS: tuple[str, ...] = (
    "rank",
    "minmax",
    "zscore_sigmoid",
    "ecdf",
    "none",
)


def _as_vector(values: ArrayLike) -> FloatArray:
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

    array = _as_vector(values)
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

    array = _as_vector(values)
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
