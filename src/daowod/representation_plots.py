"""Publication figures for Representation Experiment E4.

Two kinds of figure, answering two different questions.

**Geometry summaries** (bars) answer "does the coherence premise hold in this
space", which is a comparison of numbers across spaces and is best read as a bar
chart against a reference line at 1.0.

**Embedding projections** (scatter) answer "do the tail categories form visible
groups", which is a claim about structure and needs to be seen.

Sampling, stated rather than hidden
-----------------------------------
A projection of 48 000 points where 34 are tail regions shows nothing about the
tail — the 0.07 % of points that matter are invisible under 75 % background. Every
projection is therefore produced twice:

``natural``
    a uniform random subsample, so the *real* proportions are visible. That the tail
    is almost invisible here is itself the finding, not a defect of the plot.

``balanced``
    every unknown region plus an equal-sized sample of known and background, so the
    geometry of the rare strata can be inspected at all.

Neither is the "true" picture on its own; each panel is labelled with which it is,
and the geometry *metrics* are always computed on the full pool.

UMAP is not available in this environment (`umap-learn` is not installed and there
is no network access), so projections use PCA — a linear, deterministic baseline
that cannot invent structure — and t-SNE, which can reveal local structure PCA
misses. Both are reported, because a neighbourhood claim that appears only under
t-SNE's non-linear embedding is weaker evidence than one visible in both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from daowod.audit import Strata
from daowod.plots import (
    CATEGORICAL,
    SEQUENTIAL,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _figure,
    _style,
    save,
)

FloatArray = NDArray[np.float64]

#: Colour per oracle stratum. Fixed by entity, so a stratum keeps its colour in
#: every panel of every figure.
STRATUM_COLOURS: Mapping[str, str] = {
    "tail": CATEGORICAL[4],
    "medium": CATEGORICAL[3],
    "head": CATEGORICAL[2],
    "known": CATEGORICAL[0],
    "background": "#c9c8c2",
}
STRATUM_ORDER: tuple[str, ...] = ("background", "known", "head", "medium", "tail")

#: Points drawn per projection. t-SNE is O(n log n) with a large constant; 6 000
#: keeps a figure under a minute and is plenty for judging whether groups exist.
PROJECTION_SAMPLE = 6_000


def stratum_labels(strata: Strata) -> NDArray[np.object_]:
    """One label per proposal, from the oracle strata."""

    labels = np.full(strata.is_background.shape[0], "background", dtype=object)
    labels[strata.is_known] = "known"
    labels[strata.is_head] = "head"
    labels[strata.is_medium] = "medium"
    labels[strata.is_tail] = "tail"
    return labels


def sample_indices(
    strata: Strata,
    *,
    scheme: str = "natural",
    size: int = PROJECTION_SAMPLE,
    seed: int = 0,
) -> NDArray[np.int64]:
    """Deterministic subsample for a projection.

    ``natural`` preserves the pool's composition; ``balanced`` keeps every unknown
    region and matches it with equal numbers of known and background regions, so the
    rare strata are visible at all.
    """

    generator = np.random.default_rng(seed)
    total = strata.is_background.shape[0]
    if scheme == "natural":
        take = int(min(size, total))
        return np.sort(generator.choice(total, size=take, replace=False))
    if scheme != "balanced":
        raise ValueError(f"Unknown sampling scheme {scheme!r}.")

    unknown = np.flatnonzero(strata.is_unknown)
    per_group = max(int(unknown.size), 1)
    chosen = [unknown]
    for mask in (strata.is_known, strata.is_background):
        pool = np.flatnonzero(mask)
        if pool.size:
            chosen.append(
                generator.choice(pool, size=int(min(per_group, pool.size)), replace=False)
            )
    return np.sort(np.concatenate(chosen))


def project(
    embeddings: ArrayLike,
    *,
    method: str = "pca",
    seed: int = 0,
    perplexity: float = 30.0,
) -> tuple[FloatArray, dict[str, object]]:
    """Two-dimensional coordinates, with a manifest describing how they were made."""

    matrix = np.asarray(embeddings, dtype=np.float64)
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    if method == "pca":
        model = PCA(n_components=2, random_state=seed)
        coordinates = np.asarray(model.fit_transform(matrix), dtype=np.float64)
        return coordinates, {
            "method": "pca",
            "explained_variance_ratio": [float(value) for value in model.explained_variance_ratio_],
        }
    if method == "tsne":
        # PCA to 50 dimensions first, as is standard: t-SNE on 2048 raw dimensions is
        # both slower and noisier, and the pre-reduction is deterministic.
        reduced = matrix
        if matrix.shape[1] > 50:
            reduced = PCA(n_components=50, random_state=seed).fit_transform(matrix)
        model = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=float(min(perplexity, max(5.0, (reduced.shape[0] - 1) / 3.0))),
            init="pca",
            metric="cosine",
        )
        coordinates = np.asarray(model.fit_transform(reduced), dtype=np.float64)
        return coordinates, {
            "method": "tsne",
            "perplexity": float(model.perplexity),
            "pre_reduced_to": int(reduced.shape[1]),
        }
    raise ValueError(f"Unknown projection method {method!r}.")


def _scatter_strata(axes, coordinates: FloatArray, labels: NDArray[np.object_]) -> None:
    for name in STRATUM_ORDER:
        mask = labels == name
        if not mask.any():
            continue
        # Background is drawn first, small and pale; the rare strata are drawn last
        # and larger so they are not buried under 75 % of the points.
        rare = name in ("tail", "medium", "head")
        axes.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=14 if rare else 4,
            c=STRATUM_COLOURS[name],
            alpha=0.95 if rare else 0.35,
            linewidths=0.4 if rare else 0.0,
            edgecolors=SURFACE if rare else "none",
            label=f"{name} ({int(mask.sum())})",
        )


def _scatter_continuous(axes, coordinates: FloatArray, values: ArrayLike, label: str):
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array)
    return axes.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        s=5,
        c=array[order],
        cmap=SEQUENTIAL,
        alpha=0.8,
        linewidths=0.0,
    )


def projection_figure(
    embeddings: ArrayLike,
    strata: Strata,
    directory: str | Path,
    *,
    representation: str,
    method: str = "pca",
    scheme: str = "balanced",
    extra: Mapping[str, ArrayLike] | None = None,
    seed: int = 0,
    name: str | None = None,
) -> tuple[list[Path], dict[str, object]]:
    """One projection of one space, panelled by colouring."""

    indices = sample_indices(strata, scheme=scheme, seed=seed)
    coordinates, manifest = project(
        np.asarray(embeddings, dtype=np.float64)[indices], method=method, seed=seed
    )
    labels = stratum_labels(strata)[indices]
    colourings = {key: np.asarray(values)[indices] for key, values in (extra or {}).items()}

    panels = 1 + len(colourings)
    figure = _figure(1, panels, width=4.6, height=4.2)
    axes = figure.axes[0]
    _scatter_strata(axes, coordinates, labels)
    _style(axes, title="oracle stratum", xlabel=f"{method.upper()} 1", ylabel=f"{method.upper()} 2")
    axes.legend(frameon=False, fontsize=7, labelcolor=TEXT_SECONDARY, loc="best", markerscale=1.6)

    for index, (key, values) in enumerate(colourings.items(), start=1):
        panel = figure.axes[index]
        image = _scatter_continuous(panel, coordinates, values, key)
        _style(panel, title=key, xlabel=f"{method.upper()} 1", ylabel="")
        bar = figure.colorbar(image, ax=panel, fraction=0.045, pad=0.02)
        bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)

    figure.suptitle(
        f"{representation} — {method.upper()}, {scheme} sample of {indices.size} regions",
        color=TEXT_PRIMARY,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    manifest.update(
        {
            "representation": representation,
            "scheme": scheme,
            "points": int(indices.size),
            "unknown_points": int(strata.is_unknown[indices].sum()),
            "tail_points": int(strata.is_tail[indices].sum()),
        }
    )
    written = save(figure, directory, name or f"figure_e4_{representation}_{method}_{scheme}")
    return written, manifest


def purity_bars(
    headline: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    name: str = "figure_e4_tail_purity_advantage",
) -> list[Path]:
    """The decisive statistic, per space, against the break-even line at 1.0.

    A log scale because the values span two orders of magnitude, and a horizontal
    line at 1.0 because that — not zero — is where the coherence premise starts to
    hold.
    """

    key = "tail_purity_advantage_normalised"
    rows = [row for row in headline if np.isfinite(float(row[key]))]
    if not rows:
        return []
    names = [str(row["representation"]) for row in rows]
    values = [float(row[key]) for row in rows]
    figure = _figure(1, 1, width=max(7.0, 1.05 * len(rows) + 3.0), height=4.4)
    axes = figure.axes[0]
    colours = [CATEGORICAL[1] if value >= 1.0 else CATEGORICAL[0] for value in values]
    axes.bar(np.arange(len(rows)), values, width=0.72, color=colours)
    axes.axhline(1.0, color=TEXT_SECONDARY, linewidth=1.2, linestyle="--")
    axes.set_yscale("log")
    axes.set_xticks(np.arange(len(rows)))
    axes.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    _style(
        axes,
        title="Does a tail region's neighbourhood look more like its own class than "
        "a background region's does?",
        xlabel="",
        ylabel="ceiling-normalised tail purity / background purity",
    )
    axes.annotate(
        "break-even: above this line the coherence premise holds",
        xy=(0.01, 1.05),
        xycoords=("axes fraction", "data"),
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    for position, value in enumerate(values):
        axes.text(
            position,
            value * 1.08,
            f"{value:.3f}",
            ha="center",
            fontsize=7,
            color=TEXT_PRIMARY,
        )
    return save(figure, directory, name)


def purity_panel(
    purity_rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    name: str = "figure_e4_neighbourhood_purity",
) -> list[Path]:
    """Same-label neighbour fraction per stratum, grouped by representation."""

    rows = list(purity_rows)
    if not rows:
        return []
    strata = ("tail", "medium", "head", "known", "background")
    names = [str(row["representation"]) for row in rows]
    figure = _figure(1, 1, width=max(8.0, 1.15 * len(rows) + 3.0), height=4.4)
    axes = figure.axes[0]
    width = 0.8 / len(strata)
    for index, stratum in enumerate(strata):
        values = [float(row.get(f"{stratum}_same_label", np.nan)) for row in rows]
        axes.bar(
            np.arange(len(rows)) + index * width - 0.4 + width / 2,
            values,
            width=width * 0.9,
            color=STRATUM_COLOURS[stratum],
            label=stratum,
        )
    axes.set_xticks(np.arange(len(rows)))
    axes.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    _style(
        axes,
        title="Fraction of the 10 nearest neighbours sharing the point's own label",
        xlabel="",
        ylabel="same-label neighbour fraction",
    )
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, ncols=len(strata))
    return save(figure, directory, name)


def geometry_scatter(
    cluster_rows: Sequence[Mapping[str, object]],
    compactness_rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    name: str = "figure_e4_cluster_geometry",
) -> list[Path]:
    """Silhouette and within/between distance ratio for the unknown classes.

    Two panels rather than one two-axis chart: the quantities have different scales
    and different break-even points (silhouette at 0, the ratio at 1).
    """

    silhouette = [row for row in cluster_rows if str(row.get("label_set")) == "unknown_class"]
    compact = [row for row in compactness_rows if row.get("available")]
    if not silhouette and not compact:
        return []
    figure = _figure(1, 2, width=5.4, height=4.2)

    axes = figure.axes[0]
    if silhouette:
        names = [str(row["representation"]) for row in silhouette]
        values = [float(row["silhouette"]) for row in silhouette]
        axes.bar(
            np.arange(len(names)),
            values,
            width=0.7,
            color=[CATEGORICAL[1] if value > 0 else CATEGORICAL[0] for value in values],
        )
        axes.axhline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--")
        axes.set_xticks(np.arange(len(names)))
        axes.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    _style(
        axes,
        title="Silhouette over true unknown classes",
        xlabel="",
        ylabel="silhouette (cosine); >0 means classes are separated",
    )

    axes = figure.axes[1]
    if compact:
        names = [str(row["representation"]) for row in compact]
        values = [float(row["compactness_ratio"]) for row in compact]
        axes.bar(
            np.arange(len(names)),
            values,
            width=0.7,
            color=[CATEGORICAL[1] if value < 1.0 else CATEGORICAL[0] for value in values],
        )
        axes.axhline(1.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--")
        axes.set_xticks(np.arange(len(names)))
        axes.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    _style(
        axes,
        title="Within-class / between-class distance",
        xlabel="",
        ylabel="ratio; <1 means a class is tighter than the gaps between classes",
    )
    return save(figure, directory, name)


def density_figure(
    density_rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    name: str = "figure_e4_local_density",
) -> list[Path]:
    """Which stratum a density-based coherence term would rank highest, per space."""

    rows = list(density_rows)
    if not rows:
        return []
    representations = sorted({str(row["representation"]) for row in rows})
    strata = ("tail", "medium", "head", "known", "background")
    figure = _figure(1, 1, width=max(8.0, 1.15 * len(representations) + 3.0), height=4.4)
    axes = figure.axes[0]
    width = 0.8 / len(strata)
    for index, stratum in enumerate(strata):
        values = []
        for representation in representations:
            match = [
                row
                for row in rows
                if str(row["representation"]) == representation and str(row["stratum"]) == stratum
            ]
            values.append(float(match[0]["density_rank_percentile_median"]) if match else np.nan)
        axes.bar(
            np.arange(len(representations)) + index * width - 0.4 + width / 2,
            values,
            width=width * 0.9,
            color=STRATUM_COLOURS[stratum],
            label=stratum,
        )
    axes.axhline(0.5, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--")
    axes.set_xticks(np.arange(len(representations)))
    axes.set_xticklabels(representations, rotation=35, ha="right", fontsize=8)
    _style(
        axes,
        title="Median density rank per stratum (0 = densest, 1 = most isolated)",
        xlabel="",
        ylabel="density rank percentile",
    )
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, ncols=len(strata))
    return save(figure, directory, name)


def active_learning_comparison(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    metric: str = "unknown_objects_found_mean",
    ylabel: str = "Unknown objects discovered at the largest budget",
    name: str = "figure_e4_active_learning",
) -> list[Path]:
    """Discovery per strategy, grouped by representation — Phase 5's headline.

    Colour carries the representation because that is the variable under test; the
    x-axis carries the strategy. A strategy that is representation-invariant by
    construction (random, the geometric prior) must show equal bars, which makes the
    figure its own correctness check.
    """

    entries = list(rows)
    if not entries:
        return []
    strategies = sorted({str(row["strategy"]) for row in entries})
    representations = sorted({str(row["representation"]) for row in entries})
    figure = _figure(1, 1, width=max(8.0, 1.5 * len(strategies) + 2.0), height=4.4)
    axes = figure.axes[0]
    width = 0.8 / max(len(representations), 1)
    for index, representation in enumerate(representations):
        values = []
        errors = []
        for strategy in strategies:
            match = [
                row
                for row in entries
                if str(row["representation"]) == representation and str(row["strategy"]) == strategy
            ]
            values.append(float(match[0][metric]) if match else np.nan)
            errors.append(float(match[0].get(f"{metric}_sd", 0.0) or 0.0) if match else 0.0)
        axes.bar(
            np.arange(len(strategies)) + index * width - 0.4 + width / 2,
            values,
            width=width * 0.88,
            color=CATEGORICAL[index % len(CATEGORICAL)],
            yerr=errors,
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 0.9, "capsize": 2},
            label=representation,
        )
    axes.set_xticks(np.arange(len(strategies)))
    axes.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
    _style(axes, title="", xlabel="", ylabel=ylabel)
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, ncols=2)
    return save(figure, directory, name)


def unused_grid_note() -> str:
    """Why there is no UMAP panel, recorded where a reader would look for one."""

    return (
        "UMAP is not included: umap-learn is not installed in this environment and "
        "there is no network access to add it. PCA (linear, deterministic) and t-SNE "
        "(non-linear) are both reported instead, so a structural claim can be checked "
        "against a projection that cannot manufacture clusters."
    )
