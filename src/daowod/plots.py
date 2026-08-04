"""Research figures for the annotation study.

Every figure is generated from the same row dictionaries the CSVs are written
from, so a number in a plot can always be traced to a line in a table — and each
figure ships beside its CSV, which is the table view that the palette's contrast
relief requires.

Design rules applied here (and why, since matplotlib defaults break most of them):

* **Colour identifies the strategy, not its rank.** :data:`STRATEGY_COLOURS` keys
  on the strategy name, so filtering to three strategies never repaints the
  survivors — the reader's memory of "blue is random" survives across figures.
* **One measure per axes.** Annotation precision and background rate are related
  but differently scaled, so they get two panels rather than two y-axes.
* **Marker shape as a second channel.** The five-slot palette clears the
  colour-vision gates on adjacent pairs, but a printed thesis figure may be read
  in greyscale, so each strategy also has its own marker.
* **Bands, not error bars, on curves.** Five strategies x five budgets x error
  bars is unreadable; a light ±1 sd band keeps the curve legible while still
  showing that a gap smaller than the band is not a result.

The palette is the validated default categorical set (blue, orange, aqua, yellow,
magenta) checked with the data-viz validator: worst adjacent colour-vision ΔE 9.1,
worst adjacent normal-vision ΔE 19.6, all slots inside the lightness band.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"

#: Categorical slots 1-5 of the validated default palette, in fixed order.
CATEGORICAL: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v")

#: Strategy -> colour and marker, fixed by entity so figures stay comparable.
STRATEGY_ORDER: tuple[str, ...] = (
    "v2:random",
    "v2:uncertainty",
    "v2:uncertainty_novelty",
    "v2:full_no_coherence",
    "v2:full",
)
STRATEGY_LABELS: Mapping[str, str] = {
    "v2:random": "Random",
    "v2:uncertainty": "Uncertainty",
    "v2:uncertainty_novelty": "Uncertainty + Novelty",
    "v2:full_no_coherence": "+ Rarity (ungated)",
    "v2:full": "+ Rarity x Coherence (gated) [baseline]",
    "v2:objectness_area_prior": "Objectness x box scale [free control]",
    "v2:prior_full": "Prior + cluster gate",
    "v2:prior_revealed_full": "Prior + anchored gate",
    "v2:revealed_support_only": "Anchored support only",
    "v2:revealed_no_gate": "Anchored rarity, ungated",
    "v2:revealed_full": "Anchored gate",
}
STRATEGY_COLOURS: Mapping[str, str] = dict(zip(STRATEGY_ORDER, CATEGORICAL, strict=True))
STRATEGY_MARKERS: Mapping[str, str] = dict(zip(STRATEGY_ORDER, MARKERS, strict=True))

#: Arm ordering for the eleven-arm comparison, baseline family first so a legend
#: reads in the same order as the report's tables.
COMPARISON_ORDER: tuple[str, ...] = (
    "v2:random",
    "v2:uncertainty",
    "v2:uncertainty_novelty",
    "v2:full_no_coherence",
    "v2:full",
    "v2:objectness_area_prior",
    "v2:prior_full",
    "v2:prior_revealed_full",
    "v2:revealed_support_only",
    "v2:revealed_no_gate",
    "v2:revealed_full",
)

#: The curated headline set. Eleven arms cannot each own a hue — the validated
#: palette has eight slots and only the first few clear the colour-vision gates on
#: an all-pairs list — so the headline curve figure shows four arms and the rest are
#: reached through the per-family small multiples and the family-coloured bars. This
#: is the documented "fold to Other or facet" rule, applied rather than worked
#: around by generating hues.
HEADLINE_STRATEGIES: tuple[str, ...] = (
    "v2:random",
    "v2:full",
    "v2:objectness_area_prior",
    "v2:prior_revealed_full",
)

#: Family -> colour. With more arms than hues, colour carries the *family*, which
#: is the comparison a reader is actually making: baseline versus free control
#: versus label-anchored.
FAMILY_ORDER: tuple[str, ...] = (
    "baseline",
    "free-control",
    "label-anchored",
    "prior+cluster",
    "prior+anchored",
)
FAMILY_COLOURS: Mapping[str, str] = {
    "baseline": CATEGORICAL[0],
    "free-control": CATEGORICAL[3],
    "label-anchored": CATEGORICAL[2],
    "prior+cluster": CATEGORICAL[1],
    "prior+anchored": CATEGORICAL[4],
}

#: Sequential blue ramp, steps 100 -> 700, for magnitude (the ablation heatmap).
SEQUENTIAL_STEPS: tuple[str, ...] = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)
SEQUENTIAL = LinearSegmentedColormap.from_list("daowod_blue", SEQUENTIAL_STEPS)


def _style(axes: plt.Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=8)
    axes.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    axes.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
    axes.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)


def _figure(rows: int, columns: int, *, width: float = 5.2, height: float = 3.6) -> Figure:
    figure, _ = plt.subplots(
        rows,
        columns,
        figsize=(width * columns, height * rows),
        squeeze=False,
        facecolor=SURFACE,
    )
    return figure


def _colour(strategy: str, fallback_index: int = 0) -> str:
    return STRATEGY_COLOURS.get(strategy, CATEGORICAL[fallback_index % len(CATEGORICAL)])


def _marker(strategy: str, fallback_index: int = 0) -> str:
    return STRATEGY_MARKERS.get(strategy, MARKERS[fallback_index % len(MARKERS)])


def _label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)


def save(figure: Figure, directory: str | Path, name: str) -> list[Path]:
    """Write one figure as PNG (for the notebook) and PDF (for the thesis)."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = root / f"{name}.{suffix}"
        figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


def _series(
    rows: Sequence[Mapping[str, object]],
    *,
    severity: str,
    strategy: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Budget, mean and sd of one metric for one (severity, strategy) cell."""

    by_budget: dict[int, list[float]] = {}
    for row in rows:
        if str(row.get("imbalance_setting")) != severity:
            continue
        if str(row.get("strategy")) != strategy:
            continue
        try:
            value = float(row[metric])  # type: ignore[arg-type]
            budget = int(row["budget"])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            by_budget.setdefault(budget, []).append(value)
    budgets = np.array(sorted(by_budget), dtype=np.float64)
    means = np.array([float(np.mean(by_budget[int(b)])) for b in budgets], dtype=np.float64)
    sds = np.array(
        [
            float(np.std(by_budget[int(b)], ddof=1)) if len(by_budget[int(b)]) > 1 else 0.0
            for b in budgets
        ],
        dtype=np.float64,
    )
    return budgets, means, sds


def _severities(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        {str(row.get("imbalance_setting", "")) for row in rows if row.get("imbalance_setting")}
    )


def _strategies(rows: Sequence[Mapping[str, object]]) -> list[str]:
    present = {str(row.get("strategy", "")) for row in rows if row.get("strategy")}
    ordered = [name for name in STRATEGY_ORDER if name in present]
    return ordered + sorted(present - set(ordered))


def discovery_curves(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    metric: str = "tail_discovery_recall",
    name: str = "figure_tail_discovery_vs_budget",
    ylabel: str = "Tail discovery recall",
) -> list[Path]:
    """The headline figure: discovery recall against annotation budget, per severity."""

    severities = _severities(rows)
    strategies = _strategies(rows)
    if not severities or not strategies:
        return []
    figure = _figure(1, len(severities))
    for column, severity in enumerate(severities):
        axes = figure.axes[column]
        for index, strategy in enumerate(strategies):
            budgets, means, sds = _series(rows, severity=severity, strategy=strategy, metric=metric)
            if budgets.size == 0:
                continue
            colour = _colour(strategy, index)
            axes.fill_between(
                budgets, means - sds, means + sds, color=colour, alpha=0.14, linewidth=0
            )
            axes.plot(
                budgets,
                means,
                color=colour,
                linewidth=2.0,
                marker=_marker(strategy, index),
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=_label(strategy),
            )
        _style(
            axes,
            title=f"severity: {severity}",
            xlabel="Annotated regions (oracle calls)",
            ylabel=ylabel if column == 0 else "",
        )
        axes.set_ylim(bottom=0.0)
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncols=min(len(labels), 3),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, -0.16),
    )
    return save(figure, directory, name)


def group_discovery_panels(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    severity: str,
    name: str = "figure_group_discovery",
) -> list[Path]:
    """All / head / medium / tail discovery for one severity, four panels."""

    groups = ("all", "head", "medium", "tail")
    strategies = _strategies(rows)
    if not strategies:
        return []
    figure = _figure(2, 2, width=4.6, height=3.2)
    for index, group in enumerate(groups):
        axes = figure.axes[index]
        for position, strategy in enumerate(strategies):
            budgets, means, sds = _series(
                rows,
                severity=severity,
                strategy=strategy,
                metric=f"{group}_discovery_recall",
            )
            if budgets.size == 0:
                continue
            colour = _colour(strategy, position)
            axes.fill_between(
                budgets, means - sds, means + sds, color=colour, alpha=0.12, linewidth=0
            )
            axes.plot(
                budgets,
                means,
                color=colour,
                linewidth=2.0,
                marker=_marker(strategy, position),
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=_label(strategy),
            )
        _style(
            axes,
            title=f"{group} unknowns",
            xlabel="Annotated regions",
            ylabel="Discovery recall",
        )
        axes.set_ylim(bottom=0.0)
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncols=min(len(labels), 3),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, -0.06),
    )
    figure.suptitle(
        f"Discovery by frequency group — severity {severity}",
        color=TEXT_PRIMARY,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return save(figure, directory, name)


def annotation_efficiency(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    severity: str,
    name: str = "figure_annotation_efficiency",
) -> list[Path]:
    """Annotation precision, background rate and isolated-outlier rate.

    Three panels rather than three lines on one axes: they answer different
    questions (was the annotation useful, was it wasted on background, was it
    wasted on an isolated outlier) and share only the x-axis.
    """

    panels = (
        (
            "annotation_precision",
            "Annotation precision",
            "fraction of annotations on a true unknown",
        ),
        ("background_selection_rate", "Background selection rate", "fraction on no object"),
        ("isolated_selection_rate", "Isolated-outlier selection rate", "fraction flagged isolated"),
    )
    strategies = _strategies(rows)
    if not strategies:
        return []
    figure = _figure(1, len(panels), width=4.6, height=3.4)
    for index, (metric, title, ylabel) in enumerate(panels):
        axes = figure.axes[index]
        for position, strategy in enumerate(strategies):
            budgets, means, sds = _series(rows, severity=severity, strategy=strategy, metric=metric)
            if budgets.size == 0:
                continue
            colour = _colour(strategy, position)
            axes.fill_between(
                budgets, means - sds, means + sds, color=colour, alpha=0.12, linewidth=0
            )
            axes.plot(
                budgets,
                means,
                color=colour,
                linewidth=2.0,
                marker=_marker(strategy, position),
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=_label(strategy),
            )
        _style(axes, title=title, xlabel="Annotated regions", ylabel=ylabel)
        axes.set_ylim(bottom=0.0)
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncols=min(len(labels), 3),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, -0.18),
    )
    figure.suptitle(
        f"Annotation efficiency — severity {severity}",
        color=TEXT_PRIMARY,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return save(figure, directory, name)


def unique_classes(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    severity: str,
    name: str = "figure_unique_classes",
) -> list[Path]:
    """Distinct unknown classes reached, overall and in the tail group."""

    strategies = _strategies(rows)
    if not strategies:
        return []
    figure = _figure(1, 2, width=4.8, height=3.4)
    for index, (metric, title) in enumerate(
        (("all_unique_classes", "All unknown classes"), ("tail_unique_classes", "Tail classes"))
    ):
        axes = figure.axes[index]
        for position, strategy in enumerate(strategies):
            budgets, means, sds = _series(rows, severity=severity, strategy=strategy, metric=metric)
            if budgets.size == 0:
                continue
            colour = _colour(strategy, position)
            axes.fill_between(
                budgets, means - sds, means + sds, color=colour, alpha=0.12, linewidth=0
            )
            axes.plot(
                budgets,
                means,
                color=colour,
                linewidth=2.0,
                marker=_marker(strategy, position),
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=_label(strategy),
            )
        _style(axes, title=title, xlabel="Annotated regions", ylabel="Distinct classes discovered")
        axes.set_ylim(bottom=0.0)
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncols=min(len(labels), 3),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, -0.18),
    )
    figure.suptitle(
        f"Class coverage — severity {severity}",
        color=TEXT_PRIMARY,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return save(figure, directory, name)


def auc_bars(
    auc_rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    metric: str = "tail_discovery_auc",
    name: str = "figure_tail_auc_by_severity",
    ylabel: str = "Tail discovery AUC (mean over budgets)",
) -> list[Path]:
    """Grouped bars: one group per severity, one bar per strategy, ±1 sd."""

    severities = _severities(auc_rows)
    strategies = _strategies(auc_rows)
    if not severities or not strategies:
        return []
    figure = _figure(1, 1, width=2.4 * len(severities) + 4.0, height=4.0)
    axes = figure.axes[0]
    group_width = 0.8
    bar_width = group_width / max(len(strategies), 1)
    for index, strategy in enumerate(strategies):
        means: list[float] = []
        errors: list[float] = []
        for severity in severities:
            values = [
                float(row[metric])  # type: ignore[arg-type]
                for row in auc_rows
                if str(row.get("imbalance_setting")) == severity
                and str(row.get("strategy")) == strategy
                and _is_number(row.get(metric))
            ]
            means.append(float(np.mean(values)) if values else float("nan"))
            errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        positions = np.arange(len(severities)) + index * bar_width - group_width / 2 + bar_width / 2
        axes.bar(
            positions,
            means,
            width=bar_width * 0.88,  # the 12 % remainder is the surface gap between bars
            color=_colour(strategy, index),
            label=_label(strategy),
            yerr=errors,
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.0, "capsize": 2},
        )
    axes.set_xticks(np.arange(len(severities)))
    axes.set_xticklabels(severities)
    _style(axes, title="", xlabel="Long-tail severity", ylabel=ylabel)
    axes.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncols=2, loc="upper right")
    return save(figure, directory, name)


def _is_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def component_distributions(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    strategy: str = "v2:full",
    severity: str | None = None,
    name: str = "figure_component_distributions",
) -> list[Path]:
    """Per-stratum component values: the mechanism's ordering claim, measured.

    The claim is not "rarity is high" but "the gated term ranks a true tail region
    above an isolated outlier". That is a statement about strata, so the figure
    shows the median and the p10-p90 span of each component within each stratum.
    """

    strata = (
        "true_tail",
        "true_medium",
        "true_head",
        "known_object",
        "background",
        "isolated_outlier",
    )
    selected = [
        row
        for row in rows
        if str(row.get("strategy")) == strategy
        and (severity is None or str(row.get("imbalance_setting")) == severity)
    ]
    components = [
        component
        for component in ("uncertainty", "rarity", "coherence", "gated")
        if any(str(row.get("component")) == component for row in selected)
    ]
    if not components:
        return []
    figure = _figure(1, len(components), width=3.6, height=3.8)
    for index, component in enumerate(components):
        axes = figure.axes[index]
        medians: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        labels: list[str] = []
        for stratum in strata:
            matches = [
                row
                for row in selected
                if str(row.get("component")) == component
                and str(row.get("stratum")) == stratum
                and _is_number(row.get("median"))
            ]
            if not matches:
                continue
            labels.append(stratum.replace("_", "\n"))
            medians.append(float(np.mean([float(row["median"]) for row in matches])))  # type: ignore[arg-type]
            lows.append(float(np.mean([float(row["p10"]) for row in matches])))  # type: ignore[arg-type]
            highs.append(float(np.mean([float(row["p90"]) for row in matches])))  # type: ignore[arg-type]
        if not medians:
            continue
        positions = np.arange(len(medians))
        centres = np.asarray(medians)
        axes.bar(
            positions,
            centres,
            width=0.7,
            color=CATEGORICAL[index % len(CATEGORICAL)],
            yerr=[centres - np.asarray(lows), np.asarray(highs) - centres],
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.0, "capsize": 2},
        )
        axes.set_xticks(positions)
        # Rotated because six stratum names on a narrow panel collide when level;
        # a colliding label is worse than a tilted one.
        axes.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
        _style(axes, title=component, xlabel="", ylabel="median (p10-p90)" if index == 0 else "")
    figure.suptitle(
        f"Component values by oracle stratum — {_label(strategy)}",
        color=TEXT_PRIMARY,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return save(figure, directory, name)


def gate_suppression(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    name: str = "figure_gate_suppression",
) -> list[Path]:
    """What the gate removed from, and added to, the same top-K budget."""

    if not rows:
        return []
    figure = _figure(1, 1, width=7.0, height=4.0)
    axes = figure.axes[0]
    labels = [f"{row.get('imbalance_setting', '?')}\nseed {row.get('seed', '?')}" for row in rows]
    metrics = (
        ("suppressed_isolated", "Isolated, removed by the gate", CATEGORICAL[1]),
        ("promoted_true_unknown", "True unknown, added by the gate", CATEGORICAL[2]),
        ("promoted_tail", "Tail object, added by the gate", CATEGORICAL[0]),
    )
    positions = np.arange(len(labels))
    width = 0.8 / len(metrics)
    for index, (key, label, colour) in enumerate(metrics):
        values = [float(row.get(key, 0) or 0) for row in rows]
        axes.bar(
            positions + index * width - 0.4 + width / 2,
            values,
            width=width * 0.88,
            color=colour,
            label=label,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=8)
    _style(
        axes,
        title="Gate counterfactual: ungated rarity ranking vs gated interaction",
        xlabel="",
        ylabel="Proposals in the top-K budget",
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    return save(figure, directory, name)


def class_frequency(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    name: str = "figure_long_tail_protocol",
) -> list[Path]:
    """The constructed long-tail distribution: object count per class, per severity."""

    severities = sorted({str(row.get("imbalance_setting", "")) for row in rows})
    if not severities:
        return []
    figure = _figure(1, len(severities), width=4.4, height=3.4)
    for column, severity in enumerate(severities):
        axes = figure.axes[column]
        members = [row for row in rows if str(row.get("imbalance_setting")) == severity]
        members.sort(key=lambda row: -int(row.get("objects_before", 0) or 0))
        before = [int(row.get("objects_before", 0) or 0) for row in members]
        after = [int(row.get("objects_after", 0) or 0) for row in members]
        positions = np.arange(len(members))
        axes.bar(positions, before, width=0.8, color=GRID, label="natural (reachable)")
        axes.bar(positions, after, width=0.8 * 0.6, color=CATEGORICAL[0], label="retained")
        _style(
            axes,
            title=f"severity: {severity}",
            xlabel="Unknown class, ranked by natural frequency",
            ylabel="Objects" if column == 0 else "",
        )
        if column == 0:
            axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    return save(figure, directory, name)


def ablation_heatmap(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    metric: str = "tail_discovery_auc",
    name: str = "figure_ablation_heatmap",
) -> list[Path]:
    """Gate form x coherence definition, coloured by tail discovery AUC.

    Sequential single hue because the quantity is a magnitude with a meaningful
    zero; a diverging map would imply a neutral midpoint that does not exist.
    """

    cells: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if not _is_number(row.get(metric)):
            continue
        gate = str(row.get("gate_form", "?"))
        method = str(row.get("coherence_method", "?"))
        neighbours = row.get("neighbour_count", "?")
        column = gate if gate != "multiplicative_gate" else f"{method}\nk={neighbours}"
        cells.setdefault((f"gamma={row.get('gamma', '?')}", column), []).append(
            float(row[metric])  # type: ignore[arg-type]
        )
    if not cells:
        return []
    row_labels = sorted({key[0] for key in cells})
    column_labels = sorted({key[1] for key in cells})
    matrix = np.full((len(row_labels), len(column_labels)), np.nan)
    for (row_label, column_label), values in cells.items():
        matrix[row_labels.index(row_label), column_labels.index(column_label)] = float(
            np.mean(values)
        )

    figure = _figure(1, 1, width=1.3 * len(column_labels) + 3.0, height=0.7 * len(row_labels) + 2.6)
    axes = figure.axes[0]
    image = axes.imshow(matrix, cmap=SEQUENTIAL, aspect="auto")
    axes.set_xticks(np.arange(len(column_labels)))
    axes.set_xticklabels(column_labels, fontsize=8)
    axes.set_yticks(np.arange(len(row_labels)))
    axes.set_yticklabels(row_labels, fontsize=8)
    finite = matrix[np.isfinite(matrix)]
    # White text only on the darkest third of the ramp. Using the mean as the
    # switch point puts white on mid-blue cells, where it lands near 3:1 — legible
    # for a large label, not for a three-decimal number.
    threshold = float(finite.min() + 0.62 * (finite.max() - finite.min())) if finite.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                continue
            axes.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if value > threshold else TEXT_PRIMARY,
            )
    axes.grid(False)
    axes.set_title(
        "Ablation: gate form and coherence definition", color=TEXT_PRIMARY, fontsize=11, loc="left"
    )
    bar = figure.colorbar(image, ax=axes, fraction=0.04, pad=0.02)
    bar.set_label(metric, color=TEXT_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    return save(figure, directory, name)


def cost_to_target(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    name: str = "figure_cost_to_target",
) -> list[Path]:
    """Median annotation budget needed to reach a fixed tail discovery level."""

    reached = [row for row in rows if row.get("reached")]
    if not reached:
        return []
    severities = _severities(reached)
    strategies = _strategies(reached)
    figure = _figure(1, 1, width=1.6 * len(severities) + 4.0, height=4.0)
    axes = figure.axes[0]
    width = 0.8 / max(len(strategies), 1)
    for index, strategy in enumerate(strategies):
        values: list[float] = []
        for severity in severities:
            budgets = [
                float(row["budget_to_reach"])  # type: ignore[arg-type]
                for row in reached
                if str(row.get("imbalance_setting")) == severity
                and str(row.get("strategy")) == strategy
            ]
            values.append(float(np.median(budgets)) if budgets else float("nan"))
        positions = np.arange(len(severities)) + index * width - 0.4 + width / 2
        axes.bar(
            positions,
            values,
            width=width * 0.88,
            color=_colour(strategy, index),
            label=_label(strategy),
        )
    axes.set_xticks(np.arange(len(severities)))
    axes.set_xticklabels(severities)
    target = reached[0].get("target", "?")
    _style(
        axes,
        title=f"Annotation cost to reach tail discovery recall {target}",
        xlabel="Long-tail severity",
        ylabel="Annotated regions (lower is better)",
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncols=2)
    return save(figure, directory, name)


def _family_of(strategy: str, rows: Sequence[Mapping[str, object]]) -> str:
    for row in rows:
        if str(row.get("strategy")) == strategy and row.get("strategy_family"):
            return str(row["strategy_family"])
    return "baseline"


def headline_curves(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    metric: str = "all_discovery_recall",
    ylabel: str = "Unknown discovery recall",
    name: str = "figure_headline_comparison",
    strategies: Sequence[str] = HEADLINE_STRATEGIES,
) -> list[Path]:
    """Four arms only: reference, baseline, free control, best new method.

    Four series because the palette's all-pairs gates hold for a handful of hues,
    not for eleven. Everything else is in the family panels and the family-coloured
    bars, so no arm is hidden — it is just not competing for a hue it cannot have.
    """

    severities = _severities(rows)
    present = [name_ for name_ in strategies if any(str(r.get("strategy")) == name_ for r in rows)]
    if not severities or not present:
        return []
    palette = dict(zip(present, CATEGORICAL, strict=False))
    markers = dict(zip(present, MARKERS, strict=False))
    figure = _figure(1, len(severities))
    for column, severity in enumerate(severities):
        axes = figure.axes[column]
        for strategy in present:
            budgets, means, sds = _series(rows, severity=severity, strategy=strategy, metric=metric)
            if budgets.size == 0:
                continue
            colour = palette[strategy]
            axes.fill_between(
                budgets, means - sds, means + sds, color=colour, alpha=0.14, linewidth=0
            )
            axes.plot(
                budgets,
                means,
                color=colour,
                linewidth=2.0,
                marker=markers[strategy],
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=_label(strategy),
            )
        _style(
            axes,
            title=f"severity: {severity}",
            xlabel="Annotated regions (oracle calls)",
            ylabel=ylabel if column == 0 else "",
        )
        axes.set_ylim(bottom=0.0)
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncols=min(len(labels), 2),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, -0.22),
    )
    return save(figure, directory, name)


def family_panels(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    severity: str,
    metric: str = "all_discovery_recall",
    ylabel: str = "Unknown discovery recall",
    name: str = "figure_family_panels",
) -> list[Path]:
    """One panel per strategy family, with the baseline repeated in every panel.

    Small multiples are the documented answer to "more series than hues". Repeating
    the baseline as a grey reference in each panel is what makes the panels
    comparable to one another rather than four separate charts.
    """

    families: dict[str, list[str]] = {}
    for row in rows:
        strategy = str(row.get("strategy", ""))
        if not strategy:
            continue
        families.setdefault(str(row.get("strategy_family", "baseline")), [])
        if strategy not in families[str(row.get("strategy_family", "baseline"))]:
            families[str(row.get("strategy_family", "baseline"))].append(strategy)
    ordered = [family for family in FAMILY_ORDER if family in families]
    ordered += [family for family in sorted(families) if family not in ordered]
    if not ordered:
        return []

    baseline_budgets, baseline_means, _ = _series(
        rows, severity=severity, strategy="v2:full", metric=metric
    )
    columns = min(len(ordered), 3)
    rows_needed = (len(ordered) + columns - 1) // columns
    figure = _figure(rows_needed, columns, width=4.6, height=3.4)
    for index, family in enumerate(ordered):
        axes = figure.axes[index]
        if baseline_budgets.size:
            axes.plot(
                baseline_budgets,
                baseline_means,
                color=TEXT_SECONDARY,
                linewidth=1.4,
                linestyle="--",
                label="baseline v2:full",
            )
        members = [
            strategy for strategy in COMPARISON_ORDER if strategy in families[family]
        ] or families[family]
        for position, strategy in enumerate(members):
            budgets, means, sds = _series(rows, severity=severity, strategy=strategy, metric=metric)
            if budgets.size == 0:
                continue
            colour = CATEGORICAL[position % len(CATEGORICAL)]
            axes.fill_between(
                budgets, means - sds, means + sds, color=colour, alpha=0.12, linewidth=0
            )
            axes.plot(
                budgets,
                means,
                color=colour,
                linewidth=2.0,
                marker=MARKERS[position % len(MARKERS)],
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=_label(strategy),
            )
        _style(axes, title=family, xlabel="Annotated regions", ylabel=ylabel)
        axes.set_ylim(bottom=0.0)
        axes.legend(frameon=False, fontsize=7, labelcolor=TEXT_SECONDARY, loc="upper left")
    for spare in range(len(ordered), len(figure.axes)):
        figure.axes[spare].axis("off")
    figure.suptitle(
        f"Every arm by family — severity {severity} (dashed: the baseline)",
        color=TEXT_PRIMARY,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return save(figure, directory, name)


def family_auc_bars(
    auc_rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    *,
    metric: str = "all_discovery_auc",
    ylabel: str = "Unknown discovery AUC (mean over budgets)",
    name: str = "figure_all_arms_auc",
) -> list[Path]:
    """Every arm as one bar, coloured by family, faceted by severity.

    Eleven bars can coexist because colour is not carrying arm identity here — the
    axis labels do that — it is carrying the family, which is the grouping the
    comparison is about.
    """

    severities = _severities(auc_rows)
    arms = [
        strategy
        for strategy in COMPARISON_ORDER
        if any(str(row.get("strategy")) == strategy for row in auc_rows)
    ]
    arms += sorted(
        {
            str(row.get("strategy"))
            for row in auc_rows
            if str(row.get("strategy")) not in arms and row.get("strategy")
        }
    )
    if not severities or not arms:
        return []
    # A tall, narrow stack is unreadable: with three severities the panels need to
    # be wider than they are tall, and the rotated arm labels need room.
    figure = _figure(len(severities), 1, width=max(8.0, 0.75 * len(arms)), height=2.9)
    for index, severity in enumerate(severities):
        axes = figure.axes[index]
        means: list[float] = []
        errors: list[float] = []
        colours: list[str] = []
        for strategy in arms:
            values = [
                float(row[metric])
                for row in auc_rows
                if str(row.get("imbalance_setting")) == severity
                and str(row.get("strategy")) == strategy
                and _is_number(row.get(metric))
            ]
            means.append(float(np.mean(values)) if values else float("nan"))
            errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            colours.append(FAMILY_COLOURS.get(_family_of(strategy, auc_rows), CATEGORICAL[0]))
        positions = np.arange(len(arms))
        axes.bar(
            positions,
            means,
            width=0.78,
            color=colours,
            yerr=errors,
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.0, "capsize": 2},
        )
        axes.set_xticks(positions)
        # Arm names only under the bottom panel, and the y label only beside the
        # middle one: repeating either on every panel makes three stacked axes
        # overlap each other's text.
        bottom = index == len(severities) - 1
        axes.set_xticklabels(
            [_label(strategy) if bottom else "" for strategy in arms],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        _style(
            axes,
            title=f"severity: {severity}",
            xlabel="",
            ylabel=ylabel if index == len(severities) // 2 else "",
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour)
        for family, colour in FAMILY_COLOURS.items()
        if any(_family_of(strategy, auc_rows) == family for strategy in arms)
    ]
    labels = [
        family
        for family in FAMILY_COLOURS
        if any(_family_of(strategy, auc_rows) == family for strategy in arms)
    ]
    figure.legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        fontsize=8,
        labelcolor=TEXT_SECONDARY,
        ncols=len(labels),
    )
    return save(figure, directory, name)


def render_all(
    *,
    curve_rows: Sequence[Mapping[str, object]],
    auc_rows: Sequence[Mapping[str, object]],
    distribution_rows: Sequence[Mapping[str, object]],
    gate_rows: Sequence[Mapping[str, object]],
    class_frequency_rows: Sequence[Mapping[str, object]],
    ablation_rows: Sequence[Mapping[str, object]],
    cost_rows: Sequence[Mapping[str, object]],
    directory: str | Path,
    headline_severity: str | None = None,
) -> list[Path]:
    """Every figure, returning the paths written.

    The per-severity panels use ``headline_severity``, defaulting to the most
    imbalanced setting that was run — the regime the contribution is about.
    """

    severities = _severities(curve_rows)
    severity = headline_severity or (severities[-1] if severities else "")
    written: list[Path] = []
    written += discovery_curves(curve_rows, directory)
    written += discovery_curves(
        curve_rows,
        directory,
        metric="all_discovery_recall",
        name="figure_unknown_discovery_vs_budget",
        ylabel="Unknown discovery recall (all groups)",
    )
    if severity:
        written += group_discovery_panels(curve_rows, directory, severity=severity)
        written += annotation_efficiency(curve_rows, directory, severity=severity)
        written += unique_classes(curve_rows, directory, severity=severity)
    written += headline_curves(curve_rows, directory)
    if severity:
        written += family_panels(curve_rows, directory, severity=severity)
    written += family_auc_bars(auc_rows, directory)
    written += auc_bars(auc_rows, directory)
    written += auc_bars(
        auc_rows,
        directory,
        metric="all_discovery_auc",
        name="figure_unknown_auc_by_severity",
        ylabel="Unknown discovery AUC (mean over budgets)",
    )
    written += component_distributions(distribution_rows, directory, severity=severity or None)
    written += gate_suppression(gate_rows, directory)
    written += class_frequency(class_frequency_rows, directory)
    written += ablation_heatmap(ablation_rows, directory)
    written += cost_to_target(cost_rows, directory)
    return written
