"""Controlled long-tail construction over an already-built candidate pool.

The Stage-2 protocol disabled the long-tail transformation
(``protocol.long_tail_transformation=none``) to keep real-data comparability, so
no measurement in this repository has ever varied the imbalance severity. The
plan requires exactly that: "a kontrollált tailt szándékos alulmintavételezéssel
állítjuk elő (előre definiált kiegyenlítettlenségi arány mellett)".

Where the ground truth is allowed to be used
--------------------------------------------
Building the evaluation pool is a *protocol* step, not an acquisition step. The
oracle's class labels select which proposals exist; the acquisition strategies
then see only PROB outputs for the surviving proposals. This is the same
licence the plan grants ("a ground-truth osztály csak az evaluation pool
felépítéséhez és az oracle-kiértékeléshez használható") and the same one the
long-tail literature uses to build LT-CIFAR or LVIS-style splits.

The subtle failure mode this module avoids: undersampling *proposals* of a tail
class while leaving the class's other proposals in place would make the tail
easier, not harder, because the surviving proposals would still be locally
supported by their removed neighbours' objects. Undersampling therefore happens
at the level of ground-truth **objects** — every proposal matched to a dropped
object leaves the pool with it, so a suppressed class genuinely has fewer
supporting regions in feature space, which is what the coherence gate has to
cope with.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from daowod.oracle import OracleTable

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class LongTailError(ValueError):
    """Raised when an imbalance specification cannot be satisfied."""


RetentionProfile = Literal["absolute", "relative"]
RETENTION_PROFILES: tuple[str, ...] = ("absolute", "relative")


@dataclass(frozen=True)
class ImbalanceSpec:
    """One long-tail severity.

    ``imbalance_ratio`` is the target ratio between the retained object count of
    the most frequent unknown class and that of the least frequent one. The
    retention profile is exponential in the class's frequency rank, which is the
    standard long-tail construction (Cui et al., Liu et al.): class at rank ``i``
    of ``n`` gets an exponentially decaying target
    ``ratio ** (-i / (n - 1))``.

    ``minimum_objects_per_class`` keeps a suppressed class from vanishing
    entirely — a class with zero objects is not a hard tail class, it is a
    removed class, and it would silently shrink the discovery denominator.

    Two profiles, because one formula cannot move the imbalance in both
    directions
    -------------------------------------------------------------------------
    ``absolute`` anchors the decay on ``head_cap_fraction * max_count``, so the
    target is an absolute object count. This is the profile that can *flatten*:
    capping the head is the only way to bring the head:tail ratio down.

    ``relative`` multiplies each class's *own* count by the decay, so a class is
    never given a target above what it has and the decay always bites. This is
    the profile that can *sharpen*.

    The distinction is forced by a measured property of the data, not by taste.
    On the real 3 500-image S-OWODB Task-1 export the reachable unknown
    distribution is already extreme (head class 73 objects, many classes at 1),
    so an absolute exponential target sits *above* the natural count for most
    middle and tail classes; ``min(target, available)`` then keeps everything and
    ``severe`` silently reproduces ``natural`` — measured head:tail 15.64 versus
    15.44, a 1 % gap. With ``relative`` the same request cuts every non-head
    class by its rank factor and the severity separates.
    """

    name: str
    imbalance_ratio: float
    head_cap_fraction: float = 1.0
    minimum_objects_per_class: int = 1
    protect_known: bool = True
    profile: RetentionProfile = "absolute"

    def __post_init__(self) -> None:
        if not self.name:
            raise LongTailError("An imbalance setting must be named.")
        if self.imbalance_ratio < 1.0:
            raise LongTailError(
                f"{self.name}: imbalance_ratio must be >= 1 "
                f"(1.0 means 'leave the natural distribution alone')."
            )
        if not 0.0 < self.head_cap_fraction <= 1.0:
            raise LongTailError(f"{self.name}: head_cap_fraction must lie in (0, 1].")
        if self.minimum_objects_per_class < 1:
            raise LongTailError(f"{self.name}: minimum_objects_per_class must be >= 1.")
        if self.profile not in RETENTION_PROFILES:
            raise LongTailError(
                f"{self.name}: unknown retention profile {self.profile!r}. "
                f"Supported: {list(RETENTION_PROFILES)}"
            )
        if self.profile == "relative" and self.head_cap_fraction < 1.0:
            raise LongTailError(
                f"{self.name}: head_cap_fraction has no meaning for the relative "
                "profile, which scales each class by its own count. Use the "
                "absolute profile to cap the head."
            )

    @property
    def is_identity(self) -> bool:
        """True when the setting leaves the natural distribution untouched."""

        return self.imbalance_ratio <= 1.0 and self.head_cap_fraction >= 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "imbalance_ratio": self.imbalance_ratio,
            "head_cap_fraction": self.head_cap_fraction,
            "minimum_objects_per_class": self.minimum_objects_per_class,
            "protect_known": self.protect_known,
            "profile": self.profile,
        }


#: Three regimes spanning the imbalance axis in both directions from the natural
#: distribution. ``natural`` is not a throwaway control: real S-OWODB unknowns are
#: already long-tailed, so it is itself a valid long-tail regime and the anchor the
#: other two are read against.
#:
#: * ``moderate`` flattens, by capping the head at 30 % of its natural count;
#: * ``natural`` is the untouched reachable distribution;
#: * ``severe`` sharpens, by cutting each class relative to its own count.
#:
#: Ordered by increasing head:tail object ratio, which is the quantity
#: :func:`validate_settings_distinct` requires to actually differ.
DEFAULT_IMBALANCE_SETTINGS: tuple[ImbalanceSpec, ...] = (
    ImbalanceSpec(name="moderate", imbalance_ratio=3.0, head_cap_fraction=0.3),
    ImbalanceSpec(name="natural", imbalance_ratio=1.0),
    ImbalanceSpec(name="severe", imbalance_ratio=20.0, profile="relative"),
)

#: A flatten-only axis, expressible at *any* pool size.
#:
#: Sharpening requires tail classes that still have objects to lose. Measured: on
#: a 500-image export the reachable tail group holds 8 objects across 7 classes,
#: i.e. it is already at the one-object floor, so no profile can sharpen it and
#: ``DEFAULT_IMBALANCE_SETTINGS`` legitimately fails
#: :func:`validate_settings_distinct` there. Flattening only removes objects from
#: classes that have many, so it always has headroom: this axis measures
#: head:tail 1.75 / 3.22 / 4.89 on 500 images and 5.08 / 11.04 / 15.64 on 3 500.
#: Small-pool modes therefore use this axis instead of pretending to sharpen.
FLATTENING_IMBALANCE_SETTINGS: tuple[ImbalanceSpec, ...] = (
    ImbalanceSpec(name="balanced", imbalance_ratio=1.5, head_cap_fraction=0.12),
    ImbalanceSpec(name="moderate", imbalance_ratio=3.0, head_cap_fraction=0.4),
    ImbalanceSpec(name="natural", imbalance_ratio=1.0),
)

#: Axes tried in order by :func:`choose_axis`, most informative first.
CANDIDATE_AXES: tuple[tuple[str, tuple[ImbalanceSpec, ...]], ...] = (
    ("default", DEFAULT_IMBALANCE_SETTINGS),
    ("flattening", FLATTENING_IMBALANCE_SETTINGS),
)


@dataclass(frozen=True)
class LongTailPool:
    """A pool restricted to a controlled long-tail unknown distribution."""

    keep_mask: BoolArray
    spec: ImbalanceSpec
    retained_objects_by_class: Mapping[str, int]
    original_objects_by_class: Mapping[str, int]
    report: Mapping[str, object]

    @property
    def size(self) -> int:
        return int(self.keep_mask.sum())


def retention_profile(
    class_counts: Mapping[str, int],
    *,
    imbalance_ratio: float,
    head_cap_fraction: float = 1.0,
    minimum_objects_per_class: int = 1,
    profile: RetentionProfile = "absolute",
) -> dict[str, int]:
    """Target retained object count per class for one severity.

    Classes are ranked by descending frequency (ties by name, so the profile is
    deterministic) and the rank ``i`` of ``n`` decays as
    ``decay = imbalance_ratio ** (-i / (n - 1))``. The two profiles differ only in
    what the decay multiplies:

    ``absolute``
        ``round(cap * decay)`` with ``cap = round(head_cap_fraction * max_count)``
        — an absolute object count, so a head cap flattens the distribution.

    ``relative``
        ``round(available * decay)`` — a fraction of the class's own count, so the
        decay always bites and the distribution sharpens.

    Targets are clamped into ``[min(minimum_objects_per_class, available),
    available]``: a class is never *given* objects it does not have, and never
    emptied completely.
    """

    if not class_counts:
        raise LongTailError("Cannot build a retention profile for zero classes.")
    if not 0.0 < head_cap_fraction <= 1.0:
        raise LongTailError("head_cap_fraction must lie in (0, 1].")
    if profile not in RETENTION_PROFILES:
        raise LongTailError(
            f"Unknown retention profile {profile!r}. Supported: {list(RETENTION_PROFILES)}"
        )
    ordered = sorted(class_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    total = len(ordered)
    largest = max(int(value) for value in class_counts.values())
    cap = max(1, int(round(float(head_cap_fraction) * largest)))
    targets: dict[str, int] = {}
    for rank, (class_name, available) in enumerate(ordered):
        available = int(available)
        decay = 1.0 if total == 1 else float(imbalance_ratio) ** (-rank / (total - 1))
        anchor = cap if profile == "absolute" else available
        target = int(round(anchor * decay))
        floor = min(int(minimum_objects_per_class), available)
        targets[class_name] = max(min(target, available), floor)
    return targets


def build_long_tail_pool(
    oracle: OracleTable,
    *,
    spec: ImbalanceSpec,
    seed: int = 0,
    base_mask: ArrayLike | None = None,
    class_groups: Mapping[str, str] | None = None,
) -> LongTailPool:
    """Undersample unknown ground-truth objects to hit a target imbalance.

    Returns a boolean mask over the *pool* proposals. Known-object and background
    proposals are retained untouched when ``spec.protect_known`` is set, because
    the experiment studies the unknown distribution; changing the background rate
    at the same time would confound the outlier-robustness metrics.
    """

    count = oracle.proposal_count
    mask = (
        np.ones(count, dtype=np.bool_)
        if base_mask is None
        else np.asarray(base_mask, dtype=np.bool_).copy()
    )
    if mask.shape != (count,):
        raise LongTailError("base_mask must be parallel to the oracle table.")

    objects_by_class: dict[str, list[int]] = {}
    for index, item in enumerate(oracle.objects):
        if item.is_known:
            continue
        objects_by_class.setdefault(item.class_name, []).append(index)

    # Only objects that some retained proposal actually reaches can be discovered,
    # so the imbalance is defined over the *reachable* objects. Suppressing an
    # object no proposal touches would change the printed ratio and nothing else.
    reachable = {
        int(value)
        for value in oracle.gt_object_index[mask & oracle.gt_is_unknown].tolist()
        if value >= 0
    }
    reachable_by_class = {
        class_name: sorted(index for index in indices if index in reachable)
        for class_name, indices in objects_by_class.items()
    }
    reachable_by_class = {
        class_name: indices for class_name, indices in reachable_by_class.items() if indices
    }
    if not reachable_by_class:
        raise LongTailError(
            "No unknown ground-truth object is reachable from the candidate pool; "
            "the discovery experiment would have an empty denominator."
        )

    original = {name: len(indices) for name, indices in reachable_by_class.items()}
    targets = retention_profile(
        original,
        imbalance_ratio=spec.imbalance_ratio,
        head_cap_fraction=spec.head_cap_fraction,
        minimum_objects_per_class=spec.minimum_objects_per_class,
        profile=spec.profile,
    )

    generator = np.random.default_rng(seed)
    dropped_objects: set[int] = set()
    retained: dict[str, int] = {}
    for class_name in sorted(reachable_by_class):
        indices = reachable_by_class[class_name]
        target = int(targets[class_name])
        retained[class_name] = target
        if target >= len(indices):
            continue
        keep_positions = generator.choice(len(indices), size=target, replace=False)
        keep_set = {indices[int(position)] for position in keep_positions}
        dropped_objects.update(index for index in indices if index not in keep_set)

    if dropped_objects:
        dropped = np.isin(oracle.gt_object_index, np.fromiter(dropped_objects, dtype=np.int64))
        mask &= ~dropped

    achieved = sorted(retained.values(), reverse=True)
    achieved_ratio = (
        float(achieved[0] / achieved[-1]) if achieved and achieved[-1] > 0 else float("inf")
    )
    # A requested ratio can exceed what the data can express: if the most frequent
    # reachable class has H objects, no per-class integer profile floored at 1 can
    # realise a ratio above H. Reporting the ceiling stops a run from claiming a
    # severity it never reached.
    head_objects = max(original.values()) if original else 0
    expressible = float(head_objects) if head_objects > 0 else 1.0
    report = {
        "setting": spec.name,
        "requested_imbalance_ratio": spec.imbalance_ratio,
        "achieved_imbalance_ratio": achieved_ratio,
        "maximum_expressible_ratio": expressible,
        "imbalance_ratio_saturated": bool(spec.imbalance_ratio > expressible),
        "unknown_classes": len(retained),
        "unknown_objects_before": int(sum(original.values())),
        "unknown_objects_after": int(sum(retained.values())),
        "proposals_before": int(
            np.asarray(base_mask, dtype=np.bool_).sum() if base_mask is not None else count
        ),
        "proposals_after": int(mask.sum()),
        "dropped_objects": len(dropped_objects),
        "seed": seed,
        "spec": spec.as_dict(),
    }
    if class_groups:
        # Group-level object counts are the severity the metrics actually see; a
        # per-class ratio can look unchanged while the head/tail mass shifts.
        for group in ("head", "medium", "tail"):
            members = [name for name in retained if str(class_groups.get(name, "")) == group]
            report[f"{group}_objects_after"] = int(sum(retained[name] for name in members))
            report[f"{group}_objects_before"] = int(sum(original.get(name, 0) for name in members))
        tail_mass = int(report.get("tail_objects_after", 0))
        head_mass = int(report.get("head_objects_after", 0))
        report["head_to_tail_object_ratio"] = (
            float(head_mass / tail_mass) if tail_mass > 0 else float("inf")
        )
    return LongTailPool(
        keep_mask=mask,
        spec=spec,
        retained_objects_by_class=retained,
        original_objects_by_class=original,
        report=report,
    )


def describe_settings(
    oracle: OracleTable,
    settings: Sequence[ImbalanceSpec],
    *,
    class_groups: Mapping[str, str],
    seed: int = 0,
    base_mask: ArrayLike | None = None,
) -> list[dict[str, object]]:
    """Build each setting once and report the imbalance it actually achieved.

    Used by the notebook to state, rather than assume, that the severities are
    distinct. If two settings land on the same head:tail ratio the contrast
    between them is not measurable and the run should say so.
    """

    rows: list[dict[str, object]] = []
    for spec in settings:
        pool = build_long_tail_pool(
            oracle, spec=spec, seed=seed, base_mask=base_mask, class_groups=class_groups
        )
        rows.append(dict(pool.report))
    return rows


def settings_are_distinct(
    reports: Sequence[Mapping[str, object]], *, minimum_relative_gap: float = 0.15
) -> tuple[bool, str]:
    """Whether the achieved head:tail ratios differ enough to be contrasted.

    Every *pair* must differ, not merely the extremes: a run with three settings
    where two coincide reports two severities' worth of evidence under three
    labels, which is the failure mode measured on real data before the relative
    retention profile existed (``natural`` 15.64 versus ``severe`` 15.44).
    """

    ratios = [float(report.get("head_to_tail_object_ratio", float("nan"))) for report in reports]
    names = [str(report.get("setting", "?")) for report in reports]
    finite = [
        (name, value) for name, value in zip(names, ratios, strict=True) if np.isfinite(value)
    ]
    if len(finite) < 2:
        return False, (
            "Fewer than two settings produced a finite head:tail object ratio, so "
            "no severity contrast can be measured."
        )
    if any(value <= 0 for _, value in finite):
        empty = sorted(name for name, value in finite if value <= 0)
        return False, f"These settings retained no tail objects at all: {empty}."

    finite.sort(key=lambda item: item[1])
    collapsed: list[str] = []
    for (low_name, low), (high_name, high) in zip(finite, finite[1:], strict=False):
        gap = (high - low) / low
        if gap < minimum_relative_gap:
            collapsed.append(f"{low_name} ({low:.2f}) vs {high_name} ({high:.2f}), gap {gap:.3f}")
    span = (finite[-1][1] - finite[0][1]) / finite[0][1]
    summary = (
        "head:tail object ratio "
        + ", ".join(f"{name}={value:.2f}" for name, value in finite)
        + f"; full span {span:.2f}"
    )
    if collapsed:
        return False, summary + "; indistinguishable pairs: " + "; ".join(collapsed)
    return True, summary


def validate_settings_distinct(
    reports: Sequence[Mapping[str, object]], *, minimum_relative_gap: float = 0.15
) -> str:
    """Raise unless every severity pair is measurably different.

    The error names the pairs that collapsed, the ratio each achieved, whether the
    request saturated, and the ceiling the data can express, because those four
    facts determine the fix: sharpen with ``profile="relative"``, flatten with a
    smaller ``head_cap_fraction``, or export more images so the head has the
    object mass a steeper ratio needs.
    """

    distinct, message = settings_are_distinct(reports, minimum_relative_gap=minimum_relative_gap)
    if distinct:
        return message
    detail = "\n".join(
        f"  - {report.get('setting', '?')}: requested ratio "
        f"{report.get('requested_imbalance_ratio', '?')} "
        f"({(report.get('spec') or {}).get('profile', 'absolute')} profile), "
        f"achieved per-class {float(report.get('achieved_imbalance_ratio', float('nan'))):.2f}, "
        f"head:tail objects "
        f"{float(report.get('head_to_tail_object_ratio', float('nan'))):.2f}, "
        f"tail objects {report.get('tail_objects_after', '?')}, "
        f"saturated={report.get('imbalance_ratio_saturated')}, "
        f"maximum expressible per-class ratio "
        f"{report.get('maximum_expressible_ratio', '?')}"
        for report in reports
    )
    raise LongTailError(
        "The requested long-tail severities are not measurably different, so "
        "comparing them would report the same regime under different names.\n"
        f"{message}\n{detail}\n"
        "Fix one of: use profile='relative' with a larger imbalance_ratio to "
        "sharpen; lower head_cap_fraction to flatten; or export more images so "
        "the head classes carry the object mass a steeper ratio requires."
    )


def class_frequency_rows(
    pool: LongTailPool,
    *,
    class_groups: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """Rows for ``class_frequency_report.csv``: before/after counts and group."""

    rows: list[dict[str, object]] = []
    for class_name in sorted(
        pool.original_objects_by_class,
        key=lambda name: (-int(pool.original_objects_by_class[name]), name),
    ):
        rows.append(
            {
                "imbalance_setting": pool.spec.name,
                "class_name": class_name,
                "group": (class_groups or {}).get(class_name, ""),
                "objects_before": int(pool.original_objects_by_class[class_name]),
                "objects_after": int(pool.retained_objects_by_class.get(class_name, 0)),
            }
        )
    return rows


def discoverable_objects(oracle: OracleTable, keep_mask: ArrayLike) -> dict[str, set[int]]:
    """Per-group sets of unknown object indices reachable inside a pool.

    These sets are the *denominators* of every discovery-recall metric. Computing
    them from the pool rather than from the annotations is essential: an unknown
    object that no candidate proposal covers can never be found by any strategy,
    so including it would depress every recall equally and hide the contrast the
    experiment is measuring.
    """

    mask = np.asarray(keep_mask, dtype=np.bool_)
    selected = mask & oracle.gt_is_unknown
    groups: dict[str, set[int]] = {"all": set(), "head": set(), "medium": set(), "tail": set()}
    for index, group in zip(
        oracle.gt_object_index[selected].tolist(),
        oracle.gt_group[selected].tolist(),
        strict=True,
    ):
        if int(index) < 0:
            continue
        groups["all"].add(int(index))
        name = str(group)
        if name in groups:
            groups[name].add(int(index))
    return groups


def unknown_classes_present(oracle: OracleTable, keep_mask: ArrayLike) -> dict[str, set[str]]:
    """Per-group sets of unknown class names reachable inside a pool."""

    mask = np.asarray(keep_mask, dtype=np.bool_)
    selected = mask & oracle.gt_is_unknown
    classes: dict[str, set[str]] = {"all": set(), "head": set(), "medium": set(), "tail": set()}
    for class_name, group in zip(
        oracle.gt_class[selected].tolist(), oracle.gt_group[selected].tolist(), strict=True
    ):
        name = str(class_name)
        if not name:
            continue
        classes["all"].add(name)
        group_name = str(group)
        if group_name in classes:
            classes[group_name].add(name)
    return classes


def choose_axis(
    oracle: OracleTable,
    *,
    class_groups: Mapping[str, str],
    axes: Sequence[tuple[str, Sequence[ImbalanceSpec]]] = CANDIDATE_AXES,
    seed: int = 0,
    base_mask: ArrayLike | None = None,
    minimum_relative_gap: float = 0.15,
) -> tuple[str, tuple[ImbalanceSpec, ...], list[dict[str, object]], str]:
    """First axis in ``axes`` whose severities are measurably distinct.

    Opt-in only. The default path validates the *requested* axis and fails, per
    the protocol requirement that a run may not silently substitute a severity it
    can express for the one that was asked for. This exists so a small pilot can
    still produce a three-severity contrast, and it returns the axis name so the
    run reports which one it used.
    """

    attempts: list[str] = []
    for name, settings in axes:
        rows = describe_settings(
            oracle, list(settings), class_groups=class_groups, seed=seed, base_mask=base_mask
        )
        distinct, message = settings_are_distinct(rows, minimum_relative_gap=minimum_relative_gap)
        attempts.append(f"{name}: {'distinct' if distinct else 'collapsed'} — {message}")
        if distinct:
            return name, tuple(settings), rows, "; ".join(attempts)
    raise LongTailError(
        "No candidate severity axis is expressible on this pool:\n  "
        + "\n  ".join(attempts)
        + "\nExport more images so the head classes carry more object mass, or "
        "raise per_image_limit so more unknown objects become reachable."
    )


def resolve_budgets(requested: Sequence[int], *, pool_size: int) -> list[int]:
    """Clamp the requested annotation budgets to what the pool can supply."""

    if pool_size < 1:
        raise LongTailError("Cannot resolve budgets for an empty pool.")
    usable = sorted(
        {int(value) for value in requested if int(value) > 0 and int(value) <= pool_size}
    )
    if not usable:
        raise LongTailError(
            f"None of the requested budgets {list(requested)} fits a pool of {pool_size}."
        )
    return usable
