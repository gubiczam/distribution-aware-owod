"""Contribution B — distribution-aware exemplar allocation.

The proposal's rule, from section *B) Eloszlás-tudatos exemplar-allokáció az
inkrementális frissítésben*: for a class ``c`` of size ``n_c``, the exemplar
allocation ``m_c`` is proportional to ``n_c ** alpha``, subject to ``sum(m_c) = M``
for a fixed total memory ``M``. The exponent interpolates between strategies:

    alpha = 0   uniform — the current standard, equal exemplars per class
    alpha = 1   proportional to class size — head-favouring
    alpha < 0   tail-favouring

The research question (H-B1) is what ``alpha`` is optimal in OWOD, where classes
become known incrementally out of the unknowns, and how that optimum moves with
tail severity. The proposal also requires both granularities (H-B2): "egy eltárolt
kép több objektumot (több osztályt) tartalmazhat", so *exemplars per class* is
meaningful at object level and at image level.

Scope of this module, stated plainly
------------------------------------
**This is the allocation mathematics only, and it does not make Contribution B
experimentally complete.** Answering H-B1 requires real incremental model updates
with PROB retraining and per-group forgetting measured across tasks. That is a
separate, documented future step (`docs/research_design.md` section 8). No offline
proxy for catastrophic forgetting is provided here, because a proxy would not
answer the question — forgetting is a property of a trained model, not of a
buffer.

What a future replay trainer needs from this module is one value:
:class:`ExemplarAllocation`. It carries the per-class exemplar counts and, for the
image-level case, the concrete image IDs to store. Both are plain data, so the
integration point is a function signature rather than a framework.

Two properties that are easy to get wrong
-----------------------------------------
**Exact integer conservation.** ``M`` is a number of stored items, so the
allocation must be integral and must sum to ``M`` exactly — not to ``M ± 1`` after
rounding. :func:`allocate` uses the largest-remainder method, which is the only
common apportionment rule that guarantees it.

**Determinism.** Ties in the remainder are broken by a declared, stable order, not
by dict insertion order or floating-point luck. An allocation that varies run to
run would make a forgetting measurement irreproducible, and the difference would
look like an effect of ``alpha``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Reported grid for the alpha sweep: tail-favouring, uniform (the standard),
#: through size-proportional (head-favouring). Declared here so a sweep cannot
#: quietly report a different grid than the one the protocol names.
ALPHA_GRID: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

#: The two granularities the proposal requires.
GRANULARITIES: tuple[str, ...] = ("object", "image")


class AllocationError(ValueError):
    """Raised for an invalid memory budget, class-count table or exponent."""


@dataclass(frozen=True)
class ExemplarAllocation:
    """How a fixed memory budget is spent. The hand-off to a replay trainer.

    ``counts`` maps class name to the number of exemplars to keep. ``image_ids`` is
    populated for image-level allocation and empty for object-level, where the
    choice of *which* objects to keep is a herding decision that belongs to the
    trainer, not to the budget split.
    """

    alpha: float
    total_memory: int
    granularity: str
    counts: Mapping[str, int]
    image_ids: tuple[str, ...] = ()
    #: Classes whose ideal share exceeded the objects they actually have, and were
    #: therefore capped. Recorded because a capped allocation no longer follows
    #: ``n_c ** alpha``, and a sweep that ignored it would mis-attribute the effect.
    capped_classes: tuple[str, ...] = ()
    #: Per-class shortfall at image level, where whole images must be stored and
    #: exact per-class quotas are generally unachievable.
    shortfall: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.granularity not in GRANULARITIES:
            raise AllocationError(
                f"unknown granularity {self.granularity!r}. Supported: {list(GRANULARITIES)}"
            )
        if any(value < 0 for value in self.counts.values()):
            raise AllocationError("exemplar counts must be non-negative.")
        # Conservation is checked on whichever quantity the budget counts, which is
        # the whole point of distinguishing the two granularities: object-level
        # spends its budget on objects, image-level on whole images.
        spent = sum(self.counts.values()) if self.granularity == "object" else len(self.image_ids)
        if spent != self.total_memory:
            unit = "exemplars" if self.granularity == "object" else "images"
            raise AllocationError(
                f"allocation spends {spent} {unit}, not the requested total_memory "
                f"{self.total_memory}. Conservation is not optional."
            )
        if self.granularity == "image" and len(set(self.image_ids)) != len(self.image_ids):
            raise AllocationError("image_ids must not repeat: an image is stored once.")

    def as_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "total_memory": self.total_memory,
            "granularity": self.granularity,
            "counts": dict(sorted(self.counts.items())),
            "image_ids": list(self.image_ids),
            "capped_classes": list(self.capped_classes),
            "shortfall": dict(sorted(self.shortfall.items())) if self.shortfall else None,
        }


def _validated_counts(class_counts: Mapping[str, int]) -> dict[str, int]:
    if not class_counts:
        raise AllocationError("class_counts must not be empty.")
    validated: dict[str, int] = {}
    for name, count in class_counts.items():
        value = int(count)
        if value < 0:
            raise AllocationError(f"class {name!r}: count must be non-negative, got {value}.")
        validated[str(name)] = value
    if sum(validated.values()) == 0:
        raise AllocationError("class_counts are all zero: there is nothing to allocate.")
    return validated


def ideal_shares(class_counts: Mapping[str, int], alpha: float) -> dict[str, float]:
    """The real-valued shares ``n_c ** alpha / sum(n ** alpha)``.

    Classes with no objects get a share of zero at every ``alpha``. That is a
    definition, not an approximation: ``0 ** alpha`` is zero for positive alpha and
    undefined (infinite) for negative alpha, and a class with nothing stored cannot
    receive exemplars either way.
    """

    counts = _validated_counts(class_counts)
    if not math.isfinite(float(alpha)):
        raise AllocationError(f"alpha must be finite, got {alpha!r}.")

    weights = {
        name: (float(count) ** float(alpha) if count > 0 else 0.0) for name, count in counts.items()
    }
    total = math.fsum(weights.values())
    if total <= 0.0:
        raise AllocationError(f"alpha={alpha} gives every class zero weight.")
    return {name: weight / total for name, weight in weights.items()}


def allocate(
    class_counts: Mapping[str, int],
    *,
    total_memory: int,
    alpha: float,
    respect_availability: bool = True,
) -> ExemplarAllocation:
    """Split ``total_memory`` across classes as ``m_c`` proportional to ``n_c ** alpha``.

    Guarantees, all asserted by the tests:

    * ``sum(counts.values()) == total_memory`` exactly, for every alpha and every
      count table;
    * the result is a deterministic function of the inputs, with remainder ties
      broken by descending remainder then ascending class name;
    * with ``respect_availability`` (the default) no class is allocated more
      exemplars than it has objects, and the freed budget is redistributed among
      classes that can still take it.

    The availability cap matters for exactly the case Contribution B is about: at
    ``alpha < 0`` a tail class of two objects can be allotted far more than two, and
    an uncapped number would describe a buffer that cannot be filled.
    """

    counts = _validated_counts(class_counts)
    budget = int(total_memory)
    if budget < 0:
        raise AllocationError(f"total_memory must be non-negative, got {budget}.")
    available = sum(counts.values())
    if respect_availability and budget > available:
        raise AllocationError(
            f"total_memory {budget} exceeds the {available} objects available across "
            "all classes; there is nothing to fill the remainder with."
        )

    eligible = {name for name, count in counts.items() if count > 0}
    capped: set[str] = set()
    allocation: dict[str, int] = dict.fromkeys(counts, 0)
    remaining = budget

    # Iterate because capping one class frees budget that may cap another.
    while remaining > 0 and eligible:
        shares = ideal_shares({name: counts[name] for name in eligible}, alpha)
        fresh = _largest_remainder({name: shares[name] for name in eligible}, remaining)
        if not respect_availability:
            for name, value in fresh.items():
                allocation[name] += value
            remaining = 0
            break

        overflowed = False
        for name, value in fresh.items():
            room = counts[name] - allocation[name]
            granted = min(value, room)
            allocation[name] += granted
            remaining -= granted
            if granted < value:
                overflowed = True
        for name in list(eligible):
            if allocation[name] >= counts[name]:
                eligible.discard(name)
                capped.add(name)
        if not overflowed and remaining > 0 and not eligible:
            break

    if remaining > 0:
        raise AllocationError(
            f"could not place {remaining} of {budget} exemplars; the count table "
            "cannot absorb the requested memory."
        )

    return ExemplarAllocation(
        alpha=float(alpha),
        total_memory=budget,
        granularity="object",
        counts=allocation,
        capped_classes=tuple(sorted(capped)),
    )


def _largest_remainder(shares: Mapping[str, float], budget: int) -> dict[str, int]:
    """Apportion ``budget`` integer units over real-valued shares.

    Largest remainder (Hare quota): floor everything, then hand the leftover units
    to the largest fractional parts. This is what makes the sum exact.

    The tie-break is ``(-remainder, name)`` — descending remainder, then ascending
    class name. Sorting by name rather than by insertion order is what makes the
    allocation reproducible across runs and across Python versions.
    """

    exact = {name: shares[name] * budget for name in shares}
    floors = {name: int(math.floor(value)) for name, value in exact.items()}
    leftover = budget - sum(floors.values())
    if leftover:
        order = sorted(exact, key=lambda name: (-(exact[name] - floors[name]), name))
        for name in order[:leftover]:
            floors[name] += 1
    return floors


def allocate_images(
    image_classes: Mapping[str, Sequence[str]],
    *,
    total_memory: int,
    alpha: float,
) -> ExemplarAllocation:
    """Image-level allocation: choose whole images to satisfy per-class quotas.

    ``total_memory`` counts *images* here, and ``image_classes`` maps an image ID to
    the classes its objects belong to. Object-level quotas come from the same
    ``n_c ** alpha`` rule, computed over the object counts implied by
    ``image_classes``.

    Why this cannot be exact, and what it does instead
    --------------------------------------------------
    A stored image brings *all* of its objects, so one image can serve several
    classes at once and no selection of whole images generally hits every per-class
    quota. This is precisely the ambiguity H-B2 names. The selector is therefore
    greedy and deterministic: it repeatedly takes the image that most reduces the
    total remaining shortfall, breaking ties by ascending image ID, and it records
    the residual per class in :attr:`ExemplarAllocation.shortfall` so a sweep can
    report how far from the ideal split the image-level variant actually landed
    rather than assuming it matched.
    """

    if not image_classes:
        raise AllocationError("image_classes must not be empty.")
    budget = int(total_memory)
    if budget < 0:
        raise AllocationError(f"total_memory must be non-negative, got {budget}.")
    if budget > len(image_classes):
        raise AllocationError(
            f"total_memory {budget} exceeds the {len(image_classes)} images available."
        )

    per_image = {
        str(name): tuple(str(value) for value in classes) for name, classes in image_classes.items()
    }
    object_counts: dict[str, int] = {}
    for classes in per_image.values():
        for class_name in classes:
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

    # The budget counts images, but the quota is per class, i.e. per object. Convert
    # with the pool's own mean objects-per-image rather than treating one image as one
    # object: under-scaled quotas would sit at zero for most classes and the greedy
    # step below would have no signal left to act on, degenerating to alphabetical
    # order. This is an estimate of storable objects, and `shortfall` reports what the
    # selection actually achieved against it.
    total_objects = sum(object_counts.values())
    storable = max(1, round(budget * total_objects / len(per_image)))
    quota = allocate(
        object_counts,
        total_memory=min(storable, total_objects),
        alpha=alpha,
        respect_availability=True,
    )
    need = dict(quota.counts)

    chosen: list[str] = []
    kept_counts: dict[str, int] = dict.fromkeys(object_counts, 0)
    candidates = sorted(per_image)
    for _ in range(budget):
        best_name, best_gain = None, -1
        for name in candidates:
            if name in chosen:
                continue
            gain = sum(1 for c in per_image[name] if kept_counts[c] < need.get(c, 0))
            if gain > best_gain:
                best_name, best_gain = name, gain
        if best_name is None:
            break
        chosen.append(best_name)
        for class_name in per_image[best_name]:
            kept_counts[class_name] += 1

    shortfall = {
        name: max(0, need.get(name, 0) - kept_counts.get(name, 0)) for name in sorted(object_counts)
    }
    return ExemplarAllocation(
        alpha=float(alpha),
        total_memory=len(chosen),
        granularity="image",
        # Objects actually retained per class, which is what a replay trainer sees.
        # It does not equal the ideal quota, and `shortfall` says by how much.
        counts={name: kept_counts[name] for name in sorted(object_counts)},
        image_ids=tuple(chosen),
        capped_classes=quota.capped_classes,
        shortfall=shortfall,
    )


def sweep(
    class_counts: Mapping[str, int],
    *,
    total_memory: int,
    alphas: Sequence[float] = ALPHA_GRID,
) -> list[ExemplarAllocation]:
    """Every allocation on the declared alpha grid, for one memory budget.

    This is the input to H-B1's experiment, not the experiment: it produces the
    buffers whose effect on forgetting a future incremental run must measure.
    """

    return [allocate(class_counts, total_memory=total_memory, alpha=alpha) for alpha in alphas]
