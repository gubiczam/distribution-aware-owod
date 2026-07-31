"""Frequency-group (head / medium / tail) bookkeeping for long-tail evaluation.

The authoritative source is the ``class_stats.csv`` written by
:func:`daowod.dataset.build_long_tail_pool`, which assigns every task class a
contiguous rank group. Groups are loaded once, validated loudly, and then passed
to the metric functions; no metric function is allowed to invent a group or to
skip a class it cannot place.
"""

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

GROUP_NAMES: tuple[str, ...] = ("head", "medium", "tail")


class GroupError(ValueError):
    """Raised when a frequency-group mapping is missing, ambiguous or partial."""


@dataclass(frozen=True)
class ClassGroups:
    """An immutable, validated ``class name -> frequency group`` mapping."""

    groups: Mapping[str, str]
    source: str

    def __post_init__(self) -> None:
        unknown = sorted({group for group in self.groups.values() if group not in GROUP_NAMES})
        if unknown:
            raise GroupError(
                f"{self.source}: unsupported frequency groups {unknown}; "
                f"expected only {list(GROUP_NAMES)}."
            )
        if not self.groups:
            raise GroupError(f"{self.source}: the frequency-group mapping is empty.")

    @classmethod
    def from_class_stats_csv(cls, path: str | Path) -> "ClassGroups":
        """Load the mapping from a ``class_stats.csv`` long-tail artifact."""

        source = str(path)
        rows = _read_class_stats(Path(path), source)
        groups: dict[str, str] = {}
        for class_name, group in rows:
            previous = groups.get(class_name)
            if previous is not None and previous != group:
                raise GroupError(
                    f"{source}: class {class_name!r} is assigned to both "
                    f"{previous!r} and {group!r}; every class must have exactly "
                    "one group."
                )
            if previous is not None:
                raise GroupError(f"{source}: class {class_name!r} appears more than once.")
            groups[class_name] = group
        return cls(groups=groups, source=source)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str], *, source: str) -> "ClassGroups":
        return cls(groups=dict(mapping), source=source)

    def group_of(self, class_name: str) -> str:
        try:
            return self.groups[class_name]
        except KeyError as error:
            raise GroupError(
                f"{self.source}: class {class_name!r} has no frequency group."
            ) from error

    def members(self, group: str) -> list[str]:
        if group not in GROUP_NAMES:
            raise GroupError(f"Unsupported frequency group: {group!r}")
        return sorted(name for name, value in self.groups.items() if value == group)

    def require_covers(self, class_names: Iterable[str], *, context: str) -> None:
        """Fail loudly when an evaluated class cannot be placed in a group.

        Silently dropping such a class is the failure mode this guards against:
        it would make grouped recall look complete while quietly excluding part
        of the ground truth.
        """

        requested = sorted(set(class_names))
        missing = [name for name in requested if name not in self.groups]
        if missing:
            raise GroupError(
                f"{context}: {len(missing)} evaluated class(es) have no frequency "
                f"group in {self.source}: {missing}. Regenerate the long-tail "
                "protocol or pass the matching class_stats.csv."
            )

    def counts(self) -> dict[str, int]:
        return {
            group: sum(1 for value in self.groups.values() if value == group)
            for group in GROUP_NAMES
        }

    def as_dict(self) -> dict[str, str]:
        return dict(self.groups)


def _read_class_stats(path: Path, source: str) -> list[tuple[str, str]]:
    if not path.exists():
        raise GroupError(f"Missing long-tail class statistics: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        for required in ("class_name", "group"):
            if required not in fieldnames:
                raise GroupError(f"{source}: column {required!r} is missing; found {fieldnames}.")
        rows: list[tuple[str, str]] = []
        for number, row in enumerate(reader, start=2):
            class_name = (row.get("class_name") or "").strip()
            group = (row.get("group") or "").strip()
            if not class_name:
                raise GroupError(f"{source}: line {number} has an empty class_name.")
            if not group:
                raise GroupError(f"{source}: line {number} has no group for {class_name!r}.")
            rows.append((class_name, group))
    if not rows:
        raise GroupError(f"{source}: no class rows were found.")
    return rows
