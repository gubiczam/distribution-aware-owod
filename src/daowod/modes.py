"""The three execution modes, as data rather than as notebook branches.

A mode fixes every quantity that trades runtime against statistical power:
how many images are inferred, how many proposals per image survive the candidate
filter, the annotation budget grid, the round count, the seeds, and which
severity axis is used. The notebook picks a mode name; nothing else in the
pipeline asks "am I in debug?".

Why each mode exists
--------------------
``DEBUG``
    Correctness only. Small enough to finish in seconds on a CPU, so the whole
    pipeline — including plots, CSVs and the ZIP — can be exercised before a GPU
    is booked. Its numbers are not research results and the summary says so.

``FAST``
    A real but under-powered measurement: every strategy, every severity, fewer
    seeds and a smaller pool. Used to confirm the contrast has the expected sign
    before spending the full session, and as the fallback when the runtime
    estimate says ``MAIN`` will not fit.

``MAIN``
    The reported experiment, sized for one NVIDIA T4 in roughly four to five
    hours: 4 000 images split into disjoint reference / pilot / evaluation pools,
    five strategies, three severities, three seeds, plus the ablation grid.

Measured facts behind the numbers
---------------------------------
On the real S-OWODB Task-1 export (100 queries per image, ``per_image_limit=20``):

* 500 images give 10 000 candidates, 104 reachable unknown objects and a tail
  group of 8 objects — enough to see a trend, too few to resolve one;
* 3 500 images give 70 000 candidates, 508 reachable unknown objects across 44
  classes and a tail group of 25 objects, which does resolve a discovery curve;
* the sharpening severity is only expressible on the larger pool, so ``DEBUG``
  and ``FAST`` use the flatten-only axis (see
  :data:`daowod.longtail.FLATTENING_IMBALANCE_SETTINGS`) rather than requesting a
  severity the data cannot deliver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from daowod import longtail
from daowod.annotation_study import COMPARISON_STRATEGIES, PRIMARY_STRATEGIES, StudyConfig
from daowod.candidates import CandidatePoolSpec
from daowod.longtail import ImbalanceSpec

#: The three modes the notebook offers. :func:`register` can add more — a name is
#: a label, and the registry is the only thing that decides what ``resolve_mode``
#: accepts, so a custom size does not need this tuple edited.
MODE_NAMES: tuple[str, ...] = ("DEBUG", "FAST", "MAIN", "MAINREVEALED")


class ModeError(ValueError):
    """Raised for an unknown or internally inconsistent execution mode."""


@dataclass(frozen=True)
class ExecutionMode:
    """One complete, self-consistent experiment size."""

    name: str
    description: str
    evaluation_images: int
    pilot_images: int
    reference_images: int
    per_image_limit: int
    budgets: tuple[int, ...]
    rounds: int
    seeds: tuple[int, ...]
    strategies: tuple[str, ...]
    imbalance_settings: tuple[ImbalanceSpec, ...]
    run_pilot: bool
    run_ablations: bool
    ablation_seeds: tuple[int, ...]
    reference_limit: int
    runtime_budget_seconds: float
    research_grade: bool

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.upper():
            raise ModeError(f"A mode name must be non-empty and upper-case, got {self.name!r}.")
        if min(self.evaluation_images, self.reference_images) < 1:
            raise ModeError(f"{self.name}: image counts must be positive.")
        if self.pilot_images < 0:
            raise ModeError(f"{self.name}: pilot_images must be non-negative.")
        if self.run_pilot and self.pilot_images < 1:
            raise ModeError(f"{self.name}: a pilot needs pilot_images >= 1.")
        if not self.budgets or min(self.budgets) < 1:
            raise ModeError(f"{self.name}: budgets must be positive.")
        if not self.seeds:
            raise ModeError(f"{self.name}: at least one seed is required.")
        if len(self.imbalance_settings) < 2:
            raise ModeError(
                f"{self.name}: at least two severities are required for a long-tail contrast."
            )

    @property
    def total_images(self) -> int:
        """Images the detector must be run over — what the GPU time buys."""

        return self.evaluation_images + self.pilot_images + self.reference_images

    def study_config(self) -> StudyConfig:
        """The :class:`~daowod.annotation_study.StudyConfig` this mode implies."""

        return StudyConfig(
            budgets=tuple(self.budgets),
            rounds=self.rounds,
            seeds=tuple(self.seeds),
            strategies=tuple(self.strategies),
            imbalance_settings=tuple(self.imbalance_settings),
            candidate_spec=CandidatePoolSpec(per_image_limit=self.per_image_limit),
            reference_limit=self.reference_limit,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "evaluation_images": self.evaluation_images,
            "pilot_images": self.pilot_images,
            "reference_images": self.reference_images,
            "total_images": self.total_images,
            "per_image_limit": self.per_image_limit,
            "budgets": list(self.budgets),
            "rounds": self.rounds,
            "seeds": list(self.seeds),
            "strategies": list(self.strategies),
            "imbalance_settings": [spec.as_dict() for spec in self.imbalance_settings],
            "run_pilot": self.run_pilot,
            "run_ablations": self.run_ablations,
            "ablation_seeds": list(self.ablation_seeds),
            "reference_limit": self.reference_limit,
            "runtime_budget_seconds": self.runtime_budget_seconds,
            "research_grade": self.research_grade,
        }


DEBUG_MODE = ExecutionMode(
    name="DEBUG",
    description=(
        "Smallest run that exercises every stage end to end. Not a research "
        "result: two seeds and a 200-image evaluation pool cannot separate "
        "strategies. The size is not arbitrary — below about 150 evaluation "
        "images the reachable tail group falls under four objects and no severity "
        "axis is expressible, so the run would stop at the severity validation "
        "rather than exercising the stages after it."
    ),
    evaluation_images=200,
    pilot_images=60,
    reference_images=150,
    per_image_limit=12,
    budgets=(25, 50, 100),
    rounds=2,
    seeds=(0, 1),
    strategies=PRIMARY_STRATEGIES,
    imbalance_settings=longtail.FLATTENING_IMBALANCE_SETTINGS,
    run_pilot=True,
    run_ablations=True,
    ablation_seeds=(0,),
    reference_limit=4_000,
    runtime_budget_seconds=15 * 60,
    research_grade=False,
)

FAST_MODE = ExecutionMode(
    name="FAST",
    description=(
        "Under-powered but real: all five strategies and three severities on a "
        "500-image evaluation pool with two seeds. Confirms the sign of the "
        "contrast; report it as a pilot, not as the headline."
    ),
    evaluation_images=500,
    pilot_images=150,
    reference_images=350,
    per_image_limit=20,
    budgets=(50, 100, 200, 400),
    rounds=4,
    seeds=(0, 1),
    strategies=PRIMARY_STRATEGIES,
    imbalance_settings=longtail.FLATTENING_IMBALANCE_SETTINGS,
    run_pilot=True,
    run_ablations=True,
    ablation_seeds=(0,),
    reference_limit=10_000,
    runtime_budget_seconds=60 * 60,
    research_grade=False,
)

MAIN_MODE = ExecutionMode(
    name="MAIN",
    description=(
        "The reported experiment, sized for one NVIDIA T4 in about four to five "
        "hours: 4 000 inferred images split into disjoint reference / pilot / "
        "evaluation pools, five strategies, three severities, three seeds, plus "
        "the ablation grid."
    ),
    evaluation_images=2_400,
    pilot_images=600,
    reference_images=1_000,
    per_image_limit=20,
    budgets=(100, 250, 500, 1_000, 2_000),
    rounds=5,
    seeds=(0, 1, 2),
    strategies=PRIMARY_STRATEGIES,
    imbalance_settings=longtail.DEFAULT_IMBALANCE_SETTINGS,
    run_pilot=True,
    run_ablations=True,
    ablation_seeds=(0, 1),
    reference_limit=20_000,
    runtime_budget_seconds=4.5 * 3_600,
    research_grade=True,
)

#: The A/B mode for the label-anchored follow-up: the same MAIN protocol with the
#: three anchored strategies added alongside the five baseline ones, so baseline
#: and new method share one pool, one severity axis, one seed set and one budget
#: grid. The ablation grid is off because the anchored variants *are* the ablation
#: (support only / rarity ungated / gated), and the baseline's gate-form grid was
#: already measured in the first run.
MAIN_REVEALED_MODE = replace(
    MAIN_MODE,
    name="MAINREVEALED",
    description=(
        "Label-anchored distribution estimation versus the unsupervised baseline, "
        "on one shared MAIN-scale pool: 8 strategies x 3 severities x 3 seeds."
    ),
    strategies=COMPARISON_STRATEGIES,
    run_ablations=False,
)

MODES: dict[str, ExecutionMode] = {
    "DEBUG": DEBUG_MODE,
    "FAST": FAST_MODE,
    "MAIN": MAIN_MODE,
    "MAINREVEALED": MAIN_REVEALED_MODE,
}


def resolve_mode(name: str) -> ExecutionMode:
    """Look up a mode by name, case-insensitively."""

    key = str(name).strip().upper()
    if key not in MODES:
        raise ModeError(f"Unknown execution mode {name!r}. Supported: {sorted(MODES)}")
    return MODES[key]


def register(mode: ExecutionMode, *, replace_existing: bool = False) -> ExecutionMode:
    """Add a custom size to the registry so ``PipelineConfig(mode=...)`` accepts it.

    Overwriting one of the three shipped modes is refused unless asked for
    explicitly: a run that reports ``MAIN`` must mean the ``MAIN`` in this file.
    """

    key = mode.name.strip().upper()
    if key in MODES and not replace_existing:
        raise ModeError(
            f"Mode {key!r} already exists. Pass replace_existing=True to override it, "
            "or choose another name."
        )
    MODES[key] = mode
    return mode


def scaled(
    mode: ExecutionMode,
    *,
    evaluation_images: int | None = None,
    pilot_images: int | None = None,
    per_image_limit: int | None = None,
    seeds: Sequence[int] | None = None,
) -> ExecutionMode:
    """A copy of ``mode`` resized by the runtime planner.

    Budgets are clamped to the resized pool by
    :func:`daowod.longtail.resolve_budgets` at run time rather than here, so a
    downscaled run keeps the largest budget its pool can actually supply instead
    of silently dropping the top of the curve.
    """

    updates: dict[str, object] = {}
    if evaluation_images is not None:
        updates["evaluation_images"] = max(1, int(evaluation_images))
    if pilot_images is not None:
        updates["pilot_images"] = max(0, int(pilot_images))
        if int(pilot_images) < 1:
            updates["run_pilot"] = False
    if per_image_limit is not None:
        updates["per_image_limit"] = max(1, int(per_image_limit))
    if seeds is not None:
        chosen = tuple(int(value) for value in seeds)
        if not chosen:
            raise ModeError("A resized mode must keep at least one seed.")
        updates["seeds"] = chosen
        updates["ablation_seeds"] = tuple(
            value for value in mode.ablation_seeds if value in chosen
        ) or (chosen[0],)
    return replace(mode, **updates)
