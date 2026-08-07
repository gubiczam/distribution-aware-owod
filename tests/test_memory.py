"""Contribution B — the exemplar allocation core.

The properties asserted here are the ones a wrong allocation would violate
silently, and every one of them would corrupt an alpha sweep rather than crash it:

* the budget is conserved exactly, at every alpha and every count table;
* alpha = 0 is uniform, alpha = 1 is size-proportional, alpha < 0 favours the tail;
* the result is a deterministic function of its inputs;
* no class is allocated more exemplars than it has objects.
"""

import pytest

from daowod.memory import (
    ALPHA_GRID,
    AllocationError,
    allocate,
    allocate_images,
    ideal_shares,
    sweep,
)

#: A long-tailed table in the shape the proposal targets: a few frequent classes,
#: many rare ones.
LONG_TAIL = {"head": 120, "upper_mid": 60, "mid": 30, "lower_mid": 12, "tail": 6, "rare": 2}


# --- conservation -------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("budget", [0, 1, 7, 30, 100, 229, 230])
def test_budget_is_conserved_exactly(alpha: float, budget: int) -> None:
    """sum(m_c) == M, not M +/- 1. Rounding each share independently would fail this."""

    result = allocate(LONG_TAIL, total_memory=budget, alpha=alpha)
    assert sum(result.counts.values()) == budget


def test_conservation_holds_for_awkward_budgets_and_class_counts() -> None:
    """Budgets that cannot divide evenly are where largest-remainder earns its place."""

    counts = {"a": 7, "b": 5, "c": 3, "d": 1}
    for budget in range(sum(counts.values()) + 1):
        for alpha in (-1.0, -0.25, 0.0, 0.33, 1.0):
            result = allocate(counts, total_memory=budget, alpha=alpha)
            assert sum(result.counts.values()) == budget


def test_a_budget_larger_than_the_available_objects_is_refused() -> None:
    with pytest.raises(AllocationError, match="exceeds the"):
        allocate({"a": 3, "b": 2}, total_memory=99, alpha=0.0)


# --- the three named strategies ----------------------------------------------


def test_alpha_zero_is_uniform_the_current_standard() -> None:
    """alpha = 0: every class gets the same share. This is the baseline to beat."""

    shares = ideal_shares(LONG_TAIL, 0.0)
    assert len(set(round(value, 12) for value in shares.values())) == 1

    result = allocate(LONG_TAIL, total_memory=len(LONG_TAIL) * 2, alpha=0.0)
    assert set(result.counts.values()) == {2}


def test_alpha_one_is_proportional_to_class_size() -> None:
    """alpha = 1: head-favouring. Shares must match the class frequencies."""

    shares = ideal_shares(LONG_TAIL, 1.0)
    total = sum(LONG_TAIL.values())
    for name, count in LONG_TAIL.items():
        assert shares[name] == pytest.approx(count / total)

    result = allocate(LONG_TAIL, total_memory=115, alpha=1.0)
    assert result.counts["head"] > result.counts["rare"]


def test_negative_alpha_favours_the_tail() -> None:
    """alpha < 0: the rare classes get MORE than the frequent ones. The contribution."""

    shares = ideal_shares(LONG_TAIL, -1.0)
    assert shares["rare"] > shares["tail"] > shares["mid"] > shares["head"]

    uniform = allocate(LONG_TAIL, total_memory=60, alpha=0.0).counts
    tailward = allocate(LONG_TAIL, total_memory=60, alpha=-1.0).counts
    assert tailward["rare"] >= uniform["rare"]
    assert tailward["head"] <= uniform["head"]


def test_alpha_orders_the_head_share_monotonically() -> None:
    """Rising alpha must move budget towards the head, with no inversions."""

    head_shares = [ideal_shares(LONG_TAIL, alpha)["head"] for alpha in ALPHA_GRID]
    assert head_shares == sorted(head_shares)


# --- determinism --------------------------------------------------------------


def test_allocation_is_deterministic() -> None:
    for alpha in ALPHA_GRID:
        first = allocate(LONG_TAIL, total_memory=37, alpha=alpha)
        second = allocate(LONG_TAIL, total_memory=37, alpha=alpha)
        assert first.counts == second.counts


def test_ties_break_by_class_name_not_by_insertion_order() -> None:
    """Equal counts must not be ordered by how the dict happened to be built.

    Four classes of equal size share one leftover unit. The winner must be the same
    whichever order the mapping is constructed in, or a forgetting measurement would
    not reproduce and the noise would look like an effect of alpha.
    """

    forward = {"a": 10, "b": 10, "c": 10, "d": 10}
    backward = {"d": 10, "c": 10, "b": 10, "a": 10}
    left = allocate(forward, total_memory=9, alpha=0.0)
    right = allocate(backward, total_memory=9, alpha=0.0)
    assert left.counts == right.counts
    # The single leftover unit goes to the alphabetically first class.
    assert left.counts["a"] == 3


# --- the availability cap -----------------------------------------------------


def test_no_class_receives_more_exemplars_than_it_has_objects() -> None:
    """The case Contribution B is about: alpha < 0 over-allots a two-object class."""

    counts = {"head": 500, "rare": 2}
    result = allocate(counts, total_memory=100, alpha=-1.0)
    assert result.counts["rare"] <= 2
    assert "rare" in result.capped_classes
    assert sum(result.counts.values()) == 100


def test_capping_redistributes_rather_than_losing_the_budget() -> None:
    counts = {"a": 1, "b": 1, "c": 50}
    result = allocate(counts, total_memory=20, alpha=-2.0)
    assert result.counts["a"] == 1
    assert result.counts["b"] == 1
    assert result.counts["c"] == 18
    assert sum(result.counts.values()) == 20


def test_the_cap_can_be_disabled_for_an_unconstrained_ideal_split() -> None:
    result = allocate({"a": 1, "b": 100}, total_memory=50, alpha=-1.0, respect_availability=False)
    assert result.counts["a"] > 1
    assert sum(result.counts.values()) == 50


# --- empty and degenerate classes --------------------------------------------


def test_a_class_with_no_objects_gets_nothing_at_every_alpha() -> None:
    """0 ** negative is infinite; a class with nothing stored must simply get zero."""

    counts = {"present": 10, "absent": 0}
    for alpha in ALPHA_GRID:
        result = allocate(counts, total_memory=5, alpha=alpha)
        assert result.counts["absent"] == 0
        assert result.counts["present"] == 5


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(AllocationError, match="must not be empty"):
        allocate({}, total_memory=1, alpha=0.0)
    with pytest.raises(AllocationError, match="non-negative"):
        allocate({"a": -1}, total_memory=1, alpha=0.0)
    with pytest.raises(AllocationError, match="all zero"):
        allocate({"a": 0}, total_memory=0, alpha=0.0)
    with pytest.raises(AllocationError, match="non-negative"):
        allocate({"a": 5}, total_memory=-1, alpha=0.0)
    with pytest.raises(AllocationError, match="finite"):
        allocate({"a": 5}, total_memory=1, alpha=float("inf"))


# --- image-level allocation (H-B2) -------------------------------------------


def test_image_level_allocation_stores_whole_images_and_reports_its_shortfall() -> None:
    """One image carries several classes, so per-class quotas cannot all be met."""

    image_classes = {
        "img_head_1": ["head", "head"],
        "img_head_2": ["head"],
        "img_mixed": ["head", "tail"],
        "img_tail": ["tail"],
        "img_rare": ["rare"],
    }
    result = allocate_images(image_classes, total_memory=3, alpha=-1.0)
    assert result.granularity == "image"
    assert len(result.image_ids) == 3
    assert len(set(result.image_ids)) == 3, "an image is stored once"
    assert result.shortfall is not None
    assert all(value >= 0 for value in result.shortfall.values())


def test_image_selection_is_deterministic() -> None:
    image_classes = {f"img{index}": ["head" if index % 2 else "tail"] for index in range(8)}
    first = allocate_images(image_classes, total_memory=4, alpha=-1.0)
    second = allocate_images(image_classes, total_memory=4, alpha=-1.0)
    assert first.image_ids == second.image_ids


def test_image_level_prefers_images_that_serve_an_unmet_quota() -> None:
    """A tail-favouring alpha must reach for the tail image, not another head image."""

    image_classes = {
        "head_a": ["head"],
        "head_b": ["head"],
        "head_c": ["head"],
        "tail_only": ["tail"],
    }
    result = allocate_images(image_classes, total_memory=2, alpha=-1.0)
    assert "tail_only" in result.image_ids


def test_image_budget_larger_than_the_image_pool_is_refused() -> None:
    with pytest.raises(AllocationError, match="exceeds the"):
        allocate_images({"a": ["x"]}, total_memory=5, alpha=0.0)


# --- the sweep, and the explicit non-claim -----------------------------------


def test_sweep_covers_the_declared_grid_and_conserves_every_budget() -> None:
    results = sweep(LONG_TAIL, total_memory=50)
    assert [result.alpha for result in results] == list(ALPHA_GRID)
    for result in results:
        assert sum(result.counts.values()) == 50


def test_the_module_does_not_pretend_to_measure_forgetting() -> None:
    """Guards the scope boundary in docs/research_design.md section 8.

    H-B1 - the optimal alpha and its dependence on tail severity - needs real
    incremental model updates. If a `forgetting` helper ever appears in this module,
    either it is a proxy that cannot answer the question, or the module has grown
    into the incremental-learning framework this refactor deliberately did not build.
    """

    from daowod import memory

    exported = {name for name in dir(memory) if not name.startswith("_")}
    assert not {name for name in exported if "forget" in name.lower()}
    assert not {name for name in exported if "retrain" in name.lower()}
