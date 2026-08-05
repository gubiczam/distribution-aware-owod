"""Every input the experiments require must survive a clean `git clone`.

This exists because a Colab run once died on a config that pointed into the ignored
`outputs/` tree: the clone was fine, the asset simply was not in it. Somebody fixed
that one path, and the next run died on the next one. Checking the whole list here
fails once, locally, with everything named.

Two properties are asserted, and they are different:

* **presence and content** — the file is in the work tree and its bytes are the ones
  the reported results were computed from, pinned by digest;
* **clean-clone reachability** — `git check-ignore` does not claim it, so a fresh
  clone actually carries it.
"""

import subprocess
from pathlib import Path

import pytest

from daowod.dataset import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Version-controlled protocol inputs, with the digests the results were computed
#: from. A digest change means the protocol changed and the numbers in
#: docs/results.md no longer describe this tree.
REQUIRED_INPUTS: dict[str, str] = {
    "data/protocol/stage1b/stage1b_candidate_500.txt": (
        "70fa185514dcbbba8397781d85275362c888e6ea0c4d6c1325ad6c82fa18aac6"
    ),
    "data/protocol/stage1b/stage1b_reference_3500.txt": (
        "25a1b33614bcb77c8ef9b238ab878950b62861d0fc048fc58574c7fd0c6df762"
    ),
}

#: Required inputs whose content is expected to evolve, so only reachability is
#: asserted here. Their semantics are covered by their own tests.
REQUIRED_PATHS: tuple[str, ...] = (
    "pyproject.toml",
    "data/protocol/stage2/stage2_class_groups.csv",
)


def _is_ignored(relative_path: str) -> bool:
    """True when git would exclude the path from a clean clone."""

    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    # 0 = ignored, 1 = not ignored, 128 = error (e.g. not a repository).
    if result.returncode not in (0, 1):
        pytest.skip(f"git check-ignore unavailable: {result.stderr.decode().strip()}")
    return result.returncode == 0


@pytest.mark.parametrize("relative_path", sorted({*REQUIRED_INPUTS, *REQUIRED_PATHS}))
def test_required_input_survives_a_clean_clone(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).exists(), f"{relative_path} is missing from the work tree"
    assert not _is_ignored(relative_path), (
        f"{relative_path} is gitignored and would not survive a clean clone"
    )


@pytest.mark.parametrize("relative_path,expected", sorted(REQUIRED_INPUTS.items()))
def test_protocol_split_content_is_pinned(relative_path: str, expected: str) -> None:
    """A silent edit to a split file would invalidate every reported number."""

    assert file_sha256(REPO_ROOT / relative_path) == expected


def test_class_groups_cover_head_medium_and_tail() -> None:
    """A partial mapping silently drops classes from the grouped metrics."""

    import csv

    path = REPO_ROOT / "data/protocol/stage2/stage2_class_groups.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "class groups CSV is empty"
    assert {row["group"] for row in rows} == {"head", "medium", "tail"}
    assert len({row["class_name"] for row in rows}) == len(rows), "duplicate class in mapping"
