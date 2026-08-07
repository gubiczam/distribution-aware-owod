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

import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from nbformat.validator import validate as validate_nbformat

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


# --- notebooks ----------------------------------------------------------------

NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
MASTER_NOTEBOOK = REPO_ROOT / "notebooks" / "contribution_a_master_colab.ipynb"
CONTRIBUTION_B_NOTEBOOK = REPO_ROOT / "notebooks" / "contribution_b_colab.ipynb"

MAX_NOTEBOOK_CELL_CHARS = 12_000
MAX_NOTEBOOK_LINE_CHARS = 4_000
LOCAL_PATH_PATTERNS = (
    r"/Users/[A-Za-z0-9_.-]+",
    r"/home/(?!runner\b)[A-Za-z0-9_.-]+",
    r"[A-Z]:\\\\Users",
    r"/private/(?:tmp|var)/",
)
MASTER_FORBIDDEN_IMPLEMENTATIONS = (
    "def score_pool",
    "def compute_coherence",
    "def build_candidate_pool",
    "def match_proposals",
    "def allocate(",
    "KMeans(",
    "def run_campaign",
)
MASTER_REQUIRED_DELEGATION = (
    "from daowod.pipeline import",
    "run_pipeline(",
    '"experiments/contribution_a.py"',
    '"experiments/component_audit.py"',
    '"experiments/representation_geometry.py"',
)


def _load_notebook(path: Path) -> dict:
    notebook = nbformat.read(path, as_version=4)
    validate_nbformat(notebook)
    return notebook


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def _code_sources(notebook: dict) -> list[str]:
    return [_cell_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_the_repository_ships_one_notebook_per_contribution() -> None:
    """One authoritative notebook per contribution, and no superseded generations.

    The Contribution A notebook is the master Colab entrypoint; the shim it replaced
    was deleted rather than left beside it, because two notebooks for one experiment
    is how they drift apart.
    """

    assert [path.name for path in NOTEBOOKS] == [
        "contribution_a_master_colab.ipynb",
        "contribution_b_colab.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_json_is_valid_and_reviewable(path: Path) -> None:
    """Every notebook is valid, output-free and small enough to review as source."""

    notebook = _load_notebook(path)
    for index, cell in enumerate(notebook["cells"]):
        text = _cell_source(cell)
        assert cell.get("id"), f"{path.name} cell {index} has no id"
        assert cell["cell_type"] in {"markdown", "code"}
        assert len(text) <= MAX_NOTEBOOK_CELL_CHARS, f"{path.name} cell {index} is huge"
        assert all(len(line) <= MAX_NOTEBOOK_LINE_CHARS for line in text.splitlines()), (
            f"{path.name} cell {index} has a giant line"
        )
        assert not re.search(r"base64|b'\\x89PNG|data:image/", text), (
            f"{path.name} cell {index} embeds large data"
        )
        for pattern in LOCAL_PATH_PATTERNS:
            assert not re.search(pattern, text), f"{path.name} cell {index} has a local path"
        if cell["cell_type"] == "code":
            assert not cell.get("outputs"), f"{path.name} cell {index} has stored outputs"
            assert cell.get("execution_count") is None


def test_master_notebook_delegates_to_repository_code_and_validator() -> None:
    """The master notebook is a research notebook, not a parallel implementation."""

    notebook = _load_notebook(MASTER_NOTEBOOK)
    joined = "\n".join(_code_sources(notebook))
    for forbidden in MASTER_FORBIDDEN_IMPLEMENTATIONS:
        assert forbidden not in joined, f"master notebook reimplements {forbidden!r}"
    for expected in MASTER_REQUIRED_DELEGATION:
        assert expected in joined, f"master notebook does not use {expected}"

    result = subprocess.run(
        [sys.executable, "experiments/validate_master_notebook.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_contribution_b_notebook_stays_a_thin_shim() -> None:
    """Contribution B is a compact demo/entrypoint over daowod.memory."""

    notebook = _load_notebook(CONTRIBUTION_B_NOTEBOOK)
    assert len(notebook["cells"]) <= 15
    joined = "\n".join(_code_sources(notebook))
    assert "from daowod.memory import" in joined
    assert "experiments/contribution_b.py" in joined
    assert "def allocate" not in joined


def test_notebooks_reference_only_paths_that_exist() -> None:
    """Catches a rename that left a notebook pointing at a deleted script."""

    for path in NOTEBOOKS:
        text = json.dumps(json.loads(path.read_text(encoding="utf-8")))
        for reference in sorted(set(re.findall(r"(?:experiments|configs|docs)/[\w./-]+", text))):
            target = reference.rstrip(".,)")
            assert (REPO_ROOT / target).exists(), f"{path.name} references missing {target}"


# --- interface surface --------------------------------------------------------


def test_the_cli_only_inspects_and_says_so() -> None:
    """`daowod-run` must not grow experiment subcommands without the docs following.

    The CLI is registry inspection. Experiments live in experiments/, and README and
    docs/reproduction.md say so explicitly. If a subcommand is added here, those two
    documents have to change in the same commit.
    """

    from daowod.cli import build_parser

    actions = [
        action
        for action in build_parser()._actions  # noqa: SLF001 - argparse has no public API
        if hasattr(action, "choices") and action.choices
    ]
    assert len(actions) == 1
    assert set(actions[0].choices) == {"strategies"}


def test_documented_commands_match_the_real_interfaces() -> None:
    """Every documented `experiments/...` invocation must parse.

    docs/reproduction.md once documented a `--stage` flag that did not exist, and
    nothing caught it because no test read the documentation.
    """

    import re
    import subprocess
    import sys

    text = "\n".join(
        (REPO_ROOT / name).read_text(encoding="utf-8")
        for name in ("README.md", "docs/reproduction.md", "docs/artifacts.md")
    )
    invocations = set(re.findall(r"python experiments/(\w+\.py)((?:\s+[\w\-./<>\[\]]+)*)", text))
    assert invocations, "no documented experiment commands found"

    for script, tail in sorted(invocations):
        assert (REPO_ROOT / "experiments" / script).exists(), script
        # extract_embeddings needs torch, which this environment deliberately lacks.
        if script == "extract_embeddings.py":
            continue
        subcommand = (
            [tail.split()[0]] if tail.split() and not tail.split()[0].startswith("-") else []
        )
        result = subprocess.run(
            [sys.executable, f"experiments/{script}", *subcommand, "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{script} {' '.join(subcommand)} --help failed:\n{result.stderr}"
        )
