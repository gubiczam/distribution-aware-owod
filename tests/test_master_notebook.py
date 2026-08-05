"""Tests for the master notebook and its validator.

Two halves, and the second is the one that matters:

* the real notebook must pass every check;
* each check must **fail** on a notebook that violates it.

A validator that cannot fail is decoration. Every negative case below corresponds to a
mistake this repository actually made: stored outputs that made a notebook unreviewable,
a documented flag that never existed, a dangling path left by a rename, and research
numbers pasted into a cell so a rerun could not contradict them.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from validate_master_notebook import (  # noqa: E402
    CONSERVATIVE_DEFAULTS,
    DEFAULT_NOTEBOOK,
    REQUIRED_CONFIG_KEYS,
    REQUIRED_MODES,
    REQUIRED_OUTPUTS,
    REQUIRED_SECTIONS,
    Report,
    check_commands,
    check_config,
    check_no_hardcoded_results,
    check_no_inline_data,
    check_no_local_paths,
    check_referenced_files,
    check_sections,
    check_structure,
    validate,
)

NOTEBOOK_PATH = DEFAULT_NOTEBOOK


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


# --- the real notebook --------------------------------------------------------


def test_the_master_notebook_exists_and_is_the_authoritative_one() -> None:
    assert NOTEBOOK_PATH.exists(), NOTEBOOK_PATH
    assert NOTEBOOK_PATH.name == "contribution_a_master_colab.ipynb"


def test_the_master_notebook_passes_every_validator_check() -> None:
    report = validate(NOTEBOOK_PATH)
    assert report.success, "validator failures:\n  " + "\n  ".join(report.failures)
    assert len(report.passed) >= 12, report.passed


def test_every_code_cell_parses(notebook: dict) -> None:
    import ast

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as error:  # pragma: no cover - a failure is the point
            pytest.fail(f"cell {index} line {error.lineno}: {error.msg}")


def test_the_notebook_declares_the_six_execution_modes(notebook: dict) -> None:
    text = json.dumps(notebook)
    for mode in REQUIRED_MODES:
        assert mode in text, mode


def test_the_notebook_never_claims_official_detector_metrics(notebook: dict) -> None:
    """The central honesty property: unavailable metrics are marked, never zeroed."""

    text = json.dumps(notebook)
    assert "NOT AVAILABLE" in text
    for metric in ("known_mAP", "U_Recall_official", "WI", "A_OSE"):
        assert metric in text, f"{metric} is not even mentioned as unavailable"
    # No cell may assert an improvement in a quantity the notebook cannot compute.
    for forbidden in ("known mAP improved", "U-Recall improved", "reduces forgetting"):
        assert forbidden not in text


def test_the_notebook_does_not_reimplement_the_science(notebook: dict) -> None:
    """It must orchestrate, not re-derive. A second implementation would silently drift."""

    joined = "\n".join("".join(cell["source"]) for cell in code_cells(notebook))
    # Signs of a parallel implementation living in the notebook.
    for forbidden in (
        "def score_pool",
        "def compute_coherence",
        "def build_candidate_pool",
        "def match_proposals",
        "def allocate(",
        "KMeans(",
        "def run_campaign",
    ):
        assert forbidden not in joined, f"notebook reimplements {forbidden!r}"
    # And signs that it uses the library instead.
    for expected in (
        "from daowod.pipeline import",
        "run_pipeline(",
        "experiments/component_audit.py",
    ):
        assert expected in joined, expected


def test_conservative_defaults_are_conservative(notebook: dict) -> None:
    """Parse the assignments rather than matching strings.

    The notebook aligns its `=` signs for readability, so `KEY  = False` is normal and a
    literal `"KEY = False"` match is the wrong test — which is exactly how this test
    first failed against a correct notebook.
    """

    import ast

    assignments: dict[str, str] = {}
    for cell in code_cells(notebook):
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, ast.unparse(node.value))

    for key, expected in CONSERVATIVE_DEFAULTS.items():
        assert key in assignments, f"{key} is never assigned"
        assert assignments[key] == expected, (
            f"{key} defaults to {assignments[key]}, expected {expected}"
        )


def test_the_notebook_uses_the_repository_config_for_the_protocol(notebook: dict) -> None:
    joined = "\n".join("".join(cell["source"]) for cell in code_cells(notebook))
    assert "configs/contribution_a.yaml" in joined
    assert "load_modes(" in joined or "resolve_mode(" in joined


# --- the validator must be able to fail ---------------------------------------


def test_structure_check_rejects_stored_outputs(notebook: dict) -> None:
    broken = copy.deepcopy(notebook)
    target = next(c for c in broken["cells"] if c["cell_type"] == "code")
    target["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["noise"]}]
    report = Report()
    check_structure(broken, report)
    assert not report.success
    assert any("stored outputs" in failure for failure in report.failures)


def test_structure_check_rejects_a_missing_cell_id(notebook: dict) -> None:
    broken = copy.deepcopy(notebook)
    broken["cells"][0].pop("id", None)
    report = Report()
    check_structure(broken, report)
    assert any("has no id" in failure for failure in report.failures)


def test_sections_check_rejects_a_missing_heading(notebook: dict) -> None:
    broken = copy.deepcopy(notebook)
    for cell in broken["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = [
                line for line in cell["source"] if "Component / mechanism audit" not in line
            ]
    report = Report()
    check_sections(broken, report)
    assert not report.success


def test_sections_check_rejects_reordered_headings(notebook: dict) -> None:
    broken = copy.deepcopy(notebook)
    markdown_indices = [i for i, c in enumerate(broken["cells"]) if c["cell_type"] == "markdown"]
    first, last = markdown_indices[1], markdown_indices[-1]
    broken["cells"][first], broken["cells"][last] = (
        broken["cells"][last],
        broken["cells"][first],
    )
    report = Report()
    check_sections(broken, report)
    assert not report.success


def test_config_check_rejects_a_missing_key(notebook: dict) -> None:
    broken = copy.deepcopy(notebook)
    for cell in broken["cells"]:
        if cell["cell_type"] == "code" and "CONFIGURATION" in "".join(cell["source"]):
            cell["source"] = [
                line for line in cell["source"] if not line.startswith("MAX_TEMP_DISK_GB")
            ]
    report = Report()
    check_config(broken, report)
    assert any("MAX_TEMP_DISK_GB" in failure for failure in report.failures)


def test_config_check_rejects_an_unsafe_default(notebook: dict) -> None:
    broken = copy.deepcopy(notebook)
    for cell in broken["cells"]:
        if cell["cell_type"] == "code" and "CONFIGURATION" in "".join(cell["source"]):
            cell["source"] = [
                line.replace(
                    "KEEP_LARGE_INTERMEDIATES       = False",
                    "KEEP_LARGE_INTERMEDIATES       = True",
                )
                for line in cell["source"]
            ]
    report = Report()
    check_config(broken, report)
    assert any("KEEP_LARGE_INTERMEDIATES" in failure for failure in report.failures)


def test_local_path_check_rejects_a_home_directory(notebook: dict) -> None:
    broken = copy.deepcopy(notebook)
    broken["cells"][0]["source"] = ["DATASET = '/Users/someone/data/OWOD'\n"]
    report = Report()
    check_no_local_paths(broken, report)
    assert not report.success
    assert any("/Users/someone" in failure for failure in report.failures)


def test_referenced_files_check_rejects_a_dangling_path(notebook: dict, tmp_path: Path) -> None:
    broken = copy.deepcopy(notebook)
    broken["cells"][0]["source"] = ['sh(["python", "experiments/does_not_exist.py"])\n']
    report = Report()
    check_referenced_files(broken, report, REPO_ROOT)
    assert any("does_not_exist.py" in failure for failure in report.failures)


def test_commands_check_rejects_a_nonexistent_flag(notebook: dict) -> None:
    """The phantom `--stage` class of bug."""

    broken = copy.deepcopy(notebook)
    broken["cells"] = [
        {
            "cell_type": "code",
            "id": "x",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                'sh([sys.executable, "experiments/component_audit.py",\n',
                '    "--totally-invented-flag", "x"])\n',
            ],
        }
    ]
    report = Report()
    check_commands(broken, report, REPO_ROOT)
    assert any("--totally-invented-flag" in failure for failure in report.failures)


def test_inline_data_check_rejects_an_embedded_blob(notebook: dict) -> None:
    broken = copy.deepcopy(notebook)
    broken["cells"][0]["source"] = ["BLOB = '" + "A" * 20_000 + "'\n"]
    report = Report()
    check_no_inline_data(broken, report)
    assert not report.success


def test_hardcoded_results_check_rejects_a_pasted_number(notebook: dict) -> None:
    broken = copy.deepcopy(notebook)
    broken["cells"] = [
        {
            "cell_type": "code",
            "id": "x",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["tail_discovery_recall = 0.015  # from the paper\n"],
        }
    ]
    report = Report()
    check_no_hardcoded_results(broken, report)
    assert not report.success
    assert any("0.015" in failure for failure in report.failures)


def test_validator_reports_a_missing_notebook(tmp_path: Path) -> None:
    report = validate(tmp_path / "absent.ipynb")
    assert not report.success
    assert any("not found" in failure for failure in report.failures)


def test_validator_reports_invalid_json(tmp_path: Path) -> None:
    broken = tmp_path / "broken.ipynb"
    broken.write_text("{not json", encoding="utf-8")
    report = validate(broken)
    assert not report.success
    assert any("invalid JSON" in failure for failure in report.failures)


# --- the validator's own contract --------------------------------------------


def test_the_required_lists_are_not_empty() -> None:
    assert len(REQUIRED_SECTIONS) >= 14
    assert len(REQUIRED_CONFIG_KEYS) >= 30
    assert len(REQUIRED_OUTPUTS) >= 6
    assert len(REQUIRED_MODES) == 6


def test_the_validator_cli_exits_zero_on_the_real_notebook() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "experiments/validate_master_notebook.py", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["failures"] == []
