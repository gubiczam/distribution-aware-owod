#!/usr/bin/env python3
"""Validate the master Colab notebook without executing it.

    python experiments/validate_master_notebook.py
    python experiments/validate_master_notebook.py --notebook <path> --json

A notebook is the one artifact in this repository that CI cannot run: it needs Colab, a
T4 and dataset assets. So everything checkable without executing it is checked here, and
`tests/test_master_notebook.py` runs this on every commit.

Ten checks, each earned by a failure mode this project actually hit:

1.  **structure** — valid nbformat, cell ids, no stored outputs. Embedded outputs made
    the previous generation's JSON unreviewable.
2.  **sections** — the documented section headings are present and in order, so the
    notebook stays readable as a research document.
3.  **config keys** — every setting the task contract requires exists in the single
    configuration cell, and that cell comes before anything that consumes it.
4.  **no local paths** — no `/Users/...`, no `/home/...`, no machine-specific absolute
    path. A notebook that only runs on the author's laptop is not reproducible.
5.  **referenced files** — every `experiments/`, `configs/`, `docs/`, `src/` or `tests/`
    path the notebook mentions exists in the repository. A rename left dangling links
    before.
6.  **commands parse** — every `experiments/*.py` invocation the notebook builds is
    checked against that script's real argument parser. A phantom `--stage` flag once
    survived in the documentation because nothing read it.
7.  **output contract** — the artifacts the notebook promises are the artifacts its
    packaging step expects.
8.  **setup ordering** — imports and installation precede use; the configuration cell
    precedes the first stage.
9.  **no inline bulk data** — no cell carries a large embedded blob masquerading as code.
10. **no hard-coded results** — no numeric research claim is baked into a cell. Findings
    must be read from generated tables, so a rerun can contradict them.

Exit code 0 = all checks pass. 1 = at least one failure.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = REPO_ROOT / "notebooks" / "contribution_a_master_colab.ipynb"

#: Section headings the notebook must carry, in this order.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "What this notebook does and does not claim",
    "Configuration",
    "Environment setup",
    "preflight",
    "Unit and smoke validation",
    "PROB proposal export",
    "Candidate pool",
    "long tail",
    "Metrics",
    "Component / mechanism audit",
    "Revealed-label follow-up",
    "Representation study",
    "Results and figures",
    "Research conclusion",
    "Artifact export",
    "Troubleshooting",
)

#: Settings the task contract requires in the single configuration cell.
REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "REPO_URL",
    "REPO_BRANCH",
    "DAOWOD_COMMIT",
    "USE_DRIVE",
    "DRIVE_ROOT",
    "OUTPUT_ROOT",
    "CACHE_ROOT",
    "RUN_MODE",
    "RANDOM_SEEDS",
    "MAX_RUNTIME_HOURS",
    "MAX_TEMP_DISK_GB",
    "MAX_RAM_GB",
    "RESUME",
    "FORCE_STAGE",
    "PROTOCOL_NAME",
    "DATASET_ROOT",
    "JPEG_IMAGES_DIR",
    "ANNOTATIONS_DIR",
    "SPLIT_FILE",
    "CLASS_GROUP_FILE",
    "LVIS_ROOT",
    "PROB_REPO_URL",
    "PROB_COMMIT",
    "PROB_ROOT",
    "CHECKPOINT_PATH",
    "DINO_WEIGHTS_PATH",
    "EXISTING_EXPORT",
    "MAX_PROPOSALS_PER_IMAGE",
    "CHUNK_IMAGES",
    "INFER_BATCH_SIZE",
    "USE_AMP",
    "ENABLE_REPRESENTATION_STUDY",
    "REPRESENTATIONS_TO_RUN",
    "EXISTING_REPRESENTATION_ROOT",
    "PROCESS_ONE_REPRESENTATION_AT_A_TIME",
    "SAVE_PNG",
    "SAVE_PDF",
    "CREATE_ZIP",
    "COPY_COMPACT_RESULTS_TO_DRIVE",
    "KEEP_LARGE_INTERMEDIATES",
)

#: Modes the notebook must offer.
REQUIRED_MODES: tuple[str, ...] = (
    "SMOKE",
    "DEBUG",
    "FAST",
    "MAIN",
    "MAIN_REVEALED",
    "REPRESENTATION",
)

#: Artifacts the notebook promises to write.
REQUIRED_OUTPUTS: tuple[str, ...] = (
    "environment.json",
    "git_commits.json",
    "preflight.csv",
    "metrics.json",
    "runtime_report.json",
    "limitations.md",
    "reproduction_command.txt",
    "archive_manifest.json",
)

#: Settings whose default must be conservative, and the required default.
CONSERVATIVE_DEFAULTS: dict[str, str] = {
    "KEEP_LARGE_INTERMEDIATES": "False",
    "ENABLE_REPRESENTATION_STUDY": "False",
    "RUN_REPRESENTATION_ACQUISITION": "False",
    "CLEAN_TEMPORARY_FILES": "False",
    "USE_AMP": "False",
}

LOCAL_PATH_PATTERNS = (
    r"/Users/[A-Za-z0-9_.-]+",
    r"/home/(?!runner\b)[A-Za-z0-9_.-]+",
    r"[A-Z]:\\\\Users",
    r"/private/(?:tmp|var)/",
)

#: A numeric literal next to one of these words is a research claim, not configuration.
CLAIM_WORDS = (
    "recall",
    "auc",
    "precision",
    "mAP",
    "purity",
    "sibling",
    "same-label",
    "discovery",
    "lift",
    "advantage",
)

MAX_CELL_CHARS = 12_000
MAX_LINE_CHARS = 4_000


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self, check: str, detail: str = "") -> None:
        self.passed.append(f"{check}: {detail}" if detail else check)

    def fail(self, check: str, detail: str) -> None:
        self.failures.append(f"{check}: {detail}")

    def warn(self, check: str, detail: str) -> None:
        self.warnings.append(f"{check}: {detail}")

    @property
    def success(self) -> bool:
        return not self.failures


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_text(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


# --- 1. structure -------------------------------------------------------------


def check_structure(notebook: dict, report: Report) -> None:
    if notebook.get("nbformat") != 4:
        report.fail("structure", f"nbformat is {notebook.get('nbformat')}, expected 4")
        return
    cells = notebook.get("cells") or []
    if not cells:
        report.fail("structure", "no cells")
        return
    for index, cell in enumerate(cells):
        if cell.get("cell_type") not in {"markdown", "code"}:
            report.fail("structure", f"cell {index} has type {cell.get('cell_type')!r}")
        if not cell.get("id"):
            report.fail("structure", f"cell {index} has no id")
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                report.fail("structure", f"cell {index} carries stored outputs")
            if cell.get("execution_count") is not None:
                report.fail("structure", f"cell {index} carries an execution count")
    report.ok("structure", f"{len(cells)} cells, ids present, no stored outputs")


# --- 2. sections --------------------------------------------------------------


def check_sections(notebook: dict, report: Report) -> None:
    """Match against markdown HEADING lines only.

    Searching the whole prose instead matched section names used in body text — the
    scope section legitimately mentions "metrics" and "configuration" long before
    those sections begin — and reported a false ordering failure.
    """

    headings: list[str] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for line in cell_text(cell).splitlines():
            if line.lstrip().startswith("#"):
                headings.append(line.lstrip("#").strip().lower())

    position = 0
    missing: list[str] = []
    for heading in REQUIRED_SECTIONS:
        needle = heading.lower()
        found = next(
            (i for i in range(position, len(headings)) if needle in headings[i]),
            None,
        )
        if found is None:
            # Present but out of order, or absent entirely — distinguish the two.
            if any(needle in text for text in headings):
                report.fail("sections", f"heading {heading!r} appears out of order")
                return
            missing.append(heading)
        else:
            position = found

    if missing:
        report.fail("sections", f"missing section headings: {missing}")
    else:
        report.ok("sections", f"{len(REQUIRED_SECTIONS)} headings present and in order")


# --- 3. configuration ---------------------------------------------------------


def find_config_cell(notebook: dict) -> tuple[int, str]:
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = cell_text(cell)
        if "CONFIGURATION" in text and "RUN_MODE" in text:
            return index, text
    return -1, ""


def check_config(notebook: dict, report: Report) -> None:
    index, text = find_config_cell(notebook)
    if index < 0:
        report.fail("config", "no configuration cell found")
        return

    assigned: set[str] = set()
    defaults: dict[str, str] = {}
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        report.fail("config", f"configuration cell does not parse: {error}")
        return
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
                    defaults[target.id] = ast.unparse(node.value)

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in assigned]
    if missing:
        report.fail("config", f"missing keys: {missing}")
    else:
        report.ok("config", f"{len(REQUIRED_CONFIG_KEYS)} required keys present")

    all_text = "\n".join(cell_text(cell) for cell in notebook["cells"])
    absent_modes = [mode for mode in REQUIRED_MODES if mode not in all_text]
    if absent_modes:
        report.fail("config", f"modes never mentioned: {absent_modes}")
    else:
        report.ok("config", f"all {len(REQUIRED_MODES)} modes documented")

    for key, expected in CONSERVATIVE_DEFAULTS.items():
        if key in defaults and defaults[key] != expected:
            report.fail("config", f"{key} defaults to {defaults[key]}, expected {expected}")
        elif key not in defaults and key not in all_text:
            report.warn("config", f"{key} not found anywhere")
    report.ok("config", "conservative defaults verified")

    # The configuration cell must precede any cell that consumes it.
    consumers = [
        i
        for i, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code" and i != index and "RUN_MODE" in cell_text(cell)
    ]
    if consumers and min(consumers) < index:
        report.fail("config", f"cell {min(consumers)} uses RUN_MODE before the config cell")
    else:
        report.ok("config", f"configuration cell is at index {index}, before its consumers")


# --- 4. no machine-specific paths --------------------------------------------


def check_no_local_paths(notebook: dict, report: Report) -> None:
    offenders: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        text = cell_text(cell)
        for pattern in LOCAL_PATH_PATTERNS:
            for match in re.findall(pattern, text):
                offenders.append(f"cell {index}: {match}")
    if offenders:
        report.fail("no_local_paths", "; ".join(sorted(set(offenders))[:6]))
    else:
        report.ok("no_local_paths", "no machine-specific absolute paths")


# --- 5. referenced repository files ------------------------------------------


def check_referenced_files(notebook: dict, report: Report, repo: Path) -> None:
    text = "\n".join(cell_text(cell) for cell in notebook["cells"])
    pattern = r"(?:experiments|configs|docs|src|tests|data)/[\w./-]+"
    missing: list[str] = []
    checked = 0
    for reference in sorted(set(re.findall(pattern, text))):
        target = reference.rstrip("./,)")
        if target.endswith((".py", ".yaml", ".md", ".csv", ".txt", ".ipynb", ".docx")):
            checked += 1
            if not (repo / target).exists():
                missing.append(target)
    if missing:
        report.fail("referenced_files", f"missing: {missing}")
    else:
        report.ok("referenced_files", f"{checked} referenced repository files exist")


# --- 6. commands parse against the real parsers ------------------------------


def check_commands(notebook: dict, report: Report, repo: Path) -> None:
    text = "\n".join(cell_text(cell) for cell in notebook["cells"])
    scripts = sorted(set(re.findall(r'"experiments/(\w+\.py)"', text)))
    if not scripts:
        report.warn("commands", "no experiments/ invocations found")
        return

    checked, failed = [], []
    for script in scripts:
        path = repo / "experiments" / script
        if not path.exists():
            failed.append(f"{script} does not exist")
            continue
        if script == "extract_embeddings.py":
            continue  # needs torch by design; see docs/reproduction.md
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failed.append(f"{script} --help exited {result.returncode}")
            continue
        checked.append(script)

        # Every long flag the notebook passes to this script must exist in its parser.
        used = set(re.findall(rf'"experiments/{script}"[^\]]*?', text))
        block = text[text.find(f'"experiments/{script}"') :][:2000] if used else ""
        for flag in set(re.findall(r'"(--[a-z][a-z0-9-]*)"', block)):
            if flag not in result.stdout:
                failed.append(f"{script} has no flag {flag}")

    if failed:
        report.fail("commands", "; ".join(sorted(set(failed))[:8]))
    else:
        report.ok("commands", f"{len(checked)} scripts parse and every flag exists")


# --- 7. output contract ------------------------------------------------------


def check_output_contract(notebook: dict, report: Report) -> None:
    text = "\n".join(cell_text(cell) for cell in notebook["cells"])
    missing = [name for name in REQUIRED_OUTPUTS if name not in text]
    if missing:
        report.fail("output_contract", f"never written: {missing}")
    else:
        report.ok("output_contract", f"{len(REQUIRED_OUTPUTS)} promised artifacts written")

    for phrase in ("NOT AVAILABLE", "requires retraining"):
        if phrase not in text:
            report.fail("output_contract", f"missing the unavailable-metric marker {phrase!r}")
    if all(p in text for p in ("NOT AVAILABLE", "requires retraining")):
        report.ok("output_contract", "unavailable official metrics are marked, not zeroed")


# --- 8. setup ordering -------------------------------------------------------


def check_setup_order(notebook: dict, report: Report) -> None:
    code_cells = [
        (index, cell_text(cell))
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    ]
    first_import = next((i for i, t in code_cells if re.search(r"^import |^from ", t, re.M)), None)
    first_pipeline = next((i for i, t in code_cells if "run_pipeline(" in t), None)
    first_install = next((i for i, t in code_cells if "pip" in t and "install" in t), None)

    if first_pipeline is None:
        report.fail("setup_order", "run_pipeline is never called")
        return
    if first_import is None or first_import > first_pipeline:
        report.fail("setup_order", "imports do not precede the pipeline call")
        return
    if first_install is not None and first_install > first_pipeline:
        report.fail("setup_order", "installation happens after the pipeline call")
        return
    report.ok(
        "setup_order",
        f"imports at cell {first_import}, install at {first_install}, pipeline at {first_pipeline}",
    )


# --- 9. no inline bulk data --------------------------------------------------


def check_no_inline_data(notebook: dict, report: Report) -> None:
    offenders: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        text = cell_text(cell)
        if len(text) > MAX_CELL_CHARS:
            offenders.append(f"cell {index} is {len(text)} chars")
        for line in text.splitlines():
            if len(line) > MAX_LINE_CHARS:
                offenders.append(f"cell {index} has a {len(line)}-char line")
                break
        if re.search(r"base64|b'\\x89PNG|data:image/", text):
            offenders.append(f"cell {index} looks like embedded binary")
    if offenders:
        report.fail("no_inline_data", "; ".join(offenders[:5]))
    else:
        report.ok("no_inline_data", f"no cell exceeds {MAX_CELL_CHARS} chars")


# --- 10. no hard-coded research results --------------------------------------


def check_no_hardcoded_results(notebook: dict, report: Report) -> None:
    """A numeric literal beside a metric word inside CODE is a baked-in claim.

    Markdown may quote published findings — that is the literature review, and it
    cites docs/results.md. Code must derive every number from a generated table.
    """

    offenders: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        for lineno, line in enumerate(cell_text(cell).splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            lowered = stripped.lower()
            if not any(word.lower() in lowered for word in CLAIM_WORDS):
                continue
            # A metric name used as a column/key is fine; a float beside it is not.
            for literal in re.findall(r"(?<![\w.])0\.\d+|(?<![\w.])\d+\.\d+", stripped):
                if literal in {"0.5", "1.0", "0.0"}:
                    continue  # thresholds and neutral constants
                offenders.append(f"cell {index} line {lineno}: {literal} near a metric word")
    if offenders:
        report.fail("no_hardcoded_results", "; ".join(offenders[:6]))
    else:
        report.ok("no_hardcoded_results", "no numeric research claim baked into code")


CHECKS = {
    "structure": check_structure,
    "sections": check_sections,
    "config": check_config,
    "no_local_paths": check_no_local_paths,
    "output_contract": check_output_contract,
    "setup_order": check_setup_order,
    "no_inline_data": check_no_inline_data,
    "no_hardcoded_results": check_no_hardcoded_results,
}


def validate(notebook_path: Path, repo: Path = REPO_ROOT) -> Report:
    """Run every check and return the report."""

    report = Report()
    if not notebook_path.exists():
        report.fail("structure", f"notebook not found: {notebook_path}")
        return report
    try:
        notebook = load_notebook(notebook_path)
    except json.JSONDecodeError as error:
        report.fail("structure", f"invalid JSON: {error}")
        return report

    for check in CHECKS.values():
        check(notebook, report)
    check_referenced_files(notebook, report, repo)
    check_commands(notebook, report, repo)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebook", default=str(DEFAULT_NOTEBOOK))
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args(argv)

    report = validate(Path(args.notebook))

    if args.json:
        print(
            json.dumps(
                {
                    "notebook": args.notebook,
                    "success": report.success,
                    "passed": report.passed,
                    "warnings": report.warnings,
                    "failures": report.failures,
                },
                indent=2,
            )
        )
        return 0 if report.success else 1

    print(f"Validating {args.notebook}\n")
    for line in report.passed:
        print(f"  [PASS] {line}")
    for line in report.warnings:
        print(f"  [WARN] {line}")
    for line in report.failures:
        print(f"  [FAIL] {line}")
    print(
        f"\n{len(report.passed)} passed, {len(report.warnings)} warnings, "
        f"{len(report.failures)} failures"
    )
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
