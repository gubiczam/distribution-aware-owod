"""Guard the invariant that a clean clone carries every required input.

These tests exist because a Stage 2 Colab run once failed on a config that
pointed into the ignored `outputs/` tree. The clone was fine; the asset simply
was not in it. Catching that here means it fails in CI as a complete list rather
than one file per Colab session.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "stage2_master_colab.ipynb"
CLASS_GROUPS_PATH = Path("data/protocol/stage2/stage2_class_groups.csv")
STAGE2_CONFIG_NAMES = (
    "smoke_stage2_t4.yaml",
    "stage2_v2_random.yaml",
    "stage2_v2_uncertainty_objectness_weighted_entropy.yaml",
    "stage2_v2_full.yaml",
    "stage2_v2_full_no_novelty.yaml",
)


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_clean_clone_assets", REPO_ROOT / "analysis" / "audit_clean_clone_assets.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit_module = _load_audit_module()


def _notebook_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def test_clean_clone_asset_audit_passes() -> None:
    report = audit_module.audit(REPO_ROOT)
    assert report["findings"] == [], "Required inputs are not clean-clone reachable:\n" + "\n".join(
        f"  [{f['problem']}] {f['path']} (required by {f['origin']}): {f['detail']}"
        for f in report["findings"]
    )
    assert report["status"] == "PASS"


@pytest.mark.parametrize("relative_path", audit_module.REQUIRED_REPO_PATHS)
def test_required_repo_path_exists_and_is_not_ignored(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).exists(), f"{relative_path} is missing from the work tree"
    assert audit_module.under_ignored_tree(relative_path) is None, (
        f"{relative_path} lives under an ignored tree and would not survive a clean clone"
    )


@pytest.mark.parametrize("config_name", STAGE2_CONFIG_NAMES)
def test_stage2_config_class_groups_path_is_version_controlled(config_name: str) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    class_groups_path = config["dataset"]["class_groups_path"]
    assert class_groups_path == CLASS_GROUPS_PATH.as_posix(), (
        f"{config_name} must resolve class groups from the tracked protocol location"
    )
    assert (REPO_ROOT / class_groups_path).exists()


def test_class_groups_csv_covers_every_unknown_class() -> None:
    """A partial mapping silently drops classes from the grouped metrics."""

    import csv

    with (REPO_ROOT / CLASS_GROUPS_PATH).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "class groups CSV is empty"
    assert {row["group"] for row in rows} == {"head", "medium", "tail"}

    mapped = {row["class_name"] for row in rows}
    for config_name in STAGE2_CONFIG_NAMES:
        config = yaml.safe_load((REPO_ROOT / "configs" / config_name).read_text(encoding="utf-8"))
        unknown = set(config["dataset"]["unknown_classes"])
        assert unknown <= mapped, (
            f"{config_name} declares unknown classes with no group: {sorted(unknown - mapped)}"
        )


def test_notebook_required_paths_match_the_audit_list() -> None:
    """The notebook and the audit must not drift apart."""

    sources = _notebook_sources()
    config_cell = next(source for source in sources if "STAGE2_CLASS_GROUPS = " in source)
    assert f'STAGE2_CLASS_GROUPS = "{CLASS_GROUPS_PATH.as_posix()}"' in config_cell

    joined = "\n".join(sources)
    for relative_path in audit_module.REQUIRED_REPO_PATHS:
        if relative_path == "pyproject.toml":
            continue
        assert relative_path in joined, (
            f"{relative_path} is audited as required but the notebook never references it"
        )


def test_notebook_pins_no_path_under_an_ignored_tree() -> None:
    """Catch a reintroduced `outputs/...` style required path in the notebook."""

    sources = _notebook_sources()
    config_cell = next(source for source in sources if "STAGE2_CLASS_GROUPS = " in source)
    offenders = [
        line.strip()
        for line in config_cell.splitlines()
        if any(f'"{prefix}' in line for prefix in audit_module.IGNORED_PREFIXES)
    ]
    assert not offenders, f"Notebook config cell points at ignored trees: {offenders}"
