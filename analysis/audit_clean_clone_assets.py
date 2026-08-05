#!/usr/bin/env python3
"""Audit that every required input survives a clean `git clone`.

The failure this prevents: a config or notebook points at a file that exists
only on the machine that generated it, because it lives under an ignored tree
such as `outputs/`. The clone succeeds, the run dies on the first missing file,
somebody fixes that one file, and the next run dies on the second one.

The audit therefore reports *every* violation at once, and enforces the rule
that each required input is either

1. repository-owned - tracked by git, under a source directory that a clean
   clone materialises (`data/protocol/`, `configs/`, `src/`, ...); or
2. runtime-supplied - an absolute path the Colab notebook rewrites to a
   Drive/runtime location and validates there before use.

Run `python analysis/audit_clean_clone_assets.py` from the repository root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Trees that .gitignore excludes. A required input reachable only through one of
# these is, by definition, absent from a clean clone.
IGNORED_PREFIXES = (
    "outputs/",
    "results/",
    "cache/",
    "tmp/",
    "runs/",
    "checkpoints/",
    "wandb/",
    "external/",
)

# Repository-owned inputs the Stage 2 Colab notebook requires after checkout.
# Keep in sync with `required_repo_paths` in notebooks/stage2_master_colab.ipynb.
#: Inputs every clean clone must carry, independent of which configs exist.
FIXED_REPO_PATHS = (
    "pyproject.toml",
    "data/protocol/stage1b/stage1b_candidate_500.txt",
    "data/protocol/stage1b/stage1b_reference_3500.txt",
    "data/protocol/stage2/stage2_class_groups.csv",
    "src/daowod/config.py",
)


def discover_configs(repo_root: Path | None = None) -> tuple[str, ...]:
    """Every *executable* config in `configs/`, discovered rather than listed.

    Two rules, both learned from failures this module exists to catch:

    * **Discovered, not listed.** A hardcoded list drifted every time a config was
      added or removed, so the audit silently stopped covering things.
    * **Executable only.** A config that declares a ``protocol:`` block names real,
      fully resolved inputs and must be clean-clone reachable. A template whose
      paths are placeholders (``/path/to/...``) has nothing to resolve, and
      auditing it would only produce noise.
    """

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    discovered = []
    for path in sorted((root / "configs").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(config, dict) and "protocol" in config:
            discovered.append(f"configs/{path.name}")
    return tuple(discovered)


STAGE2_CONFIGS = discover_configs()
REQUIRED_REPO_PATHS = (*FIXED_REPO_PATHS, *STAGE2_CONFIGS)

# Config keys that name an input the run must be able to read. `output_dir` is
# excluded on purpose: it is a destination, not an input.
CONFIG_INPUT_KEYS = (
    ("dataset", "image_set_path"),
    ("dataset", "annotations_dir"),
    ("dataset", "class_groups_path"),
    ("dataset", "known_class_groups_path"),
    ("protocol", "candidate_pool_split"),
    ("protocol", "reference_split"),
    ("protocol", "initial_labelled_split"),
    ("protocol", "data_root"),
    ("protocol", "checkpoint"),
    ("prob", "repository_path"),
    ("prob", "initial_checkpoint"),
)

# Keys the notebook's config-materialisation cell rewrites to runtime locations
# before validation. An absolute path is acceptable only for these; anywhere
# else it is a machine-local path that no other environment can resolve.
RUNTIME_OVERRIDDEN_KEYS = frozenset(
    {
        ("dataset", "annotations_dir"),
        ("protocol", "data_root"),
        ("protocol", "checkpoint"),
        ("prob", "repository_path"),
        ("prob", "initial_checkpoint"),
    }
)


def tracked_files(root: Path) -> set[str] | None:
    """Paths git tracks, or None when this is not a usable git work tree."""

    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return {entry for entry in result.stdout.split("\0") if entry}


def under_ignored_tree(relative_path: str) -> str | None:
    posix = Path(relative_path).as_posix()
    for prefix in IGNORED_PREFIXES:
        if posix.startswith(prefix):
            return prefix.rstrip("/")
    return None


def check_repo_owned(
    relative_path: str,
    *,
    root: Path,
    tracked: set[str] | None,
    origin: str,
) -> list[dict[str, str]]:
    """Return every violation for one repository-owned input."""

    findings: list[dict[str, str]] = []
    posix = Path(relative_path).as_posix()
    ignored_tree = under_ignored_tree(posix)
    if ignored_tree is not None:
        findings.append(
            {
                "path": posix,
                "origin": origin,
                "problem": "required_input_under_ignored_tree",
                "detail": (
                    f"resolves to the ignored '{ignored_tree}/' tree, so a clean clone "
                    "cannot supply it; move it under data/protocol/, configs/, or "
                    "another tracked source directory"
                ),
            }
        )
    if not (root / posix).exists():
        findings.append(
            {
                "path": posix,
                "origin": origin,
                "problem": "required_input_missing",
                "detail": "does not exist in the working tree",
            }
        )
    if tracked is not None and posix not in tracked and ignored_tree is None:
        findings.append(
            {
                "path": posix,
                "origin": origin,
                "problem": "required_input_untracked",
                "detail": "exists locally but git does not track it, so it is not cloned",
            }
        )
    return findings


def audit(root: Path = REPO_ROOT) -> dict[str, Any]:
    tracked = tracked_files(root)
    findings: list[dict[str, str]] = []
    repo_owned: list[dict[str, str]] = []
    runtime_supplied: list[dict[str, str]] = []

    for relative_path in REQUIRED_REPO_PATHS:
        origin = "notebook:required_repo_paths"
        findings.extend(check_repo_owned(relative_path, root=root, tracked=tracked, origin=origin))
        repo_owned.append({"path": relative_path, "origin": origin})

    for config_relative in STAGE2_CONFIGS:
        config_path = root / config_relative
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        for section, key in CONFIG_INPUT_KEYS:
            value = (config.get(section) or {}).get(key)
            if value in (None, ""):
                continue
            origin = f"{config_relative}:{section}.{key}"
            if Path(str(value)).is_absolute():
                if (section, key) not in RUNTIME_OVERRIDDEN_KEYS:
                    findings.append(
                        {
                            "path": str(value),
                            "origin": origin,
                            "problem": "unvalidated_absolute_path",
                            "detail": (
                                "absolute path in a key the notebook does not rewrite to a "
                                "validated runtime asset; no other machine can resolve it"
                            ),
                        }
                    )
                else:
                    runtime_supplied.append({"path": str(value), "origin": origin})
                continue
            findings.extend(check_repo_owned(str(value), root=root, tracked=tracked, origin=origin))
            repo_owned.append({"path": Path(str(value)).as_posix(), "origin": origin})

    unique_repo_paths = sorted({row["path"] for row in repo_owned})
    return {
        "schema": "clean_clone_asset_audit_v1",
        "git_tracking_checked": tracked is not None,
        "repo_owned_required_paths": unique_repo_paths,
        "repo_owned_required_path_count": len(unique_repo_paths),
        "runtime_supplied_assets": sorted(
            {(row["path"], row["origin"]) for row in runtime_supplied}
        ),
        "findings": findings,
        "status": "PASS" if not findings else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write the audit report here")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root to audit")
    args = parser.parse_args(argv)

    report = audit(args.root)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"Repository-owned required inputs: {report['repo_owned_required_path_count']}")
    for path in report["repo_owned_required_paths"]:
        print(f"  OK   {path}")
    for path, origin in report["runtime_supplied_assets"]:
        print(f"  RUNTIME  {path}  <- {origin}")
    if not report["git_tracking_checked"]:
        print("NOTE: git tracking was not verifiable here; existence was still checked.")

    if report["findings"]:
        print(f"\nclean-clone asset audit FAILED with {len(report['findings'])} finding(s):")
        for finding in report["findings"]:
            print(f"  [{finding['problem']}] {finding['path']}")
            print(f"      required by {finding['origin']}")
            print(f"      {finding['detail']}")
        return 1

    print("\nclean-clone asset audit PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
