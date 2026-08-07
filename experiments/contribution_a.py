#!/usr/bin/env python3
"""Contribution A entrypoint — distribution-aware active annotation.

One command per stage. The protocol lives in `configs/contribution_a.yaml`, not in
flags: sizes, seeds, budget grids, arms and the severity axis are all declared
there, so a run is reproducible from a file rather than from a shell history.

    # the annotation study: the reported experiment
    python experiments/contribution_a.py study --checkpoint <t1.pth> --split <ids.txt>

    # why the coherence gate failed, localised to components (docs/results.md 2-5)
    python experiments/contribution_a.py audit --export <export.npz> --annotations <dir>

    # is the failure the embedding or the formulation? (docs/results.md 7)
    python experiments/contribution_a.py representation --export <export.npz> \\
        --annotations <dir> --representations <dir>

Start with `--mode DEBUG`, which exercises every stage on a few hundred images in
minutes with no GPU. Its numbers are not reportable and the run says so.

The representation stage needs region embeddings that must be produced outside this
environment, because `daowod` deliberately has no torch dependency:

    python experiments/select_rows.py --export <export.npz> --output <dir>
    <prob-venv>/bin/python experiments/extract_embeddings.py \\
        --export <export.npz> --images <JPEGImages> --output <dir> --rows <dir>/rows.npy
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CONFIG = REPO_ROOT / "configs" / "contribution_a.yaml"


def _run_study(args: argparse.Namespace) -> int:
    from daowod.pipeline import PipelineConfig, run_pipeline

    overrides: dict[str, object] = {}
    for name in (
        "data_root",
        "split_file",
        "checkpoint",
        "output_dir",
        "cache_dir",
        "existing_export",
    ):
        value = getattr(args, name, None)
        if value:
            overrides[name] = value
    if args.mode:
        overrides["mode"] = args.mode
    if args.no_gpu:
        overrides["require_gpu"] = False
        overrides["device"] = "cpu"

    config = PipelineConfig.from_yaml(args.config, **overrides)
    mode = config.execution_mode()
    print(
        f"mode           : {mode.name}  ({'reportable' if mode.research_grade else 'NOT reportable'})"
    )
    print(f"images         : {mode.total_images}  (eval {mode.evaluation_images})")
    print(
        f"arms x sev x sd: {len(mode.strategies)} x {len(mode.imbalance_settings)} x {len(mode.seeds)}"
    )
    print(f"output         : {config.output_dir}")
    result = run_pipeline(config, progress=print)
    print(f"artifacts      : {config.output_dir}")
    return 0 if result is not None else 1


def _delegate(script: str, forwarded: list[str]) -> int:
    """Run a stage script in-process with the arguments it expects.

    The stage scripts stay independently runnable — each is a complete experiment
    with its own inputs and outputs — and this dispatcher is the one obvious way in.
    """

    path = REPO_ROOT / "experiments" / script
    if not path.exists():
        raise SystemExit(f"missing stage script: {path}")
    argv = sys.argv
    try:
        sys.argv = [str(path), *forwarded]
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as exit_code:  # a stage may exit non-zero
        return int(exit_code.code or 0)
    finally:
        sys.argv = argv
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Protocol: configs/contribution_a.yaml. Results: docs/results.md.",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    study = stages.add_parser("study", help="The annotation study (the reported experiment).")
    study.add_argument("--config", default=str(DEFAULT_CONFIG))
    study.add_argument(
        "--mode", default="", help="Override `mode:` (DEBUG / FAST / MAIN / MAINREVEALED)."
    )
    study.add_argument("--data-root", dest="data_root", default="")
    study.add_argument("--split", dest="split_file", default="")
    study.add_argument("--checkpoint", default="")
    study.add_argument("--existing-export", dest="existing_export", default="")
    study.add_argument("--output", dest="output_dir", default="")
    study.add_argument("--cache", dest="cache_dir", default="")
    study.add_argument("--no-gpu", action="store_true", help="For DEBUG on a CPU machine.")
    study.set_defaults(function=_run_study)

    # These three stages own their own flags, so this dispatcher must not try to
    # interpret them. `parse_known_args` in main() collects them and _delegate hands
    # them over untouched. (argparse.REMAINDER cannot do this: as a positional it
    # never captures a *leading* option such as `--export`, so the subparser rejects
    # it before the stage script is ever reached.)
    audit = stages.add_parser(
        "audit",
        help="Component-level audit of the acquisition signals.",
        add_help=False,
    )
    audit.set_defaults(script="component_audit.py")

    geometry = stages.add_parser(
        "representation",
        help="Feature-space geometry across representations.",
        add_help=False,
    )
    geometry.set_defaults(script="representation_geometry.py")

    acquisition = stages.add_parser(
        "representation-acquisition",
        help="The same acquisition arms in a different feature space (INCOMPLETE; see "
        "docs/results.md section 11).",
        add_help=False,
    )
    acquisition.set_defaults(script="representation_study.py")

    return parser


def main(argv: list[str] | None = None) -> int:
    args, extra = build_parser().parse_known_args(argv)
    script = getattr(args, "script", "")
    if script:
        return _delegate(script, extra)
    if extra:
        raise SystemExit(f"unrecognised arguments for `study`: {extra}")
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
