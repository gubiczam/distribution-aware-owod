"""One command reproduces an experiment.

    daowod-run validate   --config configs/experiment.yaml
    daowod-run strategies
    daowod-run campaign   --config configs/experiment.yaml
    daowod-run diagnose   --candidates round/candidate_proposals.npz \\
                          --references round/reference_proposals.npz \\
                          --output outputs/diagnostics

``validate`` and ``strategies`` need no detector, so a configuration can be
checked before any GPU time is spent.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from daowod.config import ExperimentConfig, load_config
from daowod.scoring import REQUIRED_STRATEGIES, STRATEGY_REGISTRY


def _git_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run_manifest(config: ExperimentConfig, *, extra: dict[str, object] | None = None) -> dict:
    """Everything needed to reproduce a run: commit, config, environment, seeds."""

    repository = Path(__file__).resolve().parents[2]
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(repository),
        "repository": str(repository),
        "config_source": config.source_path,
        "config_fingerprint": config.fingerprint(),
        "config": config.as_dict(),
        "seeds": list(config.active_learning.seeds),
        "strategies": list(config.acquisition.strategies),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }
    if extra:
        manifest.update(extra)
    return manifest


def _command_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    specs = config.acquisition.resolved_specs()
    print(f"configuration : {config.source_path}")
    print(f"fingerprint   : {config.fingerprint()}")
    print(f"seeds         : {list(config.active_learning.seeds)}")
    print(f"rounds        : {config.active_learning.rounds}")
    print(f"budget/round  : {config.active_learning.budget_per_round}")
    print(f"grouped metrics: {config.evaluation.grouped_metrics}")
    if config.legacy_aliases:
        print("legacy keys   :")
        for key, translation in sorted(config.legacy_aliases.items()):
            print(f"  {key}: {translation}")
    print(f"strategies    : {len(specs)}")
    for requested, spec in zip(config.acquisition.strategies, specs, strict=True):
        weights = {name: value for name, value in spec.weights().items() if value > 0}
        print(
            f"  {requested:<28} v{spec.semantics_version} "
            f"weights={weights or 'random'} "
            f"u={spec.uncertainty_method} coh={spec.coherence_method} "
            f"norm={spec.normalisation} agg={spec.image_aggregation}/{spec.top_k}"
        )
    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(
            json.dumps(run_manifest(config), indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"manifest      : {args.manifest}")
    print("configuration is valid")
    return 0


def _command_strategies(args: argparse.Namespace) -> int:
    names = REQUIRED_STRATEGIES if args.required_only else STRATEGY_REGISTRY.qualified_names()
    for name in names:
        spec = STRATEGY_REGISTRY.resolve(name)
        weights = {key: value for key, value in spec.weights().items() if value > 0}
        marker = " [deprecated]" if spec.deprecated else ""
        print(f"{name:<30} {weights or 'random'}{marker}")
        if args.verbose and spec.description:
            print(f"{'':<30} {spec.description}")
    return 0


def _command_campaign(args: argparse.Namespace) -> int:
    from daowod.experiment import ActiveLearningCampaign
    from daowod.prob_adapter import ProbAdapter

    config = load_config(args.config)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest(config), indent=2, default=str) + "\n", encoding="utf-8"
    )
    adapter = ProbAdapter(
        repository_path=config.prob.repository_path,
        train_command=config.prob.train_command,
        predict_command=config.prob.predict_command,
        evaluate_command=config.prob.evaluate_command,
        timeout_seconds=config.prob.timeout_seconds,
    )
    result = ActiveLearningCampaign(config, adapter).run()
    print(f"rounds recorded: {len(result.metrics)}")
    print(f"output         : {result.output_dir}")
    return 0


def _command_diagnose(args: argparse.Namespace) -> int:
    from daowod.offline import diagnose_pool

    report = diagnose_pool(
        candidate_proposals=args.candidates,
        reference_proposals=args.references,
        output_dir=args.output,
        strategies=tuple(args.strategies) if args.strategies else None,
        budget=args.budget,
        seeds=tuple(args.seeds),
        class_stats_path=args.class_stats,
        annotations_dir=args.annotations,
        unknown_classes=tuple(args.unknown_classes) if args.unknown_classes else (),
    )
    print(json.dumps(report["headline"], indent=2, default=str))
    print(f"artifacts: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daowod-run", description="Distribution-aware OWOD active learning."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate a configuration file.")
    validate.add_argument("--config", required=True)
    validate.add_argument("--manifest", default="", help="Write a run manifest here.")
    validate.set_defaults(function=_command_validate)

    strategies = commands.add_parser("strategies", help="List registry strategies.")
    strategies.add_argument("--required-only", action="store_true")
    strategies.add_argument("--verbose", action="store_true")
    strategies.set_defaults(function=_command_strategies)

    campaign = commands.add_parser("campaign", help="Run the full campaign.")
    campaign.add_argument("--config", required=True)
    campaign.set_defaults(function=_command_campaign)

    diagnose = commands.add_parser(
        "diagnose", help="Offline multi-seed diagnostics over exported proposals."
    )
    diagnose.add_argument("--candidates", required=True)
    diagnose.add_argument("--references", required=True)
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument("--strategies", nargs="*", default=None)
    diagnose.add_argument("--budget", type=int, default=10)
    diagnose.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    diagnose.add_argument("--class-stats", default=None)
    diagnose.add_argument("--annotations", default=None)
    diagnose.add_argument("--unknown-classes", nargs="*", default=None)
    diagnose.set_defaults(function=_command_diagnose)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
