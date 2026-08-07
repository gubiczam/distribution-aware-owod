"""Inspect the acquisition registry without running anything.

    daowod-run strategies [--verbose]

Experiments are launched from `experiments/`, not from here — one entrypoint per
contribution, each taking a config:

    python experiments/contribution_a.py --config configs/contribution_a.yaml
    python experiments/contribution_b.py --config configs/contribution_b.yaml
"""

import argparse
from collections.abc import Sequence

from daowod.scoring import REQUIRED_STRATEGIES, STRATEGY_REGISTRY


def _command_strategies(args: argparse.Namespace) -> int:
    names = REQUIRED_STRATEGIES if args.required_only else STRATEGY_REGISTRY.names()
    for name in names:
        spec = STRATEGY_REGISTRY.resolve(name)
        weights = {key: value for key, value in spec.weights().items() if value > 0}
        print(f"{name:<30} {weights or 'random'}")
        if args.verbose and spec.description:
            print(f"{'':<30} {spec.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daowod-run", description="Distribution-aware OWOD active annotation."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    strategies = commands.add_parser("strategies", help="List registry strategies.")
    strategies.add_argument(
        "--required-only", action="store_true", help="Only the reported ablation matrix."
    )
    strategies.add_argument("--verbose", action="store_true", help="Include descriptions.")
    strategies.set_defaults(function=_command_strategies)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
