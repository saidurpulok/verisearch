"""Reproducible stage-oriented CLI; heavy models imported only when needed."""

from __future__ import annotations

import argparse
from pathlib import Path

from verisearch.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="verisearch", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="action", required=True)
    prepare = data_commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    generation = commands.add_parser("generate")
    generation.add_argument("--config", type=Path, required=True)
    generation.add_argument("--partition", choices=["fit", "test"], required=True)
    verifier = commands.add_parser("verifier")
    verifier_commands = verifier.add_subparsers(dest="action", required=True)
    for name in ("train", "evaluate"):
        sub = verifier_commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name == "evaluate":
            sub.add_argument("--split", choices=["validation", "test"], default="validation")
    for name in ("tune", "eval"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
    plot = commands.add_parser("plot")
    plot.add_argument("--results", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plot":
        from verisearch.evaluation.plot import plot_results

        plot_results(args.results, args.output)
        return
    config = load_config(args.config)
    if args.command == "data":
        from verisearch.data.load import prepare_data

        prepare_data(config)
    elif args.command == "generate":
        from verisearch.experiment import generate

        generate(config, args.partition)
    elif args.command == "verifier":
        if args.action == "train":
            from verisearch.experiment import train

            train(config)
        else:
            from verisearch.experiment import evaluate_verifier

            evaluate_verifier(config, args.split)
    elif args.command == "tune":
        from verisearch.experiment import tune

        tune(config)
    elif args.command == "eval":
        from verisearch.experiment import evaluate

        evaluate(config)


if __name__ == "__main__":
    main()
