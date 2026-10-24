# Copyright 2025-2026 Fujitsu Ltd.

"""Separate research CLI; existing onecomp quantization CLI remains unchanged."""

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "analyze":
        from .metrics.__main__ import main as analyze

        return analyze(argv[1:])
    if argv and argv[0] == "calibrate":
        from .calibration.__main__ import main as calibrate

        return calibrate(argv[1:])
    parser = argparse.ArgumentParser(
        prog="onecomp-reasoning", description="Runtime reasoning compression experiments (opt-in)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "oracle"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--reasoning-compression", choices=("auto",))
        sub.add_argument("--selected-config", type=Path)
    prep = commands.add_parser("prepare-data")
    from .data import DATASETS

    prep.add_argument("dataset", choices=DATASETS)
    prep.add_argument("--revision", required=True, help="Immutable dataset Hub commit SHA")
    prep.add_argument("--output", required=True, type=Path)
    prep.add_argument("--seed", default=0, type=int)
    prep.add_argument("--cache-dir", type=Path)
    commands.add_parser("environment")
    commands.add_parser("analyze", help="Analyze paired run directories (see analyze --help)")
    commands.add_parser("calibrate", help="Choose measured minimum cost under quality constraint")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "environment":
        from .experiment import environment_metadata

        print(json.dumps(environment_metadata(), indent=2))
    elif args.command == "prepare-data":
        from .data import prepare_dataset, write_json

        info = prepare_dataset(
            args.dataset,
            args.output,
            revision=args.revision,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
        write_json(args.output.with_suffix(".manifest.json"), info)
        print(json.dumps(info, indent=2))
    else:
        from .experiment import run_experiment, run_oracle

        config = json.loads(args.config.read_text())
        selection = None
        if args.reasoning_compression == "auto":
            if not args.selected_config:
                parser.error("auto requires --selected-config from an offline calibration run")
            selection = json.loads(args.selected_config.read_text())
        elif args.selected_config:
            parser.error("--selected-config requires --reasoning-compression auto")
        if args.command == "oracle":
            if selection:
                parser.error("oracle does not apply a deployed policy selection")
            run_oracle(config, args.output)
        else:
            run_experiment(config, args.output, selection=selection)


if __name__ == "__main__":
    main()
