"""Offline analysis/calibration entry point (no model loading)."""

import argparse
import json
import sys
from pathlib import Path

from ..calibration import select_configuration
from .report import analyze_runs, load_oracle, load_run


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"analyze", "calibrate"}:
        argv.insert(0, "analyze")
    parser = argparse.ArgumentParser(description="Analyze measured reasoning-compression runs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument(
        "run_dirs", nargs="*", help="Run directories containing manifest.json/results.jsonl"
    )
    analyze.add_argument("--run", action="append", default=[])
    analyze.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    analyze.add_argument("--oracle-jsonl", "--oracle", dest="oracle_jsonl")
    analyze.add_argument("--entropy-threshold", type=float)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("run_dirs", nargs="*")
    calibrate.add_argument("--run", action="append", default=[])
    calibrate.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=0.5,
        help="Allowed accuracy drop in percentage points (0.5 means 0.005 fraction)",
    )
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument(
        "--cost", choices=("latency_s", "model_evaluated_tokens"), default="latency_s"
    )
    args = parser.parse_args(argv)
    paths = args.run_dirs + args.run
    if not paths:
        parser.error("Provide at least one run directory")
    runs = [load_run(path) for path in paths]
    if args.command == "analyze":
        oracle = []
        if args.oracle_jsonl:
            oracle = load_oracle(args.oracle_jsonl, runs[0]["manifest"])
        result = analyze_runs(
            runs, args.output_dir, oracle_records=oracle, entropy_threshold=args.entropy_threshold
        )
        print(json.dumps({"runs": len(result["summaries"]), "output_dir": args.output_dir}))
    else:
        result = select_configuration(runs, args.max_accuracy_drop / 100, cost_key=args.cost)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "selected_policy",
                        "policy_config",
                        "expected_accuracy_delta",
                        "expected_latency_reduction",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
