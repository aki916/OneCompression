"""Machine-readable analysis and a standalone measured-latency Pareto plot.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .analysis import (
    binary_metrics,
    entropy_hypothesis,
    oracle_headroom,
    pair_failures,
    pareto_frontier,
    summarize,
    validate_comparable_manifests,
)


def load_run(path: str | Path) -> dict[str, Any]:
    """Load a run directory, or JSONL beside its manifest.json."""
    path = Path(path)
    data_path = path / "results.jsonl" if path.is_dir() else path
    manifest_path = data_path.parent / "manifest.json"
    status_path = data_path.parent / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise ValueError(f"Run is incomplete or failed: {data_path.parent}")
    with data_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return {"manifest": json.loads(manifest_path.read_text(encoding="utf-8")), "records": records}


def load_oracle(path: str | Path, reference_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read oracle rows, checking any sibling manifest/status before analysis."""
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_comparable_manifests(reference_manifest, manifest)
        if manifest.get("split") != reference_manifest.get("split"):
            raise ValueError("Oracle and analyzed runs must share the same split")
    status_path = path.parent / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise ValueError("Oracle run is incomplete or failed")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def analyze_runs(
    runs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    oracle_records: Sequence[Mapping[str, Any]] = (),
    entropy_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare identical paired runs and emit summary.json plus pareto.svg.

    Oracle and signal outputs are absent if unmeasured. Signals cannot silently
    acquire filled-in zero values, and held-out model-quality claims are outside
    this report's scope.
    """
    summaries = [dict(summarize(run["records"]), run_index=i) for i, run in enumerate(runs)]
    references = [i for i, summary in enumerate(summaries) if summary.get("policy") == "none"]
    if len(references) != 1:
        raise ValueError("Analysis requires exactly one vanilla policy=none run")
    reference = runs[references[0]]
    for run, summary in zip(runs, summaries):
        validate_comparable_manifests(reference["manifest"], run["manifest"])
        if reference["manifest"].get("split") != run["manifest"].get("split"):
            raise ValueError("Paired analysis requires the same split")
        summary["failure_decomposition"] = pair_failures(reference["records"], run["records"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "manifest": reference["manifest"],
        "summaries": summaries,
        "latency_pareto_frontier": pareto_frontier(summaries),
        "model_evaluation_pareto_frontier": pareto_frontier(
            summaries, cost_key="model_evaluated_tokens"
        ),
        "cost_caution": "Measured latency includes probes; serial reference timing is not vLLM serving throughput.",
    }
    if oracle_records:
        report["oracle"] = {
            str(i): oracle_headroom(reference["records"], oracle_records, run["records"])
            for i, run in enumerate(runs)
        }
        signals = {}
        definitions = {
            "reasoning_tokens": (1, False),
            "end_logit": (1, False),
            "end_probability": (1, True),
            "end_margin": (1, False),
            "logit_margin": (-1, False),
            "probe_confidence": (1, True),
            "confidence": (1, True),
            "answer_stability": (1, True),
            "attention_entropy": (-1, False),
            "relative_entropy_change": (-1, False),
            "relative_entropy": (-1, False),
            "backward_ratio": (-1, False),
        }
        for key, (direction, probability) in definitions.items():
            available = [row for row in oracle_records if row.get(key) is not None]
            if available:
                observed_scores = sorted(direction * row[key] for row in available)
                thresholds = (
                    (0.5, 0.9, 0.95)
                    if probability
                    else sorted(
                        {
                            observed_scores[int((len(observed_scores) - 1) * fraction)]
                            for fraction in (0.1, 0.5, 0.9)
                        }
                    )
                )
                signals[key] = binary_metrics(
                    [row["safe_stop"] for row in available],
                    [direction * row[key] for row in available],
                    probabilities=probability,
                    thresholds=thresholds,
                )
                signals[key]["score_direction"] = direction
                signals[key]["missing_count"] = len(oracle_records) - len(available)
                signals[key]["threshold_scope"] = (
                    "fixed_probability_points"
                    if probability
                    else "descriptive_score_quantiles_not_calibrated_policies"
                )
        report["safe_stop_signal_metrics"] = signals
        if entropy_threshold is not None:
            report["entropy_hypothesis"] = entropy_hypothesis(oracle_records, entropy_threshold)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_pareto(summaries, destination / "pareto.svg")
    return report


def plot_pareto(points: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Export a static scientific plot; use matplotlib when available.

    A dependency-free SVG fallback retains axes, units, labels and an explicit
    measured-cost caption. It contains no inferred/extrapolated observations.
    """
    valid = [
        point
        for point in points
        if point.get("latency_s") is not None and point.get("accuracy") is not None
    ]
    frontier = pareto_frontier(valid)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _svg_pareto(valid, frontier, Path(path))
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    for point in valid:
        axis.scatter(point["latency_s"], point["accuracy"], label=_point_label(point))
    if frontier:
        axis.plot(
            [p["latency_s"] for p in frontier],
            [p["accuracy"] for p in frontier],
            color="black",
            alpha=0.4,
        )
    axis.set(
        xlabel="Mean measured end-to-end latency (s)",
        ylabel="Accuracy (fraction)",
        ylim=(0, 1.02),
        title="Reasoning compression: measured quality–latency frontier",
    )
    axis.grid(alpha=0.2)
    if valid:
        axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _point_label(point):
    return f"{point.get('policy', 'unknown')} {json.dumps(point.get('policy_config') or {}, sort_keys=True)}"


def _svg_pareto(points, frontier, path):
    width, plot_height = 1050, 430
    height = plot_height + 130 + 22 * len(points)
    left, right, top, bottom = 85, 1010, 55, plot_height
    xmax = max((float(p["latency_s"]) for p in points), default=1.0) * 1.08 or 1.0

    def x(value):
        return left + value / xmax * (right - left)

    def y(value):
        return bottom - value * (bottom - top)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#172033">',
        '<text x="85" y="28" font-size="18">Reasoning compression: measured quality–latency frontier</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        yy, xx = y(fraction), x(xmax * fraction)
        elements += [
            f'<path d="M {left} {yy} H {right}" stroke="#e4e8ed"/>',
            f'<text x="72" y="{yy + 4}" text-anchor="end" font-size="12">{fraction:.1f}</text>',
            f'<text x="{xx}" y="{bottom + 22}" text-anchor="middle" font-size="12">{xmax * fraction:.4g}</text>',
        ]
    elements += [
        f'<path d="M {left} {top} V {bottom} H {right}" stroke="#172033" fill="none"/>',
        f'<text x="{(left+right)/2}" y="{bottom+48}" text-anchor="middle">Mean measured end-to-end latency (s)</text>',
        f'<text x="23" y="{(top+bottom)/2}" text-anchor="middle" transform="rotate(-90 23 {(top+bottom)/2})">Accuracy (fraction)</text>',
    ]
    if frontier:
        coords = " ".join(f'{x(point["latency_s"])},{y(point["accuracy"])}' for point in frontier)
        elements.append(
            f'<polyline points="{coords}" fill="none" stroke="#8993a2" stroke-width="1.5"/>'
        )
    colors = ("#2563eb", "#d94801", "#059669", "#9333ea", "#db2777", "#087f8c")
    for i, point in enumerate(points):
        color = colors[i % len(colors)]
        label = html.escape(_point_label(point))
        elements += [
            f'<circle cx="{x(point["latency_s"])}" cy="{y(point["accuracy"])}" r="5" fill="{color}"><title>{label}</title></circle>',
            f'<circle cx="90" cy="{bottom+85+i*22}" r="4" fill="{color}"/>',
            f'<text x="103" y="{bottom+89+i*22}" font-size="12">{label}</text>',
        ]
    if not points:
        elements.append(
            '<text x="400" y="200">No measured latency/accuracy pairs available</text>'
        )
    elements.append("</g></svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")
