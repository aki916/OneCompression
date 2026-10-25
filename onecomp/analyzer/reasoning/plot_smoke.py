"""Replot saved measurements without rerunning inference.

Copyright 2025-2026 Fujitsu Ltd.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="cpu_smoke or cpu_profile directory")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from onecomp.reasoning.metrics import pareto_frontier

    report = json.loads((args.run / "results/summary.json").read_text())
    if report["manifest"].get("experiment_kind") != "synthetic_cpu_mechanics_only":
        raise ValueError("This plotter is explicitly limited to the artificial CPU fixture")
    rows = report["summaries"]
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    for row in rows:
        name, cfg = row["policy"], row["policy_config"]
        suffix = ""
        if name == "fixed":
            suffix = f" budget={cfg['budget']}"
        elif name == "thinkbrake":
            suffix = f" gap={cfg['threshold']}"
        elif name == "deer":
            suffix = f" confidence={cfg['confidence_threshold']}, interval={cfg['interval']}"
        ax.scatter(1000 * row["latency_s"], row["accuracy"], label=name + suffix, s=45)
    frontier = pareto_frontier(rows)
    ax.plot(
        [1000 * p["latency_s"] for p in frontier],
        [p["accuracy"] for p in frontier],
        color="0.5",
        linewidth=1,
        alpha=0.7,
    )
    ax.set(
        xlabel="Mean measured CPU latency (ms)",
        ylabel="Scripted fixture accuracy",
        ylim=(-0.03, 1.03),
        title="SYNTHETIC CPU mechanics — scripted output head",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.text(
        0.01,
        -0.025,
        "Real tiny-Qwen3 forwards; artificial answers. No Qwen3-8B or GPU quality/speedup claim.",
        fontsize=9,
    )
    output = args.run / "plots"
    output.mkdir(exist_ok=True)
    for extension in ("svg", "png"):
        fig.savefig(output / f"pareto.{extension}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    (output / "plot_metadata.json").write_text(
        json.dumps(
            {
                "matplotlib": matplotlib.__version__,
                "source": "results/summary.json",
                "script": "onecomp.analyzer.reasoning.plot_smoke",
                "inference_rerun": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
