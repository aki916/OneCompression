"""Paired weight/reasoning interactions and held-out threshold transfer.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .analysis import pair_failures, summarize, validate_comparable_manifests


def _variant(summary):
    return json.dumps([summary["policy"], summary["policy_config"]], sort_keys=True)


def compare_quantization_runs(
    bf16_runs: Sequence[Mapping[str, Any]], quantized_runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare identical policy configurations across two measured weight formats.

    Every requested configuration must exist in both formats. Report actual
    latency ratios, not a product of token and per-token speedup estimates.
    This function does not calibrate thresholds; use the transfer function for
    comparisons of independently calibrated versus transferred configurations.
    """

    def index(runs):
        indexed = {}
        for run in runs:
            summary = summarize(run["records"])
            key = _variant(summary)
            if key in indexed:
                raise ValueError("Duplicate weight-format policy/configuration variant")
            indexed[key] = (run, summary)
        return indexed

    originals, quantized = index(bf16_runs), index(quantized_runs)
    if not originals or originals.keys() != quantized.keys():
        raise ValueError(
            "Both weight formats require the same nonempty set of policy configurations"
        )
    comparisons = []
    for key, (before, before_summary) in originals.items():
        after, after_summary = quantized[key]
        validate_comparable_manifests(
            before["manifest"], after["manifest"], allow_quantization_change=True
        )
        if before["manifest"]["quantization"] == after["manifest"]["quantization"]:
            raise ValueError(
                "Weight-compression analysis requires different quantization identities"
            )
        if before["manifest"].get("split") != after["manifest"].get("split"):
            raise ValueError("Weight-format comparison requires the same split")
        paired = pair_failures(before["records"], after["records"])
        before_latency, after_latency = before_summary["latency_s"], after_summary["latency_s"]
        comparisons.append(
            {
                "policy": before_summary["policy"],
                "policy_config": before_summary["policy_config"],
                "bf16": before_summary,
                "quantized": after_summary,
                "accuracy_delta": after_summary["accuracy"] - before_summary["accuracy"],
                "reasoning_tokens_delta": (
                    after_summary["reasoning_tokens"] - before_summary["reasoning_tokens"]
                    if after_summary["reasoning_tokens"] is not None
                    and before_summary["reasoning_tokens"] is not None
                    else None
                ),
                "measured_latency_speedup": (
                    before_latency / after_latency
                    if before_latency is not None
                    and after_latency is not None
                    and after_latency > 0
                    else None
                ),
                "paired_weight_change_outcomes": paired,
            }
        )
    return {
        "comparisons": comparisons,
        "scope": "paired_same_policy_weight_format_comparison_not_multiplicative_speedup",
    }


def evaluate_threshold_transfer(
    bf16_calibration_runs: Sequence[Mapping[str, Any]],
    quantized_calibration_runs: Sequence[Mapping[str, Any]],
    quantized_evaluation_runs: Sequence[Mapping[str, Any]],
    max_accuracy_drop: float,
) -> dict[str, Any]:
    """Compare BF16-selected and 4-bit-selected policies on held-out 4-bit data.

    Evaluation examples must be disjoint from *both* calibration datasets. The
    candidate configurations selected by either calibration must actually have
    been measured in the quantized evaluation, including the vanilla baseline.
    This can also test budget transfer; no quality deltas are inferred from
    calibration accuracy alone.
    """
    from ..calibration import select_configuration, validate_selection_for_evaluation

    source = select_configuration(bf16_calibration_runs, max_accuracy_drop)
    target = select_configuration(quantized_calibration_runs, max_accuracy_drop)
    validate_comparable_manifests(
        source["calibration_manifest"],
        target["calibration_manifest"],
        allow_quantization_change=True,
    )
    if (
        source["calibration_manifest"]["quantization"]
        == target["calibration_manifest"]["quantization"]
    ):
        raise ValueError("Threshold transfer requires distinct weight-format identities")
    if source["calibration_sample_ids"] != target["calibration_sample_ids"]:
        raise ValueError("Use paired calibration examples to isolate weight-format effects")
    evaluation = {}
    for run in quantized_evaluation_runs:
        validate_selection_for_evaluation(
            source, run["manifest"], run["records"], allow_quantization_change=True
        )
        validate_selection_for_evaluation(target, run["manifest"], run["records"])
        summary = summarize(run["records"])
        key = _variant(summary)
        if key in evaluation:
            raise ValueError("Duplicate quantized evaluation configuration")
        evaluation[key] = (run, summary)

    def selected_run(selection):
        key = json.dumps(
            [selection["selected_policy"], selection["policy_config"]], sort_keys=True
        )
        if key not in evaluation:
            raise ValueError("A selected configuration has no held-out quantized measurement")
        return evaluation[key]

    transferred_run, transferred = selected_run(source)
    recalibrated_run, recalibrated = selected_run(target)
    baselines = [
        (run, summary) for run, summary in evaluation.values() if summary["policy"] == "none"
    ]
    if len(baselines) != 1:
        raise ValueError("Exactly one quantized evaluation vanilla run is required")
    vanilla_run, vanilla = baselines[0]
    return {
        "bf16_calibration_selection": source,
        "quantized_calibration_selection": target,
        "quantized_evaluation_vanilla": vanilla,
        "quantized_evaluation_bf16_transferred": transferred,
        "quantized_evaluation_recalibrated": recalibrated,
        "transferred_failure_decomposition": pair_failures(
            vanilla_run["records"], transferred_run["records"]
        ),
        "recalibrated_failure_decomposition": pair_failures(
            vanilla_run["records"], recalibrated_run["records"]
        ),
        "heldout_accuracy_gain_from_recalibration": recalibrated["accuracy"]
        - transferred["accuracy"],
        "heldout_latency_delta_from_recalibration_s": (
            recalibrated["latency_s"] - transferred["latency_s"]
            if recalibrated["latency_s"] is not None and transferred["latency_s"] is not None
            else None
        ),
    }
