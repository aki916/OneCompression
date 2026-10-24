"""Select from measured calibration runs, never generated-token proxies.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..metrics.analysis import (
    _index,
    _number,
    _ratio,
    pair_failures,
    summarize,
    validate_comparable_manifests,
)


def select_configuration(
    runs: Sequence[Mapping[str, Any]], max_accuracy_drop: float, *, cost_key: str = "latency_s"
) -> dict[str, Any]:
    """Minimize measured cost on paired calibration examples.

    Each run is ``{'manifest': {..., 'split': 'calibration'}, 'records': [...]}``.
    ``max_accuracy_drop`` is an accuracy fraction: 0.005 means 0.5 percentage
    points. The CLI converts its explicitly percentage-point argument. Selection
    describes empirical calibration accuracy, not a statistical guarantee of
    held-out accuracy. The measured vanilla run is itself a valid candidate.
    """
    epsilon = _number(max_accuracy_drop, "max_accuracy_drop", nonnegative=True)
    if epsilon > 1:
        raise ValueError("max_accuracy_drop is a fraction in [0, 1]")
    if cost_key not in {"latency_s", "model_evaluated_tokens"}:
        raise ValueError("Select using measured latency_s or total model_evaluated_tokens")
    if not runs:
        raise ValueError("At least one measured run is required")
    summaries = []
    variants = set()
    for run in runs:
        if run["manifest"].get("split") != "calibration":
            raise ValueError(
                "Policy selection requires calibration split; never tune on final/test"
            )
        summary = summarize(run["records"])
        if not summary["n"]:
            raise ValueError("Empty calibration run")
        variant = json.dumps([summary["policy"], summary["policy_config"]], sort_keys=True)
        if variant in variants:
            raise ValueError(
                "Duplicate policy/configuration variants; aggregate replicates explicitly"
            )
        variants.add(variant)
        for field in ("policy", "policy_config"):
            if field in run["manifest"] and run["manifest"][field] != summary[field]:
                raise ValueError(f"Manifest {field} disagrees with result records")
        summaries.append(summary)
    references = [i for i, summary in enumerate(summaries) if summary["policy"] == "none"]
    if len(references) != 1:
        raise ValueError("Exactly one policy=none vanilla calibration run is required")
    reference_idx = references[0]
    baseline = summaries[reference_idx]
    reference = runs[reference_idx]
    reference_ids = _index(reference["records"])
    for run in runs:
        validate_comparable_manifests(reference["manifest"], run["manifest"])
        if _index(run["records"]).keys() != reference_ids.keys():
            raise ValueError("Calibration runs must use exactly the same sample IDs")
    if baseline[cost_key] is None:
        raise ValueError(f"Vanilla is missing measured {cost_key}")
    target = baseline["accuracy"] - epsilon
    eligible = [
        i
        for i, summary in enumerate(summaries)
        if summary["accuracy"] + 1e-12 >= target and summary[cost_key] is not None
    ]
    selected_idx = min(
        eligible, key=lambda i: (summaries[i][cost_key], -summaries[i]["accuracy"], i)
    )
    selected = summaries[selected_idx]
    manifest_json = json.dumps(reference["manifest"], sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "selected_run_index": selected_idx,
        "selected_policy": selected["policy"],
        "policy_config": selected["policy_config"],
        "cost_key": cost_key,
        "selected_cost": selected[cost_key],
        "vanilla_cost": baseline[cost_key],
        "max_accuracy_drop_fraction": epsilon,
        "max_accuracy_drop_percentage_points": epsilon * 100,
        "vanilla_accuracy": baseline["accuracy"],
        "selected_accuracy": selected["accuracy"],
        "expected_accuracy_delta": selected["accuracy"] - baseline["accuracy"],
        "expected_latency_reduction": (
            _ratio(baseline["latency_s"] - selected["latency_s"], baseline["latency_s"])
            if baseline["latency_s"] is not None and selected["latency_s"] is not None
            else None
        ),
        "expected_reasoning_token_reduction": (
            _ratio(
                baseline["reasoning_tokens"] - selected["reasoning_tokens"],
                baseline["reasoning_tokens"],
            )
            if baseline["reasoning_tokens"] is not None
            and selected["reasoning_tokens"] is not None
            else None
        ),
        "failure_decomposition": pair_failures(
            reference["records"], runs[selected_idx]["records"]
        ),
        "calibration_sample_ids": [list(key) for key in sorted(reference_ids)],
        "calibration_manifest": dict(reference["manifest"]),
        "calibration_manifest_sha256": hashlib.sha256(manifest_json.encode()).hexdigest(),
        "candidate_summaries": summaries,
        "selection_scope": "empirical_calibration_constraint_not_a_held_out_quality_guarantee",
    }


def validate_selection_for_evaluation(
    selection: Mapping[str, Any],
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    allow_quantization_change: bool = False,
) -> None:
    """Check selected-policy application, allowing explicit quantization transfer.

    Dataset *identity/version* must match; a different split and disjoint sample
    IDs are mandatory. Model/decoding changes invalidate the calibration.
    """
    if manifest.get("split") not in {"validation", "evaluation", "final", "test"}:
        raise ValueError("Selected configuration evaluation requires a held-out split")
    validate_comparable_manifests(
        selection["calibration_manifest"],
        manifest,
        allow_quantization_change=allow_quantization_change,
    )
    training_ids = {tuple(key) for key in selection["calibration_sample_ids"]}
    if training_ids.intersection(_index(records)):
        raise ValueError("Evaluation samples overlap policy calibration samples")
