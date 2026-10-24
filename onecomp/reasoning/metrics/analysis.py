"""Metrics with explicit denominators, pairing, ties, and missing measurements.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean
from typing import Any


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _label(value: Any) -> int:
    if value not in (0, 1, False, True) or value is None:
        raise ValueError(f"Expected binary correctness label, got {value!r}")
    return int(value)


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return number


def _sample_key(row: Mapping[str, Any]) -> tuple[str, str]:
    if "sample_id" not in row:
        raise ValueError("Every record requires a sample_id")
    return str(row.get("dataset", "")), str(row["sample_id"])


def _index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        key = _sample_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate sample key: {key}")
        indexed[key] = row
    return indexed


def pair_failures(
    vanilla: Iterable[Mapping[str, Any]], compressed: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Pair by dataset/sample identity; reject missing/duplicated examples.

    Harm is conditional on vanilla correctness; rescue on vanilla incorrectness.
    An empty denominator is null, never an apparent zero error rate.
    """
    reference, candidate = _index(vanilla), _index(compressed)
    if reference.keys() != candidate.keys():
        raise ValueError("Paired runs must contain exactly the same dataset/sample IDs")
    counts = {"correct_correct": 0, "correct_wrong": 0, "wrong_correct": 0, "wrong_wrong": 0}
    for key, row in reference.items():
        for identity in (
            "prompt_token_ids",
            "prompt_ids",
            "prompt_sha256",
            "expected_answer",
            "reference_answer",
        ):
            if identity in row or identity in candidate[key]:
                if (
                    identity not in row
                    or identity not in candidate[key]
                    or row[identity] != candidate[key][identity]
                ):
                    raise ValueError(f"Paired sample {key} differs in {identity}")
        before = "correct" if _label(row["correct"]) else "wrong"
        after = "correct" if _label(candidate[key]["correct"]) else "wrong"
        counts[f"{before}_{after}"] += 1
    return {
        "n": len(reference),
        **counts,
        "harm_rate": _ratio(
            counts["correct_wrong"], counts["correct_correct"] + counts["correct_wrong"]
        ),
        "rescue_rate": _ratio(
            counts["wrong_correct"], counts["wrong_correct"] + counts["wrong_wrong"]
        ),
    }


_MEAN_FIELDS = (
    "reasoning_tokens",
    "answer_tokens",
    "total_generated_tokens",
    "probe_tokens",
    "normal_model_evaluated_tokens",
    "probe_model_evaluated_tokens",
    "model_evaluated_tokens",
    "latency_s",
    "time_to_first_token_s",
    "reasoning_latency_s",
    "probe_latency_s",
    "intervention_overhead_s",
    "controller_overhead_s",
    "attention_latency_s",
    "peak_memory",
    "kv_cache_peak_bytes",
)


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one policy/configuration; absent costs stay null.

    Throughput here is serial-observed throughput (sum of request wall times),
    not a concurrent-serving throughput measurement. The caller must supply
    a batch measurement separately to make a serving-performance claim.
    """
    rows = list(records)
    _index(rows)
    for key in ("policy", "policy_config", "model", "quantization"):
        if rows and any(row.get(key) != rows[0].get(key) for row in rows[1:]):
            raise ValueError(f"Cannot aggregate different {key} values")
    result: dict[str, Any] = {"n": len(rows)}
    if rows:
        result.update(
            {k: rows[0].get(k) for k in ("policy", "policy_config", "model", "quantization")}
        )
    correct = sum(_label(row["correct"]) for row in rows)
    result.update(accuracy=_ratio(correct, len(rows)), pass_at_1=_ratio(correct, len(rows)))
    available = {}
    for key in _MEAN_FIELDS:
        values = [
            _number(row[key], key, nonnegative=True) for row in rows if row.get(key) is not None
        ]
        available[key] = len(values)
        result[key] = fmean(values) if rows and len(values) == len(rows) else None
    result["measurement_counts"] = available
    latency = result["latency_s"]
    result["serial_requests_per_s"] = _ratio(1.0, latency) if latency is not None else None
    result["serial_generated_tokens_per_s"] = (
        _ratio(result["total_generated_tokens"], latency)
        if latency is not None and result["total_generated_tokens"] is not None
        else None
    )
    if rows and all(row.get("vanilla_correct") is not None for row in rows):
        result["failure_decomposition"] = pair_failures(
            [dict(row, correct=row["vanilla_correct"]) for row in rows], rows
        )
    return result


def pareto_frontier(
    points: Sequence[Mapping[str, Any]], *, cost_key: str = "latency_s"
) -> list[dict[str, Any]]:
    """Return nondominated measured points, retaining exact ties.

    Accuracy is maximized; the supplied measured cost is minimized. Unknown
    measurements cannot be compared and are omitted. Equal cost/accuracy points
    are all retained so implementation identities are not arbitrarily discarded.
    """
    valid = []
    for point in points:
        if point.get(cost_key) is None or point.get("accuracy") is None:
            continue
        cost = _number(point[cost_key], cost_key, nonnegative=True)
        quality = _number(point["accuracy"], "accuracy")
        if not 0 <= quality <= 1:
            raise ValueError("accuracy must be in [0, 1]")
        valid.append((cost, quality, dict(point)))
    valid.sort(key=lambda item: (item[0], -item[1]))
    frontier = []
    best_quality = -math.inf
    previous_cost = None
    best_at_previous_cost = None
    for cost, quality, point in valid:
        if quality > best_quality or (cost == previous_cost and quality == best_at_previous_cost):
            frontier.append(point)
            best_quality = max(best_quality, quality)
            previous_cost, best_at_previous_cost = cost, quality
    return frontier


def binary_metrics(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    thresholds: Sequence[float] = (),
    probabilities: bool = False,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Safe-stop ranking and operating points; larger scores predict safe stop.

    AUROC gives half credit to ties. ``auprc`` is non-interpolated average
    precision, grouping equal scores before integrating recall increments.
    ``false_stop_rate`` means P(unsafe | predicted stop); false-positive rate
    additionally reports P(predicted stop | unsafe). Neither is conflated with
    a per-request harmful-stop probability.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal lengths")
    if not isinstance(calibration_bins, int) or calibration_bins <= 0:
        raise ValueError("calibration_bins must be a positive integer")
    pairs = [(_number(s, "score"), _label(y)) for y, s in zip(labels, scores)]
    if probabilities and any(not 0 <= s <= 1 for s, _ in pairs):
        raise ValueError("Probability scores must be in [0, 1]")
    positives = sum(y for _, y in pairs)
    negatives = len(pairs) - positives
    groups: dict[float, list[int]] = defaultdict(list)
    for score, label in pairs:
        groups[score].append(label)
    negative_below = 0
    concordance = 0.0
    for score in sorted(groups):
        pos = sum(groups[score])
        neg = len(groups[score]) - pos
        concordance += pos * (negative_below + 0.5 * neg)
        negative_below += neg
    tp = fp = 0
    ap = 0.0
    for score in sorted(groups, reverse=True):
        new_tp = sum(groups[score])
        tp += new_tp
        fp += len(groups[score]) - new_tp
        if positives:
            ap += new_tp / positives * tp / (tp + fp)
    result: dict[str, Any] = {
        "n": len(pairs),
        "positive_count": positives,
        "negative_count": negatives,
        "prevalence": _ratio(positives, len(pairs)),
        "auroc": _ratio(concordance, positives * negatives),
        "auprc": ap if positives else None,
        "auprc_definition": "noninterpolated_average_precision_grouped_ties",
        "operating_points": [],
    }
    for threshold in thresholds:
        threshold = _number(threshold, "threshold")
        tp = sum(y == 1 and s >= threshold for s, y in pairs)
        fp = sum(y == 0 and s >= threshold for s, y in pairs)
        fn, tn = positives - tp, negatives - fp
        result["operating_points"].append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, positives),
                "false_stop_rate": _ratio(fp, tp + fp),
                "false_positive_rate": _ratio(fp, negatives),
                "false_stop_fraction_all_boundaries": _ratio(fp, len(pairs)),
            }
        )
    if probabilities:
        bins = [[] for _ in range(calibration_bins)]
        for score, label in pairs:
            bins[min(int(score * calibration_bins), calibration_bins - 1)].append((score, label))
        reliability = []
        ece = 0.0
        for i, bucket in enumerate(bins):
            confidence = fmean(s for s, _ in bucket) if bucket else None
            frequency = fmean(y for _, y in bucket) if bucket else None
            if bucket:
                ece += len(bucket) / len(pairs) * abs(confidence - frequency)
            reliability.append(
                {
                    "lower": i / calibration_bins,
                    "upper": (i + 1) / calibration_bins,
                    "n": len(bucket),
                    "mean_probability": confidence,
                    "safe_stop_frequency": frequency,
                }
            )
        result.update(
            brier_score=fmean((s - y) ** 2 for s, y in pairs) if pairs else None,
            expected_calibration_error=ece if pairs else None,
            reliability_bins=reliability,
        )
    return result


def entropy_hypothesis(
    rows: Iterable[Mapping[str, Any]], threshold: float, *, entropy_key: str = "attention_entropy"
) -> dict[str, Any]:
    """Test low entropy against oracle forced-answer correctness, by difficulty.

    This reports the hypothesis's observations without treating convergence as
    correctness. Missing entropy is counted and excluded from the denominator.
    """
    threshold = _number(threshold, "threshold")
    items = list(rows)

    def group_summary(group):
        counts = {"low_correct": 0, "low_wrong": 0, "high_correct": 0, "high_wrong": 0}
        missing = 0
        for row in group:
            entropy = row.get(entropy_key)
            if entropy is None:
                missing += 1
                continue
            low = _number(entropy, entropy_key) <= threshold
            correct = _label(row["safe_stop"])
            counts[f"{'low' if low else 'high'}_{'correct' if correct else 'wrong'}"] += 1
        return {
            **counts,
            "missing_entropy": missing,
            "p_wrong_given_convergence": _ratio(
                counts["low_wrong"], counts["low_correct"] + counts["low_wrong"]
            ),
        }

    groups = defaultdict(list)
    for row in items:
        groups[str(row.get("difficulty", "unspecified"))].append(row)
    return {
        "threshold": threshold,
        "entropy_key": entropy_key,
        "overall": group_summary(items),
        "by_difficulty": {key: group_summary(value) for key, value in sorted(groups.items())},
    }


_COMPARISON_KEYS = (
    "model",
    "model_revision",
    "tokenizer_revision",
    "dataset",
    "dataset_revision",
    "prompt_template",
    "reasoning_parser",
    "decoding",
    "backend",
    "dtype",
    "quantization",
)


def validate_comparable_manifests(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    allow_quantization_change: bool = False,
) -> None:
    """Reject accidental generation differences or absent comparison metadata.

    Missing or unresolved model/tokenizer revisions cannot support a comparison.
    Cross-quantization studies must opt in, and still match source model
    identity, prompt, tokenizer, benchmark, decoding, and runtime backend.
    """
    keys = tuple(
        k
        for k in _COMPARISON_KEYS
        if not allow_quantization_change or k not in {"quantization", "dtype"}
    )
    missing = [key for key in keys if key not in first or key not in second]
    if missing:
        raise ValueError(f"Missing reproducibility fields: {', '.join(missing)}")
    unresolved = [
        key
        for key in ("model_revision", "tokenizer_revision")
        if not first[key]
        or not second[key]
        or first[key] in {"main", "master", "latest"}
        or second[key] in {"main", "master", "latest"}
    ]
    if unresolved:
        raise ValueError(f"Unresolved model/tokenizer identity: {', '.join(unresolved)}")
    different = [key for key in keys if first[key] != second[key]]
    for key in (
        "chat_template_sha256",
        "use_chat_template",
        "grading",
        "attention_implementation",
        "attn_implementation",
        "gpu_model",
        "num_gpus",
        "hardware",
        "versions",
        "source_digest",
        "implementation_sha256",
        "gpu_count",
        "gpu_models",
        "dataset_sha256",
        "checkpoint_sha256",
        "sampling",
        "warmup_requests",
        "measurement_scope",
        "engine_config",
        "effective_gpu_count",
        "cpu_threads",
        "cprofile_enabled",
        "fixture_source_sha256",
        "probe_prefix_reuse",
    ):
        if key == "checkpoint_sha256" and allow_quantization_change:
            continue
        if key in first or key in second:
            if key not in first or key not in second or first[key] != second[key]:
                different.append(key)
    if different:
        raise ValueError(f"Incomparable manifests differ in: {', '.join(different)}")


def oracle_headroom(
    vanilla: Sequence[Mapping[str, Any]],
    boundaries: Sequence[Mapping[str, Any]],
    compressed: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Earliest *observed* safe boundary; never interpolate missing checkpoints.

    Candidate coverage is part of each result. Capture fraction is defined only
    for correct compressed answers with a positive measured oracle headroom.
    Values can exceed one if a policy stops between sampled oracle boundaries;
    that is reported rather than silently clamped to a purported theoretical bound.
    """
    base = _index(vanilla)
    candidate = _index(compressed)
    if candidate and candidate.keys() != base.keys():
        raise ValueError("Compressed and vanilla sample sets differ")
    grouped = defaultdict(list)
    for row in boundaries:
        key = _sample_key(row)
        if key not in base:
            raise ValueError(f"Oracle sample absent from vanilla: {key}")
        _label(row["safe_stop"])
        _number(row["reasoning_tokens"], "reasoning_tokens", nonnegative=True)
        grouped[key].append(row)
    result = []
    for key, row in base.items():
        points = grouped[key]
        seen = [r["reasoning_tokens"] for r in points]
        if len(seen) != len(set(seen)):
            raise ValueError(f"Duplicate oracle checkpoint for sample: {key}")
        correct_points = [r["reasoning_tokens"] for r in points if r["safe_stop"]]
        earliest = min(correct_points) if correct_points else None
        vanilla_tokens = _number(row["reasoning_tokens"], "reasoning_tokens", nonnegative=True)
        if any(t > vanilla_tokens for t in seen):
            raise ValueError("Oracle checkpoints must lie on the vanilla reasoning trajectory")
        headroom = vanilla_tokens - earliest if earliest is not None else None
        compressed_row = candidate.get(key)
        capture = None
        if compressed_row and _label(compressed_row["correct"]) and headroom and headroom > 0:
            capture = (vanilla_tokens - compressed_row["reasoning_tokens"]) / headroom
        result.append(
            {
                "dataset": key[0],
                "sample_id": key[1],
                "vanilla_reasoning_tokens": vanilla_tokens,
                "oracle_reasoning_tokens": earliest,
                "observed_headroom_tokens": headroom,
                "candidate_reasoning_tokens": sorted(seen),
                "candidate_count": len(points),
                "compressed_reasoning_tokens": (
                    compressed_row["reasoning_tokens"] if compressed_row else None
                ),
                "captured_observed_headroom_fraction": capture,
                "oracle_scope": "earliest_correct_among_observed_boundaries",
            }
        )
    return result
