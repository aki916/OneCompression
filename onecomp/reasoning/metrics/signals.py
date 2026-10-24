"""Small offline signal combinations, with sample-level holdout enforcement.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from .analysis import _label, _number, _sample_key, binary_metrics


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def fit_logistic_signal(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    *,
    iterations: int = 500,
    learning_rate: float = 0.1,
    l2: float = 0.01,
) -> dict[str, Any]:
    """Fit a deterministic logistic baseline on explicitly labeled train splits.

    This is offline *analysis* of stopping signals, not a training-based reasoning
    method. Every row must have split=development/calibration. Complete cases are
    used; coverage is recorded. Standardization sees training rows only. Supply
    e.g. confidence+attention, token_count+confidence or margin+confidence to
    estimate incremental predictive information on disjoint evaluation samples.
    """
    if not features or len(set(features)) != len(features):
        raise ValueError("features must be nonempty and unique")
    if not rows:
        raise ValueError("Training rows must not be empty")
    if any(row.get("split") not in {"development", "calibration"} for row in rows):
        raise ValueError("Signal fitting requires explicit development/calibration split rows")
    if iterations < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("Invalid logistic optimization parameters")
    complete = [row for row in rows if all(row.get(feature) is not None for feature in features)]
    if not complete:
        raise ValueError("No complete training feature vectors")
    labels = [_label(row["safe_stop"]) for row in complete]
    if len(set(labels)) < 2:
        raise ValueError("Signal fitting requires both safe and unsafe boundaries")
    raw = [[_number(row[name], name) for name in features] for row in complete]
    means = [fmean(row[j] for row in raw) for j in range(len(features))]
    scales = [
        math.sqrt(fmean((row[j] - means[j]) ** 2 for row in raw)) or 1.0
        for j in range(len(features))
    ]
    vectors = [
        [1.0] + [(value - means[j]) / scales[j] for j, value in enumerate(row)] for row in raw
    ]
    weights = [0.0] * (len(features) + 1)
    for _ in range(iterations):
        gradient = [0.0] * len(weights)
        for vector, label in zip(vectors, labels):
            error = _sigmoid(sum(a * b for a, b in zip(weights, vector))) - label
            for j, value in enumerate(vector):
                gradient[j] += error * value / len(vectors)
        for j in range(len(weights)):
            weights[j] -= learning_rate * (gradient[j] + (l2 * weights[j] if j else 0.0))
    return {
        "kind": "standardized_logistic_signal_v1",
        "features": list(features),
        "means": means,
        "scales": scales,
        "weights": weights,
        "training_sample_ids": [list(key) for key in sorted({_sample_key(row) for row in rows})],
        "training_splits": sorted({row["split"] for row in rows}),
        "training_rows": len(complete),
        "dropped_training_rows": len(rows) - len(complete),
        "optimization": {"iterations": iterations, "learning_rate": learning_rate, "l2": l2},
    }


def evaluate_signal_model(
    model: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] = (0.5, 0.9, 0.95),
) -> dict[str, Any]:
    """Evaluate only on samples absent from all fitted boundary rows.

    Splitting individual boundaries from the same question across train/test is
    leakage and is rejected even if the boundary token positions differ.
    """
    if model.get("kind") != "standardized_logistic_signal_v1":
        raise ValueError("Unsupported fitted signal model")
    training_ids = {tuple(key) for key in model["training_sample_ids"]}
    if any(_sample_key(row) in training_ids for row in rows):
        raise ValueError(
            "Evaluation overlaps training sample IDs; split by question, not boundary"
        )
    if any(row.get("split") not in {"validation", "evaluation", "final", "test"} for row in rows):
        raise ValueError("Signal evaluation requires an explicit held-out split")
    features = model["features"]
    complete = [row for row in rows if all(row.get(feature) is not None for feature in features)]
    scores = []
    for row in complete:
        vector = [1.0] + [
            (_number(row[name], name) - model["means"][j]) / model["scales"][j]
            for j, name in enumerate(features)
        ]
        scores.append(_sigmoid(sum(a * b for a, b in zip(model["weights"], vector))))
    result = binary_metrics(
        [row["safe_stop"] for row in complete], scores, thresholds=thresholds, probabilities=True
    )
    result.update(features=list(features), dropped_evaluation_rows=len(rows) - len(complete))
    return result


def analyze_signal_combinations(
    calibration_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    feature_sets: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Compare incremental signal information on one common held-out subset.

    Default combinations implement confidence, attention, confidence+attention,
    token count+confidence, and logit margin+confidence diagnostics. Requiring
    the same complete-case rows prevents availability differences from looking
    like predictive gains. Missing attention may leave no comparable examples;
    callers should then supply a smaller explicit feature_sets mapping.
    """
    feature_sets = feature_sets or {
        "confidence_only": ("confidence",),
        "attention_only": ("attention_entropy",),
        "confidence_attention": ("confidence", "attention_entropy"),
        "token_count_confidence": ("reasoning_tokens", "confidence"),
        "logit_margin_confidence": ("logit_margin", "confidence"),
    }
    if not calibration_rows or not evaluation_rows:
        raise ValueError("Signal combinations require calibration and held-out boundary rows")
    training_ids = {_sample_key(row) for row in calibration_rows}
    if any(_sample_key(row) in training_ids for row in evaluation_rows):
        raise ValueError("Signal comparison splits overlap at the question level")
    if any(row.get("split") not in {"development", "calibration"} for row in calibration_rows):
        raise ValueError("Signal fitting requires explicit development/calibration split rows")
    if any(
        row.get("split") not in {"validation", "evaluation", "final", "test"}
        for row in evaluation_rows
    ):
        raise ValueError("Signal comparison requires explicit held-out evaluation rows")
    features = set().union(*feature_sets.values())
    train = [row for row in calibration_rows if all(row.get(key) is not None for key in features)]
    evaluation = [
        row for row in evaluation_rows if all(row.get(key) is not None for key in features)
    ]
    if not train or not evaluation:
        raise ValueError("No common complete-case rows for requested signal combinations")
    comparisons = {}
    for name, names in feature_sets.items():
        model = fit_logistic_signal(train, names)
        comparisons[name] = {
            "model": model,
            "heldout_metrics": evaluate_signal_model(model, evaluation),
        }
    return {
        "comparisons": comparisons,
        "common_training_rows": len(train),
        "common_evaluation_rows": len(evaluation),
        "dropped_training_rows": len(calibration_rows) - len(train),
        "dropped_evaluation_rows": len(evaluation_rows) - len(evaluation),
        "scope": "diagnostic_logistic_signal_combinations_not_causal_or_runtime_speedup_evidence",
    }
