"""Backend-independent, measured-cost analysis for reasoning compression."""

from .analysis import (
    binary_metrics,
    entropy_hypothesis,
    oracle_headroom,
    pair_failures,
    pareto_frontier,
    summarize,
    validate_comparable_manifests,
)
from .quantization import compare_quantization_runs, evaluate_threshold_transfer
from .signals import analyze_signal_combinations, evaluate_signal_model, fit_logistic_signal

__all__ = [
    "analyze_signal_combinations",
    "binary_metrics",
    "compare_quantization_runs",
    "entropy_hypothesis",
    "evaluate_signal_model",
    "evaluate_threshold_transfer",
    "fit_logistic_signal",
    "oracle_headroom",
    "pair_failures",
    "pareto_frontier",
    "summarize",
    "validate_comparable_manifests",
]
