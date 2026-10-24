"""CPU-only numerical and leakage/accounting regression checks."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from onecomp.reasoning.calibration import select_configuration, validate_selection_for_evaluation
from onecomp.reasoning.metrics import (
    analyze_signal_combinations,
    binary_metrics,
    compare_quantization_runs,
    entropy_hypothesis,
    evaluate_signal_model,
    evaluate_threshold_transfer,
    fit_logistic_signal,
    oracle_headroom,
    pair_failures,
    pareto_frontier,
    summarize,
    validate_comparable_manifests,
)
from onecomp.reasoning.metrics.report import analyze_runs, load_oracle, load_run


def manifest(**overrides):
    value = {
        "model": "test-model",
        "model_revision": "revision-a",
        "tokenizer_revision": "revision-a",
        "dataset": "unit",
        "dataset_revision": "sha-test",
        "prompt_template": "test-prompt",
        "reasoning_parser": "explicit",
        "decoding": {"temperature": 0, "seed": 2, "max_tokens": 64},
        "backend": "fixture",
        "dtype": "float32",
        "quantization": "none",
        "split": "calibration",
    }
    value.update(overrides)
    return value


def records(labels, policy="none", latency=1.0, reasoning=10, config=None):
    return [
        {
            "sample_id": str(i),
            "dataset": "unit",
            "correct": bool(label),
            "policy": policy,
            "policy_config": config or {},
            "latency_s": latency,
            "reasoning_tokens": reasoning,
            "total_generated_tokens": reasoning + 1,
            "model_evaluated_tokens": reasoning + 9,
            "model": "test-model",
            "quantization": "none",
        }
        for i, label in enumerate(labels)
    ]


def run(labels, policy="none", latency=1.0, reasoning=10, config=None):
    return {"manifest": manifest(), "records": records(labels, policy, latency, reasoning, config)}


class MetricsTest(unittest.TestCase):
    def test_pairing_and_conditional_rates(self):
        before = records([1, 1, 0, 0])
        after = list(reversed(records([1, 0, 1, 0], "fixed")))
        result = pair_failures(before, after)
        self.assertEqual(result["harm_rate"], 0.5)
        self.assertEqual(result["rescue_rate"], 0.5)
        self.assertEqual(result["wrong_wrong"], 1)

    def test_empty_conditional_denominators_are_null(self):
        self.assertIsNone(pair_failures(records([1]), records([1]))["rescue_rate"])
        self.assertIsNone(pair_failures([], [])["harm_rate"])

    def test_pair_rejects_duplicate_missing_and_dataset_mismatch(self):
        data = records([1, 0])
        with self.assertRaises(ValueError):
            pair_failures(data, data[:1])
        with self.assertRaises(ValueError):
            pair_failures(data + data[:1], data)
        bad = copy.deepcopy(data)
        bad[0]["dataset"] = "different"
        with self.assertRaises(ValueError):
            pair_failures(data, bad)
        first, second = records([1]), records([1])
        first[0]["prompt_token_ids"] = [1, 2]
        second[0]["prompt_token_ids"] = [1, 3]
        with self.assertRaises(ValueError):
            pair_failures(first, second)

    def test_summaries_never_fabricate_missing_probe_or_latency(self):
        data = records([1, 0])
        del data[0]["latency_s"]
        result = summarize(data)
        self.assertIsNone(result["latency_s"])
        self.assertIsNone(result["probe_tokens"])
        self.assertIsNone(result["probe_model_evaluated_tokens"])
        self.assertEqual(result["measurement_counts"]["latency_s"], 1)
        self.assertEqual(result["accuracy"], 0.5)

    def test_summaries_reject_nonfinite_costs_and_mixed_configs(self):
        for value in (float("nan"), float("inf"), -0.1):
            with self.assertRaises(ValueError):
                summarize(records([1], latency=value))
        data = records([1, 0])
        data[1]["policy_config"] = {"budget": 2}
        with self.assertRaises(ValueError):
            summarize(data)

    def test_pareto_retains_exact_ties_and_rejects_dominated(self):
        points = [
            {"id": "a", "accuracy": 0.8, "latency_s": 1},
            {"id": "b", "accuracy": 0.7, "latency_s": 1},
            {"id": "c", "accuracy": 0.8, "latency_s": 2},
            {"id": "d", "accuracy": 0.9, "latency_s": 3},
            {"id": "e", "accuracy": 0.8, "latency_s": 1},
            {"id": "f", "accuracy": 1, "latency_s": None},
        ]
        self.assertEqual([p["id"] for p in pareto_frontier(points)], ["a", "e", "d"])

    def test_auc_and_ap_group_ties(self):
        tied = binary_metrics([1, 0, 1, 0], [0.5] * 4)
        self.assertEqual(tied["auroc"], 0.5)
        self.assertEqual(tied["auprc"], 0.5)
        perfect = binary_metrics([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9])
        self.assertEqual(perfect["auroc"], 1)
        self.assertEqual(perfect["auprc"], 1)
        reverse = binary_metrics([0, 1], [0.8, 0.2])
        self.assertEqual(reverse["auroc"], 0)
        self.assertEqual(reverse["auprc"], 0.5)

    def test_undefined_auc_ap_and_probability_validation(self):
        self.assertIsNone(binary_metrics([1, 1], [0.2, 0.3])["auroc"])
        self.assertIsNone(binary_metrics([0, 0], [0.2, 0.3])["auprc"])
        self.assertIsNone(binary_metrics([], [], probabilities=True)["brier_score"])
        for labels, scores in (([1], []), ([2], [0.1]), ([1], [math.nan])):
            with self.assertRaises(ValueError):
                binary_metrics(labels, scores)
        with self.assertRaises(ValueError):
            binary_metrics([1], [1.1], probabilities=True)

    def test_false_stop_definitions_and_reliability(self):
        result = binary_metrics(
            [1, 0, 0, 0], [1.0, 1.0, 0.0, 0.0], thresholds=[0.9], probabilities=True
        )
        point = result["operating_points"][0]
        self.assertEqual(point["false_stop_rate"], 0.5)
        self.assertAlmostEqual(point["false_positive_rate"], 1 / 3)
        self.assertEqual(result["brier_score"], 0.25)
        self.assertEqual(result["expected_calibration_error"], 0.25)

    def test_entropy_can_converge_to_wrong_answer(self):
        rows = [
            {"safe_stop": 1, "attention_entropy": 0.1, "difficulty": "easy"},
            {"safe_stop": 0, "attention_entropy": 0.2, "difficulty": "hard"},
            {"safe_stop": 1, "attention_entropy": 0.8, "difficulty": "hard"},
            {"safe_stop": 0, "attention_entropy": None},
        ]
        result = entropy_hypothesis(rows, 0.5)
        self.assertEqual(result["overall"]["p_wrong_given_convergence"], 0.5)
        self.assertEqual(result["by_difficulty"]["hard"]["p_wrong_given_convergence"], 1)
        self.assertEqual(result["overall"]["missing_entropy"], 1)

    def test_manifest_mismatches_missing_metadata_and_transfer(self):
        validate_comparable_manifests(manifest(), manifest())
        with self.assertRaises(ValueError):
            validate_comparable_manifests(manifest(), manifest(decoding={"temperature": 0.7}))
        with self.assertRaises(ValueError):
            validate_comparable_manifests(manifest(), {})
        with self.assertRaises(ValueError):
            validate_comparable_manifests(
                manifest(model_revision=None), manifest(model_revision=None)
            )
        with self.assertRaises(ValueError):
            validate_comparable_manifests(
                manifest(chat_template_sha256="a"), manifest(chat_template_sha256="b")
            )
        quant = manifest(quantization="mixed_gptq", dtype="float16")
        with self.assertRaises(ValueError):
            validate_comparable_manifests(manifest(), quant)
        validate_comparable_manifests(manifest(), quant, allow_quantization_change=True)
        for field in (
            "implementation_sha256",
            "gpu_models",
            "gpu_count",
            "dataset_sha256",
            "checkpoint_sha256",
        ):
            with self.assertRaises(ValueError):
                validate_comparable_manifests(
                    manifest(**{field: "one"}), manifest(**{field: "two"})
                )
        validate_comparable_manifests(
            manifest(checkpoint_sha256="one"),
            manifest(checkpoint_sha256="two", quantization="mixed_gptq"),
            allow_quantization_change=True,
        )

    def test_oracle_uses_observed_points_and_correct_policy_only(self):
        base = records([1], reasoning=10)
        boundaries = [
            {"sample_id": "0", "dataset": "unit", "reasoning_tokens": n, "safe_stop": safe}
            for n, safe in ((2, 0), (4, 1), (8, 1))
        ]
        result = oracle_headroom(base, boundaries, records([1], "fixed", reasoning=7))[0]
        self.assertEqual(result["oracle_reasoning_tokens"], 4)
        self.assertEqual(result["captured_observed_headroom_fraction"], 0.5)
        self.assertIsNone(
            oracle_headroom(base, boundaries, records([0], "fixed", reasoning=1))[0][
                "captured_observed_headroom_fraction"
            ]
        )
        with self.assertRaises(ValueError):
            oracle_headroom(base, boundaries + boundaries[:1])


class CalibrationTest(unittest.TestCase):
    def test_selects_latency_not_shortest_visible_reasoning(self):
        runs = [
            run([1, 1, 0, 1]),
            run([1, 1, 0, 1], "probe", latency=1.2, reasoning=1),
            run([1, 1, 0, 1], "fixed", latency=0.6, reasoning=6),
        ]
        selected = select_configuration(runs, 0)
        self.assertEqual(selected["selected_policy"], "fixed")
        self.assertAlmostEqual(selected["expected_latency_reduction"], 0.4)
        self.assertEqual(selected["expected_accuracy_delta"], 0)

    def test_quality_constraint_rejects_fast_harmful_policy(self):
        selected = select_configuration([run([1, 1]), run([1, 0], "fixed", latency=0.01)], 0.005)
        self.assertEqual(selected["selected_policy"], "none")

    def test_rejects_test_tuning_missing_pairs_and_duplicate_variants(self):
        base = run([1, 0])
        test = run([1, 0], "fixed")
        test["manifest"]["split"] = "test"
        with self.assertRaises(ValueError):
            select_configuration([base, test], 0)
        with self.assertRaises(ValueError):
            select_configuration([base, run([1], "fixed")], 0)
        with self.assertRaises(ValueError):
            select_configuration([base, base], 0)
        with self.assertRaises(ValueError):
            select_configuration([base], 0, cost_key="reasoning_tokens")

    def test_rejects_missing_baseline_latency_and_excludes_unknown_candidate(self):
        base = run([1])
        candidate = run([1], "fixed")
        del candidate["records"][0]["latency_s"]
        self.assertEqual(select_configuration([base, candidate], 0)["selected_policy"], "none")
        del base["records"][0]["latency_s"]
        with self.assertRaises(ValueError):
            select_configuration([base], 0)

    def test_selected_evaluation_requires_disjoint_samples(self):
        selected = select_configuration([run([1])], 0)
        heldout = records([1])
        with self.assertRaises(ValueError):
            validate_selection_for_evaluation(selected, manifest(split="test"), heldout)
        heldout[0]["sample_id"] = "heldout"
        validate_selection_for_evaluation(selected, manifest(split="test"), heldout)

    def test_quantization_comparison_and_heldout_threshold_transfer(self):
        def quantized(run_data, split="calibration"):
            result = copy.deepcopy(run_data)
            result["manifest"].update(quantization="mixed_gptq_4", split=split)
            for row in result["records"]:
                row["quantization"] = "mixed_gptq_4"
                if split == "test":
                    row["sample_id"] = "heldout-" + row["sample_id"]
            return result

        original = [
            run([1, 1]),
            run([1, 1], "fixed", 0.5, config={"budget": 2}),
            run([1, 1], "fixed", 0.8, config={"budget": 4}),
        ]
        compressed = [
            quantized(run([1, 1])),
            quantized(run([1, 0], "fixed", 0.4, config={"budget": 2})),
            quantized(run([1, 1], "fixed", 0.7, config={"budget": 4})),
        ]
        comparison = compare_quantization_runs(original, compressed)
        self.assertEqual(comparison["comparisons"][1]["accuracy_delta"], -0.5)
        evaluated = [quantized(data, split="test") for data in compressed]
        transfer = evaluate_threshold_transfer(original, compressed, evaluated, 0)
        self.assertEqual(transfer["heldout_accuracy_gain_from_recalibration"], 0.5)
        self.assertEqual(transfer["bf16_calibration_selection"]["policy_config"], {"budget": 2})
        self.assertEqual(
            transfer["quantized_calibration_selection"]["policy_config"], {"budget": 4}
        )
        with self.assertRaises(ValueError):
            evaluate_threshold_transfer(original, compressed, compressed, 0)


class SignalCombinationTest(unittest.TestCase):
    def training(self):
        return [
            {
                "dataset": "unit",
                "sample_id": str(i),
                "split": "calibration",
                "safe_stop": i % 2,
                "confidence": 0.9 if i % 2 else 0.1,
                "entropy": 0.1 if i % 2 else 0.9,
            }
            for i in range(10)
        ]

    def test_combination_fits_train_normalization_and_evaluates_holdout(self):
        train = self.training()
        model = fit_logistic_signal(train, ["confidence", "entropy"])
        heldout = [
            dict(row, sample_id="heldout-" + row["sample_id"], split="test") for row in train
        ]
        result = evaluate_signal_model(model, heldout)
        self.assertEqual(result["auroc"], 1)
        self.assertLess(result["brier_score"], 0.01)
        self.assertEqual(model["means"], [0.5, 0.5])

    def test_rejects_boundary_leakage_and_implicit_training_split(self):
        train = self.training()
        model = fit_logistic_signal(train, ["confidence"])
        with self.assertRaises(ValueError):
            evaluate_signal_model(model, [dict(train[0], split="test", reasoning_tokens=999)])
        with self.assertRaises(ValueError):
            fit_logistic_signal([dict(row, split="test") for row in train], ["confidence"])

    def test_combinations_use_common_complete_cases(self):
        train = [
            dict(
                row,
                attention_entropy=row["entropy"],
                reasoning_tokens=100 + i,
                logit_margin=1 - row["confidence"],
            )
            for i, row in enumerate(self.training())
        ]
        evaluation = [
            dict(row, sample_id="heldout-" + row["sample_id"], split="test") for row in train
        ]
        evaluation.append(dict(evaluation[0], sample_id="missing", attention_entropy=None))
        result = analyze_signal_combinations(train, evaluation)
        self.assertEqual(len(result["comparisons"]), 5)
        self.assertEqual(result["dropped_evaluation_rows"], 1)
        self.assertEqual(
            {entry["heldout_metrics"]["n"] for entry in result["comparisons"].values()}, {10}
        )

    def test_report_exports_actual_measured_points(self):
        runs = [run([1, 0]), run([1, 0], "fixed", latency=0.5)]
        with tempfile.TemporaryDirectory() as directory:
            output = analyze_runs(runs, directory)
            path = Path(directory)
            self.assertEqual(len(output["latency_pareto_frontier"]), 1)
            self.assertTrue((path / "pareto.svg").is_file())
            self.assertEqual(
                json.loads((path / "summary.json").read_text())["summaries"][1]["latency_s"], 0.5
            )
            (path / "manifest.json").write_text(json.dumps(runs[0]["manifest"]))
            (path / "results.jsonl").write_text(
                "\n".join(json.dumps(row) for row in runs[0]["records"])
            )
            self.assertEqual(load_run(path), runs[0])
            (path / "status.json").write_text(json.dumps({"status": "failed"}))
            with self.assertRaises(ValueError):
                load_run(path)

    def test_native_signal_schema_and_oracle_manifest_checks(self):
        runs = [run([1], reasoning=2), run([1], "fixed", latency=0.5, reasoning=1)]
        oracle = [
            {
                "dataset": "unit",
                "sample_id": "0",
                "reasoning_tokens": 1,
                "safe_stop": 1,
                "logit_margin": 0.1,
                "confidence": 0.9,
                "relative_entropy": -0.2,
                "attention_entropy": 0.1,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            report = analyze_runs(runs, path, oracle_records=oracle, entropy_threshold=0.2)
            self.assertIn("confidence", report["safe_stop_signal_metrics"])
            self.assertEqual(
                report["safe_stop_signal_metrics"]["logit_margin"]["score_direction"], -1
            )
            (path / "oracle.jsonl").write_text(json.dumps(oracle[0]) + "\n")
            (path / "manifest.json").write_text(json.dumps(manifest()))
            self.assertEqual(load_oracle(path / "oracle.jsonl", manifest()), oracle)
            with self.assertRaises(ValueError):
                load_oracle(path / "oracle.jsonl", manifest(model="other"))


if __name__ == "__main__":
    unittest.main()
