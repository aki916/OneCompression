"""CPU-only artifact/CLI integration with an explicitly synthetic adapter."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from onecomp.reasoning import ReasoningConfig, ReasoningFormat, RequestState
from onecomp.reasoning.cli import main
from onecomp.reasoning.data import read_jsonl, write_json, write_jsonl
from onecomp.reasoning.metrics.report import load_oracle, load_run
from onecomp.runtime.transformers.adapter import GenerationResult


class FixtureTokenizer:
    init_kwargs = {"_commit_hash": "fixture-tokenizer-revision"}
    chat_template = "fixture chat template"

    def encode(self, text, **kwargs):
        return {"<think>": [10], "</think>": [20]}.get(text, [30, 31])

    def decode(self, ids, **kwargs):
        return "".join(
            {
                1: "one ",
                2: "two. ",
                3: "three ",
                4: "four. ",
                20: "</think>",
                5: r"\boxed{42}",
            }.get(i, "prompt ")
            for i in ids
        )

    def apply_chat_template(self, messages, **kwargs):
        return [30, 31]


class FixtureAdapter:
    """A fixture for harness plumbing, never a benchmark-quality measurement."""

    def __init__(self, fail_request=None):
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                _commit_hash="fixture-model-revision", _attn_implementation="eager"
            )
        )
        self.tokenizer = FixtureTokenizer()
        self.calls = []
        self.fail_request = fail_request

    def generate(
        self,
        prompt_ids,
        *,
        config=None,
        reasoning_format=None,
        request_id="warmup",
        max_new_tokens=64,
        initial_phase=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "prompt_ids": list(prompt_ids),
                "config": config,
                "request_id": request_id,
                "initial_phase": initial_phase,
            }
        )
        if request_id == self.fail_request:
            raise RuntimeError("Fixture injected inference failure")
        config = config or ReasoningConfig()
        state = RequestState(request_id, reasoning_format or ReasoningFormat((10,), (20,)))
        if initial_phase == "answer":
            state.phase = "answer"
            tokens = [5]
        else:
            budget = (
                config.budget
                if config.policy == "fixed"
                else 2 if config.policy == "thinkbrake" else 4
            )
            tokens = [1, 2, 3, 4][:budget] + [20, 5]
        tokens = tokens[:max_new_tokens]
        for token in tokens:
            state.append(token, self.tokenizer.decode([token]))
        answer = r"\boxed{42}" if 5 in tokens else ""
        state.final_answer = answer
        return GenerationResult(
            state=state,
            text=self.tokenizer.decode(tokens),
            final_answer=answer,
            metrics={
                "latency_s": 0.01 * (len(tokens) + 1),
                "probe_latency_s": 0.0,
                "normal_model_evaluated_tokens": len(prompt_ids) + max(0, len(tokens) - 1),
                "probe_model_evaluated_tokens": 0,
                "model_evaluated_tokens": len(prompt_ids) + max(0, len(tokens) - 1),
                "probe_tokens": 0,
            },
        )

    def _sync(self):
        pass

    def _forward(self, ids):
        logits = torch.zeros((1, 1, 32))
        logits[0, 0, 20] = 0.1
        return SimpleNamespace(logits=logits)

    def probe(self, prefix, config, **kwargs):
        return SimpleNamespace(
            text=r"\boxed{42}",
            confidence=0.99,
            complete=True,
            model_evaluated_tokens=len(prefix) + 1,
            latency_s=0.01,
            signals={
                "confidence": 0.99,
                "answer": "42",
                "probe_complete": True,
                "attention_entropy": 1.0,
            },
        )


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.dataset = self.path / "samples.jsonl"
        write_jsonl(
            self.dataset,
            [
                dict(
                    sample_id=f"{split}-{i}",
                    dataset="fixture",
                    dataset_revision="fixture-dataset-revision",
                    question="What is 6 * 7?",
                    answer="42",
                    split=split,
                )
                for split in ("calibration", "evaluation")
                for i in range(2)
            ],
        )
        self.config = {
            "model": "fixture/model",
            "model_revision": "fixture-model-revision",
            "tokenizer_revision": "fixture-tokenizer-revision",
            "dataset_path": str(self.dataset),
            "split": "calibration",
            "device": "cpu",
            "dtype": "float32",
            "warmup_requests": 0,
            "max_new_tokens": 32,
            "policies": [
                {"policy": "none"},
                {"policy": "fixed", "budget": 2},
                {"policy": "thinkbrake"},
            ],
            "oracle_policy": {"policy": "deer", "boundary": "tokens", "interval": 2},
            "oracle_max_boundaries": 4,
        }
        self.config_path = self.path / "config.json"
        write_json(self.config_path, self.config)
        self.adapter = FixtureAdapter()
        self.loader = patch(
            "onecomp.reasoning.experiment.load_reference",
            return_value=(self.adapter, self.adapter.tokenizer),
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.metadata = patch(
            "onecomp.reasoning.experiment.environment_metadata",
            return_value={
                "git_commit": "fixture",
                "implementation_sha256": "fixture-source",
                "versions": {"fixture": "1"},
                "gpu_count": 0,
                "gpu_models": [],
                "measurement_kind": "synthetic_test_fixture",
            },
        )
        self.metadata.start()
        self.addCleanup(self.metadata.stop)

    def run_baselines(self):
        output = self.path / "runs"
        main(["run", "--config", str(self.config_path), "--output", str(output)])
        directories = sorted(
            path for path in output.iterdir() if (path / "results.jsonl").is_file()
        )
        return output, directories

    def test_run_cli_pairs_samples_and_writes_reproducibility_and_analysis(self):
        output, directories = self.run_baselines()
        self.assertEqual(len(directories), 3)
        for directory in directories:
            run = load_run(directory)
            self.assertEqual(run["manifest"]["measurement_kind"], "synthetic_test_fixture")
            self.assertEqual(run["manifest"]["model_revision"], "fixture-model-revision")
            self.assertEqual(run["manifest"]["tokenizer_revision"], "fixture-tokenizer-revision")
            self.assertEqual(len(run["records"]), 2)
            self.assertEqual(
                {row["sample_id"] for row in run["records"]}, {"calibration-0", "calibration-1"}
            )
            self.assertTrue(all(row["prompt_token_ids"] == [30, 31] for row in run["records"]))
            if run["manifest"]["policy"] != "none":
                self.assertTrue(all(row["vanilla_correct"] for row in run["records"]))
        self.assertTrue((output / "analysis" / "pareto.svg").is_file())
        with self.assertRaises(FileExistsError):
            main(["run", "--config", str(self.config_path), "--output", str(output)])

    def test_calibrate_then_auto_cli_requires_disjoint_evaluation(self):
        _, directories = self.run_baselines()
        selection = self.path / "selected.json"
        main(
            [
                "calibrate",
                *map(str, directories),
                "--max-accuracy-drop",
                "0.5",
                "--output",
                str(selection),
            ]
        )
        selected = json.loads(selection.read_text())
        self.assertEqual(selected["max_accuracy_drop_fraction"], 0.005)
        self.assertIn(selected["selected_policy"], {"fixed", "thinkbrake"})
        evaluation = dict(self.config, split="evaluation")
        eval_config = self.path / "evaluation.json"
        write_json(eval_config, evaluation)
        main(
            [
                "run",
                "--config",
                str(eval_config),
                "--output",
                str(self.path / "evaluation"),
                "--reasoning-compression",
                "auto",
                "--selected-config",
                str(selection),
            ]
        )
        with self.assertRaises(ValueError):
            main(
                [
                    "run",
                    "--config",
                    str(self.config_path),
                    "--output",
                    str(self.path / "leak"),
                    "--reasoning-compression",
                    "auto",
                    "--selected-config",
                    str(selection),
                ]
            )

    def test_oracle_cli_forces_answer_phase_and_accounts_offline_forwards(self):
        output = self.path / "oracle"
        main(["oracle", "--config", str(self.config_path), "--output", str(output)])
        rows = read_jsonl(output / "results.jsonl")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertTrue(manifest["offline_oracle"])
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["reasoning_tokens"] for row in rows}, {2, 4})
        self.assertTrue(all(row["safe_stop"] == 1 for row in rows))
        self.assertTrue(all(row["probe_model_evaluated_tokens"] > 0 for row in rows))
        self.assertTrue(all(row["signal_replay_model_evaluated_tokens"] >= 4 for row in rows))
        self.assertEqual(load_oracle(output / "results.jsonl", manifest), rows)
        forced = [call for call in self.adapter.calls if call["initial_phase"] == "answer"]
        self.assertEqual(len(forced), 4)
        self.assertTrue(all(call["prompt_ids"][-1] == 20 for call in forced))
        self.assertTrue(
            all("safe_stop" not in call["config"].to_dict() for call in self.adapter.calls)
        )

    def test_failed_inference_keeps_incomplete_status_and_is_not_analyzable(self):
        self.adapter.fail_request = "calibration-1"
        output = self.path / "failure"
        with self.assertRaisesRegex(RuntimeError, "Fixture"):
            main(["run", "--config", str(self.config_path), "--output", str(output)])
        directories = list(output.iterdir())
        self.assertEqual(len(directories), 1)
        self.assertEqual(len(read_jsonl(directories[0] / "results.jsonl")), 1)
        self.assertEqual(
            json.loads((directories[0] / "status.json").read_text())["status"], "incomplete"
        )
        with self.assertRaises(ValueError):
            load_run(directories[0])


if __name__ == "__main__":
    unittest.main()
