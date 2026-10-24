# Copyright 2025-2026 Fujitsu Ltd.

"""Measured CPU mechanics experiment; NOT a language-model quality benchmark.

Every evaluation executes a real, random tiny Qwen3 backbone. A scripted output
head supplies a finite grammar and synthetic correct/wrong answers so that probes,
forced closing, attention reduction and oracle labels can all be exercised. The
synthetic accuracy and Pareto plots must never be presented as Qwen3-8B results.

Run from the repository root:
    .venv/bin/python experiments/reasoning_compression/smoke.py
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import pstats
import subprocess
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from onecomp.reasoning import ReasoningConfig, ReasoningFormat
from onecomp.reasoning.experiment import environment_metadata
from onecomp.reasoning.metrics import entropy_hypothesis
from onecomp.reasoning.metrics.report import analyze_runs
from onecomp.reasoning.signals import relative_entropy_change
from onecomp.runtime.transformers import TransformersReasoningAdapter


class FixtureTokenizer:
    """A declared toy vocabulary; no downloaded tokenizer or model weights."""

    eos_token_id = 3
    vocabulary = {
        0: "<pad>",
        1: "<think>",
        2: "</think>",
        3: "<eos>",
        4: "step ",
        5: ".\n",
        6: "Wait",
        7: "{",
        8: "}",
        9: "0",
        10: "1",
        11: "2",
        12: "3",
        13: "\\boxed",
        14: "\nFinal Answer\n",
        15: "Wait, reconsider.\n",
    }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in ("<think>", "</think>"):
            return [1 if text == "<think>" else 2]
        if "boxed" in text:
            return [14, 13]
        if text.startswith("Wait,"):
            return [15]
        raise ValueError("Fixture tokenizer only accepts declared control prompts")

    def decode(self, ids, **kwargs):
        del kwargs
        return "".join(self.vocabulary.get(int(token), f"Q{int(token) - 20}") for token in ids)


class ScriptedHeadQwen3(torch.nn.Module):
    """Real forward/attention work with explicitly artificial output decisions."""

    def __init__(self):
        super().__init__()
        self.config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=512,
            eos_token_id=3,
            pad_token_id=0,
        )
        self.config._attn_implementation = "eager"
        self.backbone = Qwen3ForCausalLM(self.config)
        self.generation_config = self.backbone.generation_config

    @staticmethod
    def answer_token(sample, reasoning_count):
        target = sample % 4
        sufficient = 6 + 2 * (sample % 3)
        correct = reasoning_count >= sufficient
        # Deliberately create both premature-stop harm and overthinking rescue.
        if sample % 4 == 0 and reasoning_count > 10:
            correct = False
        return 9 + (target if correct else (target + 1) % 4)

    def forward(self, input_ids, past_key_values=None, **kwargs):
        prefix = list(getattr(past_key_values, "_fixture_prefix", ()))
        context = prefix + input_ids[0].tolist()
        output = self.backbone(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        output.past_key_values._fixture_prefix = context
        sample = context[0] - 20
        cut = min(
            [context.index(token) for token in (2, 14) if token in context] or [len(context)]
        )
        reasoning_count = sum(token in (4, 5, 6, 15) for token in context[2:cut])
        confidence = 0.99 if reasoning_count >= 6 + 2 * (sample % 3) or sample % 5 == 0 else 0.55
        answer = self.answer_token(sample, reasoning_count)
        if 14 in context:
            generated = len(context) - context.index(14) - 2
            next_token = [7, answer, 8, 3][min(generated, 3)]
        elif 2 in context:
            generated = len(context) - context.index(2) - 1
            next_token = [13, 7, answer, 8, 3][min(generated, 4)]
        else:
            next_token = 2 if reasoning_count >= 12 else (4 if reasoning_count % 2 == 0 else 5)
        logits = torch.full_like(output.logits, math.log((1 - confidence) / 63))
        logits[:, -1, next_token] = math.log(confidence)
        if 2 not in context and 14 not in context and next_token != 2:
            margin = 0.08 if reasoning_count >= 6 + 2 * (sample % 3) else 3.0
            logits[:, -1, 2] = logits[:, -1, next_token] - margin
        output.logits = logits
        return output


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
    )


def execute(destination, examples, *, profiled=False):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = ScriptedHeadQwen3()
    tokenizer = FixtureTokenizer()
    adapter = TransformersReasoningAdapter(model, tokenizer)
    reasoning_format = ReasoningFormat((1,), (2,), True, "synthetic_single_token_v1")
    configurations = {
        "none": ReasoningConfig(policy="none", boundary="tokens", interval=2),
        **{
            f"fixed_{budget}": ReasoningConfig(policy="fixed", budget=budget)
            for budget in (4, 8, 12)
        },
        "prompt_budget": ReasoningConfig(policy="prompt_budget", budget=512),
        **{
            f"thinkbrake_{tau}": ReasoningConfig(
                policy="thinkbrake", threshold=tau, boundary="tokens", interval=2
            )
            for tau in (0.1, 1.0)
        },
        **{
            f"deer_{threshold}": ReasoningConfig(
                policy="deer",
                confidence_threshold=threshold,
                boundary="tokens",
                interval=2,
                probe_max_tokens=8,
            )
            for threshold in (0.95, 0.999)
        },
        "deer_0.999_interval4": ReasoningConfig(
            policy="deer",
            confidence_threshold=0.999,
            boundary="tokens",
            interval=4,
            probe_max_tokens=8,
        ),
        "stability": ReasoningConfig(
            policy="stability", boundary="tokens", interval=2, probe_max_tokens=8
        ),
        "asag_reference": ReasoningConfig(
            policy="asag_reference", boundary="tokens", interval=2, probe_max_tokens=8
        ),
        "progressive": ReasoningConfig(
            policy="progressive",
            boundary="tokens",
            interval=2,
            probe_max_tokens=8,
            cascade_uncertainty=4.0,
        ),
    }
    manifest = {
        **environment_metadata(),
        "schema_version": 1,
        "experiment_kind": "synthetic_cpu_mechanics_only",
        "warning": "Scripted output logits; accuracy is a fixture check, not reasoning-model quality. CPU latency includes real random tiny Qwen3 backbone forwards.",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "onecomp_version": importlib.metadata.version("onecomp"),
        "transformers_version": importlib.metadata.version("transformers"),
        "vllm_version": importlib.metadata.version("vllm"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "model": "synthetic/random-tiny-qwen3-scripted-head",
        "model_revision": "fixture-v1-seed-7",
        "tokenizer_revision": "fixture-v1",
        "model_config": model.config.to_dict(),
        "dataset": "synthetic-mechanics-v1",
        "dataset_revision": "fixture-v1",
        "dataset_hash": hashlib.sha256(json.dumps(list(range(examples))).encode()).hexdigest(),
        "split": "smoke",
        "sample_ids": [str(i) for i in range(examples)],
        "prompt_template": "[sample_token, thinking_start]",
        "reasoning_parser": "synthetic_single_token_v1",
        "decoding": {"temperature": 0, "max_new_tokens": 32, "seed": 7},
        "backend": "transformers_reference",
        "dtype": "float32",
        "quantization": "none",
        "gpu_model": None,
        "num_gpus": 0,
        "cpu_threads": 1,
        "cprofile_enabled": profiled,
        "fixture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    # Warm-up is excluded; no tokenizer/model load time enters request latency.
    adapter.generate([20, 1], reasoning_format=reasoning_format, max_new_tokens=32)
    runs = []
    vanilla_outputs = {}
    oracle_traces = {}
    identity = []
    for name, config in configurations.items():
        records = []
        for sample in range(examples):
            result = adapter.generate(
                [20 + sample, 1],
                reasoning_format=reasoning_format,
                config=config,
                request_id=str(sample),
                max_new_tokens=32,
                collect_boundaries=False,
            )
            correct = f"\\boxed{{{sample % 4}}}" in result.final_answer
            if name == "none":
                vanilla_outputs[sample] = result
                disabled = adapter.generate(
                    [20 + sample, 1], reasoning_format=reasoning_format, max_new_tokens=32
                )
                oracle_traces[sample] = adapter.generate(
                    [20 + sample, 1],
                    reasoning_format=reasoning_format,
                    config=config,
                    request_id=str(sample),
                    max_new_tokens=32,
                    collect_boundaries=True,
                )
                identity.append(
                    {
                        "sample_id": str(sample),
                        "identical_tokens": disabled.state.generated_token_ids
                        == result.state.generated_token_ids,
                        "disabled_latency_s": disabled.metrics["latency_s"],
                        "none_latency_s": result.metrics["latency_s"],
                        "oracle_trace_latency_s": oracle_traces[sample].metrics["latency_s"],
                        "trace_identical_tokens": oracle_traces[sample].state.generated_token_ids
                        == result.state.generated_token_ids,
                    }
                )
            records.append(
                {
                    "sample_id": str(sample),
                    "dataset": manifest["dataset"],
                    "model": manifest["model"],
                    "quantization": "none",
                    "policy": config.policy,
                    "policy_config": config.to_dict(),
                    "correct": correct,
                    "vanilla_correct": f"\\boxed{{{sample % 4}}}"
                    in vanilla_outputs[sample].final_answer,
                    "final_answer": result.final_answer,
                    "text": result.text,
                    "stop_boundary": result.state.stop_boundary,
                    "signals": result.state.signals,
                    "interventions": result.state.interventions,
                    "experiment_kind": "synthetic_cpu_mechanics_only",
                    **result.metrics,
                }
            )
        run_manifest = {**manifest, "policy": config.to_dict()}
        write_json(destination / "results" / name / "manifest.json", run_manifest)
        write_jsonl(destination / "results" / name / "results.jsonl", records)
        runs.append({"manifest": run_manifest, "records": records})
    oracle = []
    for sample, baseline in oracle_traces.items():
        first_entropy = None
        probe_answers = []
        for event in baseline.state.signals:
            prefix = [20 + sample, 1] + event["prefix_token_ids"]
            forced = adapter.generate(
                prefix + [2],
                reasoning_format=reasoning_format,
                initial_phase="answer",
                max_new_tokens=32,
            )
            probe = adapter.probe(prefix, configurations["asag_reference"])
            entropy = probe.signals["attention_entropy"]
            relative = relative_entropy_change(entropy, first_entropy)
            first_entropy = entropy if first_entropy is None else first_entropy
            probe_answers.append(probe.signals["answer"])
            window = probe_answers[-3:]
            oracle.append(
                {
                    "sample_id": str(sample),
                    "dataset": manifest["dataset"],
                    "split": "smoke",
                    "difficulty": ["fixture_easy", "fixture_medium", "fixture_hard"][sample % 3],
                    "reasoning_tokens": event["reasoning_tokens"],
                    "safe_stop": int(f"\\boxed{{{sample % 4}}}" in forced.final_answer),
                    "final_answer": forced.final_answer,
                    "end_logit": event["end_logit"],
                    "end_probability": event["end_probability"],
                    "end_margin": -event["logit_margin"],
                    "probe_confidence": probe.confidence,
                    "attention_entropy": entropy,
                    "relative_entropy_change": relative,
                    "answer_stability": float(len(window) == 3 and len(set(window)) == 1),
                    "oracle_latency_s": forced.metrics["latency_s"],
                    "oracle_model_evaluated_tokens": forced.metrics["model_evaluated_tokens"],
                    "signal_probe_latency_s": probe.latency_s,
                    "signal_probe_model_evaluated_tokens": probe.model_evaluated_tokens,
                    "experiment_kind": "synthetic_cpu_mechanics_only",
                }
            )
    write_jsonl(destination / "results" / "oracle.jsonl", oracle)
    write_json(destination / "results" / "none_identity.json", identity)
    report = analyze_runs(runs, destination / "plots", oracle_records=oracle)
    report["entropy_hypothesis"] = entropy_hypothesis(
        oracle, -0.1, entropy_key="relative_entropy_change"
    )
    write_json(destination / "results" / "summary.json", report)
    write_json(destination / "plots" / "summary.json", report)
    plot = destination / "plots" / "pareto.svg"
    plot.write_text(
        plot.read_text(encoding="utf-8").replace(
            "Reasoning compression: measured quality–latency frontier",
            "SYNTHETIC CPU mechanics: scripted outputs, actual tiny-model forwards",
        ),
        encoding="utf-8",
    )
    if not all(row["identical_tokens"] and row["trace_identical_tokens"] for row in identity):
        raise AssertionError("Policy none changed the vanilla token sequence")
    print(
        json.dumps(
            {
                "kind": manifest["experiment_kind"],
                "runs": len(runs),
                "examples_per_run": examples,
                "oracle_boundaries": len(oracle),
                "output": str(destination),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/reasoning_compression/cpu_smoke")
    )
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument(
        "--profile", action="store_true", help="Instrument a separate profiling run"
    )
    args = parser.parse_args()
    if not 1 <= args.examples <= 12:
        parser.error("The declared fixture supports 1–12 examples")
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    execute(args.output, args.examples, profiled=args.profile)
    if profiler:
        profiler.disable()
        args.output.joinpath("logs").mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.output / "logs" / "profile.pstats")
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(40)
        (args.output / "logs" / "profile.txt").write_text(stream.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
