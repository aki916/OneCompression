# Copyright 2025-2026 Fujitsu Ltd.

"""vLLM 0.21 serial cost runner for the cheap, batch-safe logits adapter.

Worker-local signal traces are not transported by vLLM's RequestOutput. Missing
trace, scheduler token-evaluation and memory measurements stay null. Use the
Transformers reference for oracle/signal studies and vLLM for measured latency.
"""

import json
from pathlib import Path
from time import perf_counter

from . import ReasoningFormat, RequestState
from .data import extract_answer, grade_answer, write_json


def run(config, output, samples, policies, *, selection=None):
    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    from onecomp.runtime.vllm import VLLMReasoningAdapter

    from .calibration import validate_selection_for_evaluation
    from .experiment import effective_policy, format_prompt, make_manifest, variant_name
    from .metrics.report import analyze_runs

    if config.get("backend") != "vllm":
        raise ValueError("backend must be transformers or vllm")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; vLLM GPU benchmark cannot run")
    unsupported = [
        p.policy
        for p in policies
        if p.policy not in {"none", "fixed", "prompt_budget", "thinkbrake"}
    ]
    if unsupported:
        raise ValueError(f"vLLM logits hook cannot run probe/attention policies: {unsupported}")
    model = config.get("model", "Qwen/Qwen3-8B")
    config = dict(config)
    if not Path(model).is_dir():
        resolved = AutoConfig.from_pretrained(
            model, revision=config.get("model_revision"), trust_remote_code=False
        )
        config["model_revision"] = resolved._commit_hash
        tokenizer_revision = config.get("tokenizer_revision")
        if tokenizer_revision in {None, "main", "master"}:
            config["tokenizer_revision"] = resolved._commit_hash
    kwargs = {
        "model": model,
        "revision": config.get("model_revision"),
        "tokenizer_revision": config.get("tokenizer_revision", config.get("model_revision")),
        "dtype": config.get("dtype", "bfloat16"),
        "trust_remote_code": False,
        "tensor_parallel_size": config.get("tensor_parallel_size", 1),
        "max_model_len": config.get("max_model_len", 32768),
        "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.85),
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "logits_processors": [VLLMReasoningAdapter.processor_path],
        "seed": config.get("seed", 0),
    }
    quantization = config.get("quantization", "none")
    if quantization != "none":
        kwargs["quantization"] = quantization
    engine = LLM(**kwargs)
    tokenizer = engine.get_tokenizer()
    fmt = ReasoningFormat.from_tokenizer(tokenizer, **config.get("reasoning_format", {}))
    prompts = {str(s["sample_id"]): format_prompt(tokenizer, s, config) for s in samples}
    for _ in range(config.get("warmup_requests", 1)):
        engine.generate(
            [{"prompt_token_ids": next(iter(prompts.values()))}],
            SamplingParams(temperature=0, max_tokens=16),
            use_tqdm=False,
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs, baseline = [], None
    for index, policy in enumerate(policies):
        variant = variant_name(index, policy)
        directory = output / variant
        directory.mkdir(exist_ok=False)
        manifest = make_manifest(config, samples, policy, tokenizer)
        manifest.update(
            prefix_caching=False,
            batch_size=1,
            engine_config={
                key: kwargs[key]
                for key in (
                    "tensor_parallel_size",
                    "max_model_len",
                    "gpu_memory_utilization",
                    "enable_prefix_caching",
                    "enable_chunked_prefill",
                )
            },
            effective_gpu_count=config.get("tensor_parallel_size", 1),
            runtime_limitations=[
                "worker_signals_not_exported",
                "model_evaluated_tokens_not_exposed",
                "no_concurrent_throughput_measurement",
            ],
        )
        if selection is not None:
            validate_selection_for_evaluation(
                selection,
                manifest,
                samples,
                allow_quantization_change=config.get("allow_quantization_transfer", False),
            )
            manifest["selection"] = selection
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "status.json", {"status": "incomplete"})
        records = []
        with (directory / "results.jsonl").open("w", encoding="utf-8") as stream:
            for sample in samples:
                key = str(sample["sample_id"])
                effective = effective_policy(policy, sample)
                params = SamplingParams(
                    temperature=0,
                    max_tokens=config.get("max_new_tokens", 16000),
                    seed=config.get("seed", 0),
                    extra_args=VLLMReasoningAdapter.extra_args(effective, fmt, key),
                )
                torch.cuda.synchronize()
                started = perf_counter()
                response = engine.generate(
                    [{"prompt_token_ids": prompts[key]}], params, use_tqdm=False
                )[0]
                torch.cuda.synchronize()
                latency = perf_counter() - started
                completion = response.outputs[0]
                stats = response.metrics
                ttft = getattr(stats, "first_token_latency", None)
                if ttft is not None and ttft <= 0:
                    ttft = None
                state = RequestState(key, fmt)
                for token in completion.token_ids:
                    state.append(token)
                answer_ids = [
                    token
                    for token, kind in zip(state.generated_token_ids, state.token_kinds)
                    if kind == "answer"
                ]
                answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
                row = {
                    "sample_id": key,
                    "dataset": sample["dataset"],
                    "difficulty": sample.get("difficulty", "unknown"),
                    "split": sample["split"],
                    "model": manifest["model"],
                    "quantization": quantization,
                    "policy": policy.policy,
                    "policy_config": policy.to_dict(),
                    "effective_policy_config": effective.to_dict(),
                    "correct": grade_answer(answer, sample["answer"]),
                    "vanilla_correct": baseline.get(key) if baseline else None,
                    "final_answer": answer,
                    "extracted_answer": extract_answer(answer),
                    "text": completion.text,
                    "reference_answer": sample["answer"],
                    "reasoning_tokens": state.reasoning_tokens,
                    "answer_tokens": state.answer_tokens,
                    "marker_tokens": state.marker_tokens,
                    "total_generated_tokens": state.total_generated_tokens,
                    "prompt_token_ids": prompts[key],
                    "generated_token_ids": state.generated_token_ids,
                    "probe_tokens": 0,
                    "normal_model_evaluated_tokens": None,
                    "probe_model_evaluated_tokens": 0,
                    "model_evaluated_tokens": None,
                    "latency_s": latency,
                    "time_to_first_token_s": ttft,
                    "reasoning_latency_s": None,
                    "probe_latency_s": 0.0,
                    "intervention_overhead_s": None,
                    "controller_overhead_s": None,
                    "peak_memory": None,
                    "kv_cache_peak_bytes": None,
                    "stop_boundary": None,
                    "signals": None,
                    "interventions": None,
                    "finish_reason": completion.finish_reason,
                    "backend": "vllm_0.21_serial",
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                records.append(row)
        if policy.policy == "none":
            baseline = {row["sample_id"]: row["correct"] for row in records}
        write_json(directory / "status.json", {"status": "complete", "completed": len(records)})
        runs.append({"manifest": manifest, "records": records})
    if sum(p.policy == "none" for p in policies) == 1:
        analyze_runs(runs, output / "analysis")
    return runs
