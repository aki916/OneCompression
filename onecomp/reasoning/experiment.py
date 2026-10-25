# Copyright 2025-2026 Fujitsu Ltd.

"""Reproducible serial benchmark runner and offline forced-prefix oracle."""

import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from . import ReasoningConfig, ReasoningFormat
from .boundaries import make_boundary
from .config import estimate_prompt_budget
from .data import PROMPT_TEMPLATE, extract_answer, grade_answer, load_samples, write_json
from .policies import make_policy
from .signals import logit_observation, relative_entropy_change
from .state import RequestState

logger = logging.getLogger(__name__)


def environment_metadata():
    import torch

    package_root = Path(__file__).resolve().parents[1]

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def git(*args):
        # A pip-installed package must not report an unrelated enclosing repository.
        if not (package_root.parent / ".git").exists():
            return None
        try:
            return subprocess.check_output(
                ["git", "-C", str(package_root.parent), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    sources = sorted((package_root / "reasoning").rglob("*.py")) + sorted(
        (package_root / "runtime").rglob("*.py")
    )
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(package_root)).encode())
        digest.update(path.read_bytes())
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "implementation_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("onecomp", "vllm", "transformers", "torch", "datasets")
        },
        "gpu_count": torch.cuda.device_count(),
        "gpu_models": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _file_identity(path):
    path = Path(path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def checkpoint_identity(path):
    """Content identity of local model/config/tokenizer files, computed once per run."""
    root = Path(path)
    if not root.is_dir():
        return None
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model"}
    )
    if not (root / "config.json").is_file() or not files:
        raise ValueError("Local checkpoint requires config.json and model/tokenizer files")
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(root)).encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_reference(config, policies):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from onecomp.runtime.transformers import TransformersReasoningAdapter

    device = config.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; Qwen3 GPU benchmarking cannot run in this environment"
        )
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    attention = any(make_policy(p).requires_attention for p in policies) or config.get(
        "oracle_attention", False
    )
    model_id = config.get("model", "Qwen/Qwen3-8B")
    if config.get("quantization", "none") != "none":
        from onecomp import QuantizedModelLoader

        model, tokenizer = QuantizedModelLoader.load_quantized_model(
            model_id, torch_dtype=dtype, device_map=device, trust_remote_code=False
        )
        if attention:
            model.set_attn_implementation("eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=config.get("model_revision"),
            dtype=dtype,
            device_map=device,
            attn_implementation=(
                "eager" if attention else config.get("attention_implementation", "sdpa")
            ),
            local_files_only=config.get("local_files_only", False),
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=config.get("tokenizer_revision")
            or getattr(model.config, "_commit_hash", None)
            or config.get("model_revision"),
            local_files_only=config.get("local_files_only", False),
            trust_remote_code=False,
        )
    adapter = TransformersReasoningAdapter(
        model, tokenizer, probe_prefix_reuse=config.get("probe_prefix_reuse", True)
    )
    return adapter, tokenizer


def variant_name(index, policy):
    digest = hashlib.sha256(json.dumps(policy.to_dict(), sort_keys=True).encode()).hexdigest()
    return f"{index:02d}_{policy.policy}_{digest[:8]}"


def effective_policy(policy, sample):
    """Substitute the per-prompt budget for ``prompt_budget``; otherwise unchanged."""
    if policy.policy == "prompt_budget":
        return replace(policy, budget=estimate_prompt_budget(sample["question"]))
    return policy


def make_manifest(config, samples, policy, tokenizer, model=None):
    model_revision = getattr(getattr(model, "config", None), "_commit_hash", None) or config.get(
        "model_revision"
    )
    tokenizer_revision = (
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or config.get("tokenizer_revision")
        or model_revision
    )
    checkpoint_sha = config.get("checkpoint_sha256")
    if Path(config.get("model", "Qwen/Qwen3-8B")).is_dir():
        checkpoint_sha = checkpoint_sha or checkpoint_identity(config["model"])
        model_revision = "local-checkpoint-sha256:" + checkpoint_sha
        tokenizer_sha = _file_identity(Path(config["model"]) / "tokenizer.json")
        tokenizer_revision = tokenizer_revision or (
            "local-tokenizer-sha256:" + tokenizer_sha if tokenizer_sha else model_revision
        )
    model_revision = config.get("source_model_revision") or model_revision
    if not model_revision or not tokenizer_revision:
        raise ValueError("Model and tokenizer revisions must resolve to immutable identities")
    return {
        **environment_metadata(),
        "schema_version": 1,
        "model": config.get("source_model", config.get("model", "Qwen/Qwen3-8B")),
        "model_revision": model_revision,
        "checkpoint_path": config.get("model", "Qwen/Qwen3-8B"),
        "checkpoint_sha256": checkpoint_sha,
        "tokenizer_revision": tokenizer_revision,
        "dataset": samples[0]["dataset"],
        "dataset_revision": samples[0]["dataset_revision"],
        "dataset_sha256": _file_identity(config["dataset_path"]),
        "split": config["split"],
        "prompt_template": config.get("prompt_template", PROMPT_TEMPLATE),
        "use_chat_template": config.get("use_chat_template", True),
        "chat_template_sha256": hashlib.sha256(
            str(getattr(tokenizer, "chat_template", None)).encode()
        ).hexdigest(),
        "reasoning_parser": config.get(
            "reasoning_format",
            {"name": "qwen3", "start": "<think>", "end": "</think>", "starts_in_reasoning": True},
        ),
        "decoding": {
            "temperature": 0.0,
            "max_tokens": config.get("max_new_tokens", 16000),
            "seed": config.get("seed", 0),
        },
        "backend": config.get("backend", "transformers"),
        "probe_prefix_reuse": config.get("probe_prefix_reuse", True),
        "dtype": config.get("dtype", "bfloat16"),
        "quantization": config.get("quantization", "none"),
        "attention_implementation": getattr(
            getattr(model, "config", None), "_attn_implementation", None
        ),
        "policy": policy.policy,
        "policy_config": policy.to_dict(),
        "grading": "balanced_box_exact_or_rational_v1_no_symbolic_equivalence",
        "sampling": "pass_at_1_greedy",
        "warmup_requests": config.get("warmup_requests", 1),
        "measurement_scope": "serial_request_wall_clock_including_probes_excluding_model_load",
    }


def format_prompt(tokenizer, sample, config):
    text = config.get("prompt_template", PROMPT_TEMPLATE).format(question=sample["question"])
    if config.get("use_chat_template", True):
        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=False,
        )
        if hasattr(result, "tolist"):
            result = result.tolist()
        if result and isinstance(result[0], list):
            result = result[0]
        return list(result)
    return list(tokenizer.encode(text, add_special_tokens=True))


def result_record(sample, result, manifest, prompt_ids, effective_policy):
    state = result.state
    return {
        "schema_version": 1,
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "difficulty": sample.get("difficulty", "unknown"),
        "split": sample["split"],
        "model": manifest["model"],
        "quantization": manifest["quantization"],
        "policy": manifest["policy"],
        "policy_config": manifest["policy_config"],
        "effective_policy_config": effective_policy.to_dict(),
        "correct": grade_answer(result.final_answer, sample["answer"]),
        "vanilla_correct": None,
        "reference_answer": sample["answer"],
        "final_answer": result.final_answer,
        "extracted_answer": extract_answer(result.final_answer),
        "text": result.text,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": state.generated_token_ids,
        "reasoning_tokens": state.reasoning_tokens,
        "answer_tokens": state.answer_tokens,
        "marker_tokens": state.marker_tokens,
        "total_generated_tokens": state.total_generated_tokens,
        "stop_boundary": state.stop_boundary,
        "signals": state.signals,
        "boundaries": state.boundaries,
        "interventions": state.interventions,
        **result.metrics,
    }


def run_experiment(config, output, *, selection=None):
    """Load one model, warm up, then run the same examples with every policy.

    Result files are flushed after each request and status stays incomplete on
    error. Existing artifacts are never overwritten. Policy order is fixed and
    logged; repeat randomized orders externally for serious timing comparisons.
    """
    from .calibration import validate_selection_for_evaluation
    from .metrics.report import analyze_runs

    config = dict(config)
    if Path(config.get("model", "Qwen/Qwen3-8B")).is_dir():
        config["checkpoint_sha256"] = checkpoint_identity(config["model"])
    samples = load_samples(config["dataset_path"], config["split"], config.get("limit"))
    policies = [ReasoningConfig.from_dict(p) for p in config.get("policies", [{"policy": "none"}])]
    if selection is not None:
        policies = [ReasoningConfig.from_dict(selection["policy_config"])]
    if config.get("temperature", 0) != 0:
        raise ValueError("The initial reference runner supports greedy decoding only")
    if config.get("backend", "transformers") != "transformers":
        return run_vllm_experiment(config, output, samples, policies, selection=selection)
    adapter, tokenizer = load_reference(config, policies)
    fmt = ReasoningFormat.from_tokenizer(tokenizer, **config.get("reasoning_format", {}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    baseline = None
    prompts = {str(s["sample_id"]): format_prompt(tokenizer, s, config) for s in samples}
    for _ in range(config.get("warmup_requests", 1)):
        adapter.generate(
            next(iter(prompts.values())),
            config=ReasoningConfig(),
            reasoning_format=fmt,
            max_new_tokens=min(16, config.get("max_new_tokens", 16000)),
        )
    for index, policy in enumerate(policies):
        variant = variant_name(index, policy)
        directory = output / variant
        directory.mkdir(exist_ok=False)
        manifest = make_manifest(config, samples, policy, tokenizer, adapter.model)
        if selection is not None:
            validate_selection_for_evaluation(
                selection,
                manifest,
                samples,
                allow_quantization_change=config.get("allow_quantization_transfer", False),
            )
            manifest["selection"] = selection
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "status.json", {"status": "incomplete", "completed": 0})
        records = []
        with (directory / "results.jsonl").open("w", encoding="utf-8") as stream:
            for sample in samples:
                effective = effective_policy(policy, sample)
                ids = prompts[str(sample["sample_id"])]
                result = adapter.generate(
                    ids,
                    config=effective,
                    reasoning_format=fmt,
                    request_id=str(sample["sample_id"]),
                    max_new_tokens=config.get("max_new_tokens", 16000),
                    seed=config.get("seed", 0),
                )
                row = result_record(sample, result, manifest, ids, effective)
                if baseline is not None:
                    row["vanilla_correct"] = baseline[str(sample["sample_id"])]
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                records.append(row)
                logger.info(
                    "%s %s: correct=%s latency=%.4fs",
                    variant,
                    sample["sample_id"],
                    row["correct"],
                    row["latency_s"],
                )
        if policy.policy == "none":
            baseline = {str(row["sample_id"]): row["correct"] for row in records}
        write_json(directory / "status.json", {"status": "complete", "completed": len(records)})
        runs.append({"manifest": manifest, "records": records})
    if sum(p.policy == "none" for p in policies) == 1:
        analyze_runs(runs, output / "analysis")
    return runs


def run_oracle(config, output):
    """Force every candidate prefix to answer; no correctness enters a policy.

    Each trial re-prefills the prefix and counts that cost. Oracle cost is offline
    supervision cost, not deployable inference. Failed trials are retained; safe
    stopping need not be monotone. Boundary truncation makes the minimum censored.
    """
    import torch

    config = dict(config)
    if Path(config.get("model", "Qwen/Qwen3-8B")).is_dir():
        config["checkpoint_sha256"] = checkpoint_identity(config["model"])
    samples = load_samples(config["dataset_path"], config["split"], config.get("limit"))
    if config.get("backend", "transformers") != "transformers":
        raise ValueError("Offline oracle currently requires backend=transformers")
    if config.get("temperature", 0) != 0:
        raise ValueError("The oracle reference supports greedy decoding only")
    if "oracle_answer_max_tokens" in config:
        raise ValueError(
            "Oracle uses the same remaining max_new_tokens budget as vanilla; remove oracle_answer_max_tokens"
        )
    observer = ReasoningConfig.from_dict(
        config.get("oracle_policy", {"policy": "deer", "boundary": "tokens", "interval": 128})
    )
    adapter, tokenizer = load_reference(config, [observer])
    fmt = ReasoningFormat.from_tokenizer(tokenizer, **config.get("reasoning_format", {}))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = make_manifest(config, samples, ReasoningConfig(), tokenizer, adapter.model)
    manifest.update(
        offline_oracle=True,
        oracle_observer=asdict(observer),
        candidate_limit=config.get("oracle_max_boundaries"),
        oracle_probe_signals=bool(config.get("oracle_probe_signals", True)),
        oracle_attention=bool(config.get("oracle_attention", False)),
    )
    write_json(out / "manifest.json", manifest)
    write_json(out / "status.json", {"status": "incomplete"})
    detector = make_boundary(observer)
    with (
        (out / "results.jsonl").open("w", encoding="utf-8") as stream,
        (out / "vanilla.jsonl").open("w", encoding="utf-8") as vanilla_stream,
    ):
        for sample in samples:
            prompt = format_prompt(tokenizer, sample, config)
            vanilla = adapter.generate(
                prompt,
                config=ReasoningConfig(),
                reasoning_format=fmt,
                request_id=str(sample["sample_id"]),
                max_new_tokens=config.get("max_new_tokens", 16000),
                seed=config.get("seed", 0),
            )
            vanilla_row = result_record(sample, vanilla, manifest, prompt, ReasoningConfig())
            vanilla_stream.write(json.dumps(vanilla_row, allow_nan=False) + "\n")
            vanilla_stream.flush()
            state = RequestState(str(sample["sample_id"]), fmt)
            candidates = []
            generated = vanilla.state.generated_token_ids
            for index, token in enumerate(generated):
                state.append(token, tokenizer.decode([token], skip_special_tokens=False))
                next_text = (
                    tokenizer.decode([generated[index + 1]], skip_special_tokens=False)
                    if index + 1 < len(generated)
                    else ""
                )
                if (
                    state.phase == "reasoning"
                    and state.reasoning_tokens >= observer.min_reasoning_tokens
                    and not state.pending_end_marker
                    and detector.detect(state, next_text)
                ):
                    candidates.append((index + 1, state.reasoning_tokens))
            maximum = config.get("oracle_max_boundaries")
            if maximum is not None and (type(maximum) is not int or maximum < 1):
                raise ValueError("oracle_max_boundaries must be a positive integer")
            censored = maximum is not None and len(candidates) > maximum
            if maximum is not None:
                candidates = candidates[:maximum]
            first_entropy = None
            answers = []
            previous_positions = [len(prompt)]
            for boundary, count in candidates:
                remaining = config.get("max_new_tokens", 16000) - boundary - len(fmt.end_token_ids)
                if remaining <= 0:
                    continue
                prefix = prompt + generated[:boundary]
                # This independent forward measures logits and its cost explicitly;
                # the probe then branches from its cache instead of re-prefilling.
                adapter._sync()
                signal_started = perf_counter()
                with torch.inference_mode():
                    replay = adapter._forward(prefix)
                logits = replay.logits[0, -1]
                signals = (
                    logit_observation(logits, fmt.end_token_ids[0], observer.temperature)
                    if len(fmt.end_token_ids) == 1
                    else {}
                )
                adapter._sync()
                signal_latency = perf_counter() - signal_started
                probe = None
                if config.get("oracle_probe_signals", True):
                    probe = adapter.probe(
                        prefix,
                        observer,
                        cache=getattr(replay, "past_key_values", None),
                        collect_attention=config.get("oracle_attention", False),
                        previous_span=(
                            (previous_positions[-2], previous_positions[-1])
                            if len(previous_positions) >= 2
                            else None
                        ),
                        current_span=(previous_positions[-1], len(prefix)),
                    )
                    signals.update(
                        probe.signals,
                        confidence=probe.confidence if probe.complete else None,
                        probe_complete=probe.complete,
                    )
                    answers.append(signals.get("answer", "") if probe.complete else "")
                    window = observer.stability_window
                    signals["answer_stability"] = float(
                        len(answers) >= window
                        and bool(answers[-1])
                        and len(set(answers[-window:])) == 1
                    )
                previous_positions.append(len(prefix))
                entropy = signals.get("attention_entropy")
                if entropy is not None:
                    first = first_entropy is None
                    if first:
                        first_entropy = entropy
                    signals["relative_entropy"] = (
                        None if first else relative_entropy_change(entropy, first_entropy)
                    )
                    signals["first_attention_boundary"] = first
                answer = adapter.generate(
                    prefix + list(fmt.end_token_ids),
                    config=ReasoningConfig(),
                    reasoning_format=fmt,
                    request_id=f"{sample['sample_id']}:{boundary}",
                    initial_phase="answer",
                    max_new_tokens=remaining,
                    seed=config.get("seed", 0),
                )
                row = {
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "split": sample["split"],
                    "difficulty": sample.get("difficulty", "unknown"),
                    "boundary": boundary,
                    "reasoning_tokens": count,
                    "safe_stop": int(grade_answer(answer.final_answer, sample["answer"])),
                    "final_answer": answer.final_answer,
                    "vanilla_correct": vanilla_row["correct"],
                    "vanilla_reasoning_tokens": vanilla.state.reasoning_tokens,
                    "candidate_search_censored": censored,
                    "remaining_answer_budget": remaining,
                    "forced_answer_metrics": answer.metrics,
                    "signal_replay_model_evaluated_tokens": len(prefix),
                    "signal_replay_latency_s": signal_latency,
                    "probe_model_evaluated_tokens": probe.model_evaluated_tokens if probe else 0,
                    "offline_model_evaluated_tokens": len(prefix)
                    + (probe.model_evaluated_tokens if probe else 0)
                    + answer.metrics["model_evaluated_tokens"],
                    "probe_latency_s": probe.latency_s if probe else 0,
                    **signals,
                }
                row.update(
                    end_margin=(
                        -signals["logit_margin"]
                        if signals.get("logit_margin") is not None
                        else None
                    ),
                    probe_confidence=signals.get("confidence"),
                    relative_entropy_change=signals.get("relative_entropy"),
                )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    write_json(out / "status.json", {"status": "complete", "samples": len(samples)})
    return str(out)


def run_vllm_experiment(config, output, samples, policies, *, selection=None):
    """Implemented separately to avoid importing vLLM for reference/analysis use."""
    from .vllm_experiment import run

    return run(config, output, samples, policies, selection=selection)
