"""CPU integration tests: real forwards, isolated probes, and cache correctness."""

from types import SimpleNamespace

import pytest
import torch

from onecomp.reasoning import ReasoningConfig, ReasoningFormat
from onecomp.runtime.transformers import TransformersReasoningAdapter


class TinyTokenizer:
    eos_token_id = 7
    pieces = {
        0: "",
        1: "Question",
        2: "<think>",
        3: "</think>",
        4: "x",
        5: ".",
        6: "42",
        7: "<eos>",
        8: "{",
        9: "}",
        10: "probe",
        11: "jump",
        12: ">",
    }

    def decode(self, ids, **_):
        return "".join(self.pieces.get(int(i), "?") for i in ids)

    def encode(self, text, **_):
        if "Final Answer" in text:
            return [10]
        if text.startswith("Wait,"):
            return [11]
        return [4]


class ScriptedLM(torch.nn.Module):
    """Small deterministic forward fixture; synthetic quality is not a benchmark."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.generation_config = SimpleNamespace(eos_token_id=7)
        self.forward_lengths = []

    def forward(self, input_ids, past_key_values=None, output_attentions=False, **_):
        incoming = input_ids[0].tolist()
        history = list(past_key_values or []) + incoming
        self.forward_lengths.append(len(incoming))
        if 10 in history:
            after_cue = history[history.index(10) + 1 :]
            next_id = [8, 6, 9, 7][min(len(after_cue), 3)]
        elif 3 in history:
            next_id = 7 if 6 in history[history.index(3) :] else 6
        else:
            count = max(0, len(history) - 1)
            next_id = 3 if count >= 6 else [4, 5][count % 2]
        logits = torch.full((1, len(incoming), 16), -7.0)
        logits[0, -1, next_id] = 7.0
        attentions = None
        if output_attentions:
            weights = torch.ones(len(history), len(history)).tril()
            weights /= weights.sum(-1, keepdim=True)
            attentions = tuple(weights[None, None].repeat(1, 2, 1, 1) for _ in range(4))
        return SimpleNamespace(logits=logits, past_key_values=history, attentions=attentions)


FORMAT = ReasoningFormat((2,), (3,))


def run(config=None, **kwargs):
    model = ScriptedLM()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    return model, adapter.generate(
        [1], config=config, reasoning_format=FORMAT, max_new_tokens=30, **kwargs
    )


def test_none_matches_disabled_and_streams_identical_tokens():
    _, raw = run()
    events = []
    model, enabled = run(ReasoningConfig(policy="none"), on_token=lambda i, text: events.append(i))
    assert enabled.state.generated_token_ids == raw.state.generated_token_ids == events
    assert enabled.state.signals == []
    assert enabled.state.interventions == []
    assert enabled.metrics["model_evaluated_tokens"] == sum(model.forward_lengths)


def test_fixed_budget_counts_markers_and_continues_final_answer():
    _, result = run(ReasoningConfig(policy="fixed", budget=2))
    assert result.state.reasoning_tokens == 2
    assert result.state.marker_tokens == 1
    assert result.state.answer_tokens == 2
    assert result.final_answer == "42<eos>"
    assert result.state.interventions[0]["kind"] == "stop"
    assert result.metrics["latency_s"] > 0
    assert result.metrics["reasoning_latency_s"] <= result.metrics["latency_s"]


def test_fixed_zero_and_multitoken_end_are_cache_consistent():
    model = ScriptedLM()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    fmt = ReasoningFormat((2,), (3, 12))
    result = adapter.generate(
        [1],
        config=ReasoningConfig(policy="fixed", budget=0),
        reasoning_format=fmt,
        max_new_tokens=12,
    )
    assert result.state.generated_token_ids[:2] == [3, 12]
    assert result.state.reasoning_tokens == 0
    assert result.state.marker_tokens == 2
    assert result.final_answer == "42<eos>"
    assert sum(model.forward_lengths) == result.metrics["model_evaluated_tokens"]


def test_thinkbrake_adaptive_stop():
    _, result = run(
        ReasoningConfig(
            policy="thinkbrake",
            boundary="tokens",
            interval=1,
            threshold=20,
            cascade_uncertainty=20,
        )
    )
    assert result.state.reasoning_tokens == 1
    assert result.state.signals[0]["logit_margin"] == 14
    assert result.final_answer == "42<eos>"


def test_probe_replay_counts_all_tokens_and_does_not_change_rejected_trajectory():
    _, raw = run()
    model, result = run(
        ReasoningConfig(
            policy="deer",
            boundary="tokens",
            interval=2,
            confidence_threshold=1.0,
            probe_max_tokens=6,
        )
    )
    assert result.state.generated_token_ids == raw.state.generated_token_ids
    assert result.metrics["probe_tokens"] == 9
    assert result.metrics["probe_prefill_tokens"] == 4 + 6 + 8
    assert result.metrics["probe_model_evaluated_tokens"] == 24
    assert result.metrics["model_evaluated_tokens"] == sum(model.forward_lengths)
    assert result.metrics["probe_latency_s"] > 0


def test_incomplete_probe_cannot_stop_and_confidence_excludes_braces():
    _, result = run(
        ReasoningConfig(
            policy="deer",
            boundary="tokens",
            interval=2,
            confidence_threshold=0.5,
            probe_max_tokens=1,
        )
    )
    assert not result.state.interventions
    assert all(s["confidence"] is None and not s["probe_complete"] for s in result.state.signals)
    model, result = run(
        ReasoningConfig(
            policy="deer",
            boundary="tokens",
            interval=2,
            confidence_threshold=0.95,
            probe_max_tokens=6,
        )
    )
    assert result.state.reasoning_tokens == 2
    assert result.state.probe_answers == ["42"]
    assert result.metrics["model_evaluated_tokens"] == sum(model.forward_lengths)


def test_asag_attention_reference_is_measured():
    model, result = run(
        ReasoningConfig(policy="asag_reference", boundary="tokens", interval=2, probe_max_tokens=6)
    )
    assert result.state.reasoning_tokens == 2
    assert result.state.signals[0]["attention_layers"] == 4
    assert result.state.signals[0]["attention_query_rows"] == 2
    assert result.state.signals[0]["first_attention_boundary"]
    assert result.metrics["attention_statistics_overhead_s"] > 0
    assert result.metrics["model_evaluated_tokens"] == sum(model.forward_lengths)


def test_collecting_vanilla_boundaries_does_not_change_tokens():
    _, plain = run()
    _, result = run(
        ReasoningConfig(policy="none", boundary="tokens", interval=2), collect_boundaries=True
    )
    assert result.state.generated_token_ids == plain.state.generated_token_ids
    assert result.state.boundaries == [2, 4, 6]
    assert all(len(s["prefix_token_ids"]) == s["boundary"] for s in result.state.signals)


def test_oracle_answer_from_prefix():
    adapter = TransformersReasoningAdapter(ScriptedLM(), TinyTokenizer())
    result = adapter.generate(
        [1, 4, 3], reasoning_format=FORMAT, max_new_tokens=6, initial_phase="answer"
    )
    assert result.final_answer == "42<eos>"
    assert result.state.reasoning_tokens == 0
    assert result.state.answer_tokens == 2


def test_none_matches_native_greedy_tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=31,
            pad_token_id=0,
        )
    ).eval()
    prompt = torch.tensor([[1, 5, 9]])
    with torch.inference_mode():
        native = model.generate(prompt, max_new_tokens=8, do_sample=False, pad_token_id=0)
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    result = adapter.generate(
        prompt[0].tolist(),
        reasoning_format=FORMAT,
        config=ReasoningConfig(policy="none"),
        max_new_tokens=8,
    )
    assert result.state.generated_token_ids == native[0, prompt.shape[1] :].tolist()


def test_asag_probe_on_real_tiny_qwen3_eager_model_output():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=31,
            pad_token_id=0,
            attn_implementation="eager",
        )
    ).eval()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    result = adapter.probe([1, 4], ReasoningConfig(policy="asag_reference", probe_max_tokens=3))
    assert result.signals["attention_layers"] == 1
    assert result.signals["attention_query_rows"] == 2
    assert result.signals["attention_key_tokens"] == 3
    assert result.signals["attention_entropy"] >= 0
    assert result.prefill_tokens == 3
    assert result.model_evaluated_tokens == result.prefill_tokens + result.decode_tokens
    assert result.attention_overhead_s > 0


def test_attention_requires_eager_and_adaptive_multitoken_rejected():
    model = ScriptedLM()
    model.config._attn_implementation = "sdpa"
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    with pytest.raises(ValueError, match="eager"):
        adapter.generate(
            [1], reasoning_format=FORMAT, config=ReasoningConfig(policy="asag_reference")
        )
    with pytest.raises(ValueError, match="singleton"):
        adapter.generate(
            [1],
            reasoning_format=ReasoningFormat((2,), (3, 12)),
            config=ReasoningConfig(policy="thinkbrake"),
        )


def test_asag_jump_and_bias_follow_normal_cache_path(monkeypatch):
    model = ScriptedLM()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    original_probe = adapter.probe
    calls = []

    def trapping_probe(*args, **kwargs):
        result = original_probe(*args, **kwargs)
        result.signals.update(
            attention_entropy=10.0,
            backward_ratio=2.0,
            previous_attention=0.2,
            current_attention=0.1,
        )
        return result

    monkeypatch.setattr(adapter, "probe", trapping_probe)
    result = adapter.generate(
        [1],
        reasoning_format=FORMAT,
        config=ReasoningConfig(
            policy="asag_reference",
            boundary="tokens",
            interval=2,
            confidence_threshold=1.0,
            probe_max_tokens=6,
            max_jumps=1,
        ),
        max_new_tokens=20,
    )
    assert [i["kind"] for i in result.state.interventions] == ["jump", "stop"]
    assert 11 in result.state.generated_token_ids
    assert result.metrics["model_evaluated_tokens"] == sum(model.forward_lengths)

    model = ScriptedLM()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    original_probe = adapter.probe

    def converging_probe(*args, **kwargs):
        result = original_probe(*args, **kwargs)
        result.signals.update(
            attention_entropy=10.0 if not calls else 5.0,
            backward_ratio=0.0,
            previous_attention=0.0,
            current_attention=0.1,
        )
        calls.append(1)
        return result

    monkeypatch.setattr(adapter, "probe", converging_probe)
    result = adapter.generate(
        [1],
        reasoning_format=FORMAT,
        config=ReasoningConfig(
            policy="asag_reference",
            boundary="tokens",
            interval=2,
            confidence_threshold=1.0,
            probe_max_tokens=6,
        ),
        max_new_tokens=20,
    )
    assert any(i["kind"] == "bias" for i in result.state.interventions)
    assert result.metrics["model_evaluated_tokens"] == sum(model.forward_lengths)


def test_progressive_attention_is_lazy_and_charges_its_extra_prefill(monkeypatch):
    model = ScriptedLM()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    original_probe = adapter.probe

    def uncertain_probe(*args, **kwargs):
        result = original_probe(*args, **kwargs)
        if kwargs.get("answer_probe", True):
            result.confidence = 0.92
            result.signals["confidence"] = 0.92
        return result

    monkeypatch.setattr(adapter, "probe", uncertain_probe)
    result = adapter.generate(
        [1],
        reasoning_format=FORMAT,
        config=ReasoningConfig(
            policy="progressive",
            boundary="tokens",
            interval=2,
            threshold=0.1,
            cascade_uncertainty=20.0,
            probe_max_tokens=6,
        ),
        max_new_tokens=20,
    )
    assert result.state.signals[0]["cascade_attention_requested"]
    traces = result.metrics["probe_history"]
    assert not traces[0]["attention_observed"] and traces[1]["attention_observed"]
    assert result.metrics["probe_model_evaluated_tokens"] == sum(
        t["model_evaluated_tokens"] for t in traces
    )
    assert result.metrics["model_evaluated_tokens"] == sum(model.forward_lengths)


def test_attention_and_confidence_ablations_omit_unneeded_work():
    _, attention = run(
        ReasoningConfig(
            policy="asag_reference", asag_mode="attention", boundary="tokens", interval=2
        )
    )
    assert attention.metrics["probe_tokens"] == 0
    assert attention.metrics["probe_decode_model_evaluated_tokens"] == 0
    assert attention.metrics["attention_latency_s"] > 0
    _, confidence = run(
        ReasoningConfig(
            policy="asag_reference", asag_mode="confidence", boundary="tokens", interval=2
        )
    )
    assert confidence.metrics["attention_latency_s"] == 0
    assert confidence.metrics["probe_tokens"] > 0


def _tiny_qwen3(attention="eager"):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=31,
            pad_token_id=0,
            attn_implementation=attention,
        )
    ).eval()


@pytest.mark.parametrize("policy", ["deer", "asag_reference"])
def test_cache_branch_probe_matches_replay_and_restores_cache_exactly(policy):
    model = _tiny_qwen3()
    adapter = TransformersReasoningAdapter(model, TinyTokenizer())
    prefix = [1, 4, 5, 4, 5, 4]
    config = ReasoningConfig(policy=policy, probe_max_tokens=4)
    with torch.inference_mode():
        cache = adapter._forward(prefix).past_key_values
    before = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    branched = adapter.probe(
        prefix, config, cache=cache, previous_span=(1, 3), current_span=(3, 6)
    )
    replayed = adapter.probe(prefix, config, previous_span=(1, 3), current_span=(3, 6))
    for (keys, values), layer in zip(before, cache.layers):
        assert torch.equal(keys, layer.keys) and torch.equal(values, layer.values)
    assert branched.token_ids == replayed.token_ids
    assert branched.reused_prefix_tokens > 0 and replayed.reused_prefix_tokens == 0
    assert branched.model_evaluated_tokens < replayed.model_evaluated_tokens
    if policy == "asag_reference":
        # Cue rows plus as many preceding reasoning rows, keys over the whole prefix.
        for key in ("attention_query_rows", "attention_key_tokens"):
            assert branched.signals[key] == replayed.signals[key]
        for key in ("attention_entropy", "previous_attention", "current_attention"):
            assert branched.signals[key] == pytest.approx(replayed.signals[key], rel=1e-4)


def test_prefix_reuse_keeps_main_trajectory_and_reduces_probe_compute():
    results = []
    for reuse in (True, False):
        adapter = TransformersReasoningAdapter(
            _tiny_qwen3(), TinyTokenizer(), probe_prefix_reuse=reuse
        )
        results.append(
            adapter.generate(
                [1, 5, 9],
                reasoning_format=FORMAT,
                config=ReasoningConfig(
                    policy="deer",
                    boundary="tokens",
                    interval=2,
                    confidence_threshold=1.0,
                    probe_max_tokens=3,
                ),
                max_new_tokens=10,
            )
        )
    reused, replayed = results
    assert reused.state.generated_token_ids == replayed.state.generated_token_ids
    assert reused.metrics["probe_reused_prefix_tokens"] > 0
    assert replayed.metrics["probe_reused_prefix_tokens"] == 0
    assert (
        reused.metrics["probe_model_evaluated_tokens"]
        < replayed.metrics["probe_model_evaluated_tokens"]
    )


def test_mixture_distribution_is_first_answer_content_position():
    adapter = TransformersReasoningAdapter(ScriptedLM(), TinyTokenizer())
    result = adapter.probe([1, 4], ReasoningConfig(policy="asag_reference", probe_max_tokens=6))
    assert result.token_ids[:3] == [8, 6, 9]
    # Not the distribution right after the cue, which predicts the "{" delimiter.
    assert int(result.first_distribution.argmax()) == 6
