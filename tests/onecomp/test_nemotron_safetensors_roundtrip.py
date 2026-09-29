"""CPU inference regressions for ModelOpt Nemotron -> GPTQ safetensors.

The reference uses ordinary Linear layers populated with decoded source weights
and dequantized GPTQ results, independently of the export/loader layer builders.
No model download or CUDA device is required.
"""

import copy
import json
import logging

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizerFast

from onecomp import CalibrationConfig, GPTQ, ModelConfig, QEPConfig, Runner, load_quantized_model
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
from onecomp.utils.modelopt_checkpoint import ModelOptCheckpoint
from onecomp.utils.unfuse_moe import unfuse_moe_experts
from tests.onecomp.test_nemotron_modelopt_qep import checkpoint  # noqa: F401


@pytest.fixture
def inference_checkpoint(checkpoint):
    """Include real tokenizer assets and the source's FP32 router/MTP extras."""
    state = load_file(checkpoint / "model.safetensors")
    router_key = "backbone.layers.1.mixer.gate.weight"
    # Values deliberately differ from their BF16 rounding: router precision
    # must survive export even though the main model dtype is BF16.
    state[router_key] = state[router_key].float() + 0.000013
    state["mtp.layers.0.unused.weight"] = torch.ones(4, 4)
    save_file(state, checkpoint / "model.safetensors")
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config.update(num_nextn_predict_layers=1, bos_token_id=1, eos_token_id=31, pad_token_id=0)
    config_path.write_text(json.dumps(config))
    (checkpoint / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "MIXED_PRECISION"}})
    )

    vocab = {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2}
    vocab.update({f"token{i}": i for i in range(3, 31)})
    vocab["[EOS]"] = 31
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(checkpoint)
    GenerationConfig(bos_token_id=1, eos_token_id=31, pad_token_id=0).save_pretrained(checkpoint)
    return checkpoint


def _dequantized_reference(model_config, results):
    config = copy.deepcopy(model_config.load_config())
    del config.quantization_config
    config.time_step_limit = (0.0, float("inf"))
    config.num_nextn_predict_layers = 0
    config.use_cache = True
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    assert unfuse_moe_experts(model, logging.getLogger(__name__))
    reader = ModelOptCheckpoint(model_config.get_model_id_or_path())
    reader.load_module(model, "", device="cpu")
    # Explicit source dtype is independent of the export's dtype policy.
    router = model.model.layers[1].mixer.gate
    router.weight = torch.nn.Parameter(
        reader.get_tensor("model.layers.1.mixer.gate.weight", dtype=torch.float32)
    )
    reader.close()
    with torch.no_grad():
        for name, result in results.items():
            linear = model.get_submodule(name)
            assert isinstance(linear, torch.nn.Linear)
            linear.weight.copy_(result.compute_dequantized_weight())
    return model


def _assert_materialized(model):
    assert all(not tensor.is_meta for tensor in model.parameters())
    assert all(not tensor.is_meta for tensor in model.buffers())


@pytest.mark.parametrize(
    ("wbits", "exclude_expert", "only_mamba", "pack_weights"),
    [
        (3, False, False, True),
        (4, False, False, True),
        (4, True, False, True),
        (4, False, True, True),
        (4, True, False, False),
    ],
    ids=["3bit", "4bit", "4bit-partial-experts", "4bit-mamba-only", "4bit-unpacked"],
)
def test_nemotron_qep_safetensors_inference_roundtrip(
    inference_checkpoint, monkeypatch, tmp_path, wbits, exclude_expert, only_mamba, pack_weights
):
    model_config = ModelConfig(path=str(inference_checkpoint), device="cpu")
    ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 1]])
    monkeypatch.setattr(
        "onecomp.qep._quantize_with_qep_arch.prepare_calibration_dataset",
        lambda **kwargs: {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
    )

    def prohibit_full_source_load(**kwargs):
        pytest.fail("Export must not load the full ModelOpt model in floating point")

    monkeypatch.setattr(model_config, "load_model", prohibit_full_source_load)
    quantizer = GPTQ(
        wbits=wbits,
        num_layers=2 if only_mamba else None,
        groupsize=16 if exclude_expert else -1,
        bitpack_on_quantize=True,
        exclude_layer_keywords=[".experts.0."] if exclude_expert else [],
    )
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        qep=True,
        qep_config=QEPConfig(batch_size=1, expert_hessian_max_bytes=4096),
        calibration_config=CalibrationConfig(max_length=8, num_calibration_samples=2),
        report_progress=False,
    )
    runner.run()
    names = set(quantizer.results)
    assert any(name.endswith("mixer.in_proj") for name in names)
    if not only_mamba:
        assert any(name.endswith("mixer.q_proj") for name in names)
    expected_expert_projections = 0 if only_mamba else (6 if exclude_expert else 8)
    assert sum(".experts." in name for name in names) == expected_expert_projections
    if only_mamba:
        assert len(names) == 2
    assert not any(name.endswith("mixer.gate") for name in names)
    reference = _dequantized_reference(model_config, quantizer.results)

    quantized, tokenizer = runner.create_quantized_model(
        use_gemlite=False, pack_weights=pack_weights
    )
    quantized.eval()
    _assert_materialized(quantized)
    assert quantized.config.use_cache
    assert {
        name for name, module in quantized.named_modules() if isinstance(module, GPTQLinear)
    } == names
    expected_router = reference.model.layers[1].mixer.gate.weight
    actual_router = quantized.model.layers[1].mixer.gate.weight
    assert actual_router.dtype == torch.float32
    torch.testing.assert_close(actual_router, expected_router, rtol=0, atol=0)

    output = tmp_path / "quantized"
    # The large-model save path must stream directly from results and source
    # tensors. Building a complete floating/quantized model is unnecessary.
    with monkeypatch.context() as patch:
        patch.setattr(runner, "create_quantized_model", prohibit_full_source_load)
        runner.save_quantized_model(
            str(output),
            pack_weights=pack_weights,
            max_shard_size="4KB" if exclude_expert else "5GB",
        )
    assert list(output.glob("*.safetensors"))
    if exclude_expert:
        assert (output / "model.safetensors.index.json").is_file()
        assert len(list(output.glob("*.safetensors"))) > 1
    assert not list(output.glob("*.pt"))
    assert (output / "tokenizer.json").is_file()
    assert not (output / "hf_quant_config.json").exists()
    saved_config = json.loads((output / "config.json").read_text())
    assert saved_config["num_nextn_predict_layers"] == 0
    assert saved_config["quantization_config"]["quant_method"] != "modelopt"
    assert set(saved_config["quantization_config"]["quantized_layer_names"]) == names
    saved_state = {}
    for shard in output.glob("*.safetensors"):
        saved_state.update(load_file(shard))
    assert not any("mtp" in name or "weight_scale" in name for name in saved_state)
    assert all(f"{name}.qweight" in saved_state for name in names)

    restored, restored_tokenizer = load_quantized_model(
        str(output), device_map="cpu", local_files_only=True
    )
    restored.eval()
    _assert_materialized(restored)
    assert restored.config.use_cache
    assert restored.generation_config.eos_token_id == 31
    assert restored_tokenizer("token3 token4").input_ids == tokenizer("token3 token4").input_ids
    restored_names = {
        name for name, module in restored.named_modules() if isinstance(module, GPTQLinear)
    }
    assert restored_names == names
    assert set(restored.state_dict()) == set(quantized.state_dict())
    for name, tensor in quantized.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)

    if exclude_expert or only_mamba:
        for expert in range(4 if only_mamba else 1):
            for projection in ("up_proj", "down_proj"):
                name = f"model.layers.1.mixer.experts.{expert}.{projection}"
                assert isinstance(restored.get_submodule(name), torch.nn.Linear)
                torch.testing.assert_close(
                    restored.get_submodule(name).weight,
                    reference.get_submodule(name).weight,
                    rtol=0,
                    atol=0,
                )

    prompt = ids[:1, :4]
    with torch.no_grad():
        expected_logits = reference(prompt, use_cache=False).logits
        before_logits = quantized(prompt, use_cache=False).logits
        after_logits = restored(prompt, use_cache=False).logits
    assert torch.isfinite(after_logits).all()
    torch.testing.assert_close(before_logits, expected_logits, rtol=0, atol=2e-3)
    torch.testing.assert_close(after_logits, before_logits, rtol=0, atol=0)

    # Prefill and at least two cached decoding steps exercise both Mamba's
    # convolution/SSM state and attention's KV cache across the MoE block.
    with torch.no_grad():
        reference_prefill = reference(prompt, use_cache=True)
        restored_prefill = restored(prompt, use_cache=True)
        assert restored_prefill.past_key_values is not None
        expected_cache = reference_prefill.past_key_values
        actual_cache = restored_prefill.past_key_values
        for token in (10, 11):
            token_ids = torch.tensor([[token]])
            expected_step = reference(token_ids, past_key_values=expected_cache, use_cache=True)
            actual_step = restored(token_ids, past_key_values=actual_cache, use_cache=True)
            expected_cache = expected_step.past_key_values
            actual_cache = actual_step.past_key_values
            torch.testing.assert_close(actual_step.logits, expected_step.logits, rtol=0, atol=2e-3)
        generation_kwargs = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False)
        expected_ids = reference.generate(prompt, use_cache=True, **generation_kwargs)
        actual_ids = restored.generate(prompt, use_cache=True, **generation_kwargs)
        uncached_ids = restored.generate(prompt, use_cache=False, **generation_kwargs)
    assert actual_ids.shape[-1] == prompt.shape[-1] + 3
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, uncached_ids, rtol=0, atol=0)

    if exclude_expert or only_mamba:
        # A loaded model may be saved again after a post-process. Preserve
        # the same mixed floating/GPTQ experts through that non-streaming path.
        runner.quantized_model = restored
        second_output = tmp_path / "resaved"
        runner.save_quantized_model(str(second_output))
        resaved, _ = load_quantized_model(str(second_output), device_map="cpu")
        resaved.eval()
        _assert_materialized(resaved)
        assert {
            name for name, module in resaved.named_modules() if isinstance(module, GPTQLinear)
        } == names
        with torch.no_grad():
            resaved_logits = resaved(prompt, use_cache=False).logits
        torch.testing.assert_close(resaved_logits, after_logits, rtol=0, atol=0)


@pytest.mark.parametrize("existing_model", [False, True])
def test_nemotron_save_preserves_source_checkpoint(inference_checkpoint, existing_model):
    model_config = ModelConfig(path=str(inference_checkpoint), device="cpu")
    runner = Runner(model_config=model_config, quantizer=GPTQ(wbits=4))
    if existing_model:
        runner.quantized_model = torch.nn.Linear(4, 4)
    original = {
        path.name: path.read_bytes() for path in inference_checkpoint.iterdir() if path.is_file()
    }
    with pytest.raises(ValueError, match="different directory from the source"):
        runner.save_quantized_model(str(inference_checkpoint))
    assert {
        path.name: path.read_bytes() for path in inference_checkpoint.iterdir() if path.is_file()
    } == original
