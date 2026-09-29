"""End-to-end CPU regression for a tiny mixed-precision Nemotron checkpoint."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, NemotronHConfig

from onecomp import CalibrationConfig, GPTQ, ModelConfig, QEPConfig, Runner
from onecomp.utils.unfuse_moe import unfuse_moe_experts


@pytest.fixture
def checkpoint(tmp_path):
    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=32,
        layers_block_type=["mamba", "moe", "attention"],
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        mamba_num_heads=32,
        mamba_head_dim=2,
        n_groups=2,
        ssm_state_size=8,
        chunk_size=4,
        conv_kernel=4,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        moe_latent_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        use_mamba_kernels=False,
    )
    torch.manual_seed(7)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    import logging

    assert unfuse_moe_experts(model, logging.getLogger(__name__))
    state = {}
    for name, tensor in model.state_dict().items():
        name = name.replace("model.", "backbone.", 1)
        if ".experts." in name and name.endswith(".weight"):
            rows, cols = tensor.shape
            state[name] = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
            state[name + "_scale"] = torch.ones(rows, cols // 16).to(torch.float8_e4m3fn)
            state[name + "_scale_2"] = torch.tensor(0.01)
        elif name.endswith("mixer.in_proj.weight"):
            state[name] = (tensor.float() / 0.01).to(torch.float8_e4m3fn)
            state[name + "_scale"] = torch.tensor(0.01)
        else:
            state[name] = tensor.contiguous()
    save_file(state, tmp_path / "model.safetensors")
    config.quantization_config = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}
    config.save_pretrained(tmp_path)
    # Reproduce ModelOpt's non-finite JSON encoding.
    data = json.loads((tmp_path / "config.json").read_text())
    data["time_step_limit"] = [0.0, {"__float__": "Infinity"}]
    (tmp_path / "config.json").write_text(json.dumps(data))
    return tmp_path


def test_streaming_modelopt_qep_and_result_roundtrip(checkpoint, monkeypatch):
    model_config = ModelConfig(path=str(checkpoint), device="cpu")
    assert model_config.uses_layerwise_loading()
    model = model_config.load_model_for_quantization()
    assert model.get_input_embeddings().weight.device.type == "cpu"
    assert model.model.layers[0].mixer.in_proj.weight.is_meta
    assert not hasattr(model.config, "quantization_config")
    assert model.config.time_step_limit == (0.0, float("inf"))
    assert hasattr(model_config.load_config(), "quantization_config")
    monkeypatch.setattr(model_config, "load_model_for_quantization", lambda: model)
    monkeypatch.setattr(model_config, "load_tokenizer", lambda: None)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    monkeypatch.setattr(
        "onecomp.qep._quantize_with_qep_arch.prepare_calibration_dataset",
        lambda **kwargs: {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
    )
    quantizer = GPTQ(wbits=3, bitpack_on_quantize=True)
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        qep=True,
        qep_config=QEPConfig(batch_size=1, expert_hessian_max_bytes=4096),
        calibration_config=CalibrationConfig(max_length=8, num_calibration_samples=2),
        report_progress=False,
    )
    runner.run()
    names = quantizer.results.keys()
    assert any("mixer.in_proj" in name for name in names)
    assert any("mixer.q_proj" in name for name in names)
    assert sum(".experts." in name for name in names) == 8
    assert not any("mixer.gate" in name for name in names)
    assert all(p.is_meta for block in model.model.layers for p in block.parameters())
    assert all(result.qweight_is_packed for result in quantizer.results.values())
    assert all(
        torch.isfinite(result.compute_dequantized_weight()).all()
        for result in quantizer.results.values()
    )
    output = checkpoint / "results.pt"
    runner.save_quantization_results(str(output))
    restored = GPTQ(wbits=3)
    restored.load_results(str(output), allow_unsafe_deserialization=True)
    assert restored.results.keys() == names


def test_other_models_keep_the_regular_cpu_loader(monkeypatch):
    config = ModelConfig(model_id="test-model")
    monkeypatch.setattr(config, "load_config", lambda: SimpleNamespace(model_type="llama"))
    sentinel = object()
    calls = []
    monkeypatch.setattr(config, "load_model", lambda **kw: calls.append(kw) or sentinel)
    assert config.load_model_for_quantization() is sentinel
    assert calls == [{"device_map": "cpu"}]
