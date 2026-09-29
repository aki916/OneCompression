"""Save / load tests for quantized NemotronH (non-gated MoE + Mamba) models.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json

import pytest
import torch
from torch import nn

from onecomp import GPTQ, ModelConfig, Runner
from onecomp.qep import _quantize_with_qep_arch as qep_arch
from onecomp.quantized_model_loader import QuantizedModelLoader
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
from tests.onecomp.fixtures.modelopt_nemotron_h import build_modelopt_nemotron_h_checkpoint


def _fake_calibration(**_kwargs):
    generator = torch.Generator().manual_seed(0)
    return {"input_ids": torch.randint(0, 256, (4, 32), generator=generator)}


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    checkpoint = build_modelopt_nemotron_h_checkpoint(tmp_path_factory.mktemp("nemotron_h"))
    save_dir = tmp_path_factory.mktemp("nemotron_h_gptq")

    patch = pytest.MonkeyPatch()
    patch.setattr(qep_arch, "prepare_calibration_dataset", _fake_calibration)
    try:
        model_config = ModelConfig(path=str(checkpoint), device="cpu")
        # The tiny checkpoint has no tokenizer files.
        model_config.load_tokenizer = _NoTokenizer
        runner = Runner(model_config=model_config, quantizer=GPTQ(wbits=4), qep=True)
        runner.run()
        quantized, _ = runner.create_quantized_model()
        runner.save_quantized_model(str(save_dir))
    finally:
        patch.undo()
    return quantized, save_dir


class _NoTokenizer:
    def save_pretrained(self, _directory):
        pass


def test_saved_config_keeps_infinite_time_step_limit(saved_model):
    _, save_dir = saved_model
    with open(save_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    assert config["time_step_limit"][1] == {"__float__": "Infinity"}
    names = config["quantization_config"]["modules_in_block_to_quantize"]
    assert "model.layers.1.mixer.experts.7.down_proj" in names


def test_load_quantized_nemotron_h_matches_created_model(monkeypatch, saved_model):
    quantized, save_dir = saved_model
    monkeypatch.setattr(
        "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained", lambda *_a, **_k: None
    )
    loaded, _ = QuantizedModelLoader.load_quantized_model(str(save_dir))

    assert loaded.config.time_step_limit[1] == float("inf")
    experts = loaded.model.layers[1].mixer.experts
    assert isinstance(experts[7].down_proj, GPTQLinear) and experts[7].gate_proj is None
    in_proj = loaded.model.layers[0].mixer.in_proj
    assert isinstance(in_proj, GPTQLinear) and in_proj.weight.numel() == 0
    assert not any(t.is_meta for t in list(loaded.parameters()) + list(loaded.buffers()))

    input_ids = torch.randint(0, 256, (2, 20), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(input_ids).logits, quantized(input_ids).logits, rtol=0, atol=0
        )
        generated = loaded.generate(input_ids[:1], max_new_tokens=4, do_sample=False)
    assert generated.shape == (1, 24)


def test_init_missing_tensors_keeps_loaded_ones():
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            with torch.device("meta"):
                self.linear = nn.Linear(4, 3)
            self.initialized = []

        def _init_weights(self, module):
            from transformers import initialization as init

            self.initialized.append(module)
            init.ones_(module.weight)
            init.ones_(module.bias)

    model = _Model()
    loaded = torch.full((3, 4), 2.0)
    model.linear.weight = nn.Parameter(loaded)

    QuantizedModelLoader._init_missing_tensors(model)

    assert model.initialized == [model.linear]
    assert torch.equal(model.linear.weight, loaded)
    assert torch.equal(model.linear.bias, torch.ones(3))


@pytest.mark.parametrize(
    "ckpt_key",
    [
        "model.layers.0.mlp.up_proj.qweight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "vision.blocks.0.weight",
    ],
)
def test_resolve_state_dict_key_shortcut_matches_scan(ckpt_key):
    model_keys = {
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.mlp.gate_up_proj.weight",
        "lm_head.weight",
    }
    tail_suffixes = set()
    for name in model_keys:
        last = name.rpartition(".")[2]
        tail_suffixes.update(last[i:] for i in range(len(last) + 1))
    assert QuantizedModelLoader._resolve_state_dict_key(
        ckpt_key, model_keys, tail_suffixes
    ) == QuantizedModelLoader._resolve_state_dict_key(ckpt_key, model_keys)
