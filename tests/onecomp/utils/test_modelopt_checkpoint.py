"""Tests for the ModelOpt (NVFP4 / FP8) checkpoint loader.

Copyright 2025-2026 Fujitsu Ltd.
"""

import logging

import pytest
import torch
from torch import nn

from onecomp.model_config import ModelConfig
from onecomp.utils.modelopt_checkpoint import (
    NVFP4Linear,
    dequantize_fp8,
    dequantize_nvfp4,
    is_modelopt_checkpoint,
    load_modelopt_model,
    materialize_lazy_weights,
    release_lazy_weights,
)
from onecomp.utils.unfuse_moe import _UnfusedExperts
from tests.onecomp.fixtures.modelopt_nemotron_h import (
    build_modelopt_nemotron_h_checkpoint,
    quantize_fp8,
    quantize_nvfp4,
    reference_model,
)


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory):
    return build_modelopt_nemotron_h_checkpoint(tmp_path_factory.mktemp("nemotron_h_nvfp4"))


def test_dequantize_nvfp4_nibble_order_and_scales():
    # byte 0x9A: low nibble 0xA -> -1.0 (element 0), high nibble 0x9 -> -0.5 (element 1)
    # byte 0x72: low nibble 0x2 -> 1.0, high nibble 0x7 -> 6.0
    packed = torch.tensor([[0x9A, 0x72] + [0] * 6], dtype=torch.uint8)
    scale = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)
    scale_2 = torch.tensor(0.5)
    weight = dequantize_nvfp4(packed, scale, scale_2, dtype=torch.float32)
    assert weight.shape == (1, 16)
    assert weight[0, :4].tolist() == [-1.0, -0.5, 1.0, 6.0]
    assert weight[0, 4:].abs().sum() == 0


def test_nvfp4_roundtrip_is_close():
    torch.manual_seed(0)
    weight = torch.randn(32, 64)
    dequantized = dequantize_nvfp4(*quantize_nvfp4(weight), dtype=torch.float32)
    rel_err = (dequantized - weight).norm() / weight.norm()
    assert rel_err < 0.15


def test_dequantize_fp8_per_tensor_and_per_channel():
    torch.manual_seed(0)
    weight = torch.randn(8, 16)
    fp8, scale = quantize_fp8(weight)
    per_tensor = dequantize_fp8(fp8, scale, torch.float32)
    assert torch.allclose(per_tensor, weight, rtol=0.07, atol=1e-2)
    per_channel = dequantize_fp8(fp8, scale.expand(8).clone(), torch.float32)
    assert torch.equal(per_tensor, per_channel)
    with pytest.raises(NotImplementedError):
        dequantize_fp8(fp8, torch.ones(2, 2), torch.float32)


def test_nvfp4_linear_materialize_and_release():
    torch.manual_seed(0)
    layer = NVFP4Linear(*quantize_nvfp4(torch.randn(16, 32)), dtype=torch.float32)
    assert isinstance(layer, nn.Linear)
    assert (layer.in_features, layer.out_features) == (32, 16)
    assert not layer.is_materialized
    x = torch.randn(3, 32)
    lazy_out = layer(x)

    assert materialize_lazy_weights(layer) == 1
    assert layer.is_materialized and layer.weight.shape == (16, 32)
    assert torch.equal(layer(x), lazy_out)

    layer.weight.data.zero_()
    assert layer(x).abs().sum() == 0

    release_lazy_weights(layer)
    assert not layer.is_materialized
    assert torch.equal(layer(x), lazy_out)


def test_is_modelopt_checkpoint(checkpoint_dir):
    assert ModelConfig(path=str(checkpoint_dir), device="cpu").is_modelopt_checkpoint()
    assert not is_modelopt_checkpoint(object())


def test_load_matches_dequantized_reference(checkpoint_dir, caplog):
    caplog.set_level(logging.INFO)
    model = load_modelopt_model(str(checkpoint_dir))

    assert getattr(model.config, "quantization_config", None) is None
    experts = model.model.layers[1].mixer.experts
    assert isinstance(experts, _UnfusedExperts) and len(experts) == 8
    assert isinstance(experts[3].up_proj, NVFP4Linear)
    assert isinstance(experts[3].down_proj, NVFP4Linear)
    assert experts[3].gate_proj is None
    in_proj = model.model.layers[0].mixer.in_proj
    assert type(in_proj) is nn.Linear and in_proj.weight.dtype == torch.bfloat16
    assert not any(p.device.type == "meta" for p in model.parameters())
    assert "Skipped 1 checkpoint tensors" in caplog.text

    ref = reference_model(checkpoint_dir)
    input_ids = torch.randint(0, 256, (2, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(model(input_ids).logits, ref(input_ids).logits)


def test_model_config_load_model_forces_bfloat16(checkpoint_dir):
    model = ModelConfig(path=str(checkpoint_dir), dtype="float16", device="cpu").load_model()
    assert model.model.embeddings.weight.dtype == torch.bfloat16
