"""Tests for block-wise offloading helpers and offloaded perplexity.

Copyright 2025-2026 Fujitsu Ltd.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from onecomp.utils import perplexity
from onecomp.utils.modelopt_checkpoint import load_modelopt_model
from onecomp.utils.offload import module_nbytes, temporarily_on_device
from tests.onecomp.fixtures.modelopt_nemotron_h import build_modelopt_nemotron_h_checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_temporarily_on_device_restores_original_tensors():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    model.tied = model[0]  # shared submodule: its parameters must move once
    params = {n: p.data for n, p in model.named_parameters()}
    buffers = dict(model.named_buffers())

    with temporarily_on_device(model, "cuda", exclude=model[1]):
        assert model[0].weight.device.type == "cuda"
        assert model[1].weight.device.type == "cpu"
        assert model[1].running_mean.device.type == "cpu"

    for name, param in model.named_parameters():
        assert param.device.type == "cpu"
        assert param.data.data_ptr() == params[name].data_ptr()
    for name, buf in model.named_buffers():
        assert buf is buffers[name]


def test_temporarily_on_device_restores_replaced_parameters():
    from onecomp.utils.modelopt_checkpoint import NVFP4Linear
    from tests.onecomp.fixtures.modelopt_nemotron_h import quantize_nvfp4

    layer = NVFP4Linear(*quantize_nvfp4(torch.randn(8, 32)))
    packed = layer.weight_packed
    with temporarily_on_device(layer, "cpu"):
        layer.materialize_weight()
        assert layer.is_materialized
    assert not layer.is_materialized
    assert layer.weight_packed is packed


def test_module_nbytes_counts_shared_tensors_once():
    linear = nn.Linear(4, 2, bias=False)
    model = nn.ModuleList([linear, linear])
    assert module_nbytes(model) == 4 * 2 * 4


@pytest.fixture(scope="module")
def nemotron_model(tmp_path_factory):
    path = build_modelopt_nemotron_h_checkpoint(tmp_path_factory.mktemp("nemotron_h_nvfp4"))
    return load_modelopt_model(str(path))


@pytest.mark.parametrize("stride", [48, 32])
def test_offloaded_perplexity_matches_full_model(monkeypatch, nemotron_model, stride):
    input_ids = torch.randint(0, 256, (1, 200), generator=torch.Generator().manual_seed(0))
    monkeypatch.setattr(
        perplexity, "_load_encodings", lambda *_args: SimpleNamespace(input_ids=input_ids)
    )
    kwargs = dict(max_length=48, stride=stride)
    expected = perplexity.calculate_perplexity(model=nemotron_model, tokenizer=object(), **kwargs)
    actual = perplexity.calculate_perplexity_offloaded(
        model=nemotron_model, tokenizer=object(), device="cpu", batch_size=2, **kwargs
    )
    assert actual == pytest.approx(expected, rel=1e-6)
    # blocks are restored and lazily dequantized weights released
    layers = nemotron_model.model.layers
    assert type(layers[0]).__name__ == "NemotronHBlock"
    assert not layers[1].mixer.experts[0].up_proj.is_materialized
