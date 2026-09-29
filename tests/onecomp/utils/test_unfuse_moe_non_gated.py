"""Tests for non-gated (up/down only) MoE experts, e.g. NemotronH.

Copyright 2025-2026 Fujitsu Ltd.
"""

import logging

import torch
from torch import nn

from onecomp.utils.unfuse_moe import (
    _is_non_gated_fused_experts,
    _is_non_gated_unfused_experts,
    _UnfusedExperts,
    fuse_moe_experts,
    unfuse_moe_experts,
)
from tests.onecomp.fixtures.modelopt_nemotron_h import tiny_nemotron_h_config

_LOGGER = logging.getLogger("test_unfuse_moe_non_gated")


def _tiny_model():
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(tiny_nemotron_h_config(), dtype=torch.float32)
    with torch.no_grad():
        for module in model.modules():
            if _is_non_gated_fused_experts(module):
                module.up_proj.normal_(0.0, 0.1)
                module.down_proj.normal_(0.0, 0.1)
    return model.eval()


def test_unfuse_non_gated_experts_preserves_outputs():
    model = _tiny_model()
    input_ids = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = model(input_ids).logits

    assert unfuse_moe_experts(model, _LOGGER)
    assert _is_non_gated_unfused_experts(model.model.layers[1].mixer.experts)
    experts = model.model.layers[1].mixer.experts
    assert isinstance(experts, _UnfusedExperts)
    assert experts.accumulate_in_router_dtype
    assert experts[0].gate_proj is None
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and ".experts." in n]
    assert "model.layers.1.mixer.experts.0.up_proj" in names
    assert len(names) == 2 * 8 * 2  # 2 MoE layers x 8 experts x (up, down)

    with torch.no_grad():
        actual = model(input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_fuse_keeps_non_gated_experts_unfused(caplog):
    model = _tiny_model()
    unfuse_moe_experts(model, _LOGGER)
    caplog.set_level(logging.WARNING)
    assert not fuse_moe_experts(model, _LOGGER)
    assert _is_non_gated_unfused_experts(model.model.layers[1].mixer.experts)
    assert "Keeping 2 non-gated MoE expert module(s) unfused" in caplog.text
