"""Nemotron-H hybrid-block and non-gated expert regression tests."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from onecomp.utils.blockwise import (
    _ATTN_MASK_MAP_KEY,
    _compute_per_type_attention_masks,
    forward_input,
    get_blocks_and_inputs,
)
from onecomp.utils.unfuse_moe import fuse_moe_experts, unfuse_moe_experts


def _tiny_nemotron():
    from transformers import NemotronHConfig, NemotronHForCausalLM

    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=16,
        layers_block_type=["mamba", "moe", "attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=1,
        ssm_state_size=2,
        chunk_size=4,
        conv_kernel=2,
        use_mamba_kernels=False,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        moe_latent_size=8,
        moe_shared_expert_intermediate_size=16,
        intermediate_size=16,
    )
    config._attn_implementation = "eager"
    return NemotronHForCausalLM(config).eval()


def _backbone(model):
    return getattr(model, "backbone", getattr(model, "model", None))


@pytest.mark.parametrize("padding", [False, True])
def test_nemotron_blockwise_matches_full_hybrid_forward(padding):
    torch.manual_seed(31)
    model = _tiny_nemotron()
    ids = torch.randint(1, 32, (3, 8))
    mask = torch.ones_like(ids)
    if padding:
        mask[:, -2:] = 0
    with torch.no_grad():
        expected = _backbone(model)(ids, attention_mask=mask, use_cache=False).last_hidden_state
    blocks, inps, kwargs = get_blocks_and_inputs(
        model, {"input_ids": ids, "attention_mask": mask}, batch_size=2
    )
    assert kwargs[_ATTN_MASK_MAP_KEY]["moe"] is None
    for block in blocks:
        inps = forward_input(inps, block, kwargs, batch_size=2, device=torch.device("cpu"))
    actual = _backbone(model).norm_f(inps)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_nemotron_capture_allows_unloaded_meta_blocks():
    model = _tiny_nemotron()
    for block in _backbone(model).layers:
        block.to("meta")
    ids = torch.randint(1, 32, (2, 8))
    blocks, inps, kwargs = get_blocks_and_inputs(model, {"input_ids": ids}, batch_size=1)
    torch.testing.assert_close(inps, _backbone(model).embeddings(ids))
    assert next(blocks[0].parameters()).is_meta
    assert _ATTN_MASK_MAP_KEY in kwargs


@pytest.mark.parametrize(
    "layer_types", [("mamba", "attention", "moe"), ("linear_attention", "full_attention", "moe")]
)
def test_nemotron_mask_aliases_preserve_causality_and_padding(layer_types):
    parent = nn.Linear(8, 8)
    parent.config = SimpleNamespace(model_type="nemotron_h", hidden_size=8)
    padding = torch.tensor([[1, 1, 0, 0]])
    causal = object()
    with patch("transformers.masking_utils.create_causal_mask", return_value=causal) as create:
        masks = _compute_per_type_attention_masks(
            parent,
            {"position_ids": torch.arange(4)[None]},
            set(layer_types),
            attention_mask=padding,
        )
    assert masks[layer_types[1]] is causal
    torch.testing.assert_close(masks[layer_types[0]], padding.bool())
    assert masks["moe"] is None
    assert create.call_args.args[2] is padding


def test_nemotron_non_gated_experts_match_and_roundtrip():
    torch.manual_seed(18)
    model = _tiny_nemotron()
    original = _backbone(model).layers[1].mixer.experts.to(dtype=torch.bfloat16)
    up = original.up_proj.detach().clone()
    down = original.down_proj.detach().clone()
    hidden = torch.randn(5, 8, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1], [0, 2], [1, 2], [0, 1], [1, 2]])
    weights = torch.rand(5, 2, dtype=torch.float32)
    expected = original(hidden, indices, weights)
    logger = logging.getLogger(__name__)
    assert unfuse_moe_experts(model, logger)
    unfused = _backbone(model).layers[1].mixer.experts
    assert len(unfused) == 4
    assert not hasattr(unfused[0], "gate_proj")
    with patch("torch.nn.functional.one_hot", side_effect=AssertionError("dense routing mask")):
        torch.testing.assert_close(unfused(hidden, indices, weights), expected, atol=0, rtol=0)
    assert unfuse_moe_experts(model, logger) is False
    assert fuse_moe_experts(model, logger)
    fused = _backbone(model).layers[1].mixer.experts
    torch.testing.assert_close(fused.up_proj, up)
    torch.testing.assert_close(fused.down_proj, down)
    torch.testing.assert_close(fused(hidden, indices, weights), expected, atol=0, rtol=0)


def test_nemotron_experts_can_unfuse_on_meta():
    model = _tiny_nemotron().to("meta")
    assert unfuse_moe_experts(model, logging.getLogger(__name__))
    experts = _backbone(model).layers[1].mixer.experts
    assert experts[0].up_proj.weight.is_meta
    assert experts[0].up_proj.weight.shape == (8, 8)
    assert experts[0].down_proj.weight.shape == (8, 8)
