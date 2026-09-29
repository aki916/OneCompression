"""Saved GPTQ layout inference and activation-order group restoration."""

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from onecomp.quantizer.gptq.gptq_layer import GPTQLinear, pack_zeros


def _layer(wbits, packed, groupsize, actorder):
    torch.manual_seed(53)
    in_features, out_features = 64, 32
    groups = 1 if groupsize == -1 else in_features // groupsize
    weight = torch.randint(0, 1 << wbits, (out_features, in_features), dtype=torch.int32)
    scales = torch.rand(groups, out_features).add(0.01).to(torch.float16)
    zeros = torch.arange(groups * out_features, dtype=torch.int32).reshape(groups, out_features)
    zeros = zeros % (1 << wbits)  # includes zero and maximum zero points
    perm = torch.randperm(in_features) if actorder else None
    layer = GPTQLinear(
        in_features=in_features,
        out_features=out_features,
        wbits=wbits,
        groupsize=groupsize,
        actorder=actorder,
        quantized_weight=weight,
        scale=scales,
        zero=zeros,
        perm=perm,
        bias=torch.randn(out_features, dtype=torch.float16),
        device="cpu",
        pack_weights=packed,
        use_gemlite=False,
    )
    return layer, weight, zeros, perm


def _restore(state, reference, **kwargs):
    return GPTQLinear.from_saved_state(
        state,
        in_features=reference.in_features,
        out_features=reference.out_features,
        wbits=reference.wbits,
        groupsize=reference.groupsize,
        actorder=reference.actorder,
        **kwargs,
    )


@pytest.mark.parametrize("wbits", [2, 3, 4, 8])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("groupsize,actorder", [(-1, False), (16, False), (16, True)])
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
def test_saved_layout_roundtrip_matches_dequantized_reference(
    tmp_path, wbits, packed, groupsize, actorder, checkpoint_format
):
    reference, weight, zeros, _ = _layer(wbits, packed, groupsize, actorder)
    state = reference.state_dict()
    if checkpoint_format == "gptq_v2":
        state["qzeros"] = pack_zeros(zeros, wbits) if packed else zeros
    save_file(state, tmp_path / "layer.safetensors")
    saved = load_file(tmp_path / "layer.safetensors")
    restored = _restore(saved, reference, checkpoint_format=checkpoint_format)
    assert restored._weight_is_packed is packed
    assert torch.equal(restored.g_idx, reference.g_idx)
    for name, value in saved.items():
        assert torch.equal(getattr(restored, name), value), name

    inputs = torch.randn(2, 3, reference.in_features)
    expected_weight = reference.scales[reference.g_idx].T * (
        weight.float() - zeros[reference.g_idx].T
    )
    expected = F.linear(inputs, expected_weight, reference.bias.float())
    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("wbits", [1, 5, 15])
def test_nonpackable_bit_widths_reload_as_dense(wbits):
    reference, _, _, _ = _layer(wbits, False, 16, True)
    restored = _restore(reference.state_dict(), reference)
    assert not restored._weight_is_packed
    inputs = torch.randn(2, reference.in_features)
    torch.testing.assert_close(restored(inputs), reference(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_missing_g_idx_reconstructed_from_activation_permutation(packed):
    reference, _, _, perm = _layer(4, packed, 16, True)
    state = reference.state_dict()
    del state["g_idx"]
    state["perm"] = perm
    restored = _restore(state, reference)
    assert torch.equal(restored.g_idx, reference.g_idx)
    inputs = torch.randn(2, reference.in_features)
    torch.testing.assert_close(restored(inputs), reference(inputs), rtol=0, atol=0)


def test_grouped_actorder_without_group_metadata_fails():
    reference, _, _, _ = _layer(4, False, 16, True)
    state = reference.state_dict()
    del state["g_idx"]
    with pytest.raises(ValueError, match="require g_idx or perm"):
        _restore(state, reference)


def test_saved_g_idx_takes_precedence_over_permutation():
    reference, _, _, perm = _layer(4, False, 16, True)
    state = reference.state_dict()
    state["perm"] = perm.flip(0)
    restored = _restore(state, reference)
    assert torch.equal(restored.g_idx, reference.g_idx)


def test_unpacked_weight_with_packed_zeros_is_rejected():
    reference, _, zeros, _ = _layer(4, False, 16, False)
    state = reference.state_dict()
    state["qzeros"] = pack_zeros(zeros - 1, 4)
    with pytest.raises(ValueError, match="requires unpacked qzeros"):
        _restore(state, reference)


@pytest.mark.parametrize("packed", [False, True])
def test_empty_layer_retains_layout_when_state_assigned(packed):
    reference, _, _, _ = _layer(4, packed, 16, True)
    state = reference.state_dict()
    restored = _restore(state, reference, empty=True)
    restored.load_state_dict(state, assign=True)
    assert restored._weight_is_packed is packed
    inputs = torch.randn(2, reference.in_features)
    torch.testing.assert_close(restored(inputs), reference(inputs), rtol=0, atol=0)
