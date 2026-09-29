"""CPU regressions for streaming ModelOpt checkpoint dequantization."""

import copy
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from onecomp.utils import modelopt_checkpoint
from onecomp.utils.modelopt_checkpoint import ModelOptCheckpoint, dequantize_nvfp4


def _checkpoint(tmp_path, *shards):
    mapping = {}
    for i, tensors in enumerate(shards):
        name = f"model-{i:05d}.safetensors"
        save_file(tensors, tmp_path / name)
        mapping.update({key: name for key in tensors})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return ModelOptCheckpoint(tmp_path, dtype=torch.float32, chunk_rows=2)


def test_nvfp4_codebook_nibble_order_and_both_scales():
    # Low nibble is the first value, high nibble is the second value.
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
    decoded = dequantize_nvfp4(
        packed,
        torch.tensor([[2.0]], dtype=torch.float8_e4m3fn),
        torch.tensor(0.25),
        dtype=torch.float32,
    )
    expected = (
        torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]]) / 2
    )
    torch.testing.assert_close(decoded, expected)
    assert torch.signbit(decoded[0, 8])


def test_nvfp4_block_scales_and_rounding():
    packed = torch.full((2, 16), 0x77, dtype=torch.uint8)
    scales = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn)
    global_scale = torch.tensor(0.1001)
    decoded = dequantize_nvfp4(packed, scales, global_scale)
    expected = (scales.float().repeat_interleave(16, dim=-1) * global_scale * 6).bfloat16()
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "packed,scale,scale2,message",
    [
        (torch.zeros(1, 8), torch.ones(1, 1), torch.ones(()), "packed uint8"),
        (
            torch.zeros(1, 4, dtype=torch.uint8),
            torch.ones(1, 1),
            torch.ones(()),
            "divisible by 16",
        ),
        (
            torch.zeros(1, 8, dtype=torch.uint8),
            torch.ones(1, 2),
            torch.ones(()),
            "weight_scale shape",
        ),
        (torch.zeros(1, 8, dtype=torch.uint8), torch.ones(1, 1), torch.ones(2), "global scale"),
    ],
)
def test_nvfp4_rejects_invalid_layouts(packed, scale, scale2, message):
    with pytest.raises(ValueError, match=message):
        dequantize_nvfp4(packed, scale, scale2)


def test_mixed_shards_chunked_loading_meta_module_and_legacy_names(tmp_path, monkeypatch):
    reader = _checkpoint(
        tmp_path,
        {
            "backbone.layers.0.nvfp4.weight": torch.full((5, 8), 0x32, dtype=torch.uint8),
            "backbone.layers.0.fp8.weight": torch.tensor(
                [[1.0, 2.0], [-2.0, 4.0]], dtype=torch.float8_e4m3fn
            ),
            "backbone.layers.0.fp8.bias": torch.tensor([0.5, -0.5], dtype=torch.bfloat16),
            "backbone.layers.0.counter": torch.tensor(7),
        },
        {
            "backbone.layers.0.nvfp4.weight_scale": torch.full(
                (5, 1), 2.0, dtype=torch.float8_e4m3fn
            ),
            "backbone.layers.0.fp8.weight_scale": torch.tensor(0.125),
        },
        {
            "backbone.layers.0.nvfp4.weight_scale_2": torch.tensor(0.25),
            # Input scales must never multiply the dequantized weights.
            "backbone.layers.0.nvfp4.input_scale": torch.tensor(999.0),
        },
    )
    # An unrelated block must not be read/materialized.
    reader.weight_map["backbone.layers.1.weight"] = "does-not-exist.safetensors"
    calls = []
    original = modelopt_checkpoint.dequantize_nvfp4

    def record_chunk(packed, *args, **kwargs):
        calls.append(packed.shape[0])
        return original(packed, *args, **kwargs)

    monkeypatch.setattr(modelopt_checkpoint, "dequantize_nvfp4", record_chunk)
    with torch.device("meta"):
        block = nn.Module()
        block.nvfp4 = nn.Linear(16, 5, bias=False)
        block.fp8 = nn.Linear(2, 2)
        block.register_buffer("counter", torch.tensor(0))
    block.register_buffer("temporary", torch.tensor(10), persistent=False)
    reader.load_module(block, "model.layers.0", device="cpu")
    assert calls == [2, 2, 1]
    assert len(reader._files) <= 2
    torch.testing.assert_close(block.nvfp4.weight, torch.tensor([0.5, 0.75]).repeat(5, 8))
    torch.testing.assert_close(block.fp8.weight, torch.tensor([[0.125, 0.25], [-0.25, 0.5]]))
    torch.testing.assert_close(block.fp8.bias, torch.tensor([0.5, -0.5]))
    assert block.counter.item() == 7 and block.counter.dtype == torch.int64
    assert block.temporary.item() == 10
    assert not any("scale" in key for key in block.state_dict())
    reader.close()
    assert not reader._files


def test_fp8_per_channel_scale_and_copy(tmp_path):
    reader = _checkpoint(
        tmp_path,
        {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn),
            "weight_scale": torch.tensor([0.5, 2.0]),
        },
    )
    expected = torch.tensor([[0.5, 1.0], [6.0, 8.0]])
    torch.testing.assert_close(reader.get_tensor("weight"), expected)
    copied = copy.deepcopy(reader)
    assert not copied._files
    torch.testing.assert_close(copied.get_tensor("weight"), expected)
    reader.close()
    copied.close()


def test_reader_rejects_missing_or_incompatible_tensors(tmp_path):
    reader = _checkpoint(tmp_path, {"weight": torch.zeros(2, 8, dtype=torch.uint8)})
    with pytest.raises(ValueError, match="decoded shape"):
        reader.get_tensor("weight", expected_shape=(2, 8))
    with pytest.raises(KeyError, match="weight_scale"):
        reader.get_tensor("weight", expected_shape=(2, 16))
    with pytest.raises(KeyError, match="missing"):
        reader.get_tensor("missing")
    reader.close()


def test_single_file_checkpoint_and_tied_parameters(tmp_path):
    save_file({"weight": torch.ones(2, 2, dtype=torch.bfloat16)}, tmp_path / "model.safetensors")
    reader = ModelOptCheckpoint(tmp_path)
    with torch.device("meta"):
        module = nn.Module()
        module.weight = nn.Parameter(torch.empty(2, 2, dtype=torch.bfloat16))
        module.alias = module.weight
    reader(module, "")
    assert module.weight is module.alias
    assert module.weight.dtype == torch.bfloat16
    torch.testing.assert_close(module.weight, torch.ones(2, 2, dtype=torch.bfloat16))
    reader.close()
