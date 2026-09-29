"""Streaming ModelOpt export preserves standalone inference state and bounded shards."""

import json
from types import SimpleNamespace

import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import nn

import onecomp.export.modelopt as exporter
from onecomp.export.modelopt import iter_quantized_modelopt_tensors, save_sharded_tensors
from onecomp.quantizer.gptq._gptq import GPTQ, GPTQResult
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
from onecomp.utils.modelopt_checkpoint import ModelOptCheckpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 32, dtype=torch.bfloat16)
        self.proj = nn.Linear(32, 32, bias=True, dtype=torch.bfloat16)
        self.gate = nn.Linear(32, 4, bias=False, dtype=torch.float32)
        self.norm = nn.LayerNorm(32, dtype=torch.bfloat16)
        self.register_buffer("positions", torch.arange(4))
        self.register_buffer("scratch", torch.ones(3), persistent=False)


@pytest.mark.parametrize("wbits", [3, 4])
@pytest.mark.parametrize("packed_result", [False, True])
def test_meta_model_exports_original_and_quantized_inference_tensors(
    tmp_path, wbits, packed_result
):
    torch.manual_seed(15)
    model = TinyModel()
    originals = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    source = tmp_path / "source"
    source.mkdir()
    # The weight itself is intentionally absent: it must be sourced exclusively
    # from the result, while untouched weights and bias come from the checkpoint.
    save_file(
        {name: tensor for name, tensor in originals.items() if name != "proj.weight"},
        source / "model.safetensors",
    )
    quantizer = GPTQ(wbits=wbits, groupsize=16, actorder=True, bitpack_on_quantize=packed_result)
    calibration = torch.randn(2, 20, 32, dtype=torch.bfloat16)
    hessian, _ = quantizer.calculate_hessian(model.proj, calibration)
    result = quantizer.quantize_layer(model.proj, calibration, hessian=hessian)
    quantizer.results["proj"] = result
    reference = GPTQLinear.from_quantization_result(
        result, bias=originals["proj.bias"], device="cpu", use_gemlite=False
    )
    model.to("meta")
    reader = ModelOptCheckpoint(source)
    destination = tmp_path / "exported"
    try:
        index = save_sharded_tensors(
            iter_quantized_modelopt_tensors(model, reader, quantizer),
            destination,
            max_shard_size="1KiB",
        )
    finally:
        reader.close()
    state = {}
    for filename in set(index["weight_map"].values()):
        state.update(load_file(destination / filename))
        with safe_open(destination / filename, framework="pt") as handle:
            assert handle.metadata() == {"format": "pt"}
    assert "proj.weight" not in state
    assert "scratch" not in state
    assert "proj.g_idx" in state
    assert state["gate.weight"].dtype == torch.float32
    assert state["embed.weight"].dtype == torch.bfloat16
    assert state["positions"].dtype == torch.int64
    for name, tensor in originals.items():
        if not name.startswith("proj."):
            assert torch.equal(state[name], tensor)
    reloaded = GPTQLinear.from_saved_state(
        {
            name.removeprefix("proj."): tensor
            for name, tensor in state.items()
            if name.startswith("proj.")
        },
        in_features=32,
        out_features=32,
        wbits=wbits,
        groupsize=16,
        actorder=True,
    )
    inputs = torch.randn(7, 32, dtype=torch.float16)
    torch.testing.assert_close(reloaded(inputs), reference(inputs), rtol=0, atol=0)
    assert all(parameter.is_meta for parameter in model.parameters())
    disk_index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert disk_index == index
    assert index["metadata"]["total_size"] == sum(
        value.numel() * value.element_size() for value in state.values()
    )


def test_streaming_reader_decodes_unquantized_nvfp4_and_fp8(tmp_path):
    model = nn.Module()
    model.nvfp4 = nn.Linear(32, 4, bias=False, dtype=torch.bfloat16, device="meta")
    model.fp8 = nn.Linear(32, 4, bias=False, dtype=torch.bfloat16, device="meta")
    save_file(
        {
            "nvfp4.weight": torch.full((4, 16), 0x22, dtype=torch.uint8),
            "nvfp4.weight_scale": torch.ones(4, 2).to(torch.float8_e4m3fn),
            "nvfp4.weight_scale_2": torch.tensor(2.0),
            "fp8.weight": torch.ones(4, 32).to(torch.float8_e4m3fn),
            "fp8.weight_scale": torch.tensor(3.0),
        },
        tmp_path / "model.safetensors",
    )
    reader = ModelOptCheckpoint(tmp_path)
    try:
        state = dict(iter_quantized_modelopt_tensors(model, reader, SimpleNamespace(results={})))
    finally:
        reader.close()
    assert set(state) == {"nvfp4.weight", "fp8.weight"}
    torch.testing.assert_close(state["nvfp4.weight"], torch.full((4, 32), 2, dtype=torch.bfloat16))
    torch.testing.assert_close(state["fp8.weight"], torch.full((4, 32), 3, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "name,shape,error",
    [
        ("missing", (32, 32), "unknown module"),
        ("embed", (8, 32), "nn.Linear"),
        ("proj", (16, 32), "has shape"),
    ],
)
def test_result_validation_precedes_iteration(name, shape, error):
    result = GPTQResult(wbits=4, qweight=torch.ones(shape, dtype=torch.int32))
    with pytest.raises(ValueError, match=error):
        iter_quantized_modelopt_tensors(
            TinyModel().to("meta"), None, SimpleNamespace(results={name: result})
        )


def test_unsupported_result_type_has_descriptive_error():
    with pytest.raises(TypeError, match="requires GPTQ"):
        iter_quantized_modelopt_tensors(
            TinyModel(), None, SimpleNamespace(results={"proj": object()})
        )


def test_shard_limit_and_shared_storage(tmp_path, monkeypatch):
    sizes = []
    original_save_file = exporter.save_file

    def recording_save_file(tensors, path, **kwargs):
        sizes.append(sum(t.numel() * t.element_size() for t in tensors.values()))
        original_save_file(tensors, path, **kwargs)

    monkeypatch.setattr(exporter, "save_file", recording_save_file)
    shared = torch.arange(6, dtype=torch.float32)
    entries = [("a", shared), ("b", shared), ("big", torch.ones(80)), ("c", shared[:3])]
    index = save_sharded_tensors(iter(entries), tmp_path, max_shard_size=64)
    assert sizes == [48, 320, 12]
    assert len(set(index["weight_map"].values())) == 3
    for name, expected in entries:
        actual = load_file(tmp_path / index["weight_map"][name])[name]
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("failure", ["duplicate", "iterator", "meta"])
def test_streaming_failure_leaves_existing_checkpoint_untouched(tmp_path, failure):
    save_file({"previous": torch.tensor([9])}, tmp_path / "model.safetensors")
    old = (tmp_path / "model.safetensors").read_bytes()

    def failing_stream():
        yield "new", torch.arange(32)
        if failure == "duplicate":
            yield "new", torch.arange(16)
        elif failure == "meta":
            yield "meta", torch.empty(3, device="meta")
        else:
            raise RuntimeError("reader failed")

    with pytest.raises((ValueError, RuntimeError)):
        save_sharded_tensors(failing_stream(), tmp_path, max_shard_size=32)
    assert (tmp_path / "model.safetensors").read_bytes() == old
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.safetensors"]


def test_rewrite_removes_obsolete_shards_but_preserves_other_files(tmp_path):
    (tmp_path / "config.json").write_text('{"test": true}')
    save_file({"adapter": torch.ones(2)}, tmp_path / "adapter_model.safetensors")
    entries = [("a", torch.ones(10)), ("b", torch.zeros(10))]
    save_sharded_tensors(entries, tmp_path, max_shard_size=40)
    assert (tmp_path / "model.safetensors.index.json").exists()
    save_sharded_tensors(entries, tmp_path, max_shard_size=100)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "adapter_model.safetensors",
        "config.json",
        "model.safetensors",
    ]
    assert set(load_file(tmp_path / "model.safetensors")) == {"a", "b"}
    save_sharded_tensors(entries, tmp_path, max_shard_size=40)
    assert not (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "model.safetensors.index.json").exists()


@pytest.mark.parametrize("size", [0, -1, "0GB", "banana", True])
def test_invalid_shard_size_rejected_before_writing(tmp_path, size):
    with pytest.raises(ValueError, match="max_shard_size"):
        save_sharded_tensors([("a", torch.ones(1))], tmp_path, max_shard_size=size)
    assert not list(tmp_path.iterdir())
