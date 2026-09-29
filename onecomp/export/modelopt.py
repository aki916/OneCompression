"""Stream a ModelOpt checkpoint into a standalone OneCompression checkpoint.

The floating point model is only a schema: its parameters may remain on the
meta device throughout export. Quantized layers and unquantized source tensors
are materialized one at a time, then written in bounded-size safetensors shards.
"""

from collections.abc import Iterable, Iterator
import json
import os
from pathlib import Path
import re
import tempfile

import torch
from safetensors.torch import save_file
from torch import nn

from onecomp.quantizer.gptq._gptq import GPTQResult
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear, is_packable_wbits


def iter_quantized_modelopt_tensors(
    model: nn.Module,
    checkpoint_reader,
    quantizer,
    *,
    pack_weights: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield CPU state tensors without loading a complete floating point model.

    ``model`` must have the same unfused expert structure used for quantization.
    Result names, shapes, and bit widths are validated before iteration starts.
    Original persistent tensors are read using the skeleton's floating point
    dtype, preserving FP32 modules such as the router. Integer buffers retain
    their source dtype. ModelOpt scales are consumed by the reader and are not
    copied into the new GPTQ checkpoint.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    results = quantizer.results
    for name, result in results.items():
        if not isinstance(result, GPTQResult):
            raise TypeError("Streaming ModelOpt export currently requires GPTQ results")
        module = modules.get(name)
        if module is None:
            raise ValueError(f"Quantization result refers to unknown module {name!r}")
        if not isinstance(module, nn.Linear):
            raise ValueError(f"Quantization result {name!r} must refer to an nn.Linear")
        shape = getattr(result, "qweight_original_shape", None)
        if shape is None:
            shape = tuple(result.qweight.shape)
        expected = (module.out_features, module.in_features)
        if tuple(shape) != expected:
            raise ValueError(
                f"Quantization result {name!r} has shape {tuple(shape)}, expected {expected}"
            )
        if pack_weights and not is_packable_wbits(result.wbits):
            raise ValueError(f"Cannot export packed GPTQ weights with wbits={result.wbits}")

    def read_original(name, existing):
        kwargs = {"device": "cpu", "expected_shape": existing.shape}
        if existing.is_floating_point():
            kwargs["dtype"] = existing.dtype
        return checkpoint_reader.get_tensor(name, **kwargs)

    def generate():
        for module_name, module in modules.items():
            prefix = module_name + "." if module_name else ""
            if module_name in results:
                bias = None
                if module.bias is not None:
                    bias = read_original(prefix + "bias", module.bias)
                layer = GPTQLinear.from_quantization_result(
                    results[module_name],
                    bias=bias,
                    device="cpu",
                    pack_weights=pack_weights,
                    use_gemlite=False,
                )
                for name, tensor in layer.state_dict().items():
                    yield prefix + name, tensor.detach().cpu()
                # Do not retain the preceding layer while reading the next one.
                del layer, bias
                continue
            for is_parameter, tensors in ((True, module._parameters), (False, module._buffers)):
                for name, existing in tensors.items():
                    if existing is None or (
                        not is_parameter and name in module._non_persistent_buffers_set
                    ):
                        continue
                    yield prefix + name, read_original(prefix + name, existing)

    return generate()


def _shard_size_bytes(value: int | str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        size = value
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?I?B)\s*", value, flags=re.I)
        if match is None:
            raise ValueError(f"Invalid max_shard_size {value!r}; use bytes or e.g. '5GB'")
        number, unit = match.groups()
        unit = unit.upper()
        exponent = 0 if unit == "B" else "KMGT".index(unit[0]) + 1
        size = int(float(number) * (1024 if "I" in unit else 1000) ** exponent)
    else:
        raise ValueError("max_shard_size must be a positive byte count or size string")
    if size <= 0:
        raise ValueError("max_shard_size must be positive")
    return size


def save_sharded_tensors(
    tensors: Iterable[tuple[str, torch.Tensor]],
    save_dir: str | Path,
    max_shard_size: int | str = "5GB",
) -> dict:
    """Write an iterator to HF safetensors shards and return the index metadata.

    One shard plus the next tensor is held at a time. A tensor exceeding the
    limit occupies a shard by itself. Files are staged inside the destination;
    iterator/serialization errors leave an existing checkpoint untouched. On
    success stale canonical model weight files are removed, while unrelated
    files (including config/tokenizer files) are preserved. A single shard uses
    ``model.safetensors`` without an index; the return value always includes its
    weight map. Each file is atomically replaced after streaming completes.
    """
    limit = _shard_size_bytes(max_shard_size)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    total_size = 0
    seen = set()
    shard_names = []
    with tempfile.TemporaryDirectory(prefix=".onecomp-weights-", dir=save_dir) as temp_dir:
        staging = Path(temp_dir)
        shard = {}
        shard_size = 0

        def flush():
            nonlocal shard, shard_size
            if not shard:
                return
            filename = f"part-{len(shard_names):05d}.safetensors"
            save_file(shard, staging / filename, metadata={"format": "pt"})
            shard_names.append((filename, tuple(shard)))
            shard = {}
            shard_size = 0

        for name, value in tensors:
            if not isinstance(name, str) or not name:
                raise ValueError("Checkpoint tensor names must be nonempty strings")
            if name in seen:
                raise ValueError(f"Duplicate checkpoint tensor {name!r}")
            seen.add(name)
            if not isinstance(value, torch.Tensor) or value.device.type == "meta":
                raise ValueError(f"Checkpoint tensor {name!r} must be a materialized tensor")
            size = value.numel() * value.element_size()
            if shard and shard_size + size > limit:
                flush()
            # Own the CPU storage so tied tensors/views are valid safetensors
            # entries and later iterator mutations cannot change buffered data.
            shard[name] = value.detach().to(device="cpu").contiguous().clone()
            shard_size += size
            total_size += size
            if shard_size >= limit:
                flush()
            del value
        flush()
        if not shard_names:
            raise ValueError("Cannot save an empty checkpoint")

        weight_map = {}
        staged_files = []
        count = len(shard_names)
        for index, (temporary_name, names) in enumerate(shard_names, start=1):
            filename = (
                "model.safetensors"
                if count == 1
                else f"model-{index:05d}-of-{count:05d}.safetensors"
            )
            os.replace(staging / temporary_name, staging / filename)
            staged_files.append(filename)
            weight_map.update((name, filename) for name in names)
        index_data = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        if count > 1:
            index_name = "model.safetensors.index.json"
            (staging / index_name).write_text(json.dumps(index_data, indent=2) + "\n")
            staged_files.append(index_name)
        for filename in staged_files:
            os.replace(staging / filename, save_dir / filename)
        canonical = re.compile(r"model(?:-\d{5}-of-\d{5})?\.safetensors(?:\.index\.json)?")
        for previous in save_dir.iterdir():
            if (
                previous.is_file()
                and canonical.fullmatch(previous.name)
                and previous.name not in staged_files
            ):
                previous.unlink()
    return index_data
