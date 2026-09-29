"""Read ModelOpt NVFP4/FP8 checkpoints without materializing the whole model.

The NVFP4 layout is ModelOpt's unswizzled HF export: the low nibble holds
the even element, with one E4M3 scale per 16 elements and an FP32 global
scale. See NVIDIA/TensorRT-Edge-LLM's checkpoint/repacking.py.
"""

from collections import OrderedDict
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

_FP8_STORAGE_DTYPES = {"F8_E4M3", "F8_E4M3FNUZ", "F8_E5M2", "F8_E5M2FNUZ"}


def dequantize_nvfp4(
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode packed E2M1 weights, applying both scales before rounding."""
    if packed.dtype != torch.uint8 or packed.ndim < 1:
        raise ValueError("NVFP4 weights must be packed uint8 tensors")
    width = packed.shape[-1] * 2
    if width % 16:
        raise ValueError(f"NVFP4 input dimension must be divisible by 16, got {width}")
    scale_shape = (*packed.shape[:-1], width // 16)
    if tuple(weight_scale.shape) != scale_shape:
        raise ValueError(
            f"NVFP4 weight_scale shape must be {scale_shape}, got {tuple(weight_scale.shape)}"
        )
    if weight_scale_2.numel() != 1:
        raise ValueError("NVFP4 weight_scale_2 must contain exactly one global scale")
    if not dtype.is_floating_point:
        raise ValueError("Dequantized weights require a floating point dtype")

    codebook = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=packed.device,
        dtype=torch.float32,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1)
    values = codebook[codes.long()].reshape(*packed.shape[:-1], width // 16, 16)
    scales = weight_scale.to(device=packed.device, dtype=torch.float32)
    scales = scales * weight_scale_2.to(device=packed.device, dtype=torch.float32).reshape(())
    values.mul_(scales.unsqueeze(-1))
    return values.reshape(*packed.shape[:-1], width).to(dtype=dtype)


class ModelOptCheckpoint:
    """Lazily load ordinary, FP8, and NVFP4 tensors from local safetensors.

    Only the requested module is materialized. Large tensors are converted
    in row chunks, so FP32 dequantization temporaries never span a whole
    block. At most two shard handles are retained; the checkpoint is read
    only and the caller owns module placement and release.
    """

    def __init__(
        self,
        path: str | Path,
        dtype: torch.dtype = torch.bfloat16,
        *,
        chunk_rows: int = 128,
    ):
        self.path = Path(path)
        if not dtype.is_floating_point:
            raise ValueError("Dequantized weights require a floating point dtype")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        self.dtype = dtype
        self.chunk_rows = chunk_rows
        self._files = OrderedDict()
        index = self.path / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            filename = self.path / "model.safetensors"
            if not filename.is_file():
                raise FileNotFoundError(f"No safetensors checkpoint found in {self.path}")
            with safe_open(filename, framework="pt", device="cpu") as handle:
                self.weight_map = {name: filename.name for name in handle.keys()}

    def __getstate__(self):
        # Models carrying this reader can still be copied/serialized without
        # trying to pickle safetensors' native mmap handles.
        return {**self.__dict__, "_files": OrderedDict()}

    def close(self):
        """Release cached file handles; subsequent reads reopen them."""
        while self._files:
            _, handle = self._files.popitem(last=False)
            handle.__exit__(None, None, None)

    def _file(self, name):
        try:
            filename = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"Tensor {name!r} is missing from {self.path}") from exc
        if filename in self._files:
            self._files.move_to_end(filename)
            return self._files[filename]
        handle = safe_open(self.path / filename, framework="pt", device="cpu")
        handle.__enter__()
        self._files[filename] = handle
        while len(self._files) > 2:
            _, previous = self._files.popitem(last=False)
            previous.__exit__(None, None, None)
        return handle

    def _raw_tensor(self, name):
        return self._file(name).get_tensor(name)

    def get_tensor(
        self,
        name: str,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
        expected_shape=None,
    ) -> torch.Tensor:
        """Read one tensor, dequantizing weights and validating its shape.

        Integer buffers retain their integer dtype. Activation and KV-cache
        scales are never applied to weights: these are fresh floating point
        modules used for recalibration, not ModelOpt inference modules.
        """
        dtype = self.dtype if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError("Dequantized weights require a floating point dtype")
        # Nemotron-H's HF backbone was renamed from backbone to model.
        if name not in self.weight_map and name.startswith("model."):
            legacy_name = "backbone." + name[len("model.") :]
            if legacy_name in self.weight_map:
                name = legacy_name
        handle = self._file(name)
        tensor_slice = handle.get_slice(name)
        shape = tuple(tensor_slice.get_shape())
        storage_dtype = tensor_slice.get_dtype()
        is_weight = name == "weight" or name.endswith(".weight")
        is_nvfp4 = is_weight and storage_dtype == "U8"
        is_fp8 = is_weight and storage_dtype in _FP8_STORAGE_DTYPES
        output_shape = (*shape[:-1], shape[-1] * 2) if is_nvfp4 and shape else shape
        if expected_shape is not None and output_shape != tuple(expected_shape):
            raise ValueError(
                f"Checkpoint tensor {name!r} has decoded shape {output_shape}, "
                f"expected {tuple(expected_shape)}"
            )

        if not (is_nvfp4 or is_fp8):
            value = handle.get_tensor(name)
            return value.to(
                device=device, dtype=dtype if value.is_floating_point() else value.dtype
            )

        if len(shape) != 2:
            raise ValueError(f"Quantized weight {name!r} must be a matrix, got shape {shape}")
        scale_name = name + "_scale"
        scale = self._raw_tensor(scale_name)
        if is_nvfp4:
            scale_2 = self._raw_tensor(name + "_scale_2")
            scale_shape = (shape[0], output_shape[1] // 16)
            if output_shape[1] % 16 or tuple(scale.shape) != scale_shape:
                raise ValueError(
                    f"Invalid NVFP4 weight_scale shape for {name!r}: {tuple(scale.shape)}"
                )
            if scale_2.numel() != 1:
                raise ValueError(f"NVFP4 global scale for {name!r} must be scalar")
        else:
            # ModelOpt FP8 uses a scalar (or one scale per output channel).
            # Block-scaled MXFP8 is a different format and must not silently
            # broadcast into this decoder.
            if scale.numel() == 1:
                scale = scale.reshape(())
            elif tuple(scale.shape) in ((shape[0],), (shape[0], 1)):
                scale = scale.reshape(shape[0], 1)
            else:
                raise ValueError(
                    f"Unsupported FP8 weight_scale shape for {name!r}: {tuple(scale.shape)}"
                )

        # Reading two scales can evict the original shard from the cache.
        tensor_slice = self._file(name).get_slice(name)
        output = torch.empty(output_shape, dtype=dtype, device=device)
        for start in range(0, shape[0], self.chunk_rows):
            end = min(start + self.chunk_rows, shape[0])
            packed = tensor_slice[start:end].to(device=device)
            if is_nvfp4:
                chunk = dequantize_nvfp4(packed, scale[start:end], scale_2, dtype=dtype)
            else:
                chunk_scale = scale if scale.ndim == 0 else scale[start:end]
                chunk = packed.float() * chunk_scale.to(device=device, dtype=torch.float32)
            output[start:end].copy_(chunk)
        return output

    @torch.no_grad()
    def load_module(
        self,
        module: nn.Module,
        prefix: str,
        *,
        device: str | torch.device = "cpu",
    ) -> nn.Module:
        """Populate a meta module using its original checkpoint name.

        Only parameters and persistent buffers declared by the floating
        point module are loaded, leaving quantization metadata on disk.
        Missing tensors and incompatible shapes are errors.
        """
        prefix = prefix.rstrip(".")
        loaded = {}
        for module_name, child in module.named_modules():
            for is_parameter, tensors in ((True, child._parameters), (False, child._buffers)):
                for local_name, existing in list(tensors.items()):
                    if existing is None or (
                        not is_parameter and local_name in child._non_persistent_buffers_set
                    ):
                        continue
                    key = ".".join(part for part in (prefix, module_name, local_name) if part)
                    if id(existing) in loaded:
                        tensors[local_name] = loaded[id(existing)]
                        continue
                    value = self.get_tensor(
                        key,
                        dtype=existing.dtype if existing.is_floating_point() else self.dtype,
                        device=device,
                        expected_shape=existing.shape,
                    )
                    if is_parameter:
                        value = nn.Parameter(value, requires_grad=existing.requires_grad)
                    tensors[local_name] = value
                    loaded[id(existing)] = value
        return module

    __call__ = load_module
