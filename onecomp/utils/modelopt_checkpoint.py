"""
Loader for NVIDIA ModelOpt (``quant_method="modelopt"``) checkpoints.

Copyright 2025-2026 Fujitsu Ltd.

ModelOpt checkpoints such as ``nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4``
mix several weight formats (``quant_algo="MIXED_PRECISION"``):

- NVFP4 layers (e.g. routed MoE experts): ``weight`` is uint8 holding two
  E2M1 values per byte (low nibble first), ``weight_scale`` is an FP8-E4M3
  scale per 16 input columns, and ``weight_scale_2`` is a per-tensor FP32
  scale, i.e. ``W = e2m1(weight) * weight_scale * weight_scale_2``.
- FP8 layers (e.g. Mamba ``in_proj``/``out_proj``, shared experts):
  ``weight`` is FP8-E4M3 with a per-tensor (or per-channel) FP32
  ``weight_scale``, i.e. ``W = weight * weight_scale``.
- Unquantized tensors stored as BF16/FP32.

transformers cannot load these checkpoints, so OneComp dequantizes them
itself.  FP8 layers are dequantized to the model dtype at load time.  NVFP4
layers are kept packed in :class:`NVFP4Linear` (about 0.56 bytes/param
instead of 2), because the dequantized routed experts of the 550B model
alone would need ~1 TB of host memory.  ``NVFP4Linear`` dequantizes on the
fly in ``forward``; callers that need a dense ``weight`` (GPTQ, block-wise
evaluation) call :func:`materialize_lazy_weights` on a block after moving it
to the accelerator and :func:`release_lazy_weights` before moving it back.

Activation / KV-cache scales (``input_scale``, ``k_scale``, ``v_scale``)
are ignored: OneComp runs the dequantized model in BF16.  Multi-token
prediction (``mtp.*``) weights are skipped, matching transformers.
"""

import json
import re
from collections import defaultdict
from logging import getLogger
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

logger = getLogger(__name__)

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_LUT = torch.tensor(_E2M1_VALUES + tuple(-v for v in _E2M1_VALUES), dtype=torch.float32)

# Tensor suffixes consumed while building a quantized layer or ignored on purpose.
_SCALE_SUFFIXES = ("weight_scale", "weight_scale_2")
_IGNORED_SUFFIXES = ("input_scale", "k_scale", "v_scale")
_SUPPORTED_QUANT_ALGOS = {"FP8", "NVFP4", "MIXED_PRECISION"}
_PER_EXPERT_KEY_RE = re.compile(r"\.experts\.\d+\.")


def get_modelopt_quantization_config(config) -> dict | None:
    """Return the ModelOpt quantization_config dict of *config*, else None."""
    qcfg = getattr(config, "quantization_config", None)
    if isinstance(qcfg, dict) and qcfg.get("quant_method") == "modelopt":
        return qcfg
    return None


def is_modelopt_checkpoint(config) -> bool:
    """Return True if *config* describes a ModelOpt-quantized checkpoint."""
    return get_modelopt_quantization_config(config) is not None


def dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a ModelOpt NVFP4 weight to a dense ``(out, in)`` tensor."""
    out_features, half_in = weight_packed.shape
    in_features = half_in * 2
    num_blocks = weight_scale.shape[-1]
    if in_features % num_blocks != 0:
        raise ValueError(
            f"NVFP4 weight with {in_features} input columns is not divisible "
            f"into {num_blocks} scale blocks"
        )

    lut = _E2M1_LUT.to(weight_packed.device)
    packed = weight_packed.to(torch.int32)
    values = torch.stack((lut[packed & 0x0F], lut[packed >> 4]), dim=-1)
    del packed
    scale = weight_scale.float() * weight_scale_2.float()
    values = values.view(out_features, num_blocks, -1) * scale.unsqueeze(-1)
    return values.view(out_features, in_features).to(dtype)


def dequantize_fp8(
    weight: torch.Tensor, weight_scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize an FP8 weight with a per-tensor or per-channel scale."""
    scale = weight_scale.float()
    if scale.numel() == 1:
        scale = scale.reshape(())
    elif scale.ndim == 1 and scale.numel() == weight.shape[0]:
        scale = scale.unsqueeze(1)
    elif not (scale.ndim == 2 and scale.shape == (weight.shape[0], 1)):
        raise NotImplementedError(
            f"Unsupported FP8 weight_scale shape {tuple(scale.shape)} for weight "
            f"{tuple(weight.shape)} (only per-tensor / per-channel scales are supported)"
        )
    return (weight.float() * scale).to(dtype)


class NVFP4Linear(nn.Linear):
    """``nn.Linear`` whose weight stays in packed NVFP4 form until materialized.

    While not materialized, ``weight`` is an empty placeholder and
    ``forward`` dequantizes the packed weight on the fly.
    :meth:`materialize_weight` stores the dense weight in ``weight`` so the
    module behaves like a plain ``nn.Linear`` (e.g. for GPTQ, which reads and
    overwrites ``weight.data``); :meth:`release_weight` drops it again.
    """

    def __init__(
        self,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        bias: torch.Tensor | None = None,
    ):
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.out_features, half_in = weight_packed.shape
        self.in_features = half_in * 2
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("weight_scale_2", weight_scale_2.float().reshape(()))
        self.weight = nn.Parameter(
            torch.empty(0, dtype=dtype, device=weight_packed.device), requires_grad=False
        )
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @property
    def is_materialized(self) -> bool:
        return self.weight.numel() != 0

    def dequantize_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return dequantize_nvfp4(
            self.weight_packed,
            self.weight_scale,
            self.weight_scale_2,
            dtype=dtype or self.weight.dtype,
        )

    def materialize_weight(self) -> None:
        """Store the dense dequantized weight on the buffers' device."""
        if not self.is_materialized:
            self.weight = nn.Parameter(self.dequantize_weight(), requires_grad=False)

    def release_weight(self) -> None:
        """Drop the dense weight (including any values written into it)."""
        if self.is_materialized:
            self.weight = nn.Parameter(
                torch.empty(0, dtype=self.weight.dtype, device=self.weight_packed.device),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.is_materialized else self.dequantize_weight(x.dtype)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, nvfp4=True, materialized={self.is_materialized}"
        )


def materialize_lazy_weights(module: nn.Module) -> int:
    """Materialize every :class:`NVFP4Linear` in *module*; returns the count."""
    count = 0
    for m in module.modules():
        if isinstance(m, NVFP4Linear):
            m.materialize_weight()
            count += 1
    return count


def release_lazy_weights(module: nn.Module) -> None:
    """Release the dense weights of every :class:`NVFP4Linear` in *module*."""
    for m in module.modules():
        if isinstance(m, NVFP4Linear):
            m.release_weight()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _checkpoint_key_renamer(model_type: str):
    """Return a function applying transformers' checkpoint key renamings.

    Only pure renamings (e.g. NemotronH ``backbone.`` -> ``model.``) are used;
    weight converters such as fused-expert merges are handled by OneComp.
    """
    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightRenaming
    except ImportError:  # pragma: no cover - older transformers
        return lambda key: key

    renamings = [
        t
        for t in (get_checkpoint_conversion_mapping(model_type) or [])
        if isinstance(t, WeightRenaming)
    ]

    def rename(key: str) -> str:
        for renaming in renamings:
            key, _ = renaming.rename_source_key(key)
        return key

    return rename


def _read_weight_map(model_dir: Path) -> dict[str, str]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            return json.load(f)["weight_map"]
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"No safetensors checkpoint found under {model_dir}")
    from safetensors import safe_open

    with safe_open(str(single), framework="pt") as f:
        return {k: single.name for k in f.keys()}


class _ShardReader:
    """Read tensors by key, keeping safetensors shards open."""

    def __init__(self, model_dir: Path, weight_map: dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._handles = {}

    def get(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map[key]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.model_dir / shard), framework="pt")
            self._handles[shard] = handle
        return handle.get_tensor(key)

    def close(self) -> None:
        self._handles.clear()


def _split_key(key: str) -> tuple[str, str]:
    module_name, _, tensor_name = key.rpartition(".")
    return module_name, tensor_name


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, module)


def _assign_tensor(module: nn.Module, tensor_name: str, value: torch.Tensor) -> None:
    if tensor_name in module._parameters:  # pylint: disable=protected-access
        module._parameters[tensor_name] = nn.Parameter(  # pylint: disable=protected-access
            value, requires_grad=False
        )
    elif tensor_name in module._buffers:  # pylint: disable=protected-access
        module._buffers[tensor_name] = value  # pylint: disable=protected-access
    else:
        raise KeyError(f"{type(module).__name__} has no parameter or buffer '{tensor_name}'")


def _target_dtype(module: nn.Module, tensor_name: str, value: torch.Tensor, keep_fp32: bool):
    current = getattr(module, tensor_name, None)
    if keep_fp32 or not isinstance(current, torch.Tensor) or not value.is_floating_point():
        return value.dtype
    return current.dtype


def load_modelopt_model(
    model_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    log=None,
) -> nn.Module:
    """Load a ModelOpt FP8 / NVFP4 checkpoint on CPU.

    FP8 layers are dequantized to *dtype*; NVFP4 layers become
    :class:`NVFP4Linear` modules that keep their packed weights.

    Args:
        model_dir: Local checkpoint directory.
        dtype: Dtype of the dequantized model (bfloat16 recommended).
        log: Logger (defaults to this module's logger).

    Returns:
        The model in eval mode, with ``config.quantization_config`` removed.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    log = log or logger
    model_path = Path(model_dir)
    if not model_path.is_dir():
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_dir))

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    qcfg = get_modelopt_quantization_config(config)
    if qcfg is None:
        raise ValueError(f"{model_dir} is not a ModelOpt checkpoint")
    quant_algo = str(qcfg.get("quant_algo", "")).upper()
    if quant_algo not in _SUPPORTED_QUANT_ALGOS:
        raise NotImplementedError(
            f"ModelOpt quant_algo={quant_algo!r} is not supported "
            f"(supported: {sorted(_SUPPORTED_QUANT_ALGOS)})"
        )
    log.info(
        "ModelOpt checkpoint detected (quant_algo=%s); dequantizing FP8 layers to %s "
        "and keeping NVFP4 layers packed",
        quant_algo,
        dtype,
    )
    del config.quantization_config

    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype, trust_remote_code=True)

    keep_fp32 = set(getattr(model, "_keep_in_fp32_modules_strict", None) or []) | set(
        getattr(model, "_keep_in_fp32_modules", None) or []
    )
    ignore_patterns = [
        re.compile(p) for p in (getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])
    ]

    weight_map = _read_weight_map(model_path)
    rename = _checkpoint_key_renamer(config.model_type)

    # Group checkpoint keys by (renamed) module so that a quantized weight is
    # built together with its scales, whichever shard they live in.
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    skipped = 0
    for key in weight_map:
        if any(p.search(key) for p in ignore_patterns):
            skipped += 1
            continue
        module_name, tensor_name = _split_key(rename(key))
        groups[module_name][tensor_name] = key
    if skipped:
        log.info("Skipped %d checkpoint tensors not used by transformers (e.g. MTP)", skipped)

    # ModelOpt stores MoE experts one by one (``experts.{i}.up_proj``), so the
    # fused expert parameters of the (still empty) model are unfused first.
    if any(_PER_EXPERT_KEY_RE.search(name) for name in groups):
        from .unfuse_moe import unfuse_moe_experts

        with torch.device("meta"):
            unfuse_moe_experts(model, log)

    reader = _ShardReader(model_path, weight_map)
    num_nvfp4 = num_fp8 = 0
    try:
        # Visit modules shard by shard to keep file access mostly sequential.
        for module_name, tensors in sorted(
            groups.items(), key=lambda kv: weight_map[next(iter(kv[1].values()))]
        ):
            module = model.get_submodule(module_name) if module_name else model
            if "pre_quant_scale" in tensors:
                raise NotImplementedError(
                    f"{module_name}: AWQ-style pre_quant_scale is not supported"
                )
            if "weight_scale_2" in tensors:
                bias = reader.get(tensors["bias"]).to(dtype) if "bias" in tensors else None
                new_module = NVFP4Linear(
                    reader.get(tensors["weight"]),
                    reader.get(tensors["weight_scale"]),
                    reader.get(tensors["weight_scale_2"]),
                    dtype=dtype,
                    bias=bias,
                )
                if (new_module.out_features, new_module.in_features) != (
                    module.out_features,
                    module.in_features,
                ):
                    raise ValueError(
                        f"{module_name}: NVFP4 weight shape "
                        f"({new_module.out_features}, {new_module.in_features}) does not match "
                        f"({module.out_features}, {module.in_features})"
                    )
                _set_submodule(model, module_name, new_module)
                num_nvfp4 += 1
                continue

            for tensor_name, key in tensors.items():
                if tensor_name in _SCALE_SUFFIXES or tensor_name in _IGNORED_SUFFIXES:
                    continue
                value = reader.get(key)
                if tensor_name == "weight" and "weight_scale" in tensors:
                    value = dequantize_fp8(value, reader.get(tensors["weight_scale"]), dtype)
                    num_fp8 += 1
                fp32 = any(k in f"{module_name}.{tensor_name}" for k in keep_fp32)
                target = _target_dtype(module, tensor_name, value, fp32)
                expected = getattr(module, tensor_name).shape
                if value.shape != expected:
                    raise ValueError(
                        f"{module_name}.{tensor_name}: checkpoint shape {tuple(value.shape)} "
                        f"does not match model shape {tuple(expected)}"
                    )
                _assign_tensor(module, tensor_name, value.to(target))
    finally:
        reader.close()

    missing = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    missing += [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if missing:
        raise RuntimeError(
            f"ModelOpt checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}"
        )

    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.eval()
    log.info(
        "Loaded ModelOpt checkpoint: %d NVFP4 layers kept packed, %d FP8 layers dequantized",
        num_nvfp4,
        num_fp8,
    )
    return model
