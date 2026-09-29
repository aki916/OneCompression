"""
Compatibility shims for running quantized layers inside transformers models.

Copyright 2025-2026 Fujitsu Ltd.

"""

import torch
from torch import nn

_MAMBA_PROJECTIONS = ("in_proj", "out_proj")


def add_weight_placeholders_to_mamba_projections(model: nn.Module) -> int:
    """Give quantized Mamba projections an empty ``weight`` for dtype/device checks.

    transformers' Mamba2 mixers (e.g. NemotronH) read
    ``self.in_proj.weight.device`` to pick the CUDA-kernel path and cast
    activations to ``self.out_proj.weight.dtype``.  Quantized inference
    layers such as ``GPTQLinear`` have no ``weight``, so this registers an
    empty, non-persistent ``weight`` buffer (not saved in the state_dict)
    with the mixer's floating-point dtype on the layer's device.

    Returns:
        int: Number of layers that received a placeholder.
    """
    count = 0
    for _, mixer in model.named_modules():
        if "Mamba" not in type(mixer).__name__:
            continue
        dtype = next((p.dtype for p in mixer.parameters() if p.is_floating_point()), torch.float32)
        for attr in _MAMBA_PROJECTIONS:
            proj = getattr(mixer, attr, None)
            if proj is None or isinstance(proj, nn.Linear) or hasattr(proj, "weight"):
                continue
            tensors = list(proj.buffers()) + list(proj.parameters())
            if not tensors:
                continue
            proj.register_buffer(
                "weight", torch.empty(0, dtype=dtype, device=tensors[0].device), persistent=False
            )
            count += 1
    return count
