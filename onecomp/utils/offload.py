"""
Helpers for running models that do not fit on the accelerator.

Copyright 2025-2026 Fujitsu Ltd.

"""

from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def temporarily_on_device(module: nn.Module, device, exclude: nn.Module | None = None):
    """Move the parameters and buffers of *module* to *device* for a while.

    On exit the original tensors are put back instead of copying the device
    tensors back, so host memory is neither duplicated nor modified (unlike
    ``module.to(device)`` followed by ``module.cpu()``).  Tensors owned by
    submodules of *exclude* stay where they are.

    Parameters replaced inside the context (e.g. by
    ``NVFP4Linear.materialize_weight``) get their original data back as well.
    """
    excluded = set()
    if exclude is not None:
        excluded = {id(m) for m in exclude.modules()}

    saved = []
    seen_params = set()  # tied parameters are shared by several modules
    moved_buffers: dict[int, torch.Tensor] = {}
    for m in module.modules():
        if id(m) in excluded:
            continue
        for name, param in m._parameters.items():  # pylint: disable=protected-access
            if param is None or id(param) in seen_params:
                continue
            seen_params.add(id(param))
            saved.append((m, name, True, param.data))
            param.data = param.data.to(device)
        for name, buf in m._buffers.items():  # pylint: disable=protected-access
            if buf is None:
                continue
            saved.append((m, name, False, buf))
            if id(buf) not in moved_buffers:
                moved_buffers[id(buf)] = buf.to(device)
            m._buffers[name] = moved_buffers[id(buf)]  # pylint: disable=protected-access
    del moved_buffers
    try:
        yield module
    finally:
        for m, name, is_param, tensor in reversed(saved):
            if is_param:
                m._parameters[name].data = tensor  # pylint: disable=protected-access
            else:
                m._buffers[name] = tensor  # pylint: disable=protected-access


def module_nbytes(module: nn.Module) -> int:
    """Return the number of bytes held by the parameters and buffers of *module*."""
    seen = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total
