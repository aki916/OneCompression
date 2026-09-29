"""CPU regressions for bounded-memory MoE QEP and streamed checkpoints."""

import pytest
import torch
from torch import nn

from onecomp.qep import QEPConfig
from onecomp.qep._quantize_with_qep_arch import (
    _compute_per_module_hessians,
    _group_expert_hessians,
    run_quantize_with_qep_arch,
)
from onecomp.quantizer.gptq import GPTQ
from tests.onecomp.test_qep_expert_recovery_integration import (
    _FakeModelConfig,
    _ToyModel,
    _fake_prepare_calibration_dataset,
)


class _RepeatedProjection(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.unused = nn.Linear(4, 4, bias=False)
        self.fail = fail

    def forward(self, hidden_states):
        first = self.proj(hidden_states)
        if self.fail:
            raise RuntimeError("forward failure")
        return first + self.proj(hidden_states * 2)


def test_expert_hessian_accumulates_every_invocation_and_ignores_unused():
    block = _RepeatedProjection()
    inps = torch.randn(3, 2, 4)
    entries = _compute_per_module_hessians(block, [block.proj, block.unused], inps, {}, 2, "cpu")
    hessian, count = entries[block.proj]
    flattened = inps.reshape(-1, 4)
    expected = 2 * 5 * (flattened.T @ flattened) / (2 * flattened.shape[0])
    torch.testing.assert_close(hessian, expected)
    assert count == 2 * flattened.shape[0]
    assert entries[block.unused] is None
    assert not block.proj._forward_hooks


def test_expert_hooks_are_removed_after_forward_error():
    block = _RepeatedProjection(fail=True)
    with pytest.raises(RuntimeError, match="forward failure"):
        _compute_per_module_hessians(block, [block.proj], torch.randn(1, 2, 4), {}, 1, "cpu")
    assert not block.proj._forward_hooks


def test_expert_groups_respect_memory_budget_without_losing_large_layers():
    modules = [nn.Linear(dim, 1) for dim in (4, 4, 8, 4, 32, 4)]
    groups = list(_group_expert_hessians(modules, max_bytes=256))
    assert [module for group in groups for module in group] == modules
    assert all(
        sum(module.in_features**2 * 4 for module in group) <= 256 or len(group) == 1
        for group in groups
    )
    assert groups[0] == modules[:2]
    assert [modules[4]] in groups


@pytest.mark.parametrize("field", ["batch_size", "expert_hessian_max_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_qep_rejects_invalid_memory_settings(field, value):
    with pytest.raises(ValueError, match=field):
        QEPConfig(**{field: value})


@pytest.mark.parametrize("streamed", [False, True])
def test_bounded_expert_quantization_including_streamed_meta_blocks(monkeypatch, streamed):
    torch.manual_seed(0)
    model = _ToyModel()
    config = _FakeModelConfig(model)
    checkpoint_calls = []
    if streamed:
        state = {key: value.clone() for key, value in model.model.layers[0].state_dict().items()}
        model.model.layers[0].to_empty(device="meta")

        class Checkpoint:
            def load_module(self, block, prefix, device):
                assert all(parameter.is_meta for parameter in block.parameters())
                checkpoint_calls.append(prefix)
                block.load_state_dict(
                    {key: value.to(device) for key, value in state.items()}, assign=True
                )

        model._onecomp_checkpoint = Checkpoint()
        config.load_model_for_quantization = lambda: model

    monkeypatch.setattr(
        "onecomp.qep._quantize_with_qep_arch.prepare_calibration_dataset",
        _fake_prepare_calibration_dataset,
    )
    expected_names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and ".experts." in name
    }
    quantizer = GPTQ(wbits=4, groupsize=2, include_layer_keywords=["experts"])
    run_quantize_with_qep_arch(
        config,
        quantizer,
        QEPConfig(device="cpu", batch_size=1, expert_hessian_max_bytes=256),
        None,
        report_progress=False,
    )
    assert set(quantizer.results) == expected_names
    assert all(result.qweight_is_packed for result in quantizer.results.values())
    assert all(
        torch.isfinite(result.compute_dequantized_weight()).all()
        for result in quantizer.results.values()
    )
    if streamed:
        assert checkpoint_calls == ["model.layers.0"]
        assert all(parameter.is_meta for parameter in model.model.layers[0].parameters())
