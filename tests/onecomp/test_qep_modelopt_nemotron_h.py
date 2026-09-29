"""CPU integration tests: QEP + GPTQ on a tiny NemotronH ModelOpt (NVFP4) checkpoint.

Copyright 2025-2026 Fujitsu Ltd.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from onecomp import GPTQ, ModelConfig, Runner
from onecomp import runner as runner_module
from onecomp.qep import _quantize_with_qep_arch as qep_arch
from onecomp.qep._qep_config import QEPConfig
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
from onecomp.utils.modelopt_checkpoint import NVFP4Linear
from tests.onecomp.fixtures.modelopt_nemotron_h import build_modelopt_nemotron_h_checkpoint


def _fake_calibration(**_kwargs):
    generator = torch.Generator().manual_seed(0)
    return {"input_ids": torch.randint(0, 256, (4, 32), generator=generator)}


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory):
    return build_modelopt_nemotron_h_checkpoint(tmp_path_factory.mktemp("nemotron_h_nvfp4"))


@pytest.fixture(autouse=True)
def _patch_calibration(monkeypatch):
    monkeypatch.setattr(qep_arch, "prepare_calibration_dataset", _fake_calibration)


def _run_qep(checkpoint_dir):
    model_config = ModelConfig(path=str(checkpoint_dir), device="cpu")
    model = model_config.load_model()
    model_config.load_model = lambda device_map=None: model
    model_config.load_tokenizer = lambda: None
    quantizer = GPTQ(wbits=4)
    qep_arch.run_quantize_with_qep_arch(
        model_config=model_config,
        quantizer=quantizer,
        qep_config=QEPConfig(device="cpu"),
        calibration_config=None,
        report_progress=False,
    )
    return model, quantizer


def test_qep_quantizes_all_layers_and_releases_nvfp4_weights(checkpoint_dir):
    model, quantizer = _run_qep(checkpoint_dir)
    linear_names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name != "lm_head"
    }
    assert set(quantizer.results) == linear_names
    assert "model.layers.3.mixer.experts.7.down_proj" in quantizer.results
    assert "model.layers.0.mixer.in_proj" in quantizer.results
    nvfp4 = [m for m in model.modules() if isinstance(m, NVFP4Linear)]
    assert len(nvfp4) == 2 * 8 * 2
    assert not any(m.is_materialized for m in nvfp4)


def test_chunked_expert_hessians_match_single_pass(monkeypatch, checkpoint_dir):
    _, expected = _run_qep(checkpoint_dir)

    def one_expert_per_chunk(expert_modules, module_to_name, device):
        chunks = {}
        for module in expert_modules:
            chunks.setdefault(module_to_name[module].rpartition(".")[0], []).append(module)
        return list(chunks.values())

    monkeypatch.setattr(qep_arch, "_chunk_expert_modules", one_expert_per_chunk)
    _, actual = _run_qep(checkpoint_dir)

    assert set(actual.results) == set(expected.results)
    for name, result in expected.results.items():
        assert torch.equal(actual.results[name].qweight, result.qweight), name
        assert torch.equal(actual.results[name].scales, result.scales), name


def test_chunk_expert_modules_respects_budget_and_keeps_experts_together(monkeypatch):
    modules, names = [], {}
    for e in range(4):
        up, down = nn.Linear(16, 8, bias=False), nn.Linear(8, 16, bias=False)
        modules += [up, down]
        names[up] = f"layers.1.mixer.experts.{e}.up_proj"
        names[down] = f"layers.1.mixer.experts.{e}.down_proj"

    assert qep_arch._chunk_expert_modules(modules, names, "cpu") == [modules]

    per_expert = 4 * (16**2 + 8**2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (4 * per_expert, 0))
    chunks = qep_arch._chunk_expert_modules(modules, names, "cuda:0")
    assert chunks == [modules[:4], modules[4:]]


def test_runner_quantized_model_keeps_nvfp4_experts_quantized(checkpoint_dir):
    model_config = ModelConfig(path=str(checkpoint_dir), device="cpu")
    model_config.load_tokenizer = lambda: None
    runner = Runner(model_config=model_config, quantizer=GPTQ(wbits=4), qep=True)
    runner.run()

    quantized, _ = runner.create_quantized_model()
    experts = quantized.model.layers[1].mixer.experts
    assert isinstance(experts[0].up_proj, GPTQLinear)
    assert isinstance(experts[0].down_proj, GPTQLinear)
    assert not any(isinstance(m, NVFP4Linear) for m in quantized.modules())
    in_proj = quantized.model.layers[0].mixer.in_proj
    assert isinstance(in_proj, GPTQLinear)
    assert in_proj.weight.numel() == 0 and in_proj.weight.dtype == torch.bfloat16
    assert "model.layers.0.mixer.in_proj.weight" not in quantized.state_dict()

    dequantized = model_config.load_model()
    runner.update_model_weights(dequantized)
    input_ids = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(
            quantized(input_ids).logits, dequantized(input_ids).logits, rtol=0.05, atol=0.05
        )


def test_runner_evaluates_models_larger_than_the_gpu_block_by_block(monkeypatch, checkpoint_dir):
    model_config = ModelConfig(path=str(checkpoint_dir), device="cpu")
    model_config.load_tokenizer = lambda: None
    runner = Runner(model_config=model_config, quantizer=GPTQ(wbits=4), qep=True)
    runner.run()

    # Pretend to evaluate on a CUDA device too small for the model.
    model_config.device = "cuda:0"
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=1)
    )
    calls = []

    def fake_offloaded(model, tokenizer, device, **kwargs):
        calls.append((model, device, kwargs))
        return float(len(calls))

    def fail(**_kwargs):
        raise AssertionError("the whole model must not be evaluated on the device")

    monkeypatch.setattr(runner_module, "calc_perplexity_offloaded", fake_offloaded)
    monkeypatch.setattr(runner_module, "calc_perplexity", fail)

    ppl = runner.calculate_perplexity(original_model=True, quantized_model=True, max_samples=4)
    assert ppl == (1.0, None, 2.0)
    (original, device, kwargs), (quantized, _, _) = calls
    assert device == torch.device("cuda:0") and kwargs["max_samples"] == 4
    assert any(isinstance(m, NVFP4Linear) for m in original.modules())
    assert isinstance(quantized.model.layers[1].mixer.experts[0].up_proj, GPTQLinear)
    assert all(p.device.type == "cpu" for p in quantized.parameters())
