"""CPU save/load inference checks for packed Nemotron-H checkpoints."""

import json
from logging import getLogger
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, NemotronHConfig

from onecomp.quantized_model_loader import QuantizedModelLoader
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
from onecomp.utils.unfuse_moe import unfuse_moe_experts


def _config():
    return NemotronHConfig(
        vocab_size=32,
        hidden_size=32,
        layers_block_type=["mamba", "moe", "attention"],
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        mamba_num_heads=32,
        mamba_head_dim=2,
        n_groups=2,
        ssm_state_size=8,
        chunk_size=4,
        conv_kernel=4,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        moe_latent_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        use_mamba_kernels=False,
        attention_dropout=0.25,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


@pytest.fixture
def saved_model(tmp_path):
    torch.manual_seed(11)
    model = AutoModelForCausalLM.from_config(_config(), dtype=torch.bfloat16)
    assert unfuse_moe_experts(model, getLogger(__name__))
    names = []
    bits_table = [{} for _ in model.model.layers]
    for name, linear in list(model.named_modules()):
        if not isinstance(linear, torch.nn.Linear) or name == "lm_head":
            continue
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-4) / 7
        integers = (weight / scale).round().add(8).clamp(0, 15).to(torch.int32)
        replacement = GPTQLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            wbits=4,
            groupsize=-1,
            actorder=False,
            quantized_weight=integers,
            scale=scale,
            zero=torch.full_like(scale, 8),
            bias=linear.bias,
            device="cpu",
            use_gemlite=False,
        )
        QuantizedModelLoader._set_module_by_name(model, name, replacement)
        names.append(name)
        index, suffix = name.split(".layers.", 1)[1].split(".", 1)
        bits_table[int(index)][suffix] = {"bits": 4, "method": "gptq"}
    model.config.quantization_config = {
        "quant_method": "mixed_gptq",
        "bits": 4,
        "group_size": -1,
        "checkpoint_format": "gptq",
        "modules_in_block_to_quantize": names,
        "quantization_bits": bits_table,
    }
    model.eval()
    model.config.save_pretrained(tmp_path)
    model.generation_config.save_pretrained(tmp_path)
    state = {name: value.detach().contiguous() for name, value in model.state_dict().items()}
    weight_map = {}
    keys = list(state)
    for part in range(2):
        shard = {key: state[key] for key in keys[part::2]}
        filename = f"model-{part + 1:05d}-of-00002.safetensors"
        save_file(shard, tmp_path / filename)
        weight_map.update({key: filename for key in shard})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return model, tmp_path


def _load(path, monkeypatch, device_map="cpu"):
    monkeypatch.setattr(
        "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained", lambda *a, **kw: object()
    )
    return QuantizedModelLoader.load_quantized_model(str(path), device_map=device_map)[0]


@pytest.mark.parametrize("use_cache", [False, True])
def test_hybrid_packed_checkpoint_reloads_logits_and_generation(
    saved_model, monkeypatch, use_cache
):
    reference, path = saved_model
    model = _load(path, monkeypatch)
    assert not model.training
    assert all(not module.training for module in model.modules())
    assert all(not tensor.is_meta for tensor in model.state_dict().values())
    original = reference.state_dict()
    restored = model.state_dict()
    assert original.keys() == restored.keys()
    for name in original:
        torch.testing.assert_close(restored[name], original[name], rtol=0, atol=0)
    assert isinstance(model.model.layers[1].mixer.experts[0].up_proj, GPTQLinear)
    assert model.model.layers[1].mixer.experts[0].up_proj._weight_is_packed

    ids = torch.tensor([[1, 3, 4, 5]])
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        expected = reference(ids, attention_mask=mask, use_cache=False).logits
        actual = model(ids, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        kwargs = dict(
            attention_mask=mask,
            max_new_tokens=3,
            do_sample=False,
            use_cache=use_cache,
            eos_token_id=None,
        )
        expected_tokens = reference.generate(ids, **kwargs)
        actual_tokens = model.generate(ids, **kwargs)
    assert torch.equal(actual_tokens, expected_tokens)
    assert actual_tokens.shape == (1, 7)


def test_nemotron_skeleton_has_no_dense_parameter_storage():
    model = QuantizedModelLoader._build_empty_model_from_config(
        _config().to_dict(), torch.bfloat16
    )
    assert all(parameter.is_meta for parameter in model.parameters())
    assert all(not buffer.is_meta for buffer in model.buffers())


@pytest.mark.parametrize(
    "missing_key",
    ["model.layers.0.mixer.dt_bias", "model.layers.1.mixer.gate.e_score_correction_bias"],
)
def test_incomplete_mamba_or_router_checkpoint_fails(saved_model, monkeypatch, missing_key):
    _, path = saved_model
    state = QuantizedModelLoader._load_state_dict_from_dir(str(path))
    assert missing_key in state
    del state[missing_key]
    monkeypatch.setattr(QuantizedModelLoader, "_load_state_dict_from_dir", lambda _: state)
    with pytest.raises(RuntimeError, match="Incomplete Nemotron-H checkpoint"):
        _load(path, monkeypatch)


def test_index_ignores_unreferenced_safetensors(saved_model):
    reference, path = saved_model
    save_file({"stale.weight": torch.ones(2)}, path / "stale.safetensors")
    assert (
        QuantizedModelLoader._load_state_dict_from_dir(str(path)).keys()
        == reference.state_dict().keys()
    )


def test_missing_indexed_tensor_fails(saved_model):
    _, path = saved_model
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["missing.weight"] = "model-00001-of-00002.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="Missing indexed tensors"):
        QuantizedModelLoader._load_state_dict_from_dir(str(path))


def test_device_map_respects_explicit_mapping(monkeypatch):
    model = torch.nn.Linear(8, 8)
    requested = {"": "cpu"}
    infer = Mock(side_effect=AssertionError("An explicit device map must not be inferred"))
    dispatch = Mock(return_value=model)
    monkeypatch.setattr("accelerate.infer_auto_device_map", infer)
    monkeypatch.setattr("accelerate.dispatch_model", dispatch)
    assert QuantizedModelLoader._place_model(model, requested) is model
    dispatch.assert_called_once_with(model, device_map=requested)


def test_auto_device_map_keeps_hybrid_blocks_together(monkeypatch):
    model = torch.nn.Linear(8, 8)
    model._no_split_modules = ["NemotronHBlock"]
    infer = Mock(return_value={"": "cpu"})
    dispatch = Mock(return_value=model)
    monkeypatch.setattr("accelerate.infer_auto_device_map", infer)
    monkeypatch.setattr("accelerate.dispatch_model", dispatch)
    assert QuantizedModelLoader._place_model(model, "auto") is model
    infer.assert_called_once_with(model, no_split_module_classes=["NemotronHBlock"])


def test_gptq_replacement_reuses_checkpoint_storage(saved_model):
    reference, path = saved_model
    config = reference.config.quantization_config
    model = QuantizedModelLoader._build_empty_model_from_config(reference.config.to_dict())
    assert unfuse_moe_experts(model, getLogger(__name__))
    state = QuantizedModelLoader._load_state_dict_from_dir(str(path))
    QuantizedModelLoader._replace_quantized_layers(model, state, config)
    for name, module in model.named_modules():
        if isinstance(module, GPTQLinear):
            assert module.qweight.data_ptr() == state[f"{name}.qweight"].data_ptr()
