"""Build a tiny NemotronH checkpoint in NVIDIA ModelOpt MIXED_PRECISION format.

Mirrors the layout of ``nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4``:
routed experts are NVFP4 (per expert, ``backbone.layers.{i}.mixer.experts.{e}``),
Mamba ``in_proj``/``out_proj`` and shared experts are FP8, everything else is
BF16, and MTP weights are present but unused by transformers.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP8_MAX = 448.0


def tiny_nemotron_h_config(vocab_size: int = 256):
    from transformers import NemotronHConfig

    return NemotronHConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        layers_block_type=["mamba", "moe", "attention", "moe"],
        mtp_layers_block_type=["attention", "moe"],
        num_nextn_predict_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        mamba_num_heads=32,
        mamba_head_dim=4,
        n_groups=2,
        ssm_state_size=16,
        chunk_size=16,
        conv_kernel=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=64,
        moe_latent_size=32,
        mlp_hidden_act="relu2",
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.0,
        use_mamba_kernels=False,
        tie_word_embeddings=False,
    )


def quantize_nvfp4(weight: torch.Tensor, block_size: int = 16):
    """Quantize ``(out, in)`` to ModelOpt NVFP4 (packed uint8, fp8 scale, fp32 scale_2)."""
    w = weight.float()
    out_features, in_features = w.shape
    blocks = w.view(out_features, in_features // block_size, block_size)
    block_amax = blocks.abs().amax(dim=-1)
    scale_2 = (block_amax.max() / (6.0 * _FP8_MAX)).clamp(min=1e-12)
    scale = (block_amax / 6.0 / scale_2).clamp(min=2**-9).to(torch.float8_e4m3fn)
    scaled = blocks / (scale.float() * scale_2).unsqueeze(-1)
    codes = (scaled.abs().unsqueeze(-1) - _E2M1).abs().argmin(dim=-1)
    codes = codes | ((scaled < 0).to(torch.long) << 3)
    codes = codes.view(out_features, in_features).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), scale, scale_2.reshape(())


def quantize_fp8(weight: torch.Tensor):
    """Quantize to FP8-E4M3 with a per-tensor scale."""
    scale = (weight.float().abs().max() / _FP8_MAX).clamp(min=1e-12)
    return (weight.float() / scale).to(torch.float8_e4m3fn), scale.reshape(())


def _hf_key_to_checkpoint(key: str) -> str:
    return "backbone." + key[len("model.") :] if key.startswith("model.") else key


def build_modelopt_nemotron_h_checkpoint(
    path, tokenizer_dir=None, seed: int = 0, vocab_size: int = 256
) -> Path:
    """Write the tiny checkpoint to *path* and return it.

    Args:
        path: Output directory.
        tokenizer_dir: Optional directory whose tokenizer files are copied.
        seed: Seed for the random weights.
        vocab_size: Vocabulary size (must cover the tokenizer if one is copied).
    """
    from transformers import AutoModelForCausalLM

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = tiny_nemotron_h_config(vocab_size)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("A_log") or name.endswith(".D") or name.endswith("dt_bias"):
                continue
            if "norm" in name:
                continue
            param.normal_(0.0, 0.05)
        for module in model.modules():
            if hasattr(module, "e_score_correction_bias"):
                module.e_score_correction_bias.normal_(0.0, 0.01)

    tensors: dict[str, torch.Tensor] = {}
    quantized_layers: dict[str, dict] = {}
    for key, value in model.state_dict().items():
        ckpt_key = _hf_key_to_checkpoint(key)
        module_name = ckpt_key.rpartition(".")[0]
        if key.endswith("mixer.experts.up_proj") or key.endswith("mixer.experts.down_proj"):
            proj = key.rpartition(".")[2]
            prefix = ckpt_key.rpartition(".")[0]
            for e in range(value.shape[0]):
                packed, scale, scale_2 = quantize_nvfp4(value[e])
                name = f"{prefix}.{e}.{proj}"
                tensors[f"{name}.weight"] = packed
                tensors[f"{name}.weight_scale"] = scale
                tensors[f"{name}.weight_scale_2"] = scale_2
                tensors[f"{name}.input_scale"] = torch.tensor(1.0)
                quantized_layers[name] = {"quant_algo": "NVFP4", "group_size": 16}
        elif key.endswith(".weight") and any(
            s in key for s in ("mixer.in_proj", "mixer.out_proj", "shared_experts.")
        ):
            weight, scale = quantize_fp8(value)
            tensors[ckpt_key] = weight
            tensors[f"{module_name}.weight_scale"] = scale
            tensors[f"{module_name}.input_scale"] = torch.tensor([1.0])
            quantized_layers[module_name] = {"quant_algo": "FP8"}
        elif key.endswith("mixer.gate.weight"):
            tensors[ckpt_key] = value.float().contiguous()
        else:
            tensors[ckpt_key] = value.contiguous()
    for layer in model.model.layers:
        if layer.block_type == "full_attention":
            prefix = f"backbone.layers.{layer.layer_idx}.mixer"
            tensors[f"{prefix}.k_proj.k_scale"] = torch.tensor([1.0])
            tensors[f"{prefix}.v_proj.v_scale"] = torch.tensor([1.0])
    tensors["mtp.layers.0.mixer.q_proj.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)

    # Split into two shards so that some scales live in a different shard.
    keys = sorted(tensors)
    half = len(keys) // 2
    shards = {"model-00001-of-00002.safetensors": keys[:half]}
    shards["model-00002-of-00002.safetensors"] = keys[half:]
    weight_map = {}
    for shard, shard_keys in shards.items():
        save_file({k: tensors[k] for k in shard_keys}, str(path / shard))
        weight_map.update({k: shard for k in shard_keys})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    with open(path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f)

    config_dict = config.to_dict()
    config_dict["architectures"] = ["NemotronHForCausalLM"]
    config_dict["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": "FP8",
        "producer": {"name": "modelopt", "version": "1.0.0"},
        "quantized_layers": quantized_layers,
    }
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f)

    if tokenizer_dir is not None:
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = Path(tokenizer_dir) / name
            if src.exists():
                (path / name).write_bytes(src.read_bytes())
    return path


def reference_model(path):
    """transformers NemotronH model holding the dequantized checkpoint weights."""
    from transformers import AutoModelForCausalLM

    from onecomp.utils.modelopt_checkpoint import dequantize_fp8, dequantize_nvfp4

    path = Path(path)
    with open(path / "config.json", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict.pop("quantization_config")
    from transformers import NemotronHConfig

    config = NemotronHConfig(**config_dict)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)

    from safetensors import safe_open

    tensors = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            tensors.update({k: f.get_tensor(k) for k in f.keys()})

    state = model.state_dict()
    with torch.no_grad():
        for key in state:
            ckpt_key = _hf_key_to_checkpoint(key)
            if key.endswith("mixer.experts.up_proj") or key.endswith("mixer.experts.down_proj"):
                prefix, _, proj = ckpt_key.rpartition(".")
                for e in range(state[key].shape[0]):
                    name = f"{prefix}.{e}.{proj}"
                    state[key][e].copy_(
                        dequantize_nvfp4(
                            tensors[f"{name}.weight"],
                            tensors[f"{name}.weight_scale"],
                            tensors[f"{name}.weight_scale_2"],
                        )
                    )
            elif f"{ckpt_key.rpartition('.')[0]}.weight_scale" in tensors and key.endswith(
                ".weight"
            ):
                module_name = ckpt_key.rpartition(".")[0]
                state[key].copy_(
                    dequantize_fp8(
                        tensors[ckpt_key],
                        tensors[f"{module_name}.weight_scale"],
                        torch.bfloat16,
                    )
                )
            else:
                state[key].copy_(tensors[ckpt_key])
    model.eval()
    return model
