# QEP (Quantization Error Propagation)

QEP is a meta-algorithm that improves any layer-wise quantization method by compensating for
the error that propagates from previously quantized layers to subsequent ones.

!!! abstract "Reference"
    Yamato Arai and Yuma Ichikawa, "Quantization Error Propagation: Revisiting Layer-Wise
    Post-Training Quantization," NeurIPS 2025.
    [OpenReview](https://openreview.net/forum?id=a3l3K9khbL) |
    [Original implementation](https://github.com/FujitsuResearch/qep)

## Motivation

Standard layer-wise PTQ quantizes each layer independently using the **original** input
activations. However, after quantizing layer \(l\), the input to layer \(l+1\) is no longer
the original activation -- it is the output of the quantized layer \(l\), which contains
quantization error. This accumulated error degrades quantization quality, especially at low
bit-widths.

## How QEP Works

QEP addresses this by adjusting the weights of each layer **before** quantization to account
for the activation error introduced by previously quantized layers.

For a layer with weight \(W\), original input activations \(X\), and quantized-model input
activations \(\hat{X}\):

1. Compute the activation difference: \(\Delta = X - \hat{X}\)
2. Compute the cross-term: \(\Delta^T \hat{X}\)
3. Solve for a weight correction \(\Delta W\) via the Hessian:

\[
\Delta W = \alpha \cdot (\Delta^T \hat{X}) \cdot H^{-1}
\]

where \(H = \hat{X}^T \hat{X}\) is the Hessian matrix and \(\alpha\) is the correction
strength (`perccorr`).

4. Quantize the adjusted weight \(W + \Delta W\) using the base quantizer (e.g., GPTQ).

## Two Implementations

OneComp provides two QEP implementations, controlled by the `QEPConfig.general` parameter:

### Architecture-aware (default, `general=False`)

- Exploits the structure of transformer blocks (e.g., QKV layers sharing the same input)
- Groups layers that share input activations for efficient Hessian computation
- Processes one transformer block at a time to minimize GPU memory usage
- **Recommended** for Llama-like architectures

### Generic (`general=True`)

- Architecture-independent implementation
- Captures input activations for each layer individually
- Works with any model architecture
- Higher memory consumption and more forward passes

## Usage

### Basic QEP

```python
from onecomp import ModelConfig, Runner, GPTQ

model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf", device="cuda:0")
gptq = GPTQ(wbits=3)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
)
runner.run()
```

### Custom QEP Configuration

```python
from onecomp import QEPConfig

qep_config = QEPConfig(
    general=False,              # Architecture-aware (default)
    percdamp=0.01,              # Hessian damping
    perccorr=0.5,               # Correction strength
    exclude_layer_keywords=["mlp.down_proj"],
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
    qep_config=qep_config,
)
runner.run()
```

### Generic QEP (for non-Llama architectures)

```python
qep_config = QEPConfig(general=True)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
    qep_config=qep_config,
)
runner.run()
```

### Nemotron-H ModelOpt NVFP4/FP8 checkpoints

The architecture-aware path supports local NVIDIA Nemotron-H ModelOpt checkpoints,
including [Nemotron 3 Ultra](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard).
Use a Transformers version with native `NemotronHForCausalLM` support (tested with
5.10.2 and 5.14.1). NVFP4 expert weights and FP8 projections are dequantized to BF16
one block at a time, then quantized using GPTQ. The router remains unquantized;
auxiliary MTP heads are excluded. Routed experts use their own GPTQ Hessians without
QEP cross-terms, since routing may differ between original and quantized activations.
Experts receiving no calibration tokens use RTN.

```bash
python example/example_qep_gptq.py \
  --model-id /mnt/data/yoshida/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --device cuda:0 --batch-size 1 \
  --output nemotron_qep_gptq
```

`--calibration-dataset` accepts a local text or JSONL file as well as the default C4.
For a short smoke test, also specify `--num-calibration-samples 1 --max-length 32
--include-layer-keywords layers.1.mixer.experts.0. --num-layers 2`.
The full run processes all backbone Linear layers except the router and LM head.

The example packs each result immediately, and limits simultaneous expert Hessians
to 1 GiB. `QEPConfig.batch_size` controls activation memory;
`QEPConfig.expert_hessian_max_bytes` controls the expert Hessian budget (at least
one matrix is processed). GPU memory must still accommodate two BF16 copies of
one block, activations, and GPTQ working space. The packed 3-bit results alone
require approximately 206 GB of host memory and disk for 550B parameters.
Fast Mamba kernels reduce runtime and activation memory; the PyTorch fallback
uses substantially more memory at long sequence lengths.

The output is a complete safetensors checkpoint with `config.json`, generation
settings, and tokenizer files. `Runner.save_quantized_model()` writes bounded-size
shards directly from the source and quantization results; it does not construct
a full BF16 model. Non-quantized weights are included, and the MoE experts remain
packed. `max_shard_size="5GB"` controls the shard buffer. MTP heads are excluded.
Original ModelOpt quantization sidecars are not copied into the GPTQ checkpoint.

Use OneCompression's loader for inference:

```python
from onecomp import load_quantized_model

model, tokenizer = load_quantized_model("./nemotron_qep_gptq", device_map="auto")
inputs = tokenizer("The capital of Japan is", return_tensors="pt")
inputs = {key: value.to(model.get_input_embeddings().weight.device)
          for key, value in inputs.items()}
output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The example's `--verify-load` option performs this reload-and-generation check after
saving; `--device cpu` keeps inference on CPU, otherwise inference uses automatic
device placement. Loading constructs a meta model and assigns the saved tensors,
without allocating a full dense BF16 model. The packed model and inference working
memory must still fit across available devices/host memory. This path is tested
with the OneCompression loader; Nemotron serving through vLLM is not established
by these tests.

Whole-model perplexity is still skipped for the layerwise ModelOpt path because
its original-model evaluator materializes the source model. Use the default
`general=False`; generic QEP does not use the lazy source loader.

## Parameters

| Parameter                 | Type        | Description                                      | Default              |
|---------------------------|-------------|--------------------------------------------------|----------------------|
| `batch_size`              | `int`       | Forward batch size for architecture-aware QEP   | `16`                 |
| `expert_hessian_max_bytes` | `int`       | Simultaneous expert Hessian budget in bytes      | `1073741824`         |
| `general`                 | `bool`      | Use generic (architecture-independent) QEP       | `False`              |
| `percdamp`                | `float`     | Damping percentage for Hessian regularization     | `0.01`               |
| `perccorr`                | `float`     | Correction strength (0 = no correction, 1 = full)| `0.5`                |
| `device`                  | `str`       | GPU device for QEP computation                    | `None`           |
| `exclude_layer_keywords`  | `list[str]` | Layer keywords excluded from error propagation    | `["mlp.down_proj"]`  |

!!! note
    The default `exclude_layer_keywords` is designed for Llama-like architectures. You may need
    to adjust this for other model families.
