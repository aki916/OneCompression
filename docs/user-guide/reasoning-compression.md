# Reasoning Compression (Experimental)

`onecomp.reasoning` shortens the *thinking* phase of reasoning models such as
Qwen3 at inference time. It is training-free and opt-in: with the default
`ReasoningConfig()` (`policy="none"`), generation is token-for-token identical
to plain greedy decoding. It is independent of weight quantization and works on
original or OneComp-quantized models.

!!! warning "Research feature"
    The policies are reference implementations of published methods plus one
    experimental cascade. See [Methods and paper deviations](../reasoning-compression/papers.md)
    before comparing numbers with the papers, and [Results](../reasoning-compression/results.md)
    for what has actually been measured.

## Policies

| `policy` | Signal at each boundary | Extra model compute | Backends | Source |
|---|---|---|---|---|
| `none` | – | none | Transformers, vLLM | vanilla |
| `fixed` | reasoning-token count ≥ `budget` | none | Transformers, vLLM | budget baseline |
| `prompt_budget` | per-prompt budget from a length/operator heuristic | none | Transformers, vLLM | TALE-like heuristic (not TALE) |
| `thinkbrake` | logit gap between best non-end token and `</think>` | none | Transformers, vLLM | ThinkBrake |
| `deer` | geometric-mean confidence of a trial answer | answer probe | Transformers | DEER |
| `stability` | identical trial answers in a window | answer probe | Transformers | Dynasor-inspired |
| `asag_reference` | arithmetic confidence + attention-entropy change | answer probe + eager attention | Transformers | ASAG (reference interpretation) |
| `progressive` | logits → probe → attention, escalating only when uncertain | lazy | Transformers | OneComp experimental |

When a policy stops, the backend emits the model's own `</think>` marker through
normal decoding and the model then writes its final answer. Probes run on a
temporary branch of the KV cache: the main trajectory is restored exactly, and
every probe token the model evaluates is counted in `model_evaluated_tokens`.

## Boundaries

`boundary` selects where adaptive policies look (budgets ignore it):

- `sentence`: after a token ending in `.`, `!`, `?` or a blank line (token-level approximation).
- `transition`: before the model emits `Wait` / `Alternatively` (DEER/ASAG transition points).
- `tokens`: every `interval` reasoning tokens.

## Python API (Transformers reference)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from onecomp.reasoning import ReasoningConfig, ReasoningFormat
from onecomp.runtime.transformers import TransformersReasoningAdapter

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
adapter = TransformersReasoningAdapter(model, tokenizer)

prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 17 * 24? Put the answer in \\boxed{}."}],
    tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False,
)
result = adapter.generate(
    prompt_ids,
    reasoning_format=ReasoningFormat.from_tokenizer(tokenizer),  # <think> ... </think>
    config=ReasoningConfig(policy="deer", confidence_threshold=0.95, boundary="transition"),
    max_new_tokens=16000,
)
print(result.final_answer, result.metrics["reasoning_tokens"], result.metrics["latency_s"])
```

The reference backend decodes greedily with batch size one. `asag_reference`
(and `progressive` with `asag_mode="full"`) needs the model loaded with
`attn_implementation="eager"`. Pass `probe_prefix_reuse=False` to the adapter
to re-prefill every probe instead of branching the cache (the conservative
replay-cost reference).

## vLLM

Register the logits processor and attach per-request metadata:

```python
from vllm import LLM, SamplingParams
from onecomp.reasoning import ReasoningConfig, ReasoningFormat
from onecomp.runtime.vllm import VLLMReasoningAdapter

llm = LLM(model="Qwen/Qwen3-8B", logits_processors=[VLLMReasoningAdapter.processor_path])
fmt = ReasoningFormat.from_tokenizer(llm.get_tokenizer())
params = SamplingParams(
    temperature=0,
    max_tokens=16000,
    extra_args=VLLMReasoningAdapter.extra_args(ReasoningConfig(policy="thinkbrake"), fmt, "req-0"),
)
```

Requests without the metadata are untouched, and each request in a batch keeps
its own state. Only `none`, `fixed`, `prompt_budget` and `thinkbrake` are
supported; probe and attention policies raise `NotImplementedError`. The
processor supports vLLM 0.21.x only (other versions raise `RuntimeError` at
construction, although the `vllm` extra allows `>=0.10,<0.22`) and rejects
speculative decoding and `n > 1`.

## Configuration

| Field | Default | Used by | Meaning |
|---|---|---|---|
| `policy` | `"none"` | all | One of the policies above |
| `budget` | `2048` | fixed | Reasoning-token budget |
| `boundary` | `"sentence"` | adaptive | `sentence`, `transition` or `tokens` |
| `interval` | `128` | `boundary="tokens"` | Check every N reasoning tokens |
| `min_reasoning_tokens` | `0` | adaptive | No adaptive check before this many reasoning tokens |
| `threshold` | `0.1` | thinkbrake, progressive | Stop when the logit gap ≤ threshold |
| `temperature` | `1.0` | thinkbrake, progressive | Divides the logit gap (ThinkBrake uses the sampling temperature, 0.6 for Qwen3) |
| `confidence_threshold` | `0.95` | deer, asag, progressive | Stop when probe confidence > threshold |
| `probe_max_tokens` | `128` | probe policies | Trial-answer length limit; truncated probes are never confident |
| `relative_confidence` | `False` | deer | Non-paper variant: also stop on a new confidence maximum |
| `stability_window` | `3` | stability | Number of identical consecutive answers |
| `entropy_threshold` | `-0.1` | asag, progressive | Convergence: relative attention-entropy change below this |
| `asag_mode` | `"full"` | asag, progressive | `full`, `confidence`, `attention`, `both` (ablations) |
| `enable_injection` / `enable_trap` | `True` | asag | Probability-mixture and jump actions |
| `max_jumps` | `1` | asag | Jump prompts before a forced stop |
| `cascade_uncertainty` | `2.0` | progressive | Probe only when threshold < logit gap ≤ this |

Thresholds are fractions, never percentages.

## Benchmark CLI

`onecomp-reasoning` runs paired, reproducible experiments (model load and
warm-up are excluded from request latency):

```bash
# Freeze a dataset revision into disjoint development/calibration/evaluation splits
onecomp-reasoning prepare-data math500 --revision <dataset-commit-sha> \
    --output experiments/reasoning_compression/data/math500.jsonl

# Run every policy in the config on the same samples (writes manifest/results/status per policy)
onecomp-reasoning run --config experiments/reasoning_compression/configs/qwen3_bf16_smoke.json \
    --output experiments/reasoning_compression/results/qwen3-smoke

# Compare run directories (one policy=none run is required): summary.json + pareto.svg
onecomp-reasoning analyze experiments/reasoning_compression/results/qwen3-smoke/0* --output-dir analysis/

# Pick the cheapest measured policy within 0.5 accuracy points on the calibration split
onecomp-reasoning calibrate results/calibration/0* --max-accuracy-drop 0.5 --output selected.json
```

`oracle` forces an answer at every candidate boundary of the vanilla trajectory
to label safe stopping points offline, and `environment` prints the recorded
runtime versions. Configuration files and the full staged workflow are in
[`experiments/reasoning_compression/`](https://github.com/FujitsuResearch/OneCompression/blob/main/experiments/reasoning_compression/README.md).

## Examples

- [`example/example_reasoning_compression.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/example_reasoning_compression.py) -- Transformers reference: none / fixed / ThinkBrake / DEER on one question
- [`example/vllm_inference/example_reasoning_vllm_inference.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/vllm_inference/example_reasoning_vllm_inference.py) -- vLLM batch with per-request policies

## Further reading

- [Design](../reasoning-compression/design.md): package structure and contracts
- [Methods and paper deviations](../reasoning-compression/papers.md)
- [Runtime and vLLM integration](../reasoning-compression/runtime.md)
- [Results](../reasoning-compression/results.md)
