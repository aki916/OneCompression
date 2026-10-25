# Reasoning compression experiments

Use the repository virtual environment. Runtime compression is disabled by default;
these commands are separate from weight quantization. Read the
[user guide](../../docs/user-guide/reasoning-compression.md),
[design](../../docs/reasoning-compression/design.md),
and [paper deviations](../../docs/reasoning-compression/papers.md) before
interpreting results.

## CPU validation

```bash
.venv/bin/python -m pytest tests/onecomp/reasoning tests/onecomp/runtime
.venv/bin/python -m onecomp.reasoning environment
.venv/bin/python experiments/reasoning_compression/smoke.py --help
MPLCONFIGDIR=/tmp/onecomp-matplotlib .venv/bin/python \
  experiments/reasoning_compression/plot_smoke.py experiments/reasoning_compression/cpu_smoke
```

The CPU smoke uses a tiny random Qwen3 backbone with scripted logits and artificial
labels. It validates mechanics and measures this fixture's execution cost. It is
not a Qwen3-8B quality, literature reproduction, or GPU-performance benchmark.

## Data

Freeze an immutable Hugging Face dataset commit SHA; no silent revision defaults:

```bash
.venv/bin/python -m onecomp.reasoning prepare-data math500 \
  --revision DATASET_COMMIT_SHA \
  --cache-dir /tmp/onecomp-datasets \
  --output experiments/reasoning_compression/data/math500.jsonl
```

Supported imports: [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500),
[AMC23](https://huggingface.co/datasets/math-ai/amc23),
[AIME 2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024), and
[GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa). GPQA access is
subject to its upstream access requirements. AMC23 means the 40-row math-ai subset;
do not substitute the 83-row mixed-year dataset. GPQA choices are shuffled by a
stable per-question seed. Dataset import freezes disjoint development/calibration/
evaluation roles (20/20/60%). AIME's 30 questions cannot support a 50–100 example
development subset; report actual counts. A whole-dataset paper reproduction must
be a separate protocol and must not serve as its own threshold calibration set.

Canonical JSONL fields: `sample_id`, `dataset`, `dataset_revision`, `split`,
`question`, `answer`, optional `difficulty`. Model prompts never include answers.
The supplied grader supports balanced boxed output, exact text and rational numeric
equivalence. It is intentionally incomplete for symbolic MATH answers; audit grading
errors or supply a validated symbolic grader before claiming comparable MATH scores.

## Staged GPU runs

Start with the 12-example config; inspect JSONL before larger sweeps:

```bash
.venv/bin/python -m onecomp.reasoning run \
  --config experiments/reasoning_compression/configs/qwen3_bf16_smoke.json \
  --output experiments/reasoning_compression/results/qwen3-smoke
.venv/bin/python -m onecomp.reasoning oracle \
  --config experiments/reasoning_compression/configs/qwen3_oracle.json \
  --output experiments/reasoning_compression/results/qwen3-oracle
```

`qwen3_reference_ablation.json` contains fixed budgets, threshold sweeps, ASAG
components and cascade ablations; `qwen3_boundary_ablation.json` varies checkpoint
selection. All methods in a reference sweep share one model/attention implementation.
Full eager probe attention can use quadratic memory. Oracle trials are separate
independent re-prefills, include failed stops, and log replay costs. The default
oracle candidate cap makes the search censored; remove it only when compute permits.

Use `qwen3_vllm_calibration.json` for cheap policies on the inspected vLLM 0.21 API.
The continuous-batch logits processor supports mixed policies, slot swaps and
different stop times. This first benchmark driver submits serial requests; its
latency is measured, but it does not measure concurrent serving throughput. Worker
signal traces, scheduler token-evaluation counts and peak KV usage are unavailable
through RequestOutput and remain null. Probe/attention policies explicitly fail in
vLLM; no ASAG approximation is silently substituted.

Each run creates `manifest.json`, `results.jsonl`, and `status.json` in its own
variant directory. It records versions, source digest, resolved model revision,
tokenizer/chat template identity, GPU, decode settings, policy and dataset identity.
Existing variants are not overwritten. Analysis writes `summary.json` and
`pareto.svg`. Model load and warmup are outside request latency; probe re-prefills,
attention collection, controller, and interventions are included. Repeat runs and
vary order externally to assess timing variance; a single run is exploratory.

## Offline selection and held-out evaluation

```bash
.venv/bin/python -m onecomp.reasoning calibrate \
  experiments/reasoning_compression/results/calibration/0* \
  --max-accuracy-drop 0.5 \
  --output experiments/reasoning_compression/results/selected.json
.venv/bin/python -m onecomp.reasoning run \
  --config /path/to/held-out-evaluation-config.json \
  --reasoning-compression auto \
  --selected-config experiments/reasoning_compression/results/selected.json \
  --output experiments/reasoning_compression/results/selected-evaluation
```

`0.5` means 0.5 percentage points. Selection minimizes measured latency with an
empirical calibration accuracy constraint. It is not a held-out quality guarantee.
The evaluation config must use a held-out split, the same dataset revision/model/
prompt/decoding, and disjoint sample IDs. The Python selection API uses fractions.

For signal combinations, `fit_logistic_signal(calibration_rows, features)` and
`evaluate_signal_model(model, evaluation_rows)` reject overlapping question IDs.
Compare confidence; negative entropy; both; token count + confidence; and negative
logit gap + confidence with matched label coverage. Missing signals stay missing.
Confidence is calibrated against safe-stop correctness, never presumed to equal it.

## Weight compression interaction

Create an officially supported symmetric GPTQ 4-bit/group-128 checkpoint with
the explicit export recipe, using the quantizer already exercised by the
[Qwen3 GPTQ benchmark](../../benchmark/qwen3-8b-gptq/README.md):

```bash
.venv/bin/python experiments/reasoning_compression/quantize_gptq4.py \
  --model /path/to/pinned/Qwen3-8B-snapshot \
  --source-revision MODEL_COMMIT_SHA \
  --output artifacts/qwen3-8b-gptq4-g128
```

Point
`qwen3_onecomp_gptq4.json` at the exported checkpoint. BF16 and quantized runs each
need their own vanilla reference. This configuration uses the existing
`mixed_gptq` vLLM plugin, not an unrelated third-party 4-bit checkpoint.

Calibrate on BF16 and 4-bit separately; apply each frozen policy to disjoint 4-bit
evaluation IDs. Keep the BF16-transfer and 4-bit-recalibrated measurements separate.
For paired source identity set `source_model: "Qwen/Qwen3-8B"`,
`source_model_revision` to the original pinned model commit, and
`tokenizer_revision` to the identical tokenizer commit in the quantized run config.
The local packed checkpoint has its own separately recorded content SHA256.
`compare_quantization_runs` and `evaluate_threshold_transfer` require matching
source identities and paired examples. Set `allow_quantization_transfer: true`
only for the explicit BF16-selected configuration applied to a quantized model.
Quantization can change confidence and latency; never multiply token savings by a
weight-compression speedup. Reference packed-weight timing is distinct from vLLM
Marlin serving timing.

## Paper and deployment limits

ThinkBrake is a greedy logit-gap variant with documented sentence heuristics; DEER
uses the paper's geometric confidence but isolated re-prefill probes. Stability is
a sequential signal inspired by Dynasor, not its multi-path Certaindex scheduler.
ASAG is explicitly `asag_reference`: first-probe entropy baseline, last-four-layer
normalized entropy, probability mixing, and bounded jumps, with unresolved paper
ambiguities recorded. A validated qualitative literature trend remains a gate.
No production claim follows from unit tests or synthetic smoke timings.
