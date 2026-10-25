# Runtime reasoning compression design

## Existing architecture and extension point

`ModelConfig` loads Transformers models/tokenizers; `QuantizedModelLoader` reconstructs
OneComp packed GPTQ/DBF/OneBit checkpoints. `Runner` performs calibration and weight
compression. The `onecomp` argparse CLI is a quantization command. Existing
`onecomp-eval` uses Hydra, a vLLM HTTP server, and subprocess MT-Bench/throughput
evaluators. Neither evaluator exposes per-step model state. Quantization plugins
register through `vllm.general_plugins`; `mixed_gptq` already supports symmetric
4-bit GPTQ with group size 128 and Marlin dispatch. Qwen3-8B has an existing GPTQ
benchmark. Logging uses standard Python logging and JSON experiment artifacts.

Add independent `onecomp.reasoning` and `onecomp.runtime` packages, with a dedicated
`onecomp-reasoning` research CLI. Preserve existing quantization, evaluation, and
default inference paths. A runtime receives an already-loaded model, allowing the
same controller to operate on original and OneComp-compressed weights.

## Package layout

```text
onecomp/reasoning/
    config.py          ReasoningConfig (validated, frozen) and the prompt-budget heuristic
    state.py           ReasoningFormat (marker token IDs) and per-request RequestState
    controller.py      ReasoningCompressionController: boundary -> observation -> decision
    boundaries.py      sentence / transition / token-interval detectors
    actions.py         Decision and the stop / jump / probability-mixture actions
    signals/           logits.py (ThinkBrake gap), confidence.py (DEER/ASAG), attention.py (ASAG)
    policies/          one module per method; POLICIES registry and make_policy()
    data.py, experiment.py, vllm_experiment.py, cli.py    benchmark harness
    metrics/, calibration/                                offline analysis and selection
onecomp/runtime/
    transformers/      serial greedy reference backend (all policies)
    vllm/              vLLM 0.21 logits processor (budget and logit policies)
```

Each policy class declares what a backend must observe (`uses_logits`,
`uses_probe`, `requires_attention`, `probe_options()`, `needs_probe()`,
`needs_attention_followup()`); backends query these instead of matching policy
names. Package `__init__.py` files only re-export.

## Contracts

Separate model-specific reasoning formats, boundary detection, observed signals,
decision policies, and applied actions. A controller owns immutable configuration;
each request owns its counters, marker parser, candidate boundaries, signals,
probe history, interventions, and timing. Backends translate decisions into forced
marker tokens, logit changes, or isolated answer probes. A disabled policy is an
identity intervention. Fixed budgets do not wait for linguistic boundaries.

Use explicit token sequences for start/end markers, not substring assumptions.
Count reasoning, answer, and control-marker tokens separately. A forced close
continues final-answer decoding. Probes branch from the live KV cache and the
main trajectory's cache is restored bit-for-bit afterwards (or, for caches that
cannot be cropped, the probe replays its prefix in a new cache). Probe prefill,
decode, injected-token evaluation, and attention computation must be included in
model-token accounting; reused prefix tokens are reported separately.

Baseline order: none, fixed, heuristic prompt budget, cheap output signal,
confidence probes, stability, then attention reference and progressive cascade.
Paper identities/defaults/deviations are documented separately; an approximation
must have an explicit different name. Oracle labels are offline supervised
measurements and are never supplied to a deployed controller.

## Runtime dependencies

The project allows vLLM >=0.10,<0.22; the Linux Python 3.13 lock resolution and
installed runtime are 0.21.0. Installed Transformers is 5.14.1, torch 2.11.0+cu130.
Inspect the installed source, not an assumed historic per-request logits hook.
vLLM v1 uses `LogitsProcessor.apply`, `update_state(BatchUpdate)`, and
`is_argmax_invariant`; request slots can be removed, added and moved. Adapt this
interface explicitly, including slot swaps. A declared sampling parameter alone
does not establish working native thinking-budget support.

The Transformers reference uses eager attention only when internal statistics are
requested (or when `attention_implementation: eager` is configured so that all
policies of a sweep share one attention kernel), with real model forwards and
wall-clock timing. This is an algorithmic
reference, not a claim of production serving performance. vLLM supports cheap
logit policies first. Probe branching and attention statistics are not exposed by
the ordinary v1 logits hook: unsupported policies must fail explicitly. Do not
patch fused attention kernels without a validated reference and GPU profiling.

## Evaluation and limitations

Machine-readable JSONL includes manifest identity, request metrics, events and
answers. Compare paired samples with identical prompts/decoding/model revisions.
Measure latency directly; missing GPU/KV/serving measurements remain null.
Analysis provides Pareto frontiers, paired harm/rescue, oracle headroom, safe-stop
ranking/calibration and entropy/correctness contingency tables. Calibration selects
minimum measured cost subject to a quality constraint, using only a declared
calibration split. Save selection provenance and distinguish percentage points
from accuracy fractions. Final evaluation IDs must be held out.

Start with 10–20 examples, then development/calibration/validation/full stages.
Dataset versions and local file hashes are recorded; deterministic numerical and
multiple-choice grading is supported, with algebraic equivalence limitations made
explicit. Reference GPU timings and vLLM serving timings must not be conflated.

CPU synthetic/unit experiments validate mechanics and accounting but cannot answer
model-quality or GPU speedup questions; those come only from real-model runs. No
simulated score is to be labeled a Qwen3 or published-benchmark result.
