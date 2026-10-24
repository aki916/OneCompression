# Runtime reasoning compression: implementation and evaluation report

Updated 2026-10-24 (JST). Base repository commit:
`4c9c8cf56ec322ae0ed7b84353839d10f33ecb04`, OneComp 1.4.0.

**The framework and CPU mechanics experiments are implemented. Qwen3-8B model
quality, paper reproduction, GPU latency/throughput, optimized vLLM ASAG, and
BF16/4-bit comparisons have not been measured.** CUDA reports no usable devices;
`nvidia-smi` cannot communicate with the driver. Qwen3-8B and the requested
benchmark data are not present locally. See the machine-readable
[availability record](../experiments/reasoning_compression/results/availability.json).

## Implementation status

| Component | Implemented | Validation / remaining gate |
|---|---|---|
| Signal / policy / action controller | Request-local state, token formats, boundaries, traces and strict config | CPU tests; compression defaults to `none` |
| Vanilla / fixed / prompt budget | Identity policy, clean close-marker transition, deterministic prompt heuristic | Greedy identity and marker/cache tests; heuristic is not TALE reproduction |
| ThinkBrake | Output-logit gap at selectable boundaries | Official equation; greedy and sentence-tokenization deviations documented |
| DEER | Isolated boxed-answer probes, geometric confidence, optional relative threshold | Full replay compute counted; no literature quality replication |
| Answer stability | Consecutive normalized probe-answer agreement | Sequential signal, not Dynasor scheduler/Certaindex reproduction |
| ASAG reference | Arithmetic confidence, normalized attention entropy, first-boundary relative change, probability mixture, bounded jump | Real eager-attention mechanics; documented paper ambiguities; no reproduction claim |
| Progressive policy | Cheap gap → answer probe → optional attention replay | Experimental reference; all extra branches included in costs |
| vLLM | Inspected 0.21 v1 batched logits adapter for none/fixed/prompt/ThinkBrake | Batch moves/swaps/reuse tested; no GPU serving measurement |
| Oracle / safe-stop data | Forced prefix continuations, failed stops retained, signal observations and independent replay costs | Exact remaining vanilla output budget; earliest observed boundary, cap recorded |
| Offline analysis | Paired outcomes, latency/compute Pareto, AUROC/AP, calibration, entropy contingency, held-out signal combinations | Missing values/one-class cases explicit; no train/test question overlap |
| Offline auto mode | Select minimum measured cost subject to accuracy constraint; held-out application | Python fractions / CLI percentage points explicit |
| Weight interaction | Existing packed-loader/plugin integration, GPTQ4 export recipe, paired comparison and threshold-transfer analysis | Qwen3 BF16/GPTQ4 experiment not executed |

The original quantization CLI and evaluators retain their behavior. A separate
`onecomp-reasoning` entry point is added; the module invocation works directly
from the checkout. See the [experiment commands](../experiments/reasoning_compression/README.md),
[design](reasoning_compression_design.md), [paper audit](reasoning_compression_papers.md),
and [runtime/API audit](reasoning_compression_runtime.md).

## Experiment setup and reproducibility

Installed: torch 2.11.0+cu130, Transformers 5.14.1, vLLM 0.21.0, Python 3.13.
Only CPU measurements are available. The executed fixture uses a randomly
initialized four-layer, width-32 Qwen3 backbone, one CPU thread, greedy decoding,
and a declared scripted output head. The head deliberately creates correct,
incorrect, overconfident, and overthinking cases. **Its accuracy is synthetic;
its model forwards, attention tensors and elapsed times are real CPU work.**

The unprofiled smoke contains 13 configurations × 12 paired artificial requests,
plus 72 oracle forced-close trials. A separate two-request run enables cProfile.
Each variant saves the source digest, fixture source hash, model configuration,
versions, sample IDs, decoding and policy parameters. Profiling is marked in the
manifest and must not be pooled with unprofiled timing. This is a smoke split;
no final benchmark has been used for threshold tuning.

The real-data importer freezes immutable dataset revisions and disjoint
development/calibration/evaluation IDs. AMC23 uses math-ai's 40-question subset;
MATH-500 has 500 rows; AIME 2024 has 30, so large development subsets are not
possible there. GPQA uses a frozen choice permutation and requests an option
letter. The conservative exact/rational grader is not a complete symbolic MATH
grader; this limitation must be resolved or audited before literature comparisons.

## Correctness checks

Final test run: **81 passed, 8 skipped** (78 new reasoning/runtime checks and
three existing CLI/export checks). The skipped existing CLI variants require a
GPU or explicit opt-in. See the [test log](../experiments/reasoning_compression/logs/tests.txt).
Black, isort, copyright/header checks, compilation and `git diff --check` pass.

Tests cover marker counting (including multi-token markers), token/sentence/ATP
boundaries, per-request isolation, incomplete probes, idempotent decisions,
probability injection, trap jumps, configuration errors, paired metrics, held-out
calibration, oracle continuation and CLI artifacts. Runtime tests include real
tiny-Qwen3 greedy equivalence to native generation and eager-attention output,
plus actual installed vLLM processor loading, slot movement/swap/removal/reuse,
simultaneous requests with different budgets, and unsupported-feature rejection.

Disabled generation and `policy=none` must produce identical token sequences.
The 12-request smoke verifies that equality separately from instrumented oracle
collection; instrumentation is excluded from the primary vanilla timing.

Full smoke inspection caught a Transformers `ModelOutput` lifecycle bug during
attention cleanup that simplified test objects did not catch. The fix clears the
supported attribute/mapping entry and adds a real eager-Qwen3 regression test.
Review also fixed incomplete-probe stops, repeated-boundary evidence, nullable
entropy/trap handling, vLLM's colon-separated processor import syntax, and oracle
budget mismatch. These are correctness fixes, not changes tuned to benchmark scores.

## Measured smoke results and Pareto artifacts

Raw records are in
[`cpu_smoke/results/`](../experiments/reasoning_compression/cpu_smoke/results/).
The [summary JSON](../experiments/reasoning_compression/cpu_smoke/results/summary.json)
contains per-configuration measured latency, probe compute, harm/rescue, oracle
headroom and signal metrics. The
[Pareto SVG](../experiments/reasoning_compression/cpu_smoke/plots/pareto.svg)
is labeled **synthetic CPU mechanics**. Its quality axis is scripted and provides
no evidence about which algorithm wins on real reasoning problems.

Selected measurements from that artificial workload (means over 12 requests):

| Policy setting | Reasoning tokens | Total model-token evaluations | CPU latency (ms) |
|---|---:|---:|---:|
| None | 12.0 | 19.0 | 34.933 |
| Fixed budget 8 | 8.0 | 15.0 | 27.294 |
| ThinkBrake gap 0.1, interval 2 | 8.0 | 15.0 | 27.774 |
| DEER confidence 0.95, interval 2 | 6.5 | 49.0 | 45.266 |
| DEER confidence 0.999, interval 2 | 12.0 | 97.0 | 73.516 |
| DEER confidence 0.999, interval 4 | 12.0 | 61.0 | 54.493 |
| ASAG reference, interval 2 | 5.0 | 36.5 | 39.944 |

DEER at 0.95 demonstrates the intended distinction particularly clearly: fewer
visible reasoning tokens than vanilla, yet more evaluated model tokens and higher
measured latency. This is an execution-cost result for the fixture, not evidence
about a useful threshold or algorithm ranking on Qwen3-8B.

The useful measured finding is the accounting check: conservative DEER probes
at every two fixture tokens evaluate 97 model tokens per request versus 19 for
vanilla. Moving the probe interval to four evaluates 61, with identical fixture
final answers. The saved [smoke note](../experiments/reasoning_compression/cpu_smoke/README.md)
records the corresponding measured wall times. Shorter visible reasoning can
therefore coexist with greater actual execution cost in this reference backend.

No statistical significance or production speedup is claimed from 12 short CPU
requests. Fixed policy ordering and host load can influence timings; proper GPU
evaluation needs warmup, repeated trials, order variation and uncertainty estimates.

## Failure, oracle and signal analysis

The fixture deliberately exercises all four paired correctness outcomes; the
analysis reports harm conditional on vanilla-correct and rescue conditional on
vanilla-wrong. The oracle preserves unsafe boundaries after safe ones: correctness
is not assumed monotonic along a trajectory. Its minimum is among measured
boundaries, not an unsampled theoretical optimum. Capped real oracle runs carry a
censor flag and retain the same remaining total-generation budget as vanilla.

AUROC, tie-grouped average precision, false-stop rates, precision/recall, Brier
score, reliability bins and entropy/correctness tables execute on the fixture.
Those signal scores reflect the scripted head, not learned predictive ability.
The fixture contains no relative-entropy convergence at threshold −0.1; thus
`P(wrong | convergence)` is null. A missing conditioning population must not be
reported as zero risk. The random attention state cannot establish that entropy
adds information about real correctness.

The held-out combination API compares confidence, attention, their combination,
token count + confidence, and logit margin + confidence on the same complete-case
rows. It refuses overlapping question IDs, so boundaries from one problem cannot
leak across signal-training and evaluation partitions. No real-data combination
comparison has been performed.

## Profiling and evidence-driven iteration

The [separate profile](../experiments/reasoning_compression/cpu_profile/logs/profile.txt)
shows model-forward execution as the dominant cumulative path in this tiny CPU
workload: 675 forwards take 1.781 seconds cumulative out of a 1.994-second
instrumented run. These nested cumulative times are not additive phase totals.
The first smoke exposed repeated probe re-prefills as a large source of
extra model-token evaluations. This motivated the two-token/four-token boundary
ablation above; both evaluation counts and real latency are reported, while the
quality labels are explicitly synthetic.

The reference separates controller, probe, attention reduction, intervention and
full attention-probe forward time. The latter includes ordinary forward work;
it is not claimed as incremental attention overhead. GPU synchronization and peak
memory are recorded where possible; unavailable KV and serving values remain null.
The vLLM driver measures serial end-to-end request latency and reads TTFT when
exposed. It leaves unavailable worker traces and scheduler token-evaluation counts
null instead of estimating them from visible output length.

## Research-question answers

| Question | Current answer |
|---|---|
| RQ1: adaptive versus fixed Pareto | Not determined on a reasoning benchmark. Both are implemented and sweepable. |
| RQ2: cheap logits versus confidence | Not determined scientifically. Probe replay overhead is demonstrable in the CPU fixture. |
| RQ3: additional information from attention | Not determined. Held-out matched-coverage signal analysis is implemented. |
| RQ4: token savings → vLLM latency/throughput | Not measured; no GPU. Serial driver is ready; concurrent serving benchmark remains work. |
| RQ5: captured oracle headroom | Computed only for artificial cases; real forced-prefix experiment pending. |
| RQ6: harm to vanilla-correct | Measured only on scripted labels; real rate unknown. |
| RQ7: rescue of vanilla-wrong | Measured only on scripted labels; real rate unknown. |
| RQ8: quantization signal/calibration shifts | Not measured; paired source/checkpoint identities and transfer analysis are implemented. |
| RQ9: weight/reasoning complementarity | Not measured; never infer multiplicative speedup. |
| RQ10: progressive versus always-expensive monitoring | Mechanically implemented; real accuracy/cost benefit untested. |

## Limitations and next justified experiment

Reference ASAG uses eager full-prefix attention, which can require prohibitive
quadratic memory at Qwen3 reasoning lengths. Only selected query rows/last four
layers enter the statistic, but the backend can materialize more than that. The
paper leaves probability-vector choice/injection duration and jump-limit details
ambiguous; this implementation's interpretation is explicitly documented and is
not called a completed reproduction. The greedy probe cue/token masking and
sentence/transition heuristics also differ from some published settings.

An optimized vLLM ASAG kernel is **not implemented**. The inspected logits hook
does not expose attention or safe speculative branch scheduling. The runtime note
describes sufficient statistics and validation gates; no unvalidated kernel patch
or silently substituted approximation is included. Continuous-batch controller
tests are not GPU serving validation. The serial Transformers reference makes no
concurrent model-forward guarantee.

Next, on a working GPU with pinned Qwen3-8B and dataset snapshots, run the supplied
12-example BF16 smoke for vanilla/fixed/ThinkBrake/DEER. Verify final-answer parsing,
probe overhead, raw paired records and GPU timings. Then sweep fixed budgets and
cheap thresholds on 50–100 development examples, calibrate on disjoint IDs and
validate on 100–300 where the dataset permits. Only if probes improve the measured
frontier should attention collection and progressive escalation be expanded.
Repeat matched cheap-policy sweeps on the supported GPTQ4/group-128 checkpoint;
compare frozen BF16 thresholds with separately calibrated 4-bit thresholds on
held-out 4-bit examples. Paper-trend replication, optimized attention statistics,
concurrent serving, and full evaluation remain subsequent empirical gates.
