# Runtime reasoning compression: CPU mechanics results

The framework is validated on CPU only. Qwen3-8B model quality, paper reproduction,
GPU latency/throughput, optimized vLLM ASAG and BF16/4-bit comparisons are not
covered here.

## Fixture

The fixture uses a randomly initialized four-layer, width-32 Qwen3 backbone, one CPU
thread, greedy decoding, and a scripted output head that deliberately creates
correct, incorrect, overconfident and overthinking cases. **Its accuracy is
synthetic; its model forwards, attention tensors and elapsed times are real CPU
work.** The smoke run contains 13 configurations x 12 paired artificial requests
plus 72 oracle forced-close trials. Disabled generation and `policy=none` produce
identical token sequences.

## Measured results

Means over 12 requests:

| Policy setting | Reasoning tokens | Total model-token evaluations | CPU latency (ms) |
|---|---:|---:|---:|
| None | 12.0 | 19.0 | 34.933 |
| Fixed budget 8 | 8.0 | 15.0 | 27.294 |
| ThinkBrake gap 0.1, interval 2 | 8.0 | 15.0 | 27.774 |
| DEER confidence 0.95, interval 2 | 6.5 | 49.0 | 45.266 |
| DEER confidence 0.999, interval 2 | 12.0 | 97.0 | 73.516 |
| DEER confidence 0.999, interval 4 | 12.0 | 61.0 | 54.493 |
| ASAG reference, interval 2 | 5.0 | 36.5 | 39.944 |

Shorter visible reasoning can coexist with greater actual execution cost: DEER at
0.95 produces fewer reasoning tokens than vanilla but evaluates more model tokens
and has higher latency, and probing every two tokens evaluates 97 tokens per request
versus 19 for vanilla (61 with an interval of four, with identical fixture answers).
A profile of the tiny workload shows model forwards dominating execution time.

## Caveats

- The quality axis is scripted; the results say nothing about which algorithm wins
  on real reasoning problems or which threshold is useful on Qwen3-8B.
- Twelve short CPU requests give no statistical significance and no production
  speedup claim. GPU evaluation needs warmup, repeated trials, order variation and
  uncertainty estimates.
- Signal metrics (AUROC, average precision, calibration, entropy tables) reflect the
  scripted head, not learned predictive ability. Where a conditioning population is
  empty, the value is null, not zero risk.
- Reference ASAG uses eager full-prefix attention, which can need prohibitive
  memory at Qwen3 reasoning lengths. An optimized vLLM ASAG kernel is not
  implemented.
