# STT Whisper Model Recommendations: TherFour

Date: 2026-04-18
Author: Engineering benchmark note

## Benchmark Scope

This benchmark pass was limited to what is feasible in the current workspace environment.

Measured locally:

- macOS
- CPU only
- faster-whisper with `compute_type="int8"`
- synthetic 10-second audio inputs

Not measured locally:

- NVIDIA GPU performance
- telephony-specific real speech corpus
- concurrent-load behavior

Local GPU check result:

- `NO_NVIDIA_GPU`

## Method

Two quick benchmark styles were used:

1. No-speech fast path on 10 seconds of silence with VAD enabled
2. Forced decode path on 10 seconds of low-amplitude synthetic noise with VAD disabled

The second benchmark is more useful for model comparison because it exercises decoder work instead of early no-speech exit behavior.

## Local CPU Results

### Forced Decode Benchmark

Command settings:

- device: CPU
- compute type: `int8`
- beam size: `1`
- language: `en`
- `vad_filter=False`
- `condition_on_previous_text=False`

| Model              | Avg decode time (10 s synthetic input) | Notes                                                |
| ------------------ | -------------------------------------: | ---------------------------------------------------- |
| `tiny`             |                                0.129 s | Fastest, but quality too weak for hotline production |
| `base`             |                                1.437 s | Slower than expected in this environment             |
| `small`            |                                0.810 s | Best multilingual CPU tradeoff in this local run     |
| `distil-small.en`  |                                0.679 s | Strong English-only latency candidate                |
| `distil-medium.en` |                                1.942 s | Too slow here for the gained value                   |

### Load Time Snapshot

| Model              | Approx load time |
| ------------------ | ---------------: |
| `tiny`             |          4.193 s |
| `base`             |          4.343 s |
| `small`            |         14.354 s |
| `distil-small.en`  |         12.035 s |
| `distil-medium.en` |         22.076 s |

## Interpretation

### CPU-Only Deployments

Recommended default for multilingual TherFour:

- `small`

Why:

- already aligned with the current repo default
- multilingual capable
- materially better practical fit than `tiny`
- better local latency than `base` in this benchmark run

Recommended English-only low-latency option:

- `distil-small.en`

Why:

- fastest credible English-focused tier tested after `tiny`
- lower decode time than `small` in the local CPU run
- suitable only if the hotline deployment is explicitly English-only

Not recommended as CPU default:

- `tiny`
  - too much quality risk for hotline usage
- `base`
  - no clear advantage over `small` in this local run
- `distil-medium.en`
  - too slow for the value returned here

### GPU-Enabled Server Deployments

Because there is no NVIDIA GPU in this environment, the following are deployment recommendations rather than locally measured GPU results.

Recommended GPU default for multilingual production:

- `distil-large-v3`

Why:

- strong reputation for quality/latency balance on GPU
- better fit than small models when the server target is explicitly GPU-enabled
- open-source and compatible with the faster-whisper serving pattern already used here

Recommended GPU fallback for tighter latency or smaller GPU budgets:

- `small`

Why:

- simplest operational continuation from the current default
- lower memory footprint and predictable behavior
- easier baseline if the team wants conservative rollout risk

Recommended English-only GPU low-latency option:

- `distil-small.en`

Why:

- viable only if multilingual support is not required
- lower-latency path for narrowly scoped English deployments

## Production Recommendation

### TherFour Default Matrix

| Deployment target                | Recommended model | Notes                                            |
| -------------------------------- | ----------------- | ------------------------------------------------ |
| GPU server, multilingual         | `distil-large-v3` | Best recommended quality/latency target          |
| GPU server, conservative rollout | `small`           | Simplest and lowest-risk upgrade path            |
| CPU-only, multilingual           | `small`           | Best measured practical fit in this workspace    |
| CPU-only, English-only           | `distil-small.en` | Use only if multilingual support is out of scope |

## Rollout Advice

1. Keep `small` as the immediate production default until a real telephony benchmark corpus is in place.
2. Benchmark `distil-large-v3` on the actual GPU server target before promoting it as the default.
3. Only use `.en` distilled models for deployments that are explicitly English-only.
4. Re-run benchmarks on real hotline audio after Silero VAD segmentation is enabled, because turn-finalization quality materially affects end-to-end latency.

## Next Benchmark Tasks

- measure end-of-speech to final transcript latency on real telephony audio
- benchmark with Silero VAD enabled and disabled
- benchmark under concurrent-call load
- compare `small` versus `distil-large-v3` on the target GPU hardware
