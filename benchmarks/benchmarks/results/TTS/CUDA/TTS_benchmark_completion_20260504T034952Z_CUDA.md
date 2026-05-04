# CUDA TTS Benchmark Completion Report — 20260504T034952Z

This report documents the completed CUDA benchmark run and applies the provided MOS ratings.

## Run metadata

- Raw benchmark JSON: `benchmarks/benchmarks/results/TTS/CUDA/TTS_benchmark_raw_20260504T034952Z.json`
- Timestamp (UTC): `20260504T034952Z`
- CUDA compatibility check passed: `True`
- F5 CUDA backend available: `True`
- Benchmark runs per passage: `5`

## Objective latency summary (from benchmark)

| Backend | Passage | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) | Success | Fail |
|---|---|---:|---:|---:|---:|---:|---:|
| Piper | Easy | 586.5 | 550.7 | 745.5 | 745.5 | 5 | 0 |
| F5-TTS (CUDA) | Easy | 2240.4 | 2129.7 | 2690.2 | 2690.2 | 5 | 0 |
| Piper | Moderate | 711.5 | 712.9 | 740.9 | 740.9 | 5 | 0 |
| F5-TTS (CUDA) | Moderate | 2620.5 | 2614.7 | 2642.6 | 2642.6 | 5 | 0 |
| Piper | Difficult | 898.5 | 851.1 | 1021.9 | 1021.9 | 5 | 0 |
| F5-TTS (CUDA) | Difficult | 2831.9 | 2843.0 | 2875.3 | 2875.3 | 5 | 0 |

## MOS ratings (provided)

| Passage | Piper MOS | F5-TTS MOS | Delta (Piper - F5) |
|---|---:|---:|---:|
| Easy | 3.4 | 2.6 | 0.8 |
| Moderate | 3.8 | 2.2 | 1.6 |
| Difficult | 3.6 | 1.6 | 2.0 |

## Final MOS calculation

- Piper overall MOS = $(3.4 + 3.8 + 3.6) / 3 = 3.60$
- F5-TTS overall MOS = $(2.6 + 2.2 + 1.6) / 3 = 2.13$
- Overall MOS gap (Piper - F5-TTS) = $3.60 - 2.13 = 1.47$

## Interpretation

- Based on provided MOS ratings, Piper is preferred across all three passage difficulties.
- The largest subjective quality gap is on the Difficult passage.

## Generated CUDA artifacts

- `benchmarks/benchmarks/results/TTS/CUDA/MOS_easy.wav`
- `benchmarks/benchmarks/results/TTS/CUDA/MOS_moderate.wav`
- `benchmarks/benchmarks/results/TTS/CUDA/MOS_difficult.wav`
- `benchmarks/benchmarks/results/TTS/CUDA/MOS_combined.wav`