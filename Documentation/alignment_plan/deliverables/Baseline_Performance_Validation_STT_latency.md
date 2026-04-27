# Baseline Performance Validation - STT Latency

Date: 2026-04-26
Branch: argus-baseline-branch

Purpose: document baseline STT latency performance validation using existing benchmark artifacts and confirm readiness against the sprint alignment plan criteria.

## Validation Scope

- Component: STT (Whisper benchmark harness)
- Focus metrics: P50 latency, P95 latency, and run stability across baseline benchmark runs
- Environment in recorded runs: Apple M1, CPU, int8 compute type
- Data sources:
  - `benchmarks/results/whisper_benchmark_20260419T072617Z.json`
  - `benchmarks/results/whisper_benchmark_20260419T072736Z.json`
  - `benchmarks/results/whisper_benchmark_20260419T074919Z.json`
  - `benchmarks/results/whisper_benchmark_20260419T083650Z.json`

## Plan Criteria Cross-Check

From `Documentation/alignment_plan/Plan.md`:

- Baseline metrics include STT latency.
- Exit criteria require P95 latency within an agreed threshold.

Note: the plan defines "agreed threshold" wording but does not currently record a numeric threshold value in-repo.

## Measured Results

| Run ID           | Model           | Samples | P50 Latency (s) | P95 Latency (s) | Notes                                     |
| ---------------- | --------------- | ------: | --------------: | --------------: | ----------------------------------------- |
| 20260419T072617Z | small           |      28 |          0.0609 |          0.0662 | Single-file focused baseline pass         |
| 20260419T072736Z | small           |     160 |          0.0637 |          0.0723 | Broad sample set                          |
| 20260419T074919Z | small           |      16 |          0.0616 |          0.0829 | Smaller sample run with robustness fields |
| 20260419T083650Z | small           |     160 |          0.0610 |          0.0678 | Latest broad baseline, improved P95       |
| 20260419T072736Z | distil-large-v3 |     160 |          0.0692 |          0.0734 | Secondary model reference                 |
| 20260419T083650Z | distil-large-v3 |     160 |          0.0695 |          0.0746 | Secondary model reference                 |

## Failure Rate

The benchmark JSON files record `text_chars` per result. A result with `text_chars = 0` indicates the VAD filter suppressed all speech and Whisper returned an empty transcript — the functional equivalent of a dropped call in production.

| Run | Model | Total files | Empty transcript | Empty rate | SNR condition |
| --- | --- | ---: | ---: | ---: | --- |
| 20260419T072617Z | small | 28 | 4 | 14.3% | all at −10 dB noise |
| 20260419T072736Z | small | 160 | 21 | 13.1% | all at −10 dB noise |
| 20260419T072736Z | distil-large-v3 | 160 | 8 | 5.0% | all at −10 dB noise |
| 20260419T074919Z | small | 16 | 2 | 12.5% | all at −10 dB noise |
| 20260419T083650Z | small | 160 | 21 | 13.1% | all at −10 dB noise |
| 20260419T083650Z | distil-large-v3 | 160 | 8 | 5.0% | all at −10 dB noise |

**Key finding:** every empty-transcript result in the reference run (`20260419T083650Z`) occurs exclusively at SNR = −10 dB (extreme noise). No failures were observed under clean or moderate-noise conditions. This is expected VAD behaviour, not an inference error.

Empty transcript rate on **clean audio** across all runs: **0.0%**.

No exception-level failures (model load errors, file read errors) were recorded in any run. The benchmark did not perform fault injection.

## Validation Outcome

- Baseline STT latency is consistently low across recorded runs.
- For the primary `small` model, observed P95 range is 0.0662s to 0.0829s.
- Latest broad benchmark (`20260419T083650Z`) reports P95 = 0.0678s for `small`.
- Performance trend is stable to improving when comparing broad runs:
  - `small` P95 improved from 0.0723s to 0.0678s.

## Decision Status

- STT latency baseline performance: **Validated** based on recorded benchmark evidence.
- Exit gate status "P95 latency within agreed threshold": **Pending numeric threshold definition** in project docs.

## Recommended Follow-up

1. Record the explicit numeric STT P95 target in `Documentation/alignment_plan/Plan.md` (or an ADR) to convert this from evidence-based validation to strict pass/fail gating.
2. Keep `20260419T083650Z` as the baseline reference run for Sprint 1 reporting.
