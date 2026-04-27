# Baseline Performance Validation - TTS Latency

Date: 2026-04-26
Branch: argus-baseline-branch

Purpose: document baseline TTS latency validation for Sprint 1 alignment tracking.

## Validation Scope

- Component: TTS (Piper synthesis path)
- Focus metrics: P50 latency, P95 latency, RTF, and synthesis failure rate
- Primary code paths:
  - app/services/tts.py
  - app/core/config.py
- Existing instrumentation reference:
  - Documentation/alignment_plan/deliverables/Baseline_Observability_Checklist_2026-04-20.md

## Plan Criteria Cross-Check

From Documentation/alignment_plan/Plan.md:

- Baseline metrics include TTS latency.
- Exit criteria require P95 latency within an agreed threshold.

Note: no explicit numeric TTS P95 threshold is currently defined in-repo. This benchmark establishes the baseline for that threshold to be set.

## Runtime Readiness

Piper TTS was unblocked on 2026-04-26 by:

1. Installing `piper-tts==1.4.2` (PyPI, native `macosx_11_0_arm64` wheel) into `.venv` — provides `piper` CLI at `.venv/bin/piper`.
2. Downloading `en_US-lessac-medium` voice model via `piper.download_voices.download_voice()` (from Hugging Face `rhasspy/piper-voices`):
   - `models/en_US-lessac-medium.onnx` (60 MB)
   - `models/en_US-lessac-medium.onnx.json` (4.8 KB)

Both satisfy the defaults in `app/core/config.py`:
- `piper_binary = "piper"` — resolved via `.venv/bin/piper`
- `piper_model_path = "models/en_US-lessac-medium.onnx"` — present

Code/model source: TTS-WebUI Piper extension pattern (https://github.com/rsxdalv/TTS-WebUI), MIT license.

## Benchmark Methodology

- Benchmark date: 2026-04-26
- Engine: Piper TTS v1.4.2, `en_US-lessac-medium` ONNX model, 22 050 Hz 16-bit PCM output
- Invocation path: identical to production `app/services/tts.py` — subprocess stdin → stdout raw PCM
- Utterance set: 9 harm-reduction representative phrases across 3 length categories (3 short ~3–5 words, 3 medium ~15 words, 3 long ~40 words), each measured 3 runs, median taken
- Hardware: Apple M-series ARM64 (macOS), single process, no GPU
- Timing: wall-clock `time.perf_counter()` from subprocess start to PCM fully received

## Measured Results — Per-Utterance

| Label    | Text (truncated)                              | Latency (s) | Audio dur (s) | RTF   |
| -------- | --------------------------------------------- | ----------: | ------------: | ----: |
| short_1  | "Call 911 immediately."                       |       0.710 |          2.04 | 0.347 |
| short_2  | "Dial 911 right now."                         |       0.619 |          2.04 | 0.303 |
| short_3  | "Help is on the way."                         |       0.607 |          1.34 | 0.455 |
| medium_1 | "If you're experiencing a mental health…"     |       0.757 |          5.85 | 0.129 |
| medium_2 | "You can reach the Substance Abuse Helpline…" |       0.883 |          8.19 | 0.108 |
| medium_3 | "The National Domestic Violence Hotline…"     |       0.926 |          9.32 | 0.099 |
| long_1   | "Harm reduction means meeting people…"        |       0.986 |         11.68 | 0.084 |
| long_2   | "If you or someone you know is struggling…"   |       0.961 |         11.47 | 0.084 |
| long_3   | "During a panic attack, try to focus…"        |       1.061 |         14.92 | 0.071 |

## Measured Results — Summary Statistics

| Category         | N | Mean latency (s) | Median (s) | Min (s) | Max (s) | Stdev (s) | Mean RTF |
| ---------------- | - | ---------------: | ---------: | ------: | ------: | --------: | -------: |
| All utterances   | 9 |            0.834 |      0.883 |   0.607 |   1.061 |     0.166 |    0.187 |
| Short (~3–5 w)   | 3 |            0.645 |      0.619 |   0.607 |   0.710 |     0.056 |    0.368 |
| Medium (~15 w)   | 3 |            0.855 |      0.883 |   0.757 |   0.926 |     0.086 |    0.112 |
| Long (~40 w)     | 3 |            1.003 |      0.986 |   0.961 |   1.061 |     0.052 |    0.080 |

**P50 (median, all): 0.883 s**
**P95 (estimated, all): ~1.04 s** (interpolated between long_2=0.961 s and long_3=1.061 s at 95th pct of 9 samples)

Key observations:
- Latency is dominated by process startup + model load (~0.6 s floor), not text length.
- RTF drops sharply with longer utterances (0.37 short → 0.08 long), showing near-constant overhead.
- All 9 calls succeeded (0 failures).

## Failure Rate

| Category                  | Count | Rate  |
| ------------------------- | ----: | ----: |
| Total synthesis calls      |     9 |       |
| Subprocess errors (rc ≠ 0) |     0 |   0 % |
| Empty PCM output (0 bytes) |     0 |   0 % |
| Overall failure rate       |     0 |   0 % |

Failure rate: **0.0 %** across all test inputs. Consistent with expected behavior for well-formed ASCII text inputs.

## Validation Outcome

- Piper TTS is fully operational in this environment.
- P50 latency (0.883 s) and P95 latency (~1.04 s) are measured and documented.
- Failure rate: 0.0 % on clean text inputs.
- No numeric threshold was pre-defined in-repo; these measurements establish the Sprint 1 baseline.
- TTS observability instrumentation (`latency_ms`, `failure_reason`) is implemented and test-covered per `Baseline_Observability_Checklist_2026-04-20.md`.

## Decision Status

- TTS latency baseline performance: **COMPLETE — baseline established.**
- Exit gate "P95 latency within agreed threshold": baseline P95 ≈ 1.04 s recorded; numeric pass/fail threshold requires stakeholder agreement.

## Notes

- Process-startup overhead (~0.6 s cold floor) will be lower in production if a persistent Piper process or Python API (`piper.PiperVoice`) is used instead of per-request subprocess.
- Model: `en_US-lessac-medium` (60 MB ONNX, 22 050 Hz). Source: `https://huggingface.co/rhasspy/piper-voices`.
- piper-tts package: `https://pypi.org/project/piper-tts/`
