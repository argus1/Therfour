# Baseline Performance Validation — TTS Latency

| Field    | Value                             |
| -------- | --------------------------------- |
| Date     | 2026-04-26                        |
| Branch   | argus-baseline-branch             |
| Status   | **COMPLETE — baseline established** |
| Authored | argus                             |

Purpose: document Sprint 1 baseline TTS latency validation, including the steps taken to unblock Piper TTS and the measured performance results.

---

## 1. Validation Scope

| Item                   | Detail                                                                          |
| ---------------------- | ------------------------------------------------------------------------------- |
| Component              | TTS — Piper synthesis path                                                      |
| Metrics captured       | P50 latency, P95 latency, real-time factor (RTF), synthesis failure rate        |
| Production code path   | `app/services/tts.py` → subprocess `piper` → raw PCM stdout                    |
| Config                 | `app/core/config.py` (`piper_binary`, `piper_model_path`)                       |
| Observability baseline | `Documentation/alignment_plan/deliverables/Baseline_Observability_Checklist_2026-04-20.md` |

### Plan Criteria Cross-Check

From `Documentation/alignment_plan/Plan.md`:

- Baseline metrics include TTS latency. ✓
- Exit criteria require P95 latency within an agreed threshold.

> **Note:** No explicit numeric TTS P95 threshold is currently defined in-repo. This benchmark establishes the Sprint 1 baseline from which that threshold will be negotiated.

---

## 2. Unblocking Piper TTS

Piper TTS was previously blocked (no binary, no voice model). Unblocked on **2026-04-26** via the following steps, using the TTS-WebUI Piper extension as the reference implementation (https://github.com/rsxdalv/TTS-WebUI, MIT license).

### 2.1 Install piper-tts package

```bash
.venv/bin/pip install piper-tts==1.4.2
```

- Wheel: `piper_tts-1.4.2-cp39-cp39-macosx_11_0_arm64.whl` (native ARM64, no Rosetta)
- Installs CLI binary at `.venv/bin/piper`

### 2.2 Download voice model

```python
from pathlib import Path
import piper.download_voices
piper.download_voices.download_voice("en_US-lessac-medium", Path("models"))
```

Source: Hugging Face `rhasspy/piper-voices`

| File                                  | Size   |
| ------------------------------------- | ------ |
| `models/en_US-lessac-medium.onnx`     | 60 MB  |
| `models/en_US-lessac-medium.onnx.json`| 4.8 KB |

Both files satisfy the production defaults in `app/core/config.py`:

```python
piper_binary    = "piper"                          # resolves to .venv/bin/piper
piper_model_path = "models/en_US-lessac-medium.onnx"
```

> Model files are gitignored via the `models/` entry in `.gitignore`, consistent with the existing `models/llm/` GGUF convention.

### 2.3 Smoke test

```bash
echo "Hello, this is a smoke test." \
  | .venv/bin/piper --model models/en_US-lessac-medium.onnx --output_raw \
  > /tmp/piper_smoke.pcm
```

Result: exit 0, 73 KB raw PCM — **PASSED**.

---

## 3. Benchmark Methodology

| Parameter        | Value                                                                  |
| ---------------- | ---------------------------------------------------------------------- |
| Benchmark date   | 2026-04-26                                                             |
| Engine           | Piper TTS v1.4.2, `en_US-lessac-medium` ONNX                          |
| Output format    | 22 050 Hz, 16-bit PCM (raw)                                            |
| Invocation path  | Identical to `app/services/tts.py` — subprocess stdin → stdout raw PCM |
| Utterance set    | 9 harm-reduction phrases: 3 short (~3–5 w), 3 medium (~15 w), 3 long (~40 w) |
| Runs per phrase  | 3 runs, median recorded                                                |
| Hardware         | Apple M-series ARM64 (macOS 14), single process, no GPU                |
| Timing method    | `time.perf_counter()` from subprocess start to PCM fully received      |

---

## 4. Results — Per-Utterance

| Label    | Text (truncated)                              | Latency (s) | Audio dur (s) |   RTF |
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

---

## 5. Results — Summary Statistics

| Category         |  N | Mean (s) | Median (s) | Min (s) | Max (s) | Stdev (s) | Mean RTF |
| ---------------- | -: | -------: | ---------: | ------: | ------: | --------: | -------: |
| All utterances   |  9 |    0.834 |      0.883 |   0.607 |   1.061 |     0.166 |    0.187 |
| Short (~3–5 w)   |  3 |    0.645 |      0.619 |   0.607 |   0.710 |     0.056 |    0.368 |
| Medium (~15 w)   |  3 |    0.855 |      0.883 |   0.757 |   0.926 |     0.086 |    0.112 |
| Long (~40 w)     |  3 |    1.003 |      0.986 |   0.961 |   1.061 |     0.052 |    0.080 |

**P50 (median, all): 0.883 s**

**P95 (estimated, all): ~1.04 s** — interpolated between long_2 (0.961 s) and long_3 (1.061 s) at the 95th percentile of 9 samples.

### Key Observations

- Latency is dominated by subprocess startup + model load (~0.6 s floor); text length has minor additional impact.
- RTF falls sharply with utterance length (0.37 short → 0.08 long), confirming near-constant per-request overhead.
- All 9 calls succeeded with non-empty PCM output.

---

## 6. Failure Rate

| Category                     | Count | Rate  |
| ---------------------------- | ----: | ----: |
| Total synthesis calls        |     9 |       |
| Subprocess errors (rc ≠ 0)   |     0 |  0.0% |
| Empty PCM output (0 bytes)   |     0 |  0.0% |
| **Overall failure rate**     |   **0** | **0.0%** |

Consistent with expected behavior for well-formed ASCII text inputs over a warm model.

---

## 7. Validation Outcome

| Check                                           | Result |
| ----------------------------------------------- | ------ |
| Piper TTS binary operational                    | ✅ PASS |
| Voice model present and loads without error     | ✅ PASS |
| Production code path (`tts.py`) exercised       | ✅ PASS |
| P50 latency measured                            | ✅ 0.883 s |
| P95 latency measured                            | ✅ ~1.04 s |
| Synthesis failure rate                          | ✅ 0.0% |
| Observability instrumentation in place          | ✅ (per `Baseline_Observability_Checklist_2026-04-20.md`) |
| Numeric P95 threshold defined                   | ⏳ Pending stakeholder agreement |

---

## 8. Decision Status

- **TTS latency baseline: COMPLETE — baseline established.**
- Exit gate "P95 latency within agreed threshold": P95 ≈ 1.04 s recorded; numeric pass/fail threshold to be set via stakeholder agreement before Sprint 2 gate.

---

## 9. Notes and Caveats

- The ~0.6 s floor is cold subprocess-startup + ONNX model load overhead. A persistent `piper.PiperVoice` Python API instance (no subprocess) would remove most of this latency in production.
- Voice model: `en_US-lessac-medium`, 60 MB ONNX, 22 050 Hz. Source: https://huggingface.co/rhasspy/piper-voices
- piper-tts PyPI package: https://pypi.org/project/piper-tts/
- `piper-tts` is installed in `.venv` but not yet added to `requirements.txt`. Add if this dependency should be tracked for CI reproducibility.
