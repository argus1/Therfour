# ADR-Package: TTS Engine Decision for TherFour

## Context

TherFour is a self-hosted, privacy-first harm-reduction telephone helpline.
All inference must run locally — no audio or text data may leave the
operator's infrastructure. The TTS engine is responsible for converting
LLM reply text into speech and streaming it back to the caller via Twilio.

The current implementation uses **Piper** invoked as a subprocess. The TTS
parity analysis (2026-04-20) identified significant gaps in reliability,
multilingual support, and observability relative to the HealthCoacher target
model. This ADR records the engine decision going forward.

### Requirements driving this decision

| Requirement | Notes |
|---|---|
| Fully local inference | No cloud TTS APIs permitted |
| Open-source deployable | Must run on operator-owned hardware without licensing cost |
| CPU-only compatible | Not all deployments have GPU access |
| Low latency | Helpline calls require fast response; target < 2s synthesis for typical reply |
| Multilingual | TherFour supports multiple caller languages |
| Fallback capable | Engine failure must not silence the caller |
| Actively maintained | Engine must have ongoing community or maintainer support |

---

## Decision

**Piper remains the default production TTS engine.**

**F5-TTS is adopted as an experimental path**, available behind a feature
flag, with defined promotion gates that must be met before it can replace
Piper as the production default.

---

## Options Considered

### Option 1: Piper (current engine, hardened)

Piper is an open-source neural TTS system from the Rhasspy project. It runs
as a local subprocess, reads text from stdin, and writes raw PCM to stdout.
Voice models are distributed as `.onnx` files paired with a config JSON.

**Strengths:**

- Already integrated and working in TherFour
- Extremely lightweight — runs well on CPU-only hardware
- Wide voice model library covering many languages
- Simple deployment: single binary plus model files
- No network dependency at inference time
- Sub-second synthesis latency on typical reply lengths
- Actively maintained under the Rhasspy project

**Weaknesses:**

- Voice quality is good but below neural diffusion models like F5-TTS
- No streaming synthesis — full text must be available before synthesis begins
- Subprocess model adds process management overhead
- No built-in capability reporting or backend status contract

**Verdict:** Retain as production default. Harden with metadata schema,
typed error handling, and fallback wiring as specified in the TTS metadata
normalization plan.

---

### Option 2: F5-TTS (experimental path)

F5-TTS is a flow-matching neural TTS model. It produces significantly more
natural-sounding speech than Piper, supports voice cloning from a short
reference audio clip, and has strong multilingual capability. HealthCoacher
runs it as a locally served HTTP endpoint (`F5TTSHTTPService`) or on-device
via Core ML (`F5TTSCoreMLService`).

For TherFour, the relevant deployment path is **F5-TTS served as a local
HTTP service** on the same host or sidecar container, consistent with how
Ollama is already deployed.

**Strengths:**

- Substantially higher voice quality and naturalness
- Voice cloning from a short reference clip — enables consistent caller persona
- Strong multilingual support including Mandarin, Farsi, Japanese
- HTTP service model aligns with TherFour's existing Ollama pattern
- Active development community

**Weaknesses:**

- Higher compute cost — GPU strongly recommended for real-time latency
- CPU-only synthesis is significantly slower; may not meet latency targets
  on low-resource deployments
- Adds a new service dependency to the deployment stack
- Less battle-tested in telephony latency contexts than Piper
- Reference audio management adds operational complexity for voice cloning

**Verdict:** Adopt as an experimental path behind a feature flag. Do not
promote to production default until promotion gates are met.

---

### Option 3: Coqui TTS / XTTS

Coqui TTS (including XTTS v2) is another neural TTS option with strong
multilingual support and voice cloning capability.

**Ruled out because:**

- Coqui Inc. shut down in January 2024; the project is community-maintained
  with uncertain long-term trajectory
- XTTS v2 has a non-commercial license clause that conflicts with TherFour's
  open-source deployability requirement
- F5-TTS covers the same quality and multilingual use case with a cleaner
  license and active maintenance

---

### Option 4: espeak-ng

espeak-ng is a lightweight formant-based TTS engine with broad language
support. Extremely fast and CPU-friendly.

**Ruled out as primary because:**

- Voice quality is robotic and below acceptable standard for a helpline context
- Retained as a potential last-resort fallback option only (see fallback
  strategy below)

---

## Architecture

### Default Path (Piper)

```
LLM reply text
      │
      ▼
  tts.synthesize()
      │
      ▼
  PiperTTSBackend
  ├── subprocess.run(piper --model ...)
  ├── timeout: 30s
  ├── empty output check
  └── TTSSynthesisResult
      │
      ▼
  telephony._send_audio()
```

### Experimental Path (F5-TTS HTTP)

```
LLM reply text
      │
      ▼
  tts.synthesize()
      │
      ▼
  F5TTSHTTPBackend  (feature flag: TTS_BACKEND=f5_http)
  ├── POST /synthesize {text, voice, language}
  ├── timeout: configurable (default 10s)
  ├── empty response check
  └── TTSSynthesisResult
      │
      ▼
  telephony._send_audio()
```

### Fallback Strategy

Both paths share the same fallback contract defined in the metadata
normalization plan. On synthesis failure:

1. Log `TTSFailureReason` with structured metadata
2. If a fallback backend is configured, attempt synthesis on fallback
3. Mark `TTSSynthesisResult.fallback_used = True`
4. Keep fallback sticky for the session

Recommended fallback chain:

```
Primary (Piper or F5-TTS)
      │ failure
      ▼
espeak-ng (if installed)
      │ failure
      ▼
Drop turn with structured log — never silent
```

---

## Promotion Gates: F5-TTS → Production Default

F5-TTS may replace Piper as the production default when **all** of the
following gates are met:

### Gate 1: Latency

F5-TTS synthesis latency on a CPU-only reference deployment must be within
**1.5× of Piper** for replies up to 100 words.

Measurement: P95 synthesis wall-clock time across 200 representative LLM
replies on a reference CPU-only machine (no GPU).

Current Piper baseline: < 800 ms for 100-word reply on CPU.
F5-TTS promotion threshold: < 1200 ms on the same hardware.

### Gate 2: Reliability

F5-TTS must demonstrate a synthesis success rate of **≥ 99%** across a
1000-call soak test on the experimental flag deployment.

Failures include: HTTP errors, timeouts, empty audio responses, and
malformed audio that cannot be decoded by the telephony pipeline.

### Gate 3: Multilingual correctness

F5-TTS must produce intelligible, correctly accented output for all five
languages currently targeted by TherFour: English, Farsi, Mandarin,
Cantonese, and Japanese.

Measurement: manual listening review by at least one native speaker per
language across 10 representative reply samples per language.

### Gate 4: Fallback interoperability

F5-TTS must operate correctly within the `TTSSynthesisResult` metadata
contract and the session-sticky fallback chain.

Specifically: F5-TTS failure must trigger fallback to the system TTS
backend and emit a correctly typed `TTSFailureReason` without crashing the
telephony pipeline.

### Gate 5: Deployment complexity

F5-TTS must be deployable via a single Docker Compose addition with no
manual model download steps beyond a documented `make setup` or equivalent
command.

The existing Piper deployment must remain functional in parallel during the
transition period so operators can roll back without downtime.

---

## Configuration

The active TTS backend is controlled by a single environment variable:

| Variable | Default | Description |
|---|---|---|
| `TTS_BACKEND` | `piper` | Active backend: `piper` or `f5_http` |
| `TTS_FALLBACK_BACKEND` | `espeak` | Fallback backend on primary failure |
| `F5_TTS_ENDPOINT` | `http://localhost:8880` | F5-TTS HTTP service URL |
| `F5_TTS_VOICE` | `en_default` | Default voice hint passed to F5 |
| `PIPER_BINARY` | `piper` | Path to Piper binary (existing) |
| `PIPER_MODEL_PATH` | `models/en_US-lessac-medium.onnx` | Piper voice model (existing) |

---

## Consequences

### Accepting this decision

- Piper remains stable and well-understood for all current deployments
- F5-TTS can be evaluated in production conditions without risking call
  quality for live callers
- Promotion gates create a clear, measurable path to upgrading voice quality
- The fallback chain ensures callers always receive a spoken response
  regardless of which engine is active

### Risks

- F5-TTS may never meet the CPU latency gate, permanently limiting its use
  to GPU-enabled deployments only
- Running two TTS services in parallel during the experimental period
  increases deployment surface and documentation burden
- espeak-ng fallback voice quality is noticeably lower and may be
  jarring to callers if triggered frequently

### Out of scope for this ADR

- Streaming TTS (token-by-token synthesis during LLM generation) — this is
  a separate architectural decision dependent on Twilio media stream
  buffering behavior
- Voice cloning management for F5-TTS reference audio — deferred to a
  follow-on ADR once F5-TTS passes Gate 1 and Gate 2
- On-device Core ML TTS path (iOS/Swift backend) — not applicable to the
  Python telephony server

---

## Decision Record

| Date | Status | Note |
|---|---|---|
| 2026-04-20 | Proposed | Initial ADR authored following TTS parity analysis |
| — | Pending review | Awaiting team sign-off |
| — | Accepted / Rejected | To be updated after review |
