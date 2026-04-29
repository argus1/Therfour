# TTS Response Metadata Normalization Plan: TherFour


## Objective

Define a practical TTS response metadata and synthesis status reporting
architecture for TherFour that:

- makes TTS failures visible and distinguishable from STT and LLM failures
- provides structured synthesis result metadata for logging and observability
- aligns with the TTS parity gaps identified in the TTS parity analysis
- mirrors the metadata contract pattern used in the STT normalization plan

## Recommendation Summary

Use a structured result contract, not raw audio returns.

- `TTSSynthesisResult`: mandatory response wrapper carrying audio and metadata
- `TTSFailureReason`: typed enum replacing generic `RuntimeError`
- `TTSStatusReporter`: lightweight observability layer per call turn

This is feasible and directly addresses the core parity gap: Therfour
currently returns a raw numpy array with no accompanying status, no failure
taxonomy, and no observability.

## Why Metadata Normalization Matters Here

Right now `tts.synthesize()` either returns a float32 array or raises a
generic exception. The telephony pipeline catches everything in one broad
`except` block and drops the turn silently. This means:

- TTS failures look identical to STT and LLM failures in logs
- Empty audio output passes through without triggering any alert
- There is no way to track synthesis latency, audio duration, or which
  backend produced the audio
- Future fallback logic has no structured contract to report against

Introducing a metadata schema solves all of these at once.

---

## Recommended Architecture

### Core Idea

Wrap every synthesis call in a result object instead of returning raw audio.
The result carries the audio payload alongside synthesis metadata and a
typed status. The telephony pipeline reads the status before sending audio
to the caller.

### Response Flow

```
tts.synthesize(text, language)
        │
        ▼
  TTSSynthesisResult
  ├── audio: np.ndarray        # float32 PCM samples (empty on failure)
  ├── status: TTSSynthesisStatus
  ├── failure_reason: TTSFailureReason | None
  ├── backend_name: str
  ├── fallback_used: bool
  ├── voice_used: str
  ├── language: str
  ├── synthesis_latency_ms: int
  ├── audio_duration_ms: int
  └── audio_bytes: int
        │
        ▼
  telephony._process_turn()
  ├── if status == success → send audio
  ├── if status == fallback_used → log + send audio
  └── if status == failed → log typed reason + drop turn gracefully
```

---

## Component-by-Component Plan

### 1. TTSFailureReason (Typed Error Enum)

Replace the current two-case exception handling with a complete failure taxonomy.

**Current state:** `RuntimeError` for binary not found and non-zero exit code only.

**Proposed enum:**

```python
class TTSFailureReason(str, Enum):
    binary_not_found   = "binary_not_found"
    synthesis_failed   = "synthesis_failed"
    empty_output       = "empty_output"
    timeout            = "timeout"
    backend_unavailable = "backend_unavailable"
    unsupported_language = "unsupported_language"
```

Why each case matters:

- `binary_not_found` — Piper binary missing; configuration error, not a runtime error
- `synthesis_failed` — Piper exited non-zero; audio generation failed
- `empty_output` — Piper ran successfully but produced no audio samples; currently
  passes through silently and should be treated as a hard failure
- `timeout` — Piper exceeded the 30-second limit; currently raises a generic
  `subprocess.TimeoutExpired` that is not caught explicitly
- `backend_unavailable` — reserved for future multi-backend support
- `unsupported_language` — reserved for language routing when multiple voices are added

### 2. TTSSynthesisStatus

A simple success/failure/fallback status attached to every result.

```python
class TTSSynthesisStatus(str, Enum):
    success        = "success"
    fallback_used  = "fallback_used"
    failed         = "failed"
```

This mirrors the approach `MemoryAdaptiveSpeechSynthesizer` uses in
HealthCoacher, where the active backend and whether a fallback occurred
are always explicitly reported rather than inferred from exceptions.

### 3. TTSSynthesisResult

The unified response contract returned by every `synthesize()` call.

```python
@dataclass
class TTSSynthesisResult:
    audio: np.ndarray             # float32 PCM; empty array on failure
    status: TTSSynthesisStatus
    failure_reason: TTSFailureReason | None
    backend_name: str             # e.g. "piper", "espeak", "fallback"
    fallback_used: bool
    voice_used: str               # voice model path or name
    language: str                 # BCP-47 tag or "auto"
    synthesis_latency_ms: int     # wall-clock time for synthesis call
    audio_duration_ms: int        # duration of produced audio
    audio_bytes: int              # byte size of raw PCM output
```

This is the TTS equivalent of the STT result schema changes proposed in the
STT normalization plan (`language_confidence`, `transcript_quality_score`,
`backend_name`, `fallback_used`, `failure_reason`).

### 4. Output Quality Gate

Add an explicit empty-output check inside `_synthesize()` before returning.

**Current state:** empty stdout produces an empty numpy array that is
returned without error.

**Proposed change:** after converting stdout to samples, check length:

```python
if len(samples) == 0:
    raise TTSSynthesisError(TTSFailureReason.empty_output)
```

This matches `F5TTSCoreMLService` in HealthCoacher, which guards:

```swift
guard !waveform.isEmpty else {
    throw Self.makeConfigurationError(description: "F5 Core ML vocoder returned an empty waveform.")
}
```

### 5. TTSStatusReporter (Observability Layer)

A lightweight per-turn reporter that logs structured TTS metadata after
each synthesis call. Mirrors the `DebugTelemetry` fields HealthCoacher
surfaces per turn.

Fields to log per turn:

| Field | Source |
|---|---|
| `tts_backend` | `TTSSynthesisResult.backend_name` |
| `tts_fallback_used` | `TTSSynthesisResult.fallback_used` |
| `tts_status` | `TTSSynthesisResult.status` |
| `tts_failure_reason` | `TTSSynthesisResult.failure_reason` |
| `tts_latency_ms` | `TTSSynthesisResult.synthesis_latency_ms` |
| `tts_audio_duration_ms` | `TTSSynthesisResult.audio_duration_ms` |
| `tts_audio_bytes` | `TTSSynthesisResult.audio_bytes` |
| `tts_voice` | `TTSSynthesisResult.voice_used` |
| `tts_language` | `TTSSynthesisResult.language` |

This gives Therfour the operational equivalent of HealthCoacher's
`ttsLatencyMs`, `ttsAudioBytes`, `ttsAudioDurationMs`, `ttsPlaybackStarted`,
and `ttsOutputRoute` telemetry fields.

---

## Concrete Repo Changes

### app/models/schemas.py

Add:

- `TTSFailureReason` enum
- `TTSSynthesisStatus` enum
- `TTSSynthesisResult` dataclass

These should live alongside STT schema additions proposed in the STT
normalization plan so all audio service contracts are in one place.

### app/services/tts.py

- Change `synthesize()` return type from `np.ndarray` to `TTSSynthesisResult`
- Add `synthesis_latency_ms` timing around the subprocess call
- Add explicit `empty_output` check after PCM conversion
- Add explicit `timeout` catch for `subprocess.TimeoutExpired`
- Populate `backend_name`, `voice_used`, and `language` from settings
- Keep `_synthesize()` raising typed `TTSSynthesisError`; wrap result
  construction in the async `synthesize()` layer

### app/services/telephony.py

- Update `_process_turn()` to read `TTSSynthesisResult.status`
- Distinguish TTS dropped turns from STT and LLM dropped turns in logs
- Emit `tts_failure_reason` as a structured log field
- Pass `TTSSynthesisResult` metadata to `TTSStatusReporter` after each turn
- Guard `_send_audio()` call on `status != failed`

### app/core/config.py

Add configuration for:

- `tts_backend` — active backend name for metadata reporting
- `tts_voice` — voice model identifier to populate `voice_used`
- `tts_language` — default language tag for metadata
- `tts_empty_output_min_samples` — minimum sample count before triggering
  `empty_output` failure

### tests/test_tts.py

Add tests for:

- `TTSSynthesisResult` is returned on success with correct metadata fields
- `failure_reason == empty_output` when Piper returns no audio
- `failure_reason == timeout` when subprocess exceeds limit
- `synthesis_latency_ms` is populated and greater than zero
- `audio_duration_ms` and `audio_bytes` are consistent with output length
- telephony turn log distinguishes TTS failure from STT/LLM failure

---

## Rollout Phases

### Phase 1

Ship `TTSSynthesisResult`, `TTSFailureReason`, and `TTSSynthesisStatus`.
Update `tts.py` to return the result wrapper. Update `telephony.py` to
read status and log structured failure reasons.

Success condition:

- TTS failures are distinguishable from STT and LLM failures in logs
- Empty audio output is detected and rejected before caller delivery

### Phase 2

Add `TTSStatusReporter` and per-turn structured logging.

Success condition:

- `tts_latency_ms`, `tts_audio_duration_ms`, and `tts_backend` are emitted
  per turn in structured logs

### Phase 3

Extend schema to support multi-backend reporting once a fallback engine
is introduced.

Success condition:

- `fallback_used` and `backend_name` correctly reflect active backend
  across primary and fallback synthesis paths

---

## Acceptance Criteria

1. `tts.synthesize()` returns `TTSSynthesisResult` in all code paths,
   success and failure.
2. Empty audio output sets `failure_reason = empty_output` and does not
   reach the caller.
3. Subprocess timeout sets `failure_reason = timeout` with a structured
   log entry.
4. TTS failure log entries include a typed `failure_reason` field distinct
   from STT and LLM failures.
5. Per-turn logs include `tts_latency_ms`, `tts_audio_duration_ms`,
   `tts_audio_bytes`, and `tts_backend`.
6. All new schema types are defined in `app/models/schemas.py` alongside
   the STT schema additions.
7. Tests cover result metadata correctness, empty output rejection, timeout
   handling, and structured failure logging.

---

## Alignment with STT Normalization Plan

This plan mirrors the STT normalization plan in structure and intent:

| STT Plan | TTS Equivalent |
|---|---|
| `backend_name` field | `TTSSynthesisResult.backend_name` |
| `fallback_used` field | `TTSSynthesisResult.fallback_used` |
| `failure_reason` enum | `TTSFailureReason` enum |
| Transcript quality gate | Empty output quality gate |
| Structured dropped-turn reasons in telephony | TTS-typed failure logging in `_process_turn()` |
| `language_confidence` semantics correction | `language` field explicit BCP-47 tag |

Both plans share the same goal: replace broad exception swallowing with
typed, observable, structured result contracts at every audio service boundary.