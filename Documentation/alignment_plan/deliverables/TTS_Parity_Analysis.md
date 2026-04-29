# TTS Parity Analysis: Therfour (Current) vs HealthCoacher (Target Model)


## Scope

This analysis compares text-to-speech behavior between:

- **Therfour** current implementation
- **HealthCoacher** target implementation pattern

Focus areas:

- Architecture and abstraction design
- Retry and recovery strategy
- Fallback behavior
- Error taxonomy
- Observability
- Language and voice routing

Primary evidence reviewed:

- Therfour: `app/services/tts.py`
- Therfour: `app/services/telephony.py`
- Therfour: `tests/test_tts.py`
- HealthCoacher: `iOSApp/Sources/Audio/AudioServices.swift`
- HealthCoacher: `iOSApp/Sources/BackendCapabilities.swift`
- HealthCoacher: `iOSApp/Sources/DebugTelemetry.swift`
- HealthCoacher: `iOSApp/Sources/LanguageSupport.swift`

---

## Executive Summary

Therfour TTS is currently single-pass and minimal: one Piper subprocess call, no
retry policy, no fallback engine, no output quality check, and no error distinction
from the rest of the pipeline. TTS failures are silent from the caller's perspective.

HealthCoacher models TTS as a resilient, layered subsystem: a formal protocol
abstraction, multiple concrete backends, a memory-adaptive fallback wrapper that
switches to Apple system TTS on failure or memory pressure, per-language voice
routing, typed capability probing, and rich observability surfaced in the debug UI.

Net: Therfour has a significant parity gap across architecture, reliability
engineering, and operational visibility — not just engine choice.

---

## Parity Matrix

| Dimension | Therfour Current | HealthCoacher Model | Gap Severity |
|---|---|---|---|
| Architecture | Bare async function, no protocol | `SpeechSynthesizer` protocol with multiple concrete backends | High |
| Retry strategy | No retry — single subprocess call | No explicit retry, but `MemoryAdaptiveSpeechSynthesizer` automatically switches to fallback on any error | High |
| Fallback engine | None | `SystemSpeechSynthesizer` (Apple AVSpeech) as sticky session fallback | High |
| Memory pressure handling | None | Listens for `UIApplication.didReceiveMemoryWarningNotification`, switches to fallback immediately | High |
| Error taxonomy | Two error types only (`FileNotFoundError`, non-zero exit) | Typed `AudioServiceError`, plus descriptive `NSError` from `F5TTSCoreMLService` for asset/model/shape failures | High |
| Output quality check | None — empty or malformed audio passes through | `F5TTSCoreMLService` validates waveform is non-empty before returning; throws on empty output | Medium |
| Language / voice routing | None — single hardcoded voice model | `LanguageAwareSpeechSynthesizer` protocol; per-language `defaultTTSVoiceHint`; voice resolved by hint, language tag, then default | High |
| Capability probing | None | `TTSCapabilityProbe` protocol; each backend reports supported languages, voice hint support, and notes | Medium |
| Active backend visibility | None | `TTSBackendStatusProvider` protocol; `activeTTSBackend` string surfaced in `DebugTelemetry.ttsBackend` | Medium |
| Observability | None | `ttsLatencyMs`, `ttsAudioBytes`, `ttsAudioDurationMs`, `ttsPlaybackStarted`, `ttsOutputRoute` all tracked in telemetry | High |
| Streaming TTS | None | `StreamingSpeechPlaybackCapable` protocol; `SystemSpeechSynthesizer` speaks chunks incrementally via `AVSpeechSynthesizer` | Medium |
| Pipeline error isolation | TTS errors indistinguishable from STT/LLM errors in one broad `except` | TTS errors are typed and surface separately through the audio service layer | High |
| Test coverage | 3 tests: happy path, binary missing, non-zero exit only | Not reviewed, but typed error contracts and protocol design enable targeted unit testing per backend | Medium |

---

## Deep Dive

### 1) Architecture and Abstraction

**Therfour current behavior:**

TTS is implemented as two plain functions: `_synthesize()` (synchronous) and
`synthesize()` (async wrapper). There is no protocol, no interface contract, and
no separation between the engine and its caller. The telephony pipeline calls
`tts.synthesize()` directly with no indirection.

**HealthCoacher model behavior:**

TTS is defined by the `SpeechSynthesizer` protocol:

```swift
protocol SpeechSynthesizer {
    func synthesize(text: String, voice: String) async throws -> Data
}
```

Three concrete backends implement this contract:

- `F5TTSHTTPService` — calls a locally served F5-TTS-MLX HTTP endpoint
- `F5TTSCoreMLService` — runs F5-TTS fully on-device via Core ML (acoustic model + vocoder)
- `SystemSpeechSynthesizer` — uses Apple `AVSpeechSynthesizer` as a system fallback

All three are wrapped by `MemoryAdaptiveSpeechSynthesizer`, which manages
primary/fallback switching transparently.

**Parity implication:**

Therfour cannot swap or extend its TTS engine without modifying call sites.
HealthCoacher can add, replace, or reroute backends without touching the
pipeline layer.

---

### 2) Retry and Fallback Behavior

**Therfour current behavior:**

`_synthesize()` runs one `subprocess.run()` call. If it fails for any reason,
an exception is raised and bubbles up to `_process_turn()` in `telephony.py`,
where a single broad `except Exception` catches everything, logs it, and
silently drops the caller's turn. There is no retry and no fallback engine.

**HealthCoacher model behavior:**

`MemoryAdaptiveSpeechSynthesizer` wraps the primary backend with
`SystemSpeechSynthesizer` as a sticky fallback:

```swift
func synthesize(text: String, voice: String, language: SupportedLanguage) async throws -> Data {
    if useFallbackForSession {
        return try await fallback.synthesize(...)
    }
    do {
        return try await primary.synthesize(...)
    } catch {
        useFallbackForSession = true
        return try await fallback.synthesize(...)
    }
}
```

Once the fallback is activated, it stays active for the entire session to
prevent repeated failure loops. Additionally, an iOS memory warning
(`UIApplication.didReceiveMemoryWarningNotification`) proactively triggers
the fallback before any synthesis failure occurs.

**Parity implication:**

Therfour has no second chance when TTS fails. HealthCoacher ensures the
user always receives a spoken response, even if the primary engine is
unavailable or memory-constrained.

---

### 3) Error Taxonomy

**Therfour current behavior:**

Only two error conditions are named:

- `FileNotFoundError` → binary missing
- Non-zero `returncode` → Piper process failed

Both raise a generic `RuntimeError`. In `telephony.py`, these are caught
alongside STT and LLM errors by a single `except Exception` block. There is
no way to distinguish a TTS failure from any other pipeline failure in logs.

**HealthCoacher model behavior:**

Errors are typed at the audio service layer:

```swift
enum AudioServiceError: Error {
    case invalidResponse
    case httpError(Int)
    case decodingError
    case unsupported
}
```

`F5TTSCoreMLService` additionally throws descriptive `NSError` values for
asset-level failures: missing acoustic model, missing vocoder, empty waveform
output, unsupported mel tensor shape, missing vocabulary file, and reference
audio sample rate mismatch. Each error carries a human-readable description
pointing to the specific missing asset or configuration problem.

**Parity implication:**

Therfour has no operational visibility into why TTS failed. HealthCoacher
surfaces the exact failure cause, enabling faster diagnosis in production.

---

### 4) Output Quality Check

**Therfour current behavior:**

After Piper runs, the raw stdout bytes are converted to a float32 array and
returned with no validation. An empty stdout (zero-length audio) would
produce an empty array that proceeds through the pipeline and gets sent
back to the caller as silence.

**HealthCoacher model behavior:**

`F5TTSCoreMLService` explicitly checks the vocoder output before returning:

```swift
guard !waveform.isEmpty else {
    throw Self.makeConfigurationError(description: "F5 Core ML vocoder returned an empty waveform.")
}
```

Empty output is treated as a hard failure, not a silent pass-through.

**Parity implication:**

Therfour can silently deliver empty audio to a caller with no error logged
at the TTS layer. HealthCoacher fails loudly and triggers fallback recovery.

---

### 5) Language and Voice Routing

**Therfour current behavior:**

Piper uses a single hardcoded voice model (`en_US-lessac-medium.onnx`).
There is no language detection, no voice selection logic, and no support
for routing different languages to different voices.

**HealthCoacher model behavior:**

Language-awareness is built into the TTS contract via `LanguageAwareSpeechSynthesizer`:

```swift
protocol LanguageAwareSpeechSynthesizer: SpeechSynthesizer {
    func synthesize(text: String, voice: String, language: SupportedLanguage) async throws -> Data
}
```

Each `SupportedLanguage` defines a `defaultTTSVoiceHint` (e.g. `"fa_default"`,
`"zh_mandarin_default"`). `F5TTSCoreMLService` resolves the voice asset by
exact hint match, then language prefix match, then manifest default, then
first available — a graceful resolution chain rather than a hard failure.

**Parity implication:**

Therfour cannot serve multilingual callers without code changes.
HealthCoacher's voice routing is data-driven and language-aware by design —
directly relevant given Therfour's stated multilingual goal.

---

### 6) Observability

**Therfour current behavior:**

No TTS-specific observability exists. The only logging is a broad exception
log in `_process_turn()` if anything in the turn pipeline fails. There is no
way to know from logs whether TTS succeeded, how long it took, or how much
audio was produced.

**HealthCoacher model behavior:**

`DebugTelemetry` tracks the following TTS-specific fields, surfaced live in
the debug UI:

| Field | What it tracks |
|---|---|
| `ttsBackend` | Which backend is currently active |
| `ttsLatencyMs` | Time taken for synthesis |
| `ttsAudioBytes` | Size of audio returned |
| `ttsAudioDurationMs` | Duration of synthesized audio |
| `ttsPlaybackStarted` | Whether playback actually began |
| `ttsOutputRoute` | Audio output route (speaker, headphones, etc.) |

`TTSBackendStatusProvider` and `TTSCapabilityProbe` protocols expose the
active backend label and its capabilities to the rest of the app.

**Parity implication:**

Therfour has zero TTS observability. HealthCoacher can diagnose TTS issues
in real time without log diving.

---

## What Therfour Should Borrow to Reach HealthCoacher-Level Modeling

### A) Introduce a TTS protocol abstraction (high priority)

Define a `SpeechSynthesizer` interface in Python to decouple the Piper
implementation from the telephony pipeline. This enables backend swapping
and makes fallback wiring straightforward.

### B) Add a fallback TTS engine (high priority)

Implement a system TTS fallback (e.g. `pyttsx3` or `espeak`) activated on
Piper failure. Make the fallback sticky for the session to mirror
HealthCoacher's `MemoryAdaptiveSpeechSynthesizer` pattern.

### C) Introduce typed TTS error codes (high priority)

Replace generic `RuntimeError` with typed failure reasons:

- `binary_not_found`
- `synthesis_failed`
- `empty_output`
- `timeout`

Distinguish TTS errors from STT and LLM errors in `_process_turn()` logging.

### D) Add output quality validation (medium priority)

Check that synthesized audio is non-empty and above a minimum duration
threshold before sending to the caller. Treat empty output as a hard
failure that triggers fallback rather than silent pass-through.

### E) Add language and voice routing (medium priority)

Map `SupportedLanguage` values to Piper voice models. Allow the call session
to pass a language hint into `synthesize()`, resolving to the appropriate
voice model with a defined fallback chain.

### F) Add TTS-specific observability (medium priority)

Log TTS latency, output byte count, audio duration, and active backend per
turn. Distinguish TTS failures from other pipeline failures in structured logs.

---

## Proposed Acceptance Criteria for TTS Parity Work Item

1. Therfour TTS is defined behind a protocol/interface, not a bare function call.
2. At least one fallback TTS engine exists and activates on primary failure.
3. Fallback is sticky for the session once activated.
4. Empty audio output is detected and treated as a failure, not silent pass-through.
5. TTS failures are typed and distinguishable from STT and LLM failures in logs.
6. TTS latency and audio output are logged per turn.
7. Tests cover:
   - Primary synthesis success
   - Fallback activation on primary failure
   - Empty output rejection
   - Timeout handling
   - Typed failure reason logging

---

## Risks if No Change is Made

- Callers hear silence with no recovery when Piper fails or hangs
- TTS failures are invisible in logs and indistinguishable from LLM failures
- Multilingual callers are served by a single hardcoded English voice
- Empty audio output passes through silently with no alerting
- No operational baseline exists to detect TTS degradation over time

---

## Recommendation

Treat TTS parity as a reliability and multilingual-readiness effort.

HealthCoacher demonstrates the target pattern:

- Protocol-first abstraction enabling backend flexibility
- Memory-adaptive fallback with sticky session behavior
- Typed error taxonomy at the audio service layer
- Output quality gating before delivery
- Rich per-turn observability

Therfour should adopt these patterns incrementally, starting with typed error
taxonomy, a fallback engine, and output quality validation in the current
Python TTS and telephony pipeline.