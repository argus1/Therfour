# STT Parity Analysis: Therfour (Current) vs HealthCoacher (Target Model)

Date: 2026-04-15
Author: Engineering analysis

## Scope

This analysis compares speech-to-text behavior between:

- Therfour current implementation
- HealthCoacher target implementation pattern

Focus areas:

- Retries and recovery strategy
- Confidence handling
- Failure paths and fallback behavior

Primary evidence reviewed:

- Therfour: app/services/stt.py
- Therfour: app/services/telephony.py
- Therfour: tests/test_stt.py
- Therfour: swift-backend/Sources/swift-backend/CallTurnProcessor.swift
- Therfour: swift-backend/Sources/swift-backend/VoiceContracts.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/Audio/WhisperKitTranscriber.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/Audio/AudioServices.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/Chat/ChatViewModel.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/App/AppContainer.swift
- HealthCoacher docs: ios-avatar-rag-prototype/Docs/Eliminate_Whisper_http_calls.md

## Executive Summary

Therfour STT is currently single-pass and minimal: one transcription call, no retry policy, no confidence gating, and broad exception capture at turn orchestration level.

HealthCoacher models STT as a resilient subsystem: multi-strategy decode attempts, decode-level quality thresholds, sticky memory-aware fallback to system speech, and graceful degradation from live partial streaming to batch final transcription.

Net: Therfour has a clear parity gap in reliability engineering and operational failure semantics, not just model/runtime choice.

## Parity Matrix

| Dimension                       | Therfour current                                                                                 | HealthCoacher model                                                                                                                            | Gap severity |
| ------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| Retry strategy                  | No explicit retries around Whisper decode                                                        | Multiple decode strategies evaluated in sequence, with temperature fallback counts and best-attempt selection                                  | High         |
| Confidence handling             | Returns language_probability as confidence; no thresholding or routing logic based on confidence | Does not expose confidence score directly, but applies decode-quality gates (log prob, no-speech, compression thresholds) and strategy scoring | High         |
| Live streaming failure handling | Not applicable in Therfour telephony path (single turn on silence boundary)                      | Live chunk transcription failures are non-fatal and continue capture; finalization falls back to file transcription                            | Medium       |
| Session fallback behavior       | No STT backend fallback                                                                          | MemoryAdaptiveSpeechTranscriber flips to sticky system fallback on memory warning or inference error                                           | High         |
| Error taxonomy                  | Python exceptions bubble to telephony turn handler; broad catch logs and drops turn              | Typed audio service errors plus app-level fallback and backend-status visibility                                                               | High         |
| Observability of active backend | Implicit, not surfaced in STT contract                                                           | Explicit active backend reporting via STTBackendStatusProvider                                                                                 | Medium       |

## Deep Dive

### 1) Retries and Recovery

Therfour current behavior:

- app/services/stt.py performs one call to model.transcribe per request.
- No retry loop around decode or model invocation.
- If STT fails, exception escapes to telephony orchestrator.
- app/services/telephony.py catches broad exceptions in \_process_turn and logs error; the turn is dropped.

HealthCoacher model behavior:

- WhisperKitTranscriber builds multiple decoding strategies.
- Strategies vary language detection behavior, prefill prompt usage, thresholds, and temperature fallback count.
- bestAttempt executes strategies in sequence, returns satisfactory attempt early, otherwise keeps highest-scoring attempt.
- If all fail and no text is recovered, last error is propagated.

Parity implication:

- Therfour has no second chance when decode quality is poor or unstable.
- HealthCoacher intentionally trades a little complexity/latency for improved transcription resilience and salvage behavior.

### 2) Confidence Handling

Therfour current behavior:

- Confidence output equals Whisper language_probability.
- This value represents language ID confidence, not transcript certainty.
- No confidence thresholds to suppress low-quality transcripts.
- No fallback branching based on confidence.

HealthCoacher model behavior:

- No explicit confidence score is exposed through the STT public result path.
- Quality control is applied during decode via noSpeechThreshold, logProbThreshold, and compressionRatioThreshold.
- Strategy quality is indirectly evaluated by transcript adequacy and scoring.

Parity implication:

- Therfour currently has a potentially misleading confidence field semantics if interpreted as transcript confidence.
- HealthCoacher model emphasizes decode-time quality gating over post-hoc numeric confidence exposure.

### 3) Failure Paths

Therfour current behavior:

- Telephony pipeline catches all exceptions at turn level; no typed STT failure code paths.
- Turn processing fails closed (no user-facing fallback behavior).
- No sticky fallback backend for constrained devices or repeated failures.
- Empty transcript path is handled by returning early, but low-quality non-empty transcripts still proceed.

HealthCoacher model behavior:

- MemoryAdaptiveSpeechTranscriber switches to fallback system STT after memory warning or STT errors.
- Fallback is sticky for the session to avoid repeated failure loops.
- Live chunk errors are non-fatal; recording continues.
- Finalization tries live finalize first, then falls back to file-based full transcription.

Parity implication:

- HealthCoacher has explicit graceful degradation paths at chunk, finalize, and session levels.
- Therfour currently relies on exception logging and turn abort, with no recovery path.

## What Therfour Should Borrow to Reach HealthCoacher-Level Modeling

### A) Retry and decode-policy layer (high priority)

Add explicit decode attempt policy in Therfour STT service:

- Primary decode with current defaults.
- One or more fallback decode attempts with relaxed thresholds and/or language-detection mode.
- Return best usable attempt based on transcript quality heuristic.

### B) Confidence semantics correction (high priority)

Either:

- Rename confidence to language_confidence, or
- Add transcript_confidence separately and only expose it when meaningfully computed.

Then implement confidence/quality gates before downstream LLM call:

- Minimum transcript length
- No-speech threshold signal
- Optional quality score floor

### C) Failure taxonomy and fallback behavior (high priority)

Introduce typed STT failure reasons and explicit fallback policy:

- transient_decode_failure
- no_speech
- unsupported_language
- backend_unavailable

Add fallback behavior:

- On repeated STT failures in a call session, switch to fallback STT backend or safer decode mode.
- Keep fallback sticky for session stability.

### D) Turn-level orchestration hardening (medium priority)

In telephony turn processor:

- Distinguish STT failures from LLM/TTS failures in logs and metrics.
- Emit structured reason codes for dropped turns.
- Avoid only broad exception logging as the primary failure path.

## Proposed Acceptance Criteria for STT Parity Work Item

1. Therfour STT supports at least one fallback decode attempt path and records attempt count.
2. Confidence field semantics are explicit and non-misleading.
3. No-speech and low-quality transcript handling is gated before LLM generation.
4. Turn processor distinguishes and reports STT failure reasons.
5. Session-level fallback behavior exists for repeated STT errors.
6. Tests cover:
   - primary decode success
   - fallback decode success after primary failure
   - no-speech rejection
   - sticky fallback activation
   - structured STT failure reporting

## Risks if No Change is Made

- Higher dropped-turn rate under noisy audio and edge conditions.
- Misinterpretation of language_probability as transcript certainty.
- Reduced operational visibility into STT-specific failures.
- Lower parity confidence against HealthCoacher resilience expectations.

## Recommendation

Treat STT parity as a reliability alignment effort, not only model parity.

HealthCoacher demonstrates the target pattern:

- layered decode strategy
- decode-quality gating
- typed failure semantics
- sticky adaptive fallback

Therfour should adopt these patterns incrementally, starting with decode retries and explicit failure taxonomy in the current Python STT + telephony pipeline.
