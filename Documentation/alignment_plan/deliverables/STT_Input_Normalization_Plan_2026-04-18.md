# STT Input Normalization Plan: TherFour

Date: 2026-04-18
Author: Engineering plan

## Objective

Define a practical STT input-normalization architecture for TherFour that balances:

- low latency for conversational hotline calls
- open-source deployability for a wide user base
- good behavior on both GPU-enabled servers and CPU-only installs
- alignment with the STT parity gaps identified on 2026-04-15

## Recommendation Summary

Use a layered design, not a single monolithic STT path.

- Silero VAD: mandatory input-normalization layer
- Whisper: canonical final transcript backend
- Sherpa-ONNX: optional streaming and fallback backend

This is feasible and preferable to forcing all three into one blocking serial chain.

## Why Not Run All Three Serially for Every Turn

Running Silero VAD, then Sherpa-ONNX, then Whisper for every finalized utterance is technically possible but not operationally ideal.

Downsides:

- extra latency per turn
- duplicated compute cost
- more failure surfaces
- more complex debugging and observability

For a hotline agent, user experience is dominated by:

- fast end-of-speech detection
- low false-trigger rate
- stable final transcript quality

That makes VAD plus one primary final decoder the right default.

## Recommended Production Architecture

### Default Path

1. Twilio audio arrives as mu-law 8 kHz.
2. Decode and upsample to the working rate.
3. Run Silero VAD on short rolling frames.
4. Keep only voiced segments and finalize utterances using VAD hangover logic.
5. Send voiced utterance to Whisper for final transcript.
6. Apply transcript quality gates before invoking the LLM.

### Optional Augmented Path

Sherpa-ONNX is enabled only for one of these cases:

- live partial transcripts during speech
- fallback when Whisper fails repeatedly or is unavailable

### Priority If Scope Must Be Reduced

1. Silero VAD
2. Whisper retry and quality policy
3. Sherpa-ONNX integration

## Component-by-Component Plan

### 1. Silero VAD

Role:

- speech onset detection
- speech end detection
- silence trimming
- no-speech suppression

Why it should be mandatory:

- biggest latency and quality win for telephony audio
- open-source and lightweight
- directly addresses false turns and wasted Whisper invocations

Implementation notes:

- frame size: 20-30 ms
- keep a rolling buffer of recent audio
- add hangover logic so brief pauses do not split turns unnaturally
- preserve a small pre-roll to avoid clipping initial phonemes

Expected benefits:

- reduced dead-air transcription
- lower effective turn latency than fixed `silence_timeout_s`
- cleaner audio handed to downstream STT

### 2. Whisper

Role:

- final transcript generation
- language metadata
- primary production STT output

Why it stays primary:

- already integrated
- strong multilingual robustness
- good quality on noisy telephony inputs when paired with correct segmentation
- faster-whisper supports GPU acceleration while remaining open-source

Required changes:

- add decode policy with at least one fallback attempt
- stop treating language probability as transcript confidence
- add transcript quality gating before LLM handoff

Recommended final-transcript policy:

- primary decode: preferred language policy and current defaults
- fallback decode: more permissive language mode or adjusted decode strategy
- best-attempt selection with rejection for no-speech or low-quality text

### 3. Sherpa-ONNX

Role:

- low-latency partial transcript engine
- CPU-safe fallback backend

Why it is useful but not mandatory for sprint 1:

- adds a second STT backend and more integration complexity
- does not replace the immediate value of fixing segmentation and Whisper behavior first
- is strongest when the product needs live partials or broad CPU support

Recommended usage:

- feature-flagged backend
- session-sticky fallback after repeated Whisper failures
- optional partial transcript stream if future UX requires it

## Deployment Strategy

### GPU-Enabled Servers

Default:

- Silero VAD
- faster-whisper on CUDA
- Sherpa-ONNX disabled by default unless partials or fallback are explicitly needed

Reason:

- best balance of quality and latency
- simplest mainline operations path

### CPU-Only Servers

Default:

- Silero VAD
- smaller faster-whisper model or lower-cost decode profile
- optional Sherpa-ONNX fallback if CPU latency is more stable in benchmarks

Reason:

- maintains broad OSS accessibility
- lets deployments choose quality vs latency tradeoffs explicitly

## Concrete Repo Changes

### app/core/config.py

Add configuration for:

- `stt_backend`
- `stt_fallback_backend`
- `vad_enabled`
- `vad_threshold`
- `vad_min_speech_ms`
- `vad_min_silence_ms`
- `vad_preroll_ms`
- `whisper_primary_model`
- `whisper_fallback_policy`
- `sherpa_enabled`

### app/services/telephony.py

Add:

- rolling frame buffer
- Silero VAD gating
- VAD-based utterance finalization
- structured dropped-turn reasons

### app/services/stt.py

Refactor to support:

- backend abstraction
- Whisper retry policy
- transcript quality gates
- optional Sherpa-ONNX fallback

### app/models/schemas.py

Add or rename fields so result semantics are explicit:

- `language_confidence`
- `transcript_quality_score`
- `backend_name`
- `fallback_used`
- `failure_reason`

### swift-backend/Sources/swift-backend/VoiceContracts.swift

Mirror Python contract changes so STT behavior is consistent across stacks.

## Rollout Phases

### Phase 1

Ship Silero VAD plus Whisper hardening.

Success condition:

- improved latency and fewer bad turns without adding a new primary STT backend

### Phase 2

Integrate Sherpa-ONNX behind a feature flag.

Success condition:

- session fallback and optional partial transcript support work without API churn

### Phase 3

Benchmark and set policy defaults.

Compare:

- Silero VAD + Whisper
- Silero VAD + Sherpa-ONNX only
- Silero VAD + Sherpa-ONNX partials + Whisper final

Benchmark dimensions:

- P50 and P95 end-of-speech-to-final-transcript latency
- transcript quality on telephony/noisy audio
- no-speech rejection quality
- failure and retry rate
- CPU and GPU resource cost

## Acceptance Criteria

1. No-speech audio is rejected before LLM invocation.
2. Fixed silence timeout is no longer the main turn-finalization mechanism.
3. Whisper supports fallback decode behavior.
4. Transcript result metadata distinguishes backend, quality, and failure reasons.
5. Sherpa-ONNX can be enabled as fallback or streaming without changing public API shape.
6. Tests cover VAD segmentation, fallback decode, and sticky backend fallback.

## Final Recommendation

For TherFour sprint 1 and likely initial production:

- ship Silero VAD + Whisper as the default STT normalization path
- add Sherpa-ONNX as an optional fallback and streaming layer

This is the best fit for a low-latency, open-source hotline agent targeting a broad server deployment base.
