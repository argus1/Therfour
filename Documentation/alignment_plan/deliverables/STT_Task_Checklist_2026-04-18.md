# STT Task Checklist: TherFour

Date: 2026-04-18
Scope: STT input normalization, fallback behavior, and validation

## Status Summary

Phase 1 implementation is complete in code and validated with focused Python and Swift tests.

Completed validation:

- `pytest tests/test_stt.py tests/test_telephony.py`
- `cd swift-backend && swift test`

## Phase 1 Checklist

### Config Surface

- [x] Add VAD settings in `app/core/config.py`
- [x] Add Whisper fallback and quality-gate settings in `app/core/config.py`
- [ ] Add `.env.example` entries for the new STT/VAD settings
- [ ] Document production defaults in `README.md`

### STT Contract

- [x] Expand `TranscriptionResult` in `app/models/schemas.py`
- [x] Preserve backward compatibility with existing `confidence`
- [x] Add explicit metadata fields:
  - `language_confidence`
  - `transcript_quality_score`
  - `backend_name`
  - `fallback_used`
  - `failure_reason`
- [x] Mirror the contract in `swift-backend/Sources/swift-backend/Models.swift`
- [x] Update Swift tests in:
  - `swift-backend/Tests/swift-backendTests/swift_backendTests.swift`
  - `swift-backend/Tests/swift-backendTests/turn_processorTests.swift`

### VAD Integration

- [x] Add streaming Silero wrapper in `app/services/vad.py`
- [x] Add runtime fallback when Silero is unavailable
- [x] Add frame buffering and preroll support in `app/services/vad.py`
- [x] Add VAD-driven turn finalization in `app/services/telephony.py`
- [x] Keep silence-timeout fallback path for non-VAD runtime
- [ ] Add explicit end-of-call flush behavior tests for buffered speech
- [ ] Add observability counters for:
  - no-speech drops
  - VAD finalized turns
  - fallback to silence timeout

### Whisper Hardening

- [x] Add multi-attempt decode policy in `app/services/stt.py`
- [x] Add transcript quality scoring in `app/services/stt.py`
- [x] Reject no-speech and low-quality output before LLM handoff
- [x] Mark fallback usage in the STT result metadata
- [ ] Add attempt count to result metadata or logs
- [ ] Add structured metrics for:
  - decode latency
  - primary success rate
  - fallback success rate
  - low-quality rejection rate

### Telephony Integration

- [x] Route voiced spans directly into the turn processor in `app/services/telephony.py`
- [x] Preserve legacy buffered decode path in `app/services/telephony.py`
- [x] Add structured dropped-turn logging in `app/services/telephony.py`
- [x] Queue finalized voiced turns while another turn is processing
- [ ] Add end-to-end integration tests for multi-turn queueing under VAD mode
- [ ] Distinguish STT, LLM, and TTS failure metrics in logs or health output

### Dependency and Packaging

- [x] Add `silero-vad` to `requirements.txt`
- [ ] Verify Docker image includes all runtime dependencies for Silero VAD
- [ ] Verify CPU-only install path remains documented and supported

### Python Tests

- [x] Add fallback-decode coverage in `tests/test_stt.py`
- [x] Add low-quality rejection coverage in `tests/test_stt.py`
- [x] Add VAD speech-finalization coverage in `tests/test_telephony.py`
- [x] Add VAD flush coverage in `tests/test_telephony.py`
- [ ] Add a call-session test that exercises `CallSession._on_media()` with VAD enabled

### Swift Tests

- [x] Update transcription contract encoding test in `swift-backend/Tests/swift-backendTests/swift_backendTests.swift`
- [x] Update turn-processor tests for expanded transcription fields in `swift-backend/Tests/swift-backendTests/turn_processorTests.swift`

## Phase 2 Checklist

### Sherpa-ONNX Fallback Track

- [ ] Add `sherpa-onnx` backend abstraction in `app/services/stt.py`
- [ ] Add backend selection config in `app/core/config.py`
- [ ] Add session-sticky fallback policy in `app/services/telephony.py`
- [ ] Add tests for repeated Whisper failure leading to Sherpa fallback
- [ ] Add backend status reporting in Python and Swift contracts if needed

### Optional Live Partial Track

- [ ] Prototype Sherpa-ONNX partial transcript flow for live speech
- [ ] Define whether partial transcripts affect hotline UX or agent turn policy
- [ ] Add guardrails so partials never replace the canonical final transcript unintentionally

## Phase 3 Checklist

### Benchmarking

- [ ] Create a reproducible benchmark corpus of telephony-quality audio
- [ ] Measure P50 and P95 end-of-speech-to-final-transcript latency
- [ ] Compare:
  - Silero VAD + Whisper
  - Silero VAD + Sherpa-ONNX only
  - Silero VAD + Sherpa-ONNX partials + Whisper final
- [ ] Record CPU and GPU hardware context with every benchmark run

### Production Default Decision

- [ ] Choose default Whisper tier for GPU servers
- [ ] Choose default Whisper tier for CPU-only installs
- [ ] Decide whether Sherpa-ONNX is promoted to supported fallback by default

## Validation Commands

### Python

```bash
/Users/argussun/Documents/Therfour/.venv/bin/python -m pytest tests/test_stt.py tests/test_telephony.py
```

### Swift

```bash
cd /Users/argussun/Documents/Therfour/swift-backend
swift test
```

## Immediate Next Tasks

1. Add `.env.example` and `README.md` updates for the new VAD and STT settings.
2. Add a call-session level test that feeds media frames through VAD mode.
3. Add observability for primary decode, fallback decode, and rejected turns.
4. Benchmark Whisper model tiers on the target deployment hardware.
