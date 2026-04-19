# Baseline Contract Checklist (Pre-Refactor)

Date: 2026-04-19
Branch: argus-baseline-branch

Purpose: lock expected STT and turn-processing contract behavior before broader refactors.

## Checklist

- [x] STT returns canonical metadata fields when transcription succeeds.
  - Coverage: [tests/test_stt.py](../../../tests/test_stt.py)
  - Assertions include language_confidence, backend_name, and failure_reason defaults.

- [x] STT returns no_speech for empty decode output.
  - Coverage: [tests/test_stt.py](../../../tests/test_stt.py)

- [x] STT executes fallback decode when primary decode yields no text.
  - Coverage: [tests/test_stt.py](../../../tests/test_stt.py)

- [x] STT rejects low-quality text and returns low_quality failure reason.
  - Coverage: [tests/test_stt.py](../../../tests/test_stt.py)

- [x] STT uses only one decode attempt when fallback is disabled.
  - Coverage: [tests/test_stt.py](../../../tests/test_stt.py)

- [x] STT fallback decode switches to auto-language after explicit language primary failure.
  - Coverage: [tests/test_stt.py](../../../tests/test_stt.py)

- [x] STT propagates terminal decode errors when all attempts fail.
  - Coverage: [tests/test_stt.py](../../../tests/test_stt.py)

- [x] Telephony turn pipeline drops too-short audio before STT/LLM/TTS processing.
  - Coverage: [tests/test_telephony.py](../../../tests/test_telephony.py)

- [x] Telephony turn pipeline drops no-speech results without calling LLM or TTS.
  - Coverage: [tests/test_telephony.py](../../../tests/test_telephony.py)

- [x] Swift contract decoding supports snake_case metadata keys from Python responses.
  - Coverage: [swift-backend/Tests/swift-backendTests/swift_backendTests.swift](../../../swift-backend/Tests/swift-backendTests/swift_backendTests.swift)

- [x] Swift contract decoding allows payloads with omitted optional metadata fields.
  - Coverage: [swift-backend/Tests/swift-backendTests/swift_backendTests.swift](../../../swift-backend/Tests/swift-backendTests/swift_backendTests.swift)

## Notes

- This checklist intentionally focuses on contract guarantees and failure semantics.
- It is not a full parity matrix for STT/TTS/RAG behavior.
