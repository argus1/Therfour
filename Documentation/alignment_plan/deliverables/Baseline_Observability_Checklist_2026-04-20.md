# Baseline Observability Checklist (Pre-Refactor)

Date: 2026-04-20
Branch: argus-baseline-branch

Purpose: lock expected latency and failure-reason instrumentation behavior across STT, TTS, and RAG before broader refactors.

## Checklist

- [x] STT emits per-turn observability events with latency_ms and stage status.
  - Coverage: [app/services/stt.py](../../../app/services/stt.py)
  - Event shape includes stage=stt, status, latency_ms, and backend metadata.

- [x] STT emits explicit failure_reason values for dropped or failed transcription outcomes.
  - Coverage: [tests/test_stt.py](../../../tests/test_stt.py)
  - Assertions include no_speech and decode_error pathways.

- [x] TTS emits synthesis observability events with latency_ms and output volume metadata.
  - Coverage: [app/services/tts.py](../../../app/services/tts.py)
  - Event shape includes stage=tts, status, latency_ms, output_samples, and sample_rate.

- [x] TTS emits failure_reason when synthesis fails.
  - Coverage: [tests/test_tts.py](../../../tests/test_tts.py)
  - Assertions include synthesis_error failure instrumentation.

- [x] RAG generation path emits observability events with latency_ms and request size metadata.
  - Coverage: [app/services/llm.py](../../../app/services/llm.py)
  - Event shape includes stage=rag, status, latency_ms, message_count, and model metadata.

- [x] RAG generation path emits failure_reason when model generation fails or returns empty output.
  - Coverage: [tests/test_llm.py](../../../tests/test_llm.py)
  - Assertions include generation_error instrumentation for failed calls.

- [x] Shared observability helper normalizes event payload formatting across all stages.
  - Coverage: [app/services/observability.py](../../../app/services/observability.py)

## Notes

- This checklist focuses on baseline log instrumentation for latency and failure reasons.
- It does not yet include metrics export, dashboards, or alert thresholds.
