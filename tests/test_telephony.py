"""Tests for the telephony audio-pipeline utilities."""

from __future__ import annotations

import asyncio
import base64
from collections import deque

import numpy as np
import pytest

from app.services.telephony import (
    CallSession,
    _sanitize_model_reply,
    build_transfer_twiml,
    downsample,
    get_custom_transfer_post_call_reopen_mode,
    get_transfer_post_call_reopen_mode,
    mulaw_to_pcm16,
    parse_transfer_directive,
    pcm16_to_mulaw,
    upsample,
)
from app.services.telephony import _transfer_confirmation_prompt
from app.services import waiting_audio
from app.core.config import settings
from app.models.schemas import TranscriptionResult
from app.services.vad import StreamingSpeechDetector


def _make_pcm16(n_samples: int, value: int = 0) -> bytes:
    return np.full(n_samples, value, dtype=np.int16).tobytes()


def test_mulaw_round_trip() -> None:
    """Encoding then decoding μ-law should reproduce the original PCM closely."""
    # Use a simple sine wave
    t = np.linspace(0, 1, 8000, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    pcm_bytes = samples.tobytes()

    mulaw_bytes = pcm16_to_mulaw(pcm_bytes)
    recovered_bytes = mulaw_to_pcm16(mulaw_bytes)
    recovered = np.frombuffer(recovered_bytes, dtype=np.int16)

    # μ-law is lossy; allow a tolerance of ±512 LSBs
    assert len(recovered) == len(samples)
    max_error = np.abs(recovered.astype(np.int32) - samples.astype(np.int32)).max()
    assert max_error < 512, f"Max error {max_error} exceeds tolerance"


def test_upsample_doubles_length() -> None:
    """Upsampling 8 kHz PCM to 16 kHz should roughly double the sample count."""
    pcm = _make_pcm16(800)  # 100 ms at 8 kHz
    out = upsample(pcm, from_rate=8000, to_rate=16000)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    # resample_poly may produce slightly more or fewer samples than exactly 2×
    assert abs(len(out) - 1600) <= 2


def test_downsample_halves_length() -> None:
    """Downsampling 16 kHz to 8 kHz should roughly halve the sample count."""
    samples = np.zeros(1600, dtype=np.float32)  # 100 ms at 16 kHz
    out = downsample(samples, from_rate=16000, to_rate=8000)
    assert isinstance(out, bytes)
    n_samples = len(out) // 2  # 16-bit = 2 bytes per sample
    assert abs(n_samples - 800) <= 2


def test_upsample_same_rate_is_noop() -> None:
    """upsample with from_rate == to_rate should return the data unchanged."""
    pcm = _make_pcm16(160, value=1000)
    out = upsample(pcm, from_rate=8000, to_rate=8000)
    original = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    np.testing.assert_array_equal(out, original)


class _FakeIterator:
    def __init__(self, events: list[dict[str, int] | None]) -> None:
        self._events = deque(events)

    def __call__(self, x):
        return self._events.popleft() if self._events else None

    def reset_states(self) -> None:
        self._events.clear()


def test_streaming_speech_detector_finalizes_on_end_event() -> None:
    detector = StreamingSpeechDetector(
        vad_iterator=_FakeIterator([None, {"start": 0}, None, {"end": 1536}])
    )
    chunk = np.ones(2048, dtype=np.float32)

    finalized = detector.process_chunk(chunk)

    assert len(finalized) == 1
    assert finalized[0].dtype == np.float32
    assert len(finalized[0]) >= 1536


def test_streaming_speech_detector_flushes_active_speech() -> None:
    detector = StreamingSpeechDetector(vad_iterator=_FakeIterator([{"start": 0}, None]))
    chunk = np.ones(1024, dtype=np.float32)

    detector.process_chunk(chunk)
    flushed = detector.flush()

    assert flushed is not None
    assert flushed.dtype == np.float32
    assert len(flushed) >= 1024


class _DummyWebSocket:
    def __init__(self) -> None:
        self.sent_messages: list[str] = []

    async def iter_text(self):  # pragma: no cover - not used in this test
        if False:
            yield ""

    async def send_text(self, message: str) -> None:
        self.sent_messages.append(message)


class _FakeSpeechDetector:
    def __init__(self, finalized_turns: list[np.ndarray]) -> None:
        self._turns = list(finalized_turns)
        self.speech_active = False

    def process_chunk(self, samples: np.ndarray) -> list[np.ndarray]:
        assert isinstance(samples, np.ndarray)
        self.speech_active = bool(self._turns)
        return self._turns

    def flush(self):
        return None


@pytest.mark.asyncio
async def test_call_session_on_media_enqueues_vad_finalized_turn(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    finalized = np.ones(1600, dtype=np.float32)
    session._speech_detector = _FakeSpeechDetector([finalized])

    captured: list[np.ndarray] = []
    monkeypatch.setattr(session, "_enqueue_turn", lambda audio: captured.append(audio))

    pcm = np.full(320, 1000, dtype=np.int16).tobytes()
    mulaw = pcm16_to_mulaw(pcm)
    payload_b64 = base64.b64encode(mulaw).decode("ascii")

    await session._on_media(payload_b64)

    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], finalized)


@pytest.mark.asyncio
async def test_call_session_interrupts_active_turn_on_speech_start() -> None:
    websocket = _DummyWebSocket()
    session = CallSession(websocket)
    session._stream_sid = "MZ123"
    session._processing = True
    session._speech_detector = StreamingSpeechDetector(vad_iterator=_FakeIterator([{"start": 0}]))

    async def _long_running_turn() -> None:
        await asyncio.sleep(60)

    task = asyncio.create_task(_long_running_turn())
    session._active_turn_task = task

    pcm = np.full(320, 1000, dtype=np.int16).tobytes()
    mulaw = pcm16_to_mulaw(pcm)
    payload_b64 = base64.b64encode(mulaw).decode("ascii")

    await session._on_media(payload_b64)
    await asyncio.sleep(0)

    assert task.cancelled() is True
    assert any('"event": "clear"' in message for message in websocket.sent_messages)


@pytest.mark.asyncio
async def test_send_audio_emits_mark_for_twilio_transport() -> None:
    websocket = _DummyWebSocket()
    session = CallSession(websocket, transport_protocol="twilio")
    session._stream_sid = "MZ123"
    samples = np.zeros(22050, dtype=np.float32)

    await session._send_audio(samples)

    assert any('"event": "mark"' in message for message in websocket.sent_messages)


@pytest.mark.asyncio
async def test_send_audio_skips_mark_for_asterisk_transport() -> None:
    websocket = _DummyWebSocket()
    session = CallSession(websocket, transport_protocol="asterisk_ari")
    session._stream_sid = "MZ123"
    samples = np.zeros(22050, dtype=np.float32)

    await session._send_audio(samples)

    assert not any('"event": "mark"' in message for message in websocket.sent_messages)


@pytest.mark.asyncio
async def test_run_turn_drops_audio_shorter_than_minimum(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(100, dtype=np.float32)

    captured_reasons: list[str] = []
    monkeypatch.setattr(
        session,
        "_log_turn_drop",
        lambda reason, **kwargs: captured_reasons.append(reason),
    )

    await session._run_turn(audio)

    assert captured_reasons == ["too_short"]
    assert session._conversation == []


@pytest.mark.asyncio
async def test_run_turn_drops_no_speech_without_calling_llm(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="",
            language="en",
            confidence=0.0,
            language_confidence=0.0,
            transcript_quality_score=0.0,
            backend_name="faster-whisper",
            fallback_used=True,
            failure_reason="no_speech",
        )

    async def _unexpected_llm(_conversation):
        raise AssertionError("LLM should not be called for no-speech turns")

    async def _unexpected_tts(_text, *, language=None):
        raise AssertionError("TTS should not be called for no-speech turns")

    captured_reasons: list[str] = []
    monkeypatch.setattr(
        session,
        "_log_turn_drop",
        lambda reason, **kwargs: captured_reasons.append(reason),
    )
    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.llm.generate", _unexpected_llm)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _unexpected_tts)

    await session._run_turn(audio)

    assert captured_reasons == ["no_speech"]
    assert session._conversation == []


@pytest.mark.asyncio
async def test_run_turn_sticks_to_sherpa_after_first_sherpa_result(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)
    preferred_backends: list[str | None] = []

    responses = [
        TranscriptionResult(
            text="hello",
            language="en",
            confidence=0.9,
            language_confidence=0.0,
            transcript_quality_score=0.9,
            backend_name="sherpa-onnx",
            fallback_used=True,
            failure_reason="",
        ),
        TranscriptionResult(
            text="again",
            language="en",
            confidence=0.9,
            language_confidence=0.0,
            transcript_quality_score=0.9,
            backend_name="sherpa-onnx",
            fallback_used=True,
            failure_reason="",
        ),
    ]

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        preferred_backends.append(preferred_backend)
        return responses.pop(0)

    async def _fake_generate(_conversation, strategy=None, rag_allowed=None):
        return "ok"

    async def _fake_synthesize(_text, *, language=None):
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        return None

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.llm.generate", _fake_generate)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)

    await session._run_turn(audio)
    await session._run_turn(audio)

    assert preferred_backends == [None, "sherpa"]


@pytest.mark.asyncio
async def test_run_turn_plays_waiting_audio_during_rag_lookup(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "rag_waiting_audio_enabled", True)
    monkeypatch.setattr(settings, "rag_waiting_audio_delay_s", 0.0)

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="tell me about safer use",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    async def _fake_generate(_conversation, strategy=None, rag_allowed=None):
        await asyncio.sleep(0.01)
        return "I can help with that."

    async def _fake_synthesize(_text, *, language=None):
        return np.zeros(22050, dtype=np.float32)

    send_lengths: list[int] = []

    async def _fake_send_audio(samples):
        send_lengths.append(len(samples))

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.llm.generate", _fake_generate)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(
        "app.services.telephony.waiting_audio.build_waiting_audio",
        lambda language=None: np.ones(4000, dtype=np.float32),
    )
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)

    await session._run_turn(audio)

    assert len(send_lengths) == 2
    assert send_lengths[0] == 4000
    assert send_lengths[1] == 22050


@pytest.mark.asyncio
async def test_run_turn_routes_rapport_mode_with_no_rag(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    monkeypatch.setattr(settings, "turn_strategy_router_enabled", True)
    monkeypatch.setattr(settings, "turn_strategy_no_rag_for_rapport", True)

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="I feel overwhelmed and alone.",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    captured: dict = {}

    async def _fake_generate(_conversation, strategy=None, rag_allowed=None):
        captured["strategy"] = strategy
        captured["rag_allowed"] = rag_allowed
        return "I hear you, and I am here with you. What feels hardest right now?"

    async def _fake_synthesize(_text, *, language=None):
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        return None

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.llm.generate", _fake_generate)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)

    await session._run_turn(audio)

    assert captured["strategy"] == "rapport_building"
    assert captured["rag_allowed"] is False


@pytest.mark.asyncio
async def test_run_turn_routes_understanding_check_mode_with_no_rag(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    monkeypatch.setattr(settings, "turn_strategy_router_enabled", True)
    monkeypatch.setattr(settings, "turn_strategy_no_rag_for_understanding_check", True)

    # Seed a prior clause-dense assistant response so brief user acknowledgment
    # triggers the understanding-check strategy.
    session._conversation.append(
        {
            "role": "assistant",
            "content": (
                "Here are steps: 1. Keep naloxone nearby. 2. Do not use alone. "
                "3. Test in small amounts. 4. Call 911 if breathing slows."
            ),
        }
    )

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="Okay, I guess.",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    captured: dict = {}

    async def _fake_generate(_conversation, strategy=None, rag_allowed=None):
        captured["strategy"] = strategy
        captured["rag_allowed"] = rag_allowed
        return "Let me check we are aligned."

    async def _fake_synthesize(_text, *, language=None):
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        return None

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.llm.generate", _fake_generate)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)

    await session._run_turn(audio)

    assert captured["strategy"] == "understanding_check_no_rag"
    assert captured["rag_allowed"] is False


@pytest.mark.asyncio
async def test_run_turn_routes_explanation_mode_with_rag_optional(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    monkeypatch.setattr(settings, "turn_strategy_router_enabled", True)
    monkeypatch.setattr(settings, "turn_strategy_rag_optional_for_explanation", True)

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="I do not understand what you meant. Can you explain that again?",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    captured: dict = {}

    async def _fake_generate(_conversation, strategy=None, rag_allowed=None):
        captured["strategy"] = strategy
        captured["rag_allowed"] = rag_allowed
        return "Let me re-explain in simpler steps."

    async def _fake_synthesize(_text, *, language=None):
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        return None

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.llm.generate", _fake_generate)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)

    await session._run_turn(audio)

    assert captured["strategy"] == "explanation_rag_optional"
    assert captured["rag_allowed"] is True


@pytest.mark.asyncio
async def test_run_turn_emits_canonical_turn_response(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="hello there",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    async def _fake_generate(_conversation, strategy=None, rag_allowed=None):
        return "Hi, how can I help?"

    async def _fake_synthesize(_text, *, language=None):
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        return None

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.llm.generate", _fake_generate)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)

    await session._run_turn(audio)

    assert len(session._canonical_turn_events) >= 2
    response_events = [
        event
        for event in session._canonical_turn_events
        if event.envelope.message_type.value == "turn.response"
    ]
    request_events = [
        event
        for event in session._canonical_turn_events
        if event.envelope.message_type.value == "turn.request"
    ]
    assert response_events
    assert request_events
    response = response_events[-1]
    request = request_events[-1]
    assert response.payload.status.state.value == "ok"
    assert response.payload.processing.stt is not None
    assert response.payload.processing.tts is not None
    assert response.envelope.idempotency_key
    assert request.envelope.idempotency_key == response.envelope.idempotency_key


@pytest.mark.asyncio
async def test_run_turn_reports_vad_fallback_backend_in_canonical_event(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    session._vad_backend_name = "silence_timeout_fallback"
    audio = np.zeros(6000, dtype=np.float32)

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="check fallback",
            language="en",
            confidence=0.9,
            language_confidence=0.9,
            transcript_quality_score=0.9,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    async def _fake_generate(_conversation, strategy=None, rag_allowed=None):
        return "Fallback mode acknowledged."

    async def _fake_synthesize(_text, *, language=None):
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        return None

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.llm.generate", _fake_generate)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)

    await session._run_turn(audio)

    response_events = [
        event
        for event in session._canonical_turn_events
        if event.envelope.message_type.value == "turn.response"
    ]
    assert response_events
    assert response_events[-1].payload.processing.vad is not None
    assert response_events[-1].payload.processing.vad.backend_name == "silence_timeout_fallback"


def test_waiting_audio_selects_multilingual_phrase_assets() -> None:
    assert waiting_audio._resolve_phrase_language("en") == "en-US"
    assert waiting_audio._resolve_phrase_language("zh") == "zh-CN"
    assert waiting_audio._resolve_phrase_language("zh-CN") == "zh-CN"
    assert waiting_audio._resolve_phrase_language("yue-Hant-HK") == "yue-Hant-HK"
    assert waiting_audio._resolve_phrase_language("ja") == "ja-JP"
    assert waiting_audio._resolve_phrase_language("fr") == "en-US"


def test_parse_transfer_directive_extracts_target_and_spoken_reply() -> None:
    directive, spoken = parse_transfer_directive(
        "TRANSFER:911\nI am connecting you to emergency services now."
    )
    assert directive is not None
    assert directive.target_kind == "number"
    assert directive.target == "911"
    assert spoken == "I am connecting you to emergency services now."


def test_parse_transfer_directive_v2_extracts_metadata() -> None:
    directive, spoken = parse_transfer_directive(
        "TRANSFER:sip:sip:agent@example.com\n"
        "TRANSFER-META:forwarded-by=Terris;topic=overdose;priority=high\n"
        "Connecting now."
    )
    assert directive is not None
    assert directive.target_kind == "sip"
    assert directive.target == "sip:agent@example.com"
    assert directive.metadata == {
        "forwarded-by": "Terris",
        "topic": "overdose",
        "priority": "high",
    }
    assert spoken == "Connecting now."


def test_parse_transfer_directive_returns_plain_reply_when_no_directive() -> None:
    directive, spoken = parse_transfer_directive("Let us focus on your immediate safety.")
    assert directive is None
    assert spoken == "Let us focus on your immediate safety."


def test_is_end_call_intent_detection() -> None:
    assert CallSession._is_end_call_intent("I think that's all for today") is True
    assert CallSession._is_end_call_intent("please end the call") is True
    assert CallSession._is_end_call_intent("I'm good, you can hang up now") is True
    assert CallSession._is_end_call_intent("can you explain naloxone") is False
    assert CallSession._is_end_call_intent("bye for now") is False
    assert CallSession._is_end_call_intent("goodbye") is False
    assert CallSession._is_end_call_intent("I'm good") is False


def test_build_transfer_twiml_adds_sip_headers(monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_allow_custom_targets", True)
    monkeypatch.setattr(settings, "transfer_allowed_sip_domains", "example.com")

    twiml = build_transfer_twiml(
        "sip",
        "sip:agent@example.com",
        "Connecting now.",
        metadata={"forwarded-by": "Terris", "topic": "support", "priority": "normal"},
    )

    assert "<Sip>sip:agent@example.com?" in twiml
    assert "x-forwarded-by=Terris" in twiml
    assert "x-topic=support" in twiml
    assert "x-priority=normal" in twiml


def test_build_transfer_twiml_rejects_number_metadata_in_strict_mode(monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_metadata_mode", "strict")
    with pytest.raises(ValueError):
        build_transfer_twiml(
            "number",
            "988",
            "Connecting now.",
            metadata={"forwarded-by": "Terris"},
        )


@pytest.mark.asyncio
async def test_run_turn_executes_transfer_when_directive_present(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    monkeypatch.setattr(settings, "transfer_confirmation_required", False)

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="Please transfer me to 988",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    async def _fake_generate(_conversation, strategy=None, rag_allowed=None):
        return "TRANSFER:988\nConnecting you to 988 now."

    transfer_calls: list[tuple[str, str, str]] = []

    async def _fake_transfer(transfer, announcement: str) -> bool:
        transfer_calls.append((transfer.target_kind, transfer.target, announcement))
        return True

    async def _unexpected_tts(_text, *, language=None):
        raise AssertionError("TTS should not run when transfer succeeds")

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.llm.generate", _fake_generate)
    monkeypatch.setattr(session, "_transfer_call", _fake_transfer)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _unexpected_tts)

    await session._run_turn(audio)

    assert transfer_calls == [("number", "988", "Connecting you to 988 now.")]


@pytest.mark.asyncio
async def test_run_turn_asks_are_we_done_on_end_call_intent(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="I think that's all for now",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    synthesized_texts: list[str] = []

    async def _fake_synthesize(text, *, language=None):
        synthesized_texts.append(text)
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        pass

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)

    await session._run_turn(audio)

    assert session._awaiting_done_confirmation is True
    assert any("Are we done?" in t for t in synthesized_texts)


@pytest.mark.asyncio
async def test_done_confirmation_yes_starts_terminator_sequence(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)
    session._awaiting_done_confirmation = True

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="yes",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    synthesized_texts: list[str] = []

    async def _fake_synthesize(text, *, language=None):
        synthesized_texts.append(text)
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        pass

    async def _fake_presence_loop(_language):
        return None

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr("app.services.telephony.call_flow_phrases.random_terminator", lambda: "Test terminator")
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)
    monkeypatch.setattr(session, "_run_end_call_presence_loop", _fake_presence_loop)

    await session._run_turn(audio)

    assert session._awaiting_done_confirmation is False
    assert session._in_end_call_presence_flow is True
    assert any("Test terminator" in t for t in synthesized_texts)


def test_transfer_confirmation_prompt_returns_target_specific_message() -> None:
    assert "911 emergency services" in _transfer_confirmation_prompt("911")
    assert "988 Suicide and Crisis Lifeline" in _transfer_confirmation_prompt("988")
    assert "permission" in _transfer_confirmation_prompt("+14155551212")


def test_sanitize_model_reply_removes_think_blocks() -> None:
    raw = "<think>internal chain-of-thought</think>\nTRANSFER:988\nConnecting you now."
    assert _sanitize_model_reply(raw) == "TRANSFER:988\nConnecting you now."


def test_sanitize_model_reply_removes_think_tag_variants() -> None:
    raw = "<think1>hidden</think1>\nUseful spoken text."
    assert _sanitize_model_reply(raw) == "Useful spoken text."


def test_sanitize_model_reply_removes_emphasis_hint_lines() -> None:
    raw = 'EMPHASIS-HINTS: "naloxone" | "call 911"\nUseful spoken text.'
    assert _sanitize_model_reply(raw) == "Useful spoken text."


def test_get_transfer_post_call_reopen_mode_compat_fallback(monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_post_call_reopen_mode", "off")
    monkeypatch.setattr(settings, "transfer_stay_on_line_enabled", True)
    assert get_transfer_post_call_reopen_mode() == "auto"


def test_get_transfer_post_call_reopen_mode_prefers_explicit_mode(monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_post_call_reopen_mode", "prompt")
    monkeypatch.setattr(settings, "transfer_stay_on_line_enabled", False)
    assert get_transfer_post_call_reopen_mode() == "prompt"


def test_get_custom_transfer_post_call_reopen_mode(monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_custom_post_call_reopen_mode", "prompt")
    assert get_custom_transfer_post_call_reopen_mode() == "prompt"


@pytest.mark.asyncio
async def test_run_turn_enters_transfer_confirmation_when_required(monkeypatch) -> None:
    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    monkeypatch.setattr(settings, "transfer_confirmation_required", True)
    monkeypatch.setattr(settings, "transfer_post_call_reopen_mode", "off")

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="Please transfer me to 911",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    async def _fake_generate(_conversation, strategy=None, rag_allowed=None):
        return "TRANSFER:911\nConnecting you to emergency services now."

    synthesized_texts: list[str] = []

    async def _fake_synthesize(text, *, language=None):
        synthesized_texts.append(text)
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        pass

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.llm.generate", _fake_generate)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)

    await session._run_turn(audio)

    assert session._in_transfer_confirmation is True
    assert session._pending_transfer is not None
    assert session._pending_transfer.target == "911"
    # Confirmation question should have been spoken
    assert any("permission" in t.lower() for t in synthesized_texts)
    assert session._awaiting_transfer_permission is True
    assert session._awaiting_post_call_reopen_preference is False


@pytest.mark.asyncio
async def test_transfer_confirmation_yes_executes_transfer(monkeypatch) -> None:
    from app.services.telephony import TransferDirective

    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    # Pre-load pending transfer state
    session._in_transfer_confirmation = True
    session._pending_transfer = TransferDirective(
        target_kind="number", target="911", metadata={}
    )
    session._pending_transfer_spoken = "Connecting you to emergency services now."
    session._awaiting_transfer_permission = True
    session._pending_post_call_reopen = False

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="yes",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    transfer_calls: list[tuple] = []

    async def _fake_transfer(transfer, announcement: str, *, enable_post_call_reopen=None) -> bool:
        transfer_calls.append((transfer.target, announcement, enable_post_call_reopen))
        return True

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr(session, "_transfer_call", _fake_transfer)

    await session._run_turn(audio)

    assert transfer_calls == [("911", "Connecting you to emergency services now.", False)]
    assert session._in_transfer_confirmation is False
    assert session._pending_transfer is None


@pytest.mark.asyncio
async def test_transfer_confirmation_no_cancels_transfer(monkeypatch) -> None:
    from app.services.telephony import TransferDirective

    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    session._in_transfer_confirmation = True
    session._pending_transfer = TransferDirective(
        target_kind="number", target="988", metadata={}
    )
    session._pending_transfer_spoken = "Connecting you to 988 now."
    session._awaiting_transfer_permission = True

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="no",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    synthesized_texts: list[str] = []

    async def _fake_synthesize(text, *, language=None):
        synthesized_texts.append(text)
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        pass

    async def _unexpected_transfer(transfer, announcement):
        raise AssertionError("Transfer should not run after caller says no")

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)
    monkeypatch.setattr(session, "_transfer_call", _unexpected_transfer)

    await session._run_turn(audio)

    assert session._in_transfer_confirmation is False
    assert session._pending_transfer is None
    assert any("won't transfer" in t.lower() for t in synthesized_texts)


def test_build_transfer_twiml_includes_action_url_when_provided(monkeypatch) -> None:
    twiml = build_transfer_twiml(
        "number",
        "988",
        "Connecting you to 988.",
        action_url="https://example.com/calls/transfer-completed",
    )
    assert 'action="https://example.com/calls/transfer-completed"' in twiml
    assert "<Dial" in twiml
    assert "988" in twiml


def test_build_transfer_twiml_rejects_custom_target_not_in_service_catalog(monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_allow_custom_targets", True)
    monkeypatch.setattr(settings, "transfer_allowed_numbers", "+14155550100,+14155550101")
    monkeypatch.setattr(settings, "transfer_services_enabled", True)
    monkeypatch.setattr(
        settings,
        "transfer_services_config_path",
        "app/core/transfer_services.json",
    )
    with pytest.raises(ValueError):
        build_transfer_twiml(
            "number",
            "+14155559999",
            "Connecting now.",
        )


@pytest.mark.asyncio
async def test_custom_transfer_prompt_mode_reopen_preference_yes(monkeypatch) -> None:
    from app.services.telephony import TransferDirective

    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    monkeypatch.setattr(settings, "transfer_custom_post_call_reopen_mode", "prompt")

    session._in_transfer_confirmation = True
    session._pending_transfer = TransferDirective(
        target_kind="number", target="+14155550100", metadata={}
    )
    session._pending_transfer_spoken = "Connecting you now."
    session._awaiting_post_call_reopen_preference = True

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="yes",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    transfer_calls: list[tuple] = []

    async def _fake_transfer(transfer, announcement: str, *, enable_post_call_reopen=None) -> bool:
        transfer_calls.append((transfer.target, announcement, enable_post_call_reopen))
        return True

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr(session, "_transfer_call", _fake_transfer)

    await session._run_turn(audio)

    assert transfer_calls == [('+14155550100', 'Connecting you now.', True)]


@pytest.mark.asyncio
async def test_transfer_confirmation_yes_then_prompt_reopen_preference(monkeypatch) -> None:
    from app.services.telephony import TransferDirective

    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    monkeypatch.setattr(settings, "transfer_post_call_reopen_mode", "prompt")

    session._in_transfer_confirmation = True
    session._pending_transfer = TransferDirective(
        target_kind="number", target="988", metadata={}
    )
    session._pending_transfer_spoken = "Connecting you to 988 now."
    session._awaiting_transfer_permission = True

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="yes",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    synthesized_texts: list[str] = []

    async def _fake_synthesize(text, *, language=None):
        synthesized_texts.append(text)
        return np.zeros(22050, dtype=np.float32)

    async def _fake_send_audio(_samples):
        pass

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr("app.services.telephony.tts.synthesize", _fake_synthesize)
    monkeypatch.setattr(session, "_send_audio", _fake_send_audio)

    await session._run_turn(audio)

    assert session._awaiting_transfer_permission is False
    assert session._awaiting_post_call_reopen_preference is True
    assert any("come back on the line" in text.lower() for text in synthesized_texts)


@pytest.mark.asyncio
async def test_prompt_reopen_yes_executes_transfer_with_reopen(monkeypatch) -> None:
    from app.services.telephony import TransferDirective

    session = CallSession(_DummyWebSocket())
    audio = np.zeros(6000, dtype=np.float32)

    session._in_transfer_confirmation = True
    session._pending_transfer = TransferDirective(
        target_kind="number", target="988", metadata={}
    )
    session._pending_transfer_spoken = "Connecting you to 988 now."
    session._awaiting_post_call_reopen_preference = True

    async def _fake_transcribe(_audio, language=None, preferred_backend=None):
        return TranscriptionResult(
            text="yes",
            language="en",
            confidence=0.95,
            language_confidence=0.95,
            transcript_quality_score=0.95,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    transfer_calls: list[tuple] = []

    async def _fake_transfer(transfer, announcement: str, *, enable_post_call_reopen=None) -> bool:
        transfer_calls.append((transfer.target, announcement, enable_post_call_reopen))
        return True

    monkeypatch.setattr("app.services.telephony.stt.transcribe", _fake_transcribe)
    monkeypatch.setattr(session, "_transfer_call", _fake_transfer)

    await session._run_turn(audio)

    assert transfer_calls == [("988", "Connecting you to 988 now.", True)]
    assert session._in_transfer_confirmation is False
    assert session._pending_transfer is None
