"""Call-session orchestration and audio-pipeline utilities for Twilio Media Streams.

Audio pipeline per conversation turn
─────────────────────────────────────
Inbound  : Twilio μ-law/8 kHz → decode → PCM-16/8 kHz → float32/16 kHz → Whisper
Outbound : text → Piper/F5-TTS → float32/22 kHz → PCM-16/8 kHz → μ-law → Twilio
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from uuid import uuid4
from typing import Literal, Optional
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from xml.sax.saxutils import escape

import numpy as np
from scipy.signal import resample_poly

from app.core.config import settings
from app.models.schemas import (
    CanonicalTurn,
    CanonicalTurnEnvelope,
    CanonicalTurnInput,
    CanonicalTurnInputAudio,
    CanonicalTurnInputLanguage,
    CanonicalTurnInputText,
    CanonicalTurnMessageType,
    CanonicalTurnOutput,
    CanonicalTurnOutputAssistantAudio,
    CanonicalTurnPayload,
    CanonicalTurnProcessing,
    CanonicalTurnProcessingLLM,
    CanonicalTurnProcessingRAG,
    CanonicalTurnProcessingSTT,
    CanonicalTurnProcessingTTS,
    CanonicalTurnProcessingVAD,
    CanonicalTurnSource,
    CanonicalTurnState,
    CanonicalTurnStatus,
)
from app.services import call_flow_phrases
from app.services import llm, stt, tts
from app.services import turn_strategy
from app.services import waiting_audio
from app.services import transfer_services
from app.services.stt_confidence import LowConfidenceHandler
from app.services.tts import PIPER_SAMPLE_RATE
from app.services.vad import StreamingSpeechDetector, silero_vad_available

logger = logging.getLogger(__name__)

_TRANSFER_DIRECTIVE_PATTERN = re.compile(r"^TRANSFER:\s*(911|988)\s*$", re.IGNORECASE)
_TRANSFER_V2_PATTERN = re.compile(r"^TRANSFER:\s*(number|sip)\s*:\s*(.+)$", re.IGNORECASE)
_TRANSFER_META_PATTERN = re.compile(r"^TRANSFER-META:\s*(.+)$", re.IGNORECASE)
_E164_PATTERN = re.compile(r"^\+[1-9]\d{6,14}$")
_THINK_BLOCK_PATTERN = re.compile(r"<think[^>]*>.*?</think[^>]*>", re.IGNORECASE | re.DOTALL)
_EMPHASIS_HINTS_PATTERN = re.compile(r"^EMPHASIS-HINTS:\s*(.+)$", re.IGNORECASE)
_CLAUSE_AWARE_TTS_PLAN_DOC = (
    "Documentation/alignment_plan/Streaming_Implmentation_Considerations_and_plan.md"
)


@dataclass(frozen=True)
class TransferDirective:
    target_kind: Literal["number", "sip"]
    target: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class ClauseAwareTTSStubHints:
    """Temporary hint container for future clause-aware/emphasis TTS implementation."""

    phrases: tuple[str, ...]
    source_doc: str = _CLAUSE_AWARE_TTS_PLAN_DOC


def get_transfer_post_call_reopen_mode() -> Literal["off", "auto", "prompt"]:
    """Resolve effective post-call reopen mode with legacy compatibility."""
    mode = settings.transfer_post_call_reopen_mode
    if mode == "off" and settings.transfer_stay_on_line_enabled:
        return "auto"
    return mode


def get_custom_transfer_post_call_reopen_mode() -> Literal["off", "auto", "prompt"]:
    """Resolve post-call reopen mode for non-emergency custom transfers."""
    return settings.transfer_custom_post_call_reopen_mode

# ── μ-law codec ───────────────────────────────────────────────────────────────
try:
    import audioop  # type: ignore[import]
except ImportError:  # Python 3.13+
    import audioop_lts as audioop  # type: ignore[no-redef]


def mulaw_to_pcm16(data: bytes) -> bytes:
    """Decode 8-bit μ-law bytes to signed 16-bit PCM bytes."""
    return audioop.ulaw2lin(data, 2)


def pcm16_to_mulaw(data: bytes) -> bytes:
    """Encode signed 16-bit PCM bytes to 8-bit μ-law bytes."""
    return audioop.lin2ulaw(data, 2)


# ── Sample-rate conversion ────────────────────────────────────────────────────

def upsample(pcm16: bytes, from_rate: int, to_rate: int) -> np.ndarray:
    """Convert PCM-16 bytes to a float32 array resampled to *to_rate* Hz."""
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if from_rate == to_rate:
        return samples
    return resample_poly(samples, to_rate, from_rate)


def downsample(samples: np.ndarray, from_rate: int, to_rate: int) -> bytes:
    """Resample float32 *samples* and return signed 16-bit PCM bytes."""
    if from_rate != to_rate:
        samples = resample_poly(samples, to_rate, from_rate)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm16.tobytes()


# ── Call session ──────────────────────────────────────────────────────────────

class CallSession:
    """Manages the full lifecycle of a single Twilio voice call."""

    _OUTBOUND_CHUNK = 160

    def __init__(
        self,
        websocket,
        *,
        transport_protocol: Literal["twilio", "asterisk_ari"] = "twilio",
    ) -> None:
        self._ws = websocket
        self._transport_protocol = transport_protocol
        self._stream_sid: Optional[str] = None
        self._call_sid: Optional[str] = None
        self._trace_id: str = str(uuid4())
        self._session_id: str = self._trace_id
        self._audio_buffer: list[bytes] = []
        self._conversation: list[dict] = []
        self._canonical_turn_events: list[CanonicalTurn] = []
        self._turn_sequence: int = 0
        self._pending_turns: list[np.ndarray] = []
        self._silence_timer: Optional[asyncio.TimerHandle] = None
        self._processing = False
        self._loop = asyncio.get_event_loop()
        self._speech_detector: Optional[StreamingSpeechDetector] = None
        self._stt_backend_sticky: Optional[stt.STTBackendName] = None
        self._active_turn_task: Optional[asyncio.Task] = None
        # Low-confidence confirmation state
        self._in_confirmation_flow: bool = False
        self._confirmation_retry_count: int = 0
        self._pending_low_confidence_text: str = ""
        # Transfer confirmation state (verbal consent before 911/988/custom transfer)
        self._in_transfer_confirmation: bool = False
        self._pending_transfer: Optional[TransferDirective] = None
        self._pending_transfer_spoken: str = ""
        self._awaiting_transfer_permission: bool = False
        self._awaiting_post_call_reopen_preference: bool = False
        self._pending_post_call_reopen: Optional[bool] = None
        # End-call terminator flow state
        self._awaiting_done_confirmation: bool = False
        self._in_end_call_presence_flow: bool = False
        self._end_call_presence_task: Optional[asyncio.Task] = None

        if settings.vad_enabled:
            if silero_vad_available():
                try:
                    self._speech_detector = StreamingSpeechDetector()
                except Exception:
                    logger.exception("Failed to initialize Silero VAD; falling back to silence timer")
            else:
                logger.warning("Silero VAD is not available; using silence timeout fallback")

    # ── Public entry point ────────────────────────────────────────────────────

    async def handle(self) -> None:
        """Consume Twilio WebSocket messages for the duration of the call."""
        async for raw in self._ws.iter_text():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON message received: %.80s", raw)
                continue

            event = msg.get("event")
            if event == "connected":
                logger.info("Twilio stream connected")
            elif event == "start":
                start = msg.get("start", {})
                self._stream_sid = msg.get("streamSid") or start.get("streamSid")
                self._call_sid = msg.get("callSid") or start.get("callSid")
                if self._call_sid:
                    self._session_id = self._call_sid
                logger.info("Stream started – SID: %s", self._stream_sid)
            elif event == "media":
                await self._on_media(msg["media"]["payload"])
            elif event == "stop":
                logger.info("Stream stopped")
                self._cancel_silence_timer()
                self._cancel_end_call_presence_task()
                if self._speech_detector is not None:
                    flushed = self._speech_detector.flush()
                    if flushed is not None:
                        self._enqueue_turn(flushed)
                break

    # ── Media handling ────────────────────────────────────────────────────────

    async def _on_media(self, payload_b64: str) -> None:
        chunk = base64.b64decode(payload_b64)
        if self._speech_detector is None:
            if settings.turn_interrupt_enabled and self._processing:
                pcm8k = mulaw_to_pcm16(chunk)
                if self._chunk_has_speech(pcm8k):
                    await self._interrupt_current_turn("caller_barge_in")
            self._audio_buffer.append(chunk)
            self._reset_silence_timer()
            return

        pcm8k = mulaw_to_pcm16(chunk)
        float16k = upsample(
            pcm8k,
            settings.audio_sample_rate_twilio,
            settings.audio_sample_rate_whisper,
        )
        was_active = bool(getattr(self._speech_detector, "speech_active", False))
        finalized_turns = self._speech_detector.process_chunk(float16k)
        speech_started = (not was_active) and bool(
            getattr(self._speech_detector, "speech_active", False)
        )
        if speech_started and settings.turn_interrupt_enabled and self._processing:
            await self._interrupt_current_turn("caller_barge_in")
        for voiced_turn in finalized_turns:
            self._enqueue_turn(voiced_turn)

    def _cancel_silence_timer(self) -> None:
        if self._silence_timer is not None:
            self._silence_timer.cancel()
            self._silence_timer = None

    def _reset_silence_timer(self) -> None:
        self._cancel_silence_timer()
        self._silence_timer = self._loop.call_later(
            settings.silence_timeout_s, self._schedule_turn
        )

    def _schedule_turn(self) -> None:
        if not self._processing and self._audio_buffer:
            asyncio.ensure_future(self._process_buffered_turn())

    def _enqueue_turn(self, audio: np.ndarray) -> None:
        if self._processing:
            self._pending_turns.append(audio)
            return
        asyncio.ensure_future(self._process_audio_turn(audio))

    # ── Turn processing ───────────────────────────────────────────────────────

    async def _process_buffered_turn(self) -> None:
        self._active_turn_task = asyncio.current_task()
        self._processing = True
        audio_chunks, self._audio_buffer = self._audio_buffer, []
        try:
            pcm8k = b"".join(mulaw_to_pcm16(c) for c in audio_chunks)
            float16k = upsample(
                pcm8k,
                settings.audio_sample_rate_twilio,
                settings.audio_sample_rate_whisper,
            )
            await self._run_turn(float16k)
        except asyncio.CancelledError:
            logger.info("Buffered call turn interrupted")
        except Exception:
            logger.exception("Error during buffered call turn processing")
        finally:
            if self._active_turn_task is asyncio.current_task():
                self._active_turn_task = None
            self._finish_turn_processing()

    async def _process_audio_turn(self, audio: np.ndarray) -> None:
        self._active_turn_task = asyncio.current_task()
        self._processing = True
        try:
            await self._run_turn(audio)
        except asyncio.CancelledError:
            logger.info("Voiced call turn interrupted")
        except Exception:
            logger.exception("Error during voiced call turn processing")
        finally:
            if self._active_turn_task is asyncio.current_task():
                self._active_turn_task = None
            self._finish_turn_processing()

    async def _run_turn(self, audio: np.ndarray) -> None:
        turn_id = str(uuid4())
        self._turn_sequence += 1
        turn_idempotency_key = self._build_turn_idempotency_key()
        turn_input = CanonicalTurnInput(
            audio=CanonicalTurnInputAudio(
                codec="pcm_f32le",
                sample_rate_hz=settings.audio_sample_rate_whisper,
                duration_ms=round((len(audio) / settings.audio_sample_rate_whisper) * 1000),
            )
        )
        self._emit_canonical_turn(
            message_type=CanonicalTurnMessageType.TURN_REQUEST,
            turn_id=turn_id,
            payload=CanonicalTurnPayload(
                input=turn_input,
                status=CanonicalTurnStatus(state=CanonicalTurnState.OK),
            ),
            idempotency_key=turn_idempotency_key,
        )

        # ── 1. Minimum duration gate ──────────────────────────────────────────
        min_samples = int(settings.min_audio_duration_s * settings.audio_sample_rate_whisper)
        if len(audio) < min_samples:
            self._log_turn_drop("too_short", audio=audio)
            self._emit_canonical_turn(
                message_type=CanonicalTurnMessageType.TURN_EVENT,
                turn_id=turn_id,
                payload=CanonicalTurnPayload(
                    input=turn_input,
                    processing=CanonicalTurnProcessing(
                        vad=CanonicalTurnProcessingVAD(
                            vad_voiced_duration_ms=round(
                                (len(audio) / settings.audio_sample_rate_whisper) * 1000
                            )
                        )
                    ),
                    status=CanonicalTurnStatus(
                        state=CanonicalTurnState.DROPPED,
                        failure_reason="too_short",
                        retryable=False,
                    ),
                ),
                idempotency_key=turn_idempotency_key,
            )
            return

        # ── 2. STT ────────────────────────────────────────────────────────────
        stt_result = await stt.transcribe(audio, preferred_backend=self._stt_backend_sticky)

        if stt_result.backend_name == "sherpa-onnx" and self._stt_backend_sticky != "sherpa":
            self._stt_backend_sticky = "sherpa"
            logger.info("STT backend switched to session-sticky Sherpa-ONNX")

        text = stt_result.text.strip()
        if not text:
            self._log_turn_drop(
                stt_result.failure_reason or "no_speech",
                audio=audio,
                stt_result=stt_result,
            )
            self._emit_canonical_turn(
                message_type=CanonicalTurnMessageType.TURN_EVENT,
                turn_id=turn_id,
                payload=CanonicalTurnPayload(
                    input=turn_input,
                    processing=CanonicalTurnProcessing(
                        stt=CanonicalTurnProcessingSTT(
                            backend_name=stt_result.backend_name,
                            transcript_text="",
                            transcript_confidence=stt_result.confidence,
                            language_confidence=stt_result.language_confidence,
                            transcript_quality_score=stt_result.transcript_quality_score,
                            fallback_used=stt_result.fallback_used,
                            failure_reason=stt_result.failure_reason or "no_speech",
                        )
                    ),
                    status=CanonicalTurnStatus(
                        state=CanonicalTurnState.DROPPED,
                        failure_reason=stt_result.failure_reason or "no_speech",
                        retryable=True,
                    ),
                ),
                idempotency_key=turn_idempotency_key,
            )
            return

        logger.info(
            "Transcribed [%s via %s]: %s",
            stt_result.language,
            stt_result.backend_name,
            text,
        )

        # ── 2.5. Handle active confirmation flows (transfer takes priority) ────
        if self._in_transfer_confirmation:
            await self._handle_transfer_confirmation_response(text, stt_result)
            return

        if self._awaiting_done_confirmation:
            await self._handle_done_confirmation_response(text, stt_result)
            return

        if self._in_confirmation_flow:
            await self._handle_confirmation_response(text, stt_result)
            return

        # Any caller response during terminator-presence checks reopens dialog.
        if self._in_end_call_presence_flow:
            logger.info("Caller reinitiated conversation during end-call presence flow")
            self._in_end_call_presence_flow = False
            self._cancel_end_call_presence_task()

        if self._is_end_call_intent(text):
            await self._ask_are_we_done(stt_result.language)
            return

        # ── 3. Check for low confidence (before LLM) ──────────────────────────
        if LowConfidenceHandler.is_low_confidence(stt_result):
            self._emit_canonical_turn(
                message_type=CanonicalTurnMessageType.TURN_EVENT,
                turn_id=turn_id,
                payload=CanonicalTurnPayload(
                    input=CanonicalTurnInput(
                        audio=turn_input.audio,
                        text=CanonicalTurnInputText(text=text),
                        language=CanonicalTurnInputLanguage(
                            code=stt_result.language,
                            language_confidence=stt_result.language_confidence,
                        ),
                    ),
                    processing=CanonicalTurnProcessing(
                        stt=CanonicalTurnProcessingSTT(
                            backend_name=stt_result.backend_name,
                            transcript_text=text,
                            transcript_confidence=stt_result.confidence,
                            language_confidence=stt_result.language_confidence,
                            transcript_quality_score=stt_result.transcript_quality_score,
                            fallback_used=stt_result.fallback_used,
                            failure_reason=stt_result.failure_reason,
                        )
                    ),
                    status=CanonicalTurnStatus(
                        state=CanonicalTurnState.PARTIAL,
                        failure_reason="low_confidence_confirmation",
                        retryable=True,
                    ),
                ),
                idempotency_key=turn_idempotency_key,
            )
            await self._enter_confirmation_flow(stt_result)
            return

        # ── 4. LLM ────────────────────────────────────────────────────────────
        strategy_decision = self._select_turn_strategy(text)
        strategy = strategy_decision.strategy
        rag_allowed = self._rag_allowed_for_strategy(strategy)
        if settings.turn_strategy_debug_logging:
            logger.info(
                "Turn strategy selected: strategy=%s reason=%s rag_allowed=%s",
                strategy.value,
                strategy_decision.reason,
                rag_allowed,
            )

        self._conversation.append({"role": "user", "content": text})
        raw_reply = await self._generate_with_optional_waiting_audio(
            stt_result.language,
            strategy=strategy,
            rag_allowed=rag_allowed,
        )
        clause_tts_hints = _extract_clause_aware_tts_stub_hints(raw_reply)
        reply = _sanitize_model_reply(raw_reply)
        transfer, spoken_reply = parse_transfer_directive(reply)
        self._conversation.append({"role": "assistant", "content": spoken_reply})
        logger.info("LLM reply: %s", spoken_reply)
        if clause_tts_hints.phrases:
            logger.info(
                "Clause-aware TTS hints captured (stub): phrases=%s source=%s",
                clause_tts_hints.phrases,
                clause_tts_hints.source_doc,
            )

        if transfer is not None:
            should_prompt_post_call = (
                _post_call_reopen_mode_for_transfer(transfer) == "prompt"
            )
            if settings.transfer_confirmation_required or should_prompt_post_call:
                await self._enter_transfer_confirmation(
                    transfer, spoken_reply, stt_result.language
                )
                return
            transfer_message = spoken_reply or _default_transfer_announcement(transfer.target)
            transferred = await self._transfer_call(transfer, transfer_message)
            if transferred:
                logger.info("Call transfer initiated to %s:%s", transfer.target_kind, transfer.target)
                return
            spoken_reply = (
                f"I could not complete the transfer to {transfer.target} right now. "
                "If you are in immediate danger, call emergency services now."
            )

        # ── 5. TTS ────────────────────────────────────────────────────────────
        # The TTS service returns float32 PCM samples at PIPER_SAMPLE_RATE.
        try:
            tts_audio = await tts.synthesize(spoken_reply, language=stt_result.language)
        except Exception:
            logger.exception("Dropping turn: TTS synthesis failed")
            self._emit_canonical_turn(
                message_type=CanonicalTurnMessageType.TURN_ERROR,
                turn_id=turn_id,
                payload=CanonicalTurnPayload(
                    input=CanonicalTurnInput(
                        audio=turn_input.audio,
                        text=CanonicalTurnInputText(text=text),
                        language=CanonicalTurnInputLanguage(
                            code=stt_result.language,
                            language_confidence=stt_result.language_confidence,
                        ),
                    ),
                    processing=CanonicalTurnProcessing(
                        stt=CanonicalTurnProcessingSTT(
                            backend_name=stt_result.backend_name,
                            transcript_text=text,
                            transcript_confidence=stt_result.confidence,
                            language_confidence=stt_result.language_confidence,
                            transcript_quality_score=stt_result.transcript_quality_score,
                            fallback_used=stt_result.fallback_used,
                            failure_reason=stt_result.failure_reason,
                        ),
                        llm=CanonicalTurnProcessingLLM(
                            backend_name=settings.llm_provider,
                            strategy=strategy.value,
                        ),
                    ),
                    output=CanonicalTurnOutput(assistant_text=spoken_reply),
                    status=CanonicalTurnStatus(
                        state=CanonicalTurnState.PARTIAL,
                        failure_reason="tts_failed",
                        retryable=True,
                    ),
                ),
                idempotency_key=turn_idempotency_key,
            )
            return

        logger.info(
            "TTS synthesis completed: lang=%s samples=%d",
            stt_result.language,
            len(tts_audio),
        )

        # ── 6. Send audio to caller ───────────────────────────────────────────
        await self._send_audio(tts_audio)

        self._emit_canonical_turn(
            message_type=CanonicalTurnMessageType.TURN_RESPONSE,
            turn_id=turn_id,
            payload=CanonicalTurnPayload(
                input=CanonicalTurnInput(
                    audio=turn_input.audio,
                    text=CanonicalTurnInputText(text=text),
                    language=CanonicalTurnInputLanguage(
                        code=stt_result.language,
                        language_confidence=stt_result.language_confidence,
                    ),
                ),
                processing=CanonicalTurnProcessing(
                    vad=CanonicalTurnProcessingVAD(
                        vad_voiced_duration_ms=round(
                            (len(audio) / settings.audio_sample_rate_whisper) * 1000
                        )
                    ),
                    stt=CanonicalTurnProcessingSTT(
                        backend_name=stt_result.backend_name,
                        transcript_text=text,
                        transcript_confidence=stt_result.confidence,
                        language_confidence=stt_result.language_confidence,
                        transcript_quality_score=stt_result.transcript_quality_score,
                        fallback_used=stt_result.fallback_used,
                        failure_reason=stt_result.failure_reason,
                    ),
                    rag=CanonicalTurnProcessingRAG(
                        enabled=bool(rag_allowed and settings.rag_enabled),
                    ),
                    llm=CanonicalTurnProcessingLLM(
                        backend_name=settings.llm_provider,
                        strategy=strategy.value,
                    ),
                    tts=CanonicalTurnProcessingTTS(
                        backend_name=settings.tts_backend,
                        voice_id=settings.f5_tts_voice,
                        output_format="mulaw",
                        output_sample_rate_hz=settings.audio_sample_rate_twilio,
                    ),
                ),
                output=CanonicalTurnOutput(
                    assistant_text=spoken_reply,
                    assistant_audio=CanonicalTurnOutputAssistantAudio(
                        format="mulaw",
                        sample_rate_hz=settings.audio_sample_rate_twilio,
                        duration_ms=round(
                            (len(tts_audio) / PIPER_SAMPLE_RATE) * 1000
                        ),
                    ),
                ),
                status=CanonicalTurnStatus(state=CanonicalTurnState.OK),
            ),
            idempotency_key=turn_idempotency_key,
        )

    def _build_turn_idempotency_key(self) -> str:
        session_hint = self._session_id or self._trace_id
        return f"session:{session_hint}:turn:{self._turn_sequence}"

    def _emit_canonical_turn(
        self,
        *,
        message_type: CanonicalTurnMessageType,
        turn_id: str,
        payload: CanonicalTurnPayload,
        idempotency_key: Optional[str] = None,
    ) -> None:
        envelope = CanonicalTurnEnvelope(
            message_type=message_type,
            trace_id=self._trace_id,
            turn_id=turn_id,
            session_id=self._session_id,
            created_at=datetime.now(timezone.utc),
            source=CanonicalTurnSource.TELEPHONY,
            idempotency_key=idempotency_key,
        )
        canonical_turn = CanonicalTurn(envelope=envelope, payload=payload)
        self._canonical_turn_events.append(canonical_turn)
        logger.info(
            "canonical_turn=%s",
            canonical_turn.model_dump_json(exclude_none=True),
        )

    # ── Transfer confirmation flow ─────────────────────────────────────────────

    async def _enter_transfer_confirmation(
        self,
        transfer: TransferDirective,
        spoken_reply: str,
        language: str | None,
    ) -> None:
        """Ask the caller for verbal consent before executing a transfer."""
        logger.info(
            "Transfer to %s:%s requires confirmation; entering confirmation flow.",
            transfer.target_kind,
            transfer.target,
        )
        self._in_transfer_confirmation = True
        self._pending_transfer = transfer
        self._pending_transfer_spoken = spoken_reply

        mode = _post_call_reopen_mode_for_transfer(transfer)
        self._pending_post_call_reopen = True if mode == "auto" else False

        if settings.transfer_confirmation_required:
            self._awaiting_transfer_permission = True
            self._awaiting_post_call_reopen_preference = False
            question = _transfer_confirmation_prompt(transfer.target)
        else:
            # Prompt-only path for post-call reopen preference.
            self._awaiting_transfer_permission = False
            self._awaiting_post_call_reopen_preference = True
            self._pending_post_call_reopen = None
            question = _post_call_reopen_prompt(transfer.target)

        try:
            tts_audio = await tts.synthesize(question, language=language)
            await self._send_audio(tts_audio)
        except Exception:
            logger.exception("Failed to synthesize transfer confirmation prompt")

    async def _handle_transfer_confirmation_response(
        self, response_text: str, response_result: stt.TranscriptionResult
    ) -> None:
        """Handle the caller's yes/no response to a pending transfer confirmation."""
        logger.info("Transfer confirmation response: %s", response_text)

        transfer = self._pending_transfer
        if transfer is None:
            logger.warning("Transfer confirmation active but no pending transfer state")
            self._clear_pending_transfer_confirmation()
            return

        if self._awaiting_post_call_reopen_preference:
            if self._is_affirmative_response(response_text):
                self._pending_post_call_reopen = True
                await self._execute_pending_transfer(response_result)
                return
            if self._is_negative_response(response_text):
                self._pending_post_call_reopen = False
                await self._execute_pending_transfer(response_result)
                return

            try:
                tts_audio = await tts.synthesize(
                    "Please say yes if you want me to return after the emergency line ends, "
                    "or no if you do not.",
                    language=response_result.language,
                )
                await self._send_audio(tts_audio)
            except Exception:
                logger.exception("Failed to synthesize reopen-preference clarification")
            return

        if self._is_affirmative_response(response_text):
            mode = _post_call_reopen_mode_for_transfer(transfer)
            needs_reopen_prompt = mode == "prompt"
            if needs_reopen_prompt:
                self._awaiting_transfer_permission = False
                self._awaiting_post_call_reopen_preference = True
                self._pending_post_call_reopen = None
                try:
                    tts_audio = await tts.synthesize(
                        _post_call_reopen_prompt(transfer.target),
                        language=response_result.language,
                    )
                    await self._send_audio(tts_audio)
                except Exception:
                    logger.exception("Failed to synthesize post-call reopen prompt")
                return

            await self._execute_pending_transfer(response_result)
            return

        if self._is_negative_response(response_text):
            logger.info("Caller declined transfer; cancelling.")
            self._clear_pending_transfer_confirmation()
            cancellation = (
                "Okay, I won't transfer you. I'm still here - please let me know how I can help."
            )
            try:
                tts_audio = await tts.synthesize(cancellation, language=response_result.language)
                await self._send_audio(tts_audio)
            except Exception:
                logger.exception("Failed to synthesize transfer cancellation message")
            return

        # Unclear – re-ask active question
        clarification = (
            "I didn't quite catch that. Please say yes to connect or no to stay."
            if self._awaiting_transfer_permission
            else "Please say yes or no."
        )
        try:
            tts_audio = await tts.synthesize(clarification, language=response_result.language)
            await self._send_audio(tts_audio)
        except Exception:
            logger.exception("Failed to synthesize transfer confirmation clarification")

    async def _execute_pending_transfer(self, response_result: stt.TranscriptionResult) -> None:
        transfer = self._pending_transfer
        if transfer is None:
            self._clear_pending_transfer_confirmation()
            return

        spoken = self._pending_transfer_spoken or _default_transfer_announcement(transfer.target)
        post_call_reopen = bool(self._pending_post_call_reopen)
        self._clear_pending_transfer_confirmation()

        transferred = await self._transfer_call(
            transfer,
            spoken,
            enable_post_call_reopen=post_call_reopen,
        )
        if not transferred:
            fallback = (
                f"I was unable to complete the transfer to {transfer.target} right now. "
                "Please call directly if you are in immediate danger."
            )
            try:
                tts_audio = await tts.synthesize(fallback, language=response_result.language)
                await self._send_audio(tts_audio)
            except Exception:
                logger.exception("Failed to synthesize transfer-failed message")

    def _clear_pending_transfer_confirmation(self) -> None:
        self._in_transfer_confirmation = False
        self._pending_transfer = None
        self._pending_transfer_spoken = ""
        self._awaiting_transfer_permission = False
        self._awaiting_post_call_reopen_preference = False
        self._pending_post_call_reopen = None

    # ── End-call terminator flow ─────────────────────────────────────────────

    async def _ask_are_we_done(self, language: str | None) -> None:
        self._awaiting_done_confirmation = True
        await self._speak_text("Are we done?", language)

    async def _handle_done_confirmation_response(
        self,
        response_text: str,
        response_result: stt.TranscriptionResult,
    ) -> None:
        if self._is_affirmative_response(response_text):
            self._awaiting_done_confirmation = False
            await self._start_terminator_sequence(response_result.language)
            return

        if self._is_negative_response(response_text):
            self._awaiting_done_confirmation = False
            await self._speak_text(
                "Okay, we can keep going. What else can I help with?",
                response_result.language,
            )
            return

        await self._speak_text(
            "Please say yes if you are ready to end, or no if you want to continue.",
            response_result.language,
        )

    async def _start_terminator_sequence(self, language: str | None) -> None:
        self._in_end_call_presence_flow = True
        terminator = call_flow_phrases.random_terminator()
        await self._speak_text(terminator, language)
        self._cancel_end_call_presence_task()
        self._end_call_presence_task = asyncio.create_task(
            self._run_end_call_presence_loop(language)
        )

    def _cancel_end_call_presence_task(self) -> None:
        if self._end_call_presence_task is not None:
            self._end_call_presence_task.cancel()
            self._end_call_presence_task = None

    async def _run_end_call_presence_loop(self, language: str | None) -> None:
        delay_s = max(0.0, float(settings.call_end_presence_delay_s))
        rounds = max(0, int(settings.call_end_presence_rounds))
        try:
            for _ in range(rounds):
                await asyncio.sleep(delay_s)
                if not self._in_end_call_presence_flow:
                    return
                await self._speak_text("Are you still there?", language)

            await asyncio.sleep(delay_s)
            if self._in_end_call_presence_flow:
                await self._hangup_call()
        except asyncio.CancelledError:
            return

    async def _hangup_call(self) -> None:
        self._in_end_call_presence_flow = False
        self._cancel_end_call_presence_task()

        if self._call_sid and settings.twilio_account_sid and settings.twilio_auth_token:
            hangup_twiml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<Response><Hangup/></Response>"
            )
            try:
                await asyncio.to_thread(twilio_transfer_call_update, self._call_sid, hangup_twiml)
                return
            except Exception:
                logger.exception("Failed to end call via Twilio API")

        with contextlib.suppress(Exception):
            await self._ws.close()

    async def _speak_text(self, text: str, language: str | None) -> None:
        try:
            audio = await tts.synthesize(text, language=language)
            await self._send_audio(audio)
        except Exception:
            logger.exception("Failed to synthesize prompt: %s", text)

    # ── Low-confidence confirmation flow ──────────────────────────────────────

    async def _enter_confirmation_flow(self, stt_result: stt.TranscriptionResult) -> None:
        """Enter confirmation flow for low-confidence STT result."""
        logger.info(
            "Low-confidence STT result (confidence: %.2f, threshold: %.2f). "
            "Entering confirmation flow.",
            stt_result.language_confidence,
            settings.stt_low_confidence_threshold,
        )
        
        self._in_confirmation_flow = True
        self._confirmation_retry_count = 0
        self._pending_low_confidence_text = stt_result.text
        
        # Generate confirmation prompt
        prompt = await LowConfidenceHandler.generate_confirmation_prompt(stt_result)
        logger.info("Confirmation prompt: %s", prompt.prompt)
        
        # Synthesize and send confirmation prompt
        try:
            tts_audio = await tts.synthesize(prompt.prompt, language=stt_result.language)
            await self._send_audio(tts_audio)
        except Exception:
            logger.exception("Failed to synthesize confirmation prompt")

    async def _handle_confirmation_response(
        self, response_text: str, response_result: stt.TranscriptionResult
    ) -> None:
        """Handle user's response to confirmation prompt (yes/no)."""
        logger.info("Confirmation response: %s", response_text)
        
        if self._is_affirmative_response(response_text):
            # User confirmed - process the original transcript
            logger.info("User confirmed low-confidence transcript")
            self._in_confirmation_flow = False
            self._confirmation_retry_count = 0
            
            # Add the original (confirmed) text to conversation and process with RAG
            self._conversation.append({"role": "user", "content": self._pending_low_confidence_text})
            strategy_decision = self._select_turn_strategy(self._pending_low_confidence_text)
            strategy = strategy_decision.strategy
            rag_allowed = self._rag_allowed_for_strategy(strategy)
            if settings.turn_strategy_debug_logging:
                logger.info(
                    "Turn strategy selected (post-confirmation): strategy=%s reason=%s rag_allowed=%s",
                    strategy.value,
                    strategy_decision.reason,
                    rag_allowed,
                )

            raw_reply = await self._generate_with_optional_waiting_audio(
                response_result.language,
                strategy=strategy,
                rag_allowed=rag_allowed,
            )
            clause_tts_hints = _extract_clause_aware_tts_stub_hints(raw_reply)
            reply = _sanitize_model_reply(raw_reply)
            transfer, spoken_reply = parse_transfer_directive(reply)
            self._conversation.append({"role": "assistant", "content": spoken_reply})
            logger.info("LLM reply (after confirmation): %s", spoken_reply)
            if clause_tts_hints.phrases:
                logger.info(
                    "Clause-aware TTS hints captured (stub, post-confirmation): phrases=%s source=%s",
                    clause_tts_hints.phrases,
                    clause_tts_hints.source_doc,
                )

            if transfer is not None:
                transfer_message = spoken_reply or _default_transfer_announcement(transfer.target)
                transferred = await self._transfer_call(transfer, transfer_message)
                if transferred:
                    logger.info(
                        "Call transfer initiated after confirmation to %s:%s",
                        transfer.target_kind,
                        transfer.target,
                    )
                    return
                spoken_reply = (
                    f"I could not complete the transfer to {transfer.target} right now. "
                    "If you are in immediate danger, call emergency services now."
                )
            
            # Synthesize and send reply
            try:
                tts_audio = await tts.synthesize(spoken_reply, language=response_result.language)
                await self._send_audio(tts_audio)
            except Exception:
                logger.exception("Failed to synthesize LLM reply after confirmation")
        
        elif self._is_negative_response(response_text):
            # User denied - check if we should retry or change topic
            self._confirmation_retry_count += 1
            logger.info("User denied confirmation (retry %d/%d)", 
                       self._confirmation_retry_count, 
                       settings.stt_max_retries)
            
            if LowConfidenceHandler.should_change_topic(self._confirmation_retry_count):
                # Max retries exceeded - allow topic change
                self._in_confirmation_flow = False
                self._confirmation_retry_count = 0
                self._pending_low_confidence_text = ""
                
                prompt = LowConfidenceHandler.get_retry_prompt(self._confirmation_retry_count)
                logger.info("Max retries exceeded. Topic change prompt: %s", prompt)
                
                try:
                    tts_audio = await tts.synthesize(prompt, language=response_result.language)
                    await self._send_audio(tts_audio)
                except Exception:
                    logger.exception("Failed to synthesize topic change prompt")
            else:
                # Allow retry
                prompt = LowConfidenceHandler.get_retry_prompt(self._confirmation_retry_count)
                logger.info("Requesting retry: %s", prompt)
                
                try:
                    tts_audio = await tts.synthesize(prompt, language=response_result.language)
                    await self._send_audio(tts_audio)
                except Exception:
                    logger.exception("Failed to synthesize retry prompt")
        else:
            # Unclear response - ask again
            logger.info("Unclear response during confirmation: %s", response_text)
            try:
                tts_audio = await tts.synthesize(
                    "I didn't catch that. Please say yes or no.",
                    language=response_result.language,
                )
                await self._send_audio(tts_audio)
            except Exception:
                logger.exception("Failed to synthesize clarification prompt")

    @staticmethod
    def _is_affirmative_response(text: str) -> bool:
        """Check if response is affirmative (yes, yeah, yep, etc.)."""
        text_lower = text.lower().strip()
        affirmative_words = {"yes", "yeah", "yep", "sure", "correct", "right", "uh-huh"}
        return any(word in text_lower for word in affirmative_words)

    @staticmethod
    def _is_negative_response(text: str) -> bool:
        """Check if response is negative (no, nope, etc.)."""
        text_lower = text.lower().strip()
        negative_words = {"no", "nope", "nah", "incorrect", "wrong", "uh-uh"}
        return any(word in text_lower for word in negative_words)

    @staticmethod
    def _is_end_call_intent(text: str) -> bool:
        text_lower = text.lower().strip()
        # Strong explicit closure phrases are sufficient on their own.
        strong_intents = {
            "we're done",
            "we are done",
            "that's all",
            "that is all",
            "that's everything",
            "that is everything",
            "nothing else",
            "end the call",
            "hang up",
            "please end",
            "disconnect the call",
            "you can end the call",
            "you can hang up",
        }
        if any(phrase in text_lower for phrase in strong_intents):
            return True

        # Softer intent phrases require an explicit call-ending action cue.
        soft_intents = {
            "i'm good",
            "im good",
            "all good",
            "i got all the help i need",
            "got all the help i need",
            "no more questions",
        }
        action_cues = {
            "end",
            "hang up",
            "disconnect",
            "done",
            "wrap up",
            "finish",
        }
        if any(phrase in text_lower for phrase in soft_intents) and any(
            cue in text_lower for cue in action_cues
        ):
            return True

        return False

    def _finish_turn_processing(self) -> None:
        self._processing = False
        if self._pending_turns:
            next_audio = self._pending_turns.pop(0)
            asyncio.ensure_future(self._process_audio_turn(next_audio))

    async def _interrupt_current_turn(self, reason: str) -> None:
        task = self._active_turn_task
        if task is None or task.done():
            return

        logger.info("Interrupting current turn: %s", reason)
        task.cancel()
        await self._clear_outbound_audio()

    async def _clear_outbound_audio(self) -> None:
        if not self._stream_sid:
            return
        await self._ws.send_text(
            json.dumps(
                {
                    "event": "clear",
                    "streamSid": self._stream_sid,
                }
            )
        )

    def _select_turn_strategy(self, user_text: str) -> turn_strategy.TurnStrategyDecision:
        if not settings.turn_strategy_router_enabled:
            return turn_strategy.TurnStrategyDecision(
                strategy=turn_strategy.TurnStrategy.TASK_OR_KNOWLEDGE_RAG_ELIGIBLE,
                reason="router_disabled",
            )
        return turn_strategy.classify_turn(user_text, self._conversation)

    @staticmethod
    def _rag_allowed_for_strategy(strategy: turn_strategy.TurnStrategy) -> bool:
        if strategy == turn_strategy.TurnStrategy.RAPPORT_BUILDING:
            return not settings.turn_strategy_no_rag_for_rapport
        if strategy == turn_strategy.TurnStrategy.INFO_GATHERING_NO_RAG:
            return not settings.turn_strategy_no_rag_for_info_gathering
        if strategy == turn_strategy.TurnStrategy.UNDERSTANDING_CHECK_NO_RAG:
            return not settings.turn_strategy_no_rag_for_understanding_check
        if strategy == turn_strategy.TurnStrategy.EXPLANATION_RAG_OPTIONAL:
            return settings.turn_strategy_rag_optional_for_explanation
        return turn_strategy.rag_allowed_for_strategy(strategy)

    async def _generate_with_optional_waiting_audio(
        self,
        language: str | None,
        *,
        strategy: turn_strategy.TurnStrategy,
        rag_allowed: bool,
    ) -> str:
        if not rag_allowed or not settings.rag_enabled or not settings.rag_waiting_audio_enabled:
            return await llm.generate(
                self._conversation,
                strategy=strategy.value,
                rag_allowed=rag_allowed,
            )

        waiting_task = asyncio.create_task(self._play_waiting_audio(language))
        try:
            return await llm.generate(
                self._conversation,
                strategy=strategy.value,
                rag_allowed=rag_allowed,
            )
        finally:
            if not waiting_task.done():
                waiting_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiting_task

    async def _play_waiting_audio(self, language: str | None) -> None:
        try:
            await asyncio.sleep(max(0.0, settings.rag_waiting_audio_delay_s))
            filler_audio = await asyncio.to_thread(waiting_audio.build_waiting_audio, language)
            if len(filler_audio) > 0:
                await self._send_audio(filler_audio)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to play waiting audio during RAG lookup")

    @staticmethod
    def _chunk_has_speech(pcm16: bytes) -> bool:
        if not pcm16:
            return False
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return False
        return float(np.max(np.abs(samples))) > 500.0

    def _log_turn_drop(
        self,
        reason: str,
        *,
        audio: np.ndarray,
        stt_result: Optional[stt.TranscriptionResult] = None,
    ) -> None:
        details: dict = {
            "reason": reason,
            "audio_ms": round((len(audio) / settings.audio_sample_rate_whisper) * 1000),
        }
        if stt_result is not None:
            details.update(
                {
                    "stt_backend": stt_result.backend_name,
                    "stt_fallback_used": stt_result.fallback_used,
                    "stt_quality": f"{stt_result.transcript_quality_score:.2f}",
                }
            )
        logger.info("Dropping turn: %s", details)

    # ── Audio sending ─────────────────────────────────────────────────────────

    async def _send_audio(self, samples: np.ndarray) -> None:
        """Downsample, encode to μ-law, and stream audio chunks to Twilio."""
        pcm8k = downsample(samples, PIPER_SAMPLE_RATE, settings.audio_sample_rate_twilio)
        mulaw_bytes = pcm16_to_mulaw(pcm8k)

        for i in range(0, len(mulaw_bytes), self._OUTBOUND_CHUNK):
            chunk = mulaw_bytes[i : i + self._OUTBOUND_CHUNK]
            payload = base64.b64encode(chunk).decode("ascii")
            await self._ws.send_text(
                json.dumps(
                    {
                        "event": "media",
                        "streamSid": self._stream_sid,
                        "media": {"payload": payload},
                    }
                )
            )

        # Twilio Media Streams support marker events; Asterisk ExternalMedia does not.
        if self._transport_protocol == "twilio":
            await self._ws.send_text(
                json.dumps(
                    {
                        "event": "mark",
                        "streamSid": self._stream_sid,
                        "mark": {"name": "end_of_response"},
                    }
                )
            )

    async def _transfer_call(
        self,
        transfer: TransferDirective,
        announcement: str,
        *,
        enable_post_call_reopen: Optional[bool] = None,
    ) -> bool:
        if not self._call_sid:
            logger.warning("Cannot transfer call without call SID")
            return False

        if not settings.twilio_account_sid or not settings.twilio_auth_token:
            logger.warning("Twilio credentials missing; cannot transfer call")
            return False

        try:
            action_url: Optional[str] = None
            should_reopen = (
                _post_call_reopen_mode_for_transfer(transfer) == "auto"
                if enable_post_call_reopen is None
                else enable_post_call_reopen
            )

            if should_reopen:
                # After the 911/988 operator hangs up, Twilio POSTs to this URL so
                # Terris can re-engage the caller. Full 3-party monitoring during the
                # emergency call is NOT possible via Media Streams - the bridge is
                # direct. This only provides post-operator-hangup re-engagement.
                scheme = "https"
                action_url = f"{scheme}://{settings.public_host}/calls/transfer-completed"

            twiml = build_transfer_twiml(
                transfer.target_kind,
                transfer.target,
                announcement,
                metadata=transfer.metadata,
                action_url=action_url,
            )
        except ValueError as exc:
            logger.warning("Rejected transfer directive: %s", exc)
            return False

        try:
            await asyncio.to_thread(twilio_transfer_call_update, self._call_sid, twiml)
            return True
        except Exception:
            logger.exception("Failed to transfer call to %s", transfer.target)
            return False


def parse_transfer_directive(reply: str) -> tuple[Optional[TransferDirective], str]:
    """Extract optional transfer directive from LLM output.

    The accepted directive format is a dedicated first line:
      TRANSFER:911
      TRANSFER:988
      TRANSFER:number:+14155551212
      TRANSFER:sip:sip:agent@example.com
    Optional second line:
      TRANSFER-META:forwarded-by=Terris;topic=overdose;priority=high
    """
    stripped = reply.strip()
    if not stripped:
        return None, ""

    lines = stripped.splitlines()
    first_line = lines[0].strip()
    match = _TRANSFER_DIRECTIVE_PATTERN.match(first_line.strip())
    if match:
        target = match.group(1)
        spoken_reply = "\n".join(lines[1:]).strip()
        return TransferDirective(target_kind="number", target=target, metadata={}), spoken_reply

    v2 = _TRANSFER_V2_PATTERN.match(first_line)
    if not v2:
        return None, stripped

    target_kind = v2.group(1).lower()
    target = v2.group(2).strip()
    metadata: dict[str, str] = {}

    spoken_start_index = 1
    if len(lines) > 1:
        metadata_match = _TRANSFER_META_PATTERN.match(lines[1].strip())
        if metadata_match:
            metadata = _parse_transfer_metadata(metadata_match.group(1))
            spoken_start_index = 2

    spoken_reply = "\n".join(lines[spoken_start_index:]).strip()
    return (
        TransferDirective(
            target_kind="sip" if target_kind == "sip" else "number",
            target=target,
            metadata=metadata,
        ),
        spoken_reply,
    )


def _sanitize_model_reply(reply: str) -> str:
    """Remove hidden reasoning and normalize whitespace before parsing/speaking."""
    if not reply:
        return ""
    sanitized = _THINK_BLOCK_PATTERN.sub("", reply)
    filtered_lines: list[str] = []
    for line in sanitized.splitlines():
        if _EMPHASIS_HINTS_PATTERN.match(line.strip()):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def _extract_clause_aware_tts_stub_hints(reply: str) -> ClauseAwareTTSStubHints:
    """Extract optional emphasis hints without changing current speech behavior.

    Planned linkage:
    Documentation/alignment_plan/Streaming_Implmentation_Considerations_and_plan.md
    """
    if not reply:
        return ClauseAwareTTSStubHints(phrases=())

    phrases: list[str] = []
    for raw_line in reply.splitlines():
        match = _EMPHASIS_HINTS_PATTERN.match(raw_line.strip())
        if not match:
            continue
        for part in match.group(1).split("|"):
            phrase = part.strip().strip('"')
            if phrase:
                phrases.append(phrase)
    return ClauseAwareTTSStubHints(phrases=tuple(phrases))


def _parse_transfer_metadata(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for chunk in raw.split(";"):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_key in {"forwarded-by", "topic", "priority"} and normalized_value:
            parsed[normalized_key] = normalized_value
    return parsed


def _default_transfer_announcement(target: str) -> str:
    if target == "911":
        return "I am connecting you to 911 now."
    if target == "988":
        return "I am connecting you to 988 now."
    return "I am connecting you now."


def _transfer_confirmation_prompt(target: str) -> str:
    """Return the verbal confirmation question Terris asks before a transfer."""
    if target == "911":
        return (
            "I'd like to connect you to 911 emergency services right now. "
            "Do I have your permission to do that?"
        )
    if target == "988":
        return (
            "I'd like to connect you to the 988 Suicide and Crisis Lifeline right now. "
            "Do I have your permission to do that?"
        )
    return "I'd like to transfer your call now. Do I have your permission to do that?"


def _post_call_reopen_prompt(target: str) -> str:
    """Ask caller whether Terris should re-open after operator disconnects."""
    if target == "911":
        return (
            "If the 911 operator ends the call first, should I come back on the line "
            "to check on you? Please say yes or no."
        )
    if target == "988":
        return (
            "If the 988 counselor ends the call first, should I come back on the line "
            "to check on you? Please say yes or no."
        )
    return (
        "If the transferred line ends first, should I come back on the line "
        "to check on you? Please say yes or no."
    )


def build_transfer_twiml(
    target_kind: Literal["number", "sip"],
    target: str,
    announcement: str,
    *,
    metadata: Optional[dict[str, str]] = None,
    action_url: Optional[str] = None,
) -> str:
    normalized = _normalize_transfer_target(target_kind, target)
    metadata = metadata or {}
    _validate_transfer_target(normalized, metadata)

    sip_target = normalized
    if target_kind == "sip" and metadata:
        sip_target = _append_sip_headers(normalized, metadata)

    if target_kind == "number" and metadata:
        if settings.transfer_metadata_mode == "strict":
            raise ValueError("metadata is not supported for number transfers in strict mode")
        logger.info("Compatibility mode: PSTN transfer metadata retained in logs only: %s", metadata)

    safe_announcement = escape(announcement.strip() or _default_transfer_announcement(target))
    destination = escape(sip_target if target_kind == "sip" else normalized)
    action_attr = f' action="{escape(action_url)}"' if action_url else ""
    dial_verb = (
        f"<Dial{action_attr}><Sip>{destination}</Sip></Dial>"
        if target_kind == "sip"
        else f"<Dial{action_attr}>{destination}</Dial>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Say>{safe_announcement}</Say>"
        f"{dial_verb}"
        "</Response>"
    )


def _normalize_transfer_target(target_kind: Literal["number", "sip"], target: str) -> str:
    trimmed = target.strip()
    if target_kind == "number":
        if trimmed in {"911", "988"}:
            return trimmed
        if not _E164_PATTERN.match(trimmed):
            raise ValueError("number target must be 911, 988, or E.164")
        return trimmed

    if not trimmed.lower().startswith("sip:"):
        trimmed = f"sip:{trimmed}"
    return trimmed


def _validate_transfer_target(target: str, metadata: dict[str, str]) -> None:
    if target in {"911", "988"}:
        return

    target_kind = "sip" if target.startswith("sip:") else "number"
    if settings.transfer_services_enabled and not transfer_services.is_configured_target(
        target_kind, target
    ):
        raise ValueError("custom transfer target is not listed in transfer services catalog")

    if not settings.transfer_allow_custom_targets:
        raise ValueError("custom transfer targets are disabled")

    if target.startswith("sip:"):
        domain = _extract_sip_domain(target)
        allowed_domains = _csv_as_set(settings.transfer_allowed_sip_domains)
        if not allowed_domains or domain not in allowed_domains:
            raise ValueError(f"sip target domain is not allowlisted: {domain}")
        return

    allowed_numbers = _csv_as_set(settings.transfer_allowed_numbers)
    if not allowed_numbers or target not in allowed_numbers:
        raise ValueError(f"number target is not allowlisted: {target}")


def _post_call_reopen_mode_for_transfer(
    transfer: TransferDirective,
) -> Literal["off", "auto", "prompt"]:
    if transfer.target in {"911", "988"}:
        return get_transfer_post_call_reopen_mode()
    return get_custom_transfer_post_call_reopen_mode()


def _append_sip_headers(sip_uri: str, metadata: dict[str, str]) -> str:
    parsed = urlparse(sip_uri)
    existing = dict(parse_qsl(parsed.query, keep_blank_values=False))

    mapping = {
        "forwarded-by": "x-forwarded-by",
        "topic": "x-topic",
        "priority": "x-priority",
    }
    for key, value in metadata.items():
        header_name = mapping.get(key)
        if not header_name:
            continue
        existing[header_name] = value

    query = urlencode(existing, doseq=False, quote_via=quote, safe="")
    return urlunparse(parsed._replace(query=query))


def _csv_as_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _extract_sip_domain(sip_uri: str) -> str:
    body = sip_uri[4:] if sip_uri.lower().startswith("sip:") else sip_uri
    host_part = body.split("?", 1)[0]
    host_part = host_part.split(";", 1)[0]
    if "@" in host_part:
        host_part = host_part.split("@", 1)[1]
    if ":" in host_part:
        host_part = host_part.split(":", 1)[0]
    if not host_part:
        raise ValueError("invalid sip target")
    return host_part.lower()


def twilio_transfer_call_update(call_sid: str, twiml: str) -> None:
    from twilio.rest import Client

    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    client.calls(call_sid).update(twiml=twiml)