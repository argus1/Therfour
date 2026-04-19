"""Call-session orchestration and audio-pipeline utilities for Twilio Media Streams.

Audio pipeline per conversation turn
─────────────────────────────────────
Inbound  : Twilio μ-law/8 kHz → decode → PCM-16/8 kHz → float32/16 kHz → Whisper
Outbound : text → Piper → float32/22 kHz → PCM-16/8 kHz → μ-law → Twilio
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from app.core.config import settings
from app.services import llm, stt, tts
from app.services.tts import PIPER_SAMPLE_RATE
from app.services.vad import StreamingSpeechDetector, silero_vad_available

logger = logging.getLogger(__name__)

# ── μ-law codec ───────────────────────────────────────────────────────────────
# audioop is part of the Python standard library up to 3.12; audioop-lts
# provides a drop-in replacement for Python 3.13+.
try:
    import audioop  # type: ignore[import]
except ImportError:  # Python 3.13+
    import audioop_lts as audioop  # type: ignore[no-redef]


def mulaw_to_pcm16(data: bytes) -> bytes:
    """Decode 8-bit μ-law bytes to signed 16-bit PCM bytes."""
    return audioop.ulaw2lin(data, 2)  # sample_width=2 → 16-bit


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
    """Manages the full lifecycle of a single Twilio voice call.

    The session consumes messages from a Twilio Media Stream WebSocket,
    buffers inbound audio, runs the STT → LLM → TTS pipeline when the
    caller pauses, and streams the synthesised audio back to Twilio.
    """

    # Twilio sends μ-law audio in ~20 ms frames (160 bytes @ 8 kHz).
    _OUTBOUND_CHUNK = 160

    def __init__(self, websocket) -> None:
        self._ws = websocket
        self._stream_sid: Optional[str] = None
        self._audio_buffer: list[bytes] = []
        self._conversation: list[dict] = []
        self._pending_turns: list[np.ndarray] = []
        self._silence_timer: Optional[asyncio.TimerHandle] = None
        self._processing = False
        self._loop = asyncio.get_event_loop()
        self._speech_detector: Optional[StreamingSpeechDetector] = None

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
                logger.info("Stream started – SID: %s", self._stream_sid)
            elif event == "media":
                await self._on_media(msg["media"]["payload"])
            elif event == "stop":
                logger.info("Stream stopped")
                self._cancel_silence_timer()
                if self._speech_detector is not None:
                    flushed = self._speech_detector.flush()
                    if flushed is not None:
                        self._enqueue_turn(flushed)
                break

    # ── Media handling ────────────────────────────────────────────────────────

    async def _on_media(self, payload_b64: str) -> None:
        """Decode a Twilio media chunk and append it to the audio buffer."""
        chunk = base64.b64decode(payload_b64)
        if self._speech_detector is None:
            self._audio_buffer.append(chunk)
            self._reset_silence_timer()
            return

        pcm8k = mulaw_to_pcm16(chunk)
        float16k = upsample(
            pcm8k,
            settings.audio_sample_rate_twilio,
            settings.audio_sample_rate_whisper,
        )
        finalized_turns = self._speech_detector.process_chunk(float16k)
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
        """Called from the event loop after silence_timeout_s of silence."""
        if not self._processing and self._audio_buffer:
            asyncio.ensure_future(self._process_buffered_turn())

    def _enqueue_turn(self, audio: np.ndarray) -> None:
        if self._processing:
            self._pending_turns.append(audio)
            return
        asyncio.ensure_future(self._process_audio_turn(audio))

    # ── Turn processing ───────────────────────────────────────────────────────

    async def _process_buffered_turn(self) -> None:
        self._processing = True
        audio_chunks, self._audio_buffer = self._audio_buffer, []
        try:
            # 1. Decode μ-law and upsample to 16 kHz for Whisper
            pcm8k = b"".join(mulaw_to_pcm16(c) for c in audio_chunks)
            float16k = upsample(
                pcm8k,
                settings.audio_sample_rate_twilio,
                settings.audio_sample_rate_whisper,
            )
            await self._run_turn(float16k)
        except Exception:
            logger.exception("Error during buffered call turn processing")
        finally:
            self._finish_turn_processing()

    async def _process_audio_turn(self, audio: np.ndarray) -> None:
        self._processing = True
        try:
            await self._run_turn(audio)
        except Exception:
            logger.exception("Error during voiced call turn processing")
        finally:
            self._finish_turn_processing()

    async def _run_turn(self, audio: np.ndarray) -> None:
        min_samples = int(settings.min_audio_duration_s * settings.audio_sample_rate_whisper)
        if len(audio) < min_samples:
            self._log_turn_drop("too_short", audio=audio)
            return

        result = await stt.transcribe(audio)
        text = result.text.strip()
        if not text:
            self._log_turn_drop(result.failure_reason or "no_speech", audio=audio, result=result)
            return
        logger.info(
            "Transcribed [%s via %s]: %s",
            result.language,
            result.backend_name,
            text,
        )

        self._conversation.append({"role": "user", "content": text})
        reply = await llm.generate(self._conversation)
        self._conversation.append({"role": "assistant", "content": reply})
        logger.info("LLM reply: %s", reply)

        speech_samples = await tts.synthesize(reply)

        await self._send_audio(speech_samples)

    def _finish_turn_processing(self) -> None:
        self._processing = False
        if self._pending_turns:
            next_audio = self._pending_turns.pop(0)
            asyncio.ensure_future(self._process_audio_turn(next_audio))

    def _log_turn_drop(
        self,
        reason: str,
        *,
        audio: np.ndarray,
        result: Optional[stt.TranscriptionResult] = None,
    ) -> None:
        details = {
            "reason": reason,
            "audio_ms": round((len(audio) / settings.audio_sample_rate_whisper) * 1000),
        }
        if result is not None:
            details.update(
                {
                    "backend": result.backend_name,
                    "fallback_used": result.fallback_used,
                    "quality": f"{result.transcript_quality_score:.2f}",
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

        # Notify Twilio that the response audio has finished
        await self._ws.send_text(
            json.dumps(
                {
                    "event": "mark",
                    "streamSid": self._stream_sid,
                    "mark": {"name": "end_of_response"},
                }
            )
        )
