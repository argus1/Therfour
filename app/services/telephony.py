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
        self._silence_timer: Optional[asyncio.TimerHandle] = None
        self._processing = False
        self._loop = asyncio.get_event_loop()

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
                break

    # ── Media handling ────────────────────────────────────────────────────────

    async def _on_media(self, payload_b64: str) -> None:
        """Decode a Twilio media chunk and append it to the audio buffer."""
        self._audio_buffer.append(base64.b64decode(payload_b64))
        self._reset_silence_timer()

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
            asyncio.ensure_future(self._process_turn())

    # ── Turn processing ───────────────────────────────────────────────────────

    async def _process_turn(self) -> None:
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

            # Discard utterances that are too short to be meaningful
            min_samples = int(settings.min_audio_duration_s * settings.audio_sample_rate_whisper)
            if len(float16k) < min_samples:
                return

            # 2. Speech-to-text
            result = await stt.transcribe(float16k)
            text = result.text.strip()
            if not text:
                return
            logger.info("Transcribed [%s]: %s", result.language, text)

            # 3. LLM – append to conversation history for multi-turn context
            self._conversation.append({"role": "user", "content": text})
            reply = await llm.generate(self._conversation)
            self._conversation.append({"role": "assistant", "content": reply})
            logger.info("LLM reply: %s", reply)

            # 4. TTS
            speech_samples = await tts.synthesize(reply)

            # 5. Downsample, encode to μ-law, and stream back to Twilio
            await self._send_audio(speech_samples)

        except Exception:
            logger.exception("Error during call turn processing")
        finally:
            self._processing = False

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
