"""Text-to-speech service backed by Piper (local, open-source).

Piper is invoked as a subprocess that reads text on stdin and writes raw
16-bit PCM to stdout.  See https://github.com/rhasspy/piper for installation
and voice-model download instructions.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Protocol

import httpx
import numpy as np
from scipy.signal import resample_poly

from app.core.config import settings
from app.models.schemas import EmptyOutputError, UnsupportedError

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")

# Piper outputs 22 050 Hz by default for the lessac-medium voice.
# Update this constant if you use a different voice model.
PIPER_SAMPLE_RATE = 22050


@dataclass(frozen=True)
class VoiceSelection:
    """Resolved voice configuration used by the selected TTS backend."""

    voice_id: str
    model_path: str
    sample_rate: int


_DEFAULT_VOICE_ID = "en-US-lessac-medium"
_VOICE_ALIASES = {
    "en-us-lessac-medium": _DEFAULT_VOICE_ID,
    "en_us-lessac-medium": _DEFAULT_VOICE_ID,
    "en_us_lessac_medium": _DEFAULT_VOICE_ID,
    "lessac": _DEFAULT_VOICE_ID,
    "en-us-avamultilingualneural": _DEFAULT_VOICE_ID,
    "en-us-ava": _DEFAULT_VOICE_ID,
    "ava": _DEFAULT_VOICE_ID,
    "en_default": "en_default",
    "fa_default": "fa_default",
    "zh_mandarin_default": "zh_mandarin_default",
    "zh_cantonese_default": "zh_cantonese_default",
    "ja_default": "ja_default",
}
_OPTION_ALIASES = {
    "speaker": "speaker",
    "speaker_id": "speaker",
    "speakerid": "speaker",
    "noise_scale": "noise_scale",
    "noiseScale": "noise_scale",
    "noise": "noise_scale",
    "noise_w": "noise_w",
    "noiseW": "noise_w",
    "length_scale": "length_scale",
    "lengthScale": "length_scale",
    "rate": "rate",
    "speaking_rate": "rate",
    "speakingRate": "rate",
    "speed": "rate",
}
_DEFAULT_OPTIONS = {
    "rate": 1.0,
    "noise_scale": 0.667,
    "noise_w": 0.8,
}

_DEFAULT_F5_VOICE_BY_LANGUAGE = {
    "en": "en_default",
    "fa": "fa_default",
    "zh": "zh_mandarin_default",
    "yue": "zh_cantonese_default",
    "ja": "ja_default",
}


class SpeechSynthesizer(Protocol):
    """Formal interface for interchangeable TTS synthesizer backends."""

    backend_name: str

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Synthesize text to float32 PCM at :data:`PIPER_SAMPLE_RATE`."""


@dataclass(frozen=True)
class TTSSynthesisObservability:
    """Per-call TTS observability payload for turn-level telemetry."""

    samples: np.ndarray | None
    backend_name: str
    voice_id: str
    output_sample_rate_hz: int
    fallback_used: bool
    synthesis_latency_ms: int
    audio_bytes: int
    audio_duration_ms: int
    failure_reason: "TTSFailureReason" = ""

    @property
    def ok(self) -> bool:
        return self.samples is not None and not self.failure_reason


@dataclass(frozen=True)
class PiperSpeechSynthesizer:
    """Piper-backed implementation of :class:`SpeechSynthesizer`."""

    backend_name: str = "piper"

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            partial(_synthesize_piper, text, voice=voice, language=language, options=options),
        )


@dataclass(frozen=True)
class F5HTTPSpeechSynthesizer:
    """HTTP F5-TTS implementation of :class:`SpeechSynthesizer`."""

    backend_name: str = "f5_http"

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return await _synthesize_f5_http(text, voice=voice, language=language, options=options)


@dataclass(frozen=True)
class F5MLXLocalSpeechSynthesizer:
    """Local MLX F5-TTS implementation of :class:`SpeechSynthesizer`."""

    backend_name: str = "f5_mlx_local"

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            partial(_synthesize_f5_mlx_local_sync, text, voice=voice, language=language, options=options),
        )


@dataclass
class FallbackSpeechSynthesizer:
    """Wrap a primary synthesizer with a fallback synthesizer."""

    primary: SpeechSynthesizer
    fallback: SpeechSynthesizer
    last_call_backend_name: str = ""
    last_call_fallback_used: bool = False

    @property
    def backend_name(self) -> str:
        return self.primary.backend_name

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        try:
            samples = await self.primary.synthesize(
                text,
                voice=voice,
                language=language,
                options=options,
            )
            self.last_call_backend_name = self.primary.backend_name
            self.last_call_fallback_used = False
            return samples
        except (UnsupportedError, EmptyOutputError) as exc:
            logger.warning(
                "%s failed (%s); falling back to %s",
                self.primary.backend_name,
                exc,
                self.fallback.backend_name,
            )
            samples = await self.fallback.synthesize(
                text,
                voice=voice,
                language=language,
                options=options,
            )
            self.last_call_backend_name = self.fallback.backend_name
            self.last_call_fallback_used = True
            return samples


class SessionStickyFallbackSpeechSynthesizer:
    """Fallback wrapper that sticks to fallback after the first primary failure."""

    def __init__(self, primary: SpeechSynthesizer, fallback: SpeechSynthesizer) -> None:
        self.primary = primary
        self.fallback = fallback
        self._fallback_active = False
        self.last_call_backend_name = primary.backend_name
        self.last_call_fallback_used = False

    @property
    def backend_name(self) -> str:
        return self.fallback.backend_name if self._fallback_active else self.primary.backend_name

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if self._fallback_active:
            samples = await self.fallback.synthesize(
                text,
                voice=voice,
                language=language,
                options=options,
            )
            self.last_call_backend_name = self.fallback.backend_name
            self.last_call_fallback_used = True
            return samples

        try:
            samples = await self.primary.synthesize(
                text,
                voice=voice,
                language=language,
                options=options,
            )
            self.last_call_backend_name = self.primary.backend_name
            self.last_call_fallback_used = False
            return samples
        except (UnsupportedError, EmptyOutputError) as exc:
            self._fallback_active = True
            logger.warning(
                "%s failed (%s); enabling session-sticky fallback=%s",
                self.primary.backend_name,
                exc,
                self.fallback.backend_name,
            )
            samples = await self.fallback.synthesize(
                text,
                voice=voice,
                language=language,
                options=options,
            )
            self.last_call_backend_name = self.fallback.backend_name
            self.last_call_fallback_used = True
            return samples


def _build_backend_synthesizer(backend: str) -> SpeechSynthesizer:
    normalized = backend.lower()
    if normalized == "f5_http":
        return F5HTTPSpeechSynthesizer()
    if normalized == "f5_mlx_local":
        return F5MLXLocalSpeechSynthesizer()
    return PiperSpeechSynthesizer()


def _build_synthesizer() -> SpeechSynthesizer:
    primary = _build_backend_synthesizer(settings.tts_backend)
    if settings.tts_fallback_backend != "piper" or primary.backend_name == "piper":
        return primary
    return FallbackSpeechSynthesizer(primary=primary, fallback=PiperSpeechSynthesizer())


def build_session_synthesizer() -> SpeechSynthesizer:
    """Build a synthesizer suitable for one call session with sticky fallback."""
    primary = _build_backend_synthesizer(settings.tts_backend)
    if settings.tts_fallback_backend != "piper" or primary.backend_name == "piper":
        return primary
    return SessionStickyFallbackSpeechSynthesizer(
        primary=primary,
        fallback=PiperSpeechSynthesizer(),
    )


def _resolve_observed_backend_name(synthesizer: SpeechSynthesizer) -> str:
    observed = getattr(synthesizer, "last_call_backend_name", "")
    if observed:
        return str(observed)
    return getattr(synthesizer, "backend_name", settings.tts_backend)


def _resolve_observed_fallback_used(synthesizer: SpeechSynthesizer) -> bool:
    return bool(getattr(synthesizer, "last_call_fallback_used", False))


TTSFailureReason = Literal[
    "",
    "empty_text",
    "binary_not_found",
    "timeout",
    "backend_http_error",
    "invalid_response",
    "empty_output",
    "synthesis_failed",
    "unsupported",
    "unknown",
]


def _classify_tts_exception(exc: Exception) -> TTSFailureReason:
    if isinstance(exc, EmptyOutputError):
        message = str(exc).lower()
        if "empty text" in message:
            return "empty_text"
        return "empty_output"

    if isinstance(exc, UnsupportedError):
        message = str(exc).lower()
        if "binary not found" in message:
            return "binary_not_found"
        if "timed out" in message:
            return "timeout"
        if "http" in message:
            return "backend_http_error"
        if "base64" in message or "json response" in message or "sample width" in message:
            return "invalid_response"
        if "exited with code" in message or "generate() failed" in message:
            return "synthesis_failed"
        if "unsupported" in message:
            return "unsupported"
        return "synthesis_failed"

    return "unknown"


def _normalize_voice_id(voice: str | None) -> str:
    if not voice:
        return _DEFAULT_VOICE_ID
    normalized = voice.strip().replace("_", "-").lower()
    return _VOICE_ALIASES.get(normalized, voice.strip())


def _normalize_language(language: str | None) -> str:
    if not language:
        return "en-US"
    return language.strip()


def _resolve_f5_voice(language: str | None, requested_voice: str | None) -> str:
    normalized_language = _normalize_language(language).lower()
    requested = _normalize_voice_id(requested_voice)
    if requested in _DEFAULT_F5_VOICE_BY_LANGUAGE.values():
        return requested

    if normalized_language.startswith("yue") or "hant-hk" in normalized_language:
        return _DEFAULT_F5_VOICE_BY_LANGUAGE["yue"]

    lang_prefix = normalized_language.split("-", 1)[0]
    return _DEFAULT_F5_VOICE_BY_LANGUAGE.get(lang_prefix, settings.f5_tts_voice)


# F5-TTS MLX local backend outputs 24 kHz; resampled to PIPER_SAMPLE_RATE downstream.
_F5_MLX_SAMPLE_RATE = 24000


def _select_voice(voice: str | None, language: str | None, backend: str) -> VoiceSelection:
    """Resolve requested voice/language to backend-specific voice selection."""
    if backend in ("f5_http", "f5_mlx_local"):
        voice_id = _resolve_f5_voice(language=language, requested_voice=voice)
        return VoiceSelection(
            voice_id=voice_id,
            model_path="",
            sample_rate=_F5_MLX_SAMPLE_RATE,
        )

    requested = _normalize_voice_id(voice)
    if requested != _DEFAULT_VOICE_ID:
        logger.warning("Unsupported TTS voice '%s'; falling back to %s", requested, _DEFAULT_VOICE_ID)

    if language and not language.lower().startswith("en"):
        logger.warning("Unsupported TTS language '%s'; falling back to en-US voice", language)

    return VoiceSelection(
        voice_id=_DEFAULT_VOICE_ID,
        model_path=settings.piper_model_path,
        sample_rate=PIPER_SAMPLE_RATE,
    )


def _normalize_options(options: dict[str, Any] | None) -> dict[str, float | int]:
    """Map provider-specific option names to Piper-supported flags and defaults."""
    normalized: dict[str, float | int] = dict(_DEFAULT_OPTIONS)
    if options:
        for key, value in options.items():
            canonical = _OPTION_ALIASES.get(key)
            if canonical is None:
                continue
            if canonical == "speaker":
                try:
                    normalized[canonical] = int(value)
                except (TypeError, ValueError):
                    logger.warning("Ignoring invalid TTS speaker option: %r", value)
                continue
            try:
                normalized[canonical] = float(value)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid TTS option '%s': %r", key, value)

    rate = max(0.25, min(4.0, float(normalized.get("rate", 1.0))))
    normalized["length_scale"] = max(0.25, min(4.0, 1.0 / rate))
    normalized["noise_scale"] = max(0.0, min(2.0, float(normalized.get("noise_scale", 0.667))))
    normalized["noise_w"] = max(0.0, min(2.0, float(normalized.get("noise_w", 0.8))))
    return normalized


def _build_piper_cmd(selection: VoiceSelection, options: dict[str, float | int]) -> list[str]:
    cmd = [
        settings.piper_binary,
        "--model",
        selection.model_path,
        "--output_raw",
        "--length_scale",
        str(options["length_scale"]),
        "--noise_scale",
        str(options["noise_scale"]),
        "--noise_w",
        str(options["noise_w"]),
    ]
    if "speaker" in options:
        cmd.extend(["--speaker", str(options["speaker"])])
    return cmd


def _decode_wav_or_pcm16(raw_audio: bytes, sample_rate_hint: int) -> tuple[np.ndarray, int]:
    if raw_audio.startswith(b"RIFF"):
        with wave.open(io.BytesIO(raw_audio), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_bytes = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            raise UnsupportedError(f"F5-TTS returned unsupported sample width: {sample_width}")

        pcm = np.frombuffer(frame_bytes, dtype=np.int16)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return pcm.astype(np.float32) / 32768.0, sample_rate

    pcm = np.frombuffer(raw_audio, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, sample_rate_hint


def _resample_to_piper_rate(samples: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == PIPER_SAMPLE_RATE:
        return samples
    if source_rate <= 0:
        raise UnsupportedError(f"Invalid source sample rate from TTS backend: {source_rate}")

    resampled = resample_poly(samples, PIPER_SAMPLE_RATE, source_rate)
    return resampled.astype(np.float32)


def _synthesize_piper(
    text: str,
    *,
    voice: str | None = None,
    language: str | None = None,
    options: dict[str, Any] | None = None,
) -> np.ndarray:
    """Run piper synchronously and return float32 PCM samples."""
    if not text.strip():
        raise EmptyOutputError("Cannot synthesize empty text")

    selection = _select_voice(voice=voice, language=language, backend="piper")
    resolved_options = _normalize_options(options)
    cmd = _build_piper_cmd(selection, resolved_options)
    try:
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise UnsupportedError(
            f"Piper binary not found at '{settings.piper_binary}'. "
            "Install piper and set PIPER_BINARY in .env."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise UnsupportedError(f"Piper synthesis timed out after 30 seconds") from exc

    if result.returncode != 0:
        raise UnsupportedError(f"Piper exited with code {result.returncode}: {result.stderr.decode()}")

    if not result.stdout:
        raise EmptyOutputError("Piper produced no audio output")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def _synthesize_f5_mlx_local_sync(
    text: str,
    *,
    voice: str | None = None,
    language: str | None = None,
    options: dict[str, Any] | None = None,
) -> np.ndarray:
    """Synthesise locally via f5-tts-mlx (Apple Silicon / MLX). Sync; run in executor."""
    if not text.strip():
        raise EmptyOutputError("Cannot synthesize empty text")

    try:
        from f5_tts_mlx.generate import generate  # noqa: PLC0415
    except ImportError as exc:
        raise UnsupportedError(
            "f5-tts-mlx is not installed. Install with: pip install f5-tts-mlx"
        ) from exc

    normalized_options = _normalize_options(options)
    # length_scale is inverted speed; convert back to speed for f5-tts-mlx
    speed = 1.0 / max(0.25, float(normalized_options.get("length_scale", 1.0)))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
        try:
            generate(
                generation_text=text,
                model_name=settings.f5_mlx_model,
                speed=speed,
                output_path=tmp_wav.name,
            )
        except Exception as exc:
            raise UnsupportedError(f"f5-tts-mlx generate() failed: {exc}") from exc

        try:
            tmp_wav.seek(0)
            wav_bytes = tmp_wav.read()
        except Exception as exc:
            raise UnsupportedError(f"Failed to read f5-tts-mlx output WAV: {exc}") from exc

    if not wav_bytes:
        raise EmptyOutputError("f5-tts-mlx produced no audio output")

    samples, sample_rate = _decode_wav_or_pcm16(wav_bytes, _F5_MLX_SAMPLE_RATE)
    if len(samples) == 0:
        raise EmptyOutputError("f5-tts-mlx produced empty audio")

    return _resample_to_piper_rate(samples, sample_rate)


async def _synthesize_f5_http(
    text: str,
    *,
    voice: str | None = None,
    language: str | None = None,
    options: dict[str, Any] | None = None,
) -> np.ndarray:
    """Call F5 HTTP backend like HealthCoacher's F5TTSHTTPService."""
    if not text.strip():
        raise EmptyOutputError("Cannot synthesize empty text")

    selection = _select_voice(voice=voice, language=language, backend="f5_http")
    normalized_options = _normalize_options(options)
    payload = {
        "text": text,
        "voice": selection.voice_id,
        "language": _normalize_language(language),
        "options": normalized_options,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.f5_tts_timeout_s) as client:
            response = await client.post(settings.f5_tts_endpoint, json=payload)
    except httpx.TimeoutException as exc:
        raise UnsupportedError(f"F5-TTS request timed out after {settings.f5_tts_timeout_s} seconds") from exc
    except httpx.HTTPError as exc:
        raise UnsupportedError(f"F5-TTS HTTP request failed: {exc}") from exc

    if response.status_code < 200 or response.status_code >= 300:
        raise UnsupportedError(
            f"F5-TTS returned HTTP {response.status_code}: {response.text[:200]}"
        )

    content_type = response.headers.get("content-type", "").lower()
    audio_bytes = response.content

    if "application/json" in content_type:
        body = response.json()
        encoded_audio = body.get("audio_base64") or body.get("audio") or body.get("data")
        if not encoded_audio:
            raise EmptyOutputError("F5-TTS JSON response does not contain audio data")
        try:
            import base64

            audio_bytes = base64.b64decode(encoded_audio)
        except Exception as exc:  # pragma: no cover - defensive decode guard
            raise UnsupportedError("F5-TTS JSON audio payload is not valid base64") from exc

    if not audio_bytes:
        raise EmptyOutputError("F5-TTS produced no audio output")

    samples, sample_rate = _decode_wav_or_pcm16(audio_bytes, selection.sample_rate)
    if len(samples) == 0:
        raise EmptyOutputError("F5-TTS produced empty audio samples")

    return _resample_to_piper_rate(samples, sample_rate)


async def synthesize(
    text: str,
    *,
    voice: str | None = None,
    language: str | None = None,
    options: dict[str, Any] | None = None,
    synthesizer: SpeechSynthesizer | None = None,
) -> np.ndarray:
    """Synthesize *text* to speech, returning float32 PCM at :data:`PIPER_SAMPLE_RATE` Hz."""
    active_synthesizer = synthesizer or _build_synthesizer()
    return await active_synthesizer.synthesize(
        text,
        voice=voice,
        language=language,
        options=options,
    )


async def synthesize_with_observability(
    text: str,
    *,
    voice: str | None = None,
    language: str | None = None,
    options: dict[str, Any] | None = None,
    synthesizer: SpeechSynthesizer | None = None,
) -> TTSSynthesisObservability:
    """Synthesize text and return audio plus per-call TTS observability metadata."""
    active_synthesizer = synthesizer or _build_synthesizer()
    start = time.perf_counter()
    requested_backend = getattr(active_synthesizer, "backend_name", settings.tts_backend)

    try:
        try:
            samples = await synthesize(
                text,
                voice=voice,
                language=language,
                options=options,
                synthesizer=active_synthesizer,
            )
        except TypeError as exc:
            # Test doubles may only accept a subset of keyword arguments.
            samples = await synthesize(
                text,
                language=language,
            )
        latency_ms = round((time.perf_counter() - start) * 1000)
        observed_backend = _resolve_observed_backend_name(active_synthesizer)
        fallback_used = _resolve_observed_fallback_used(active_synthesizer)
        voice_selection = _select_voice(voice=voice, language=language, backend=observed_backend)
        return TTSSynthesisObservability(
            samples=samples,
            backend_name=observed_backend,
            voice_id=voice_selection.voice_id,
            output_sample_rate_hz=PIPER_SAMPLE_RATE,
            fallback_used=fallback_used,
            synthesis_latency_ms=latency_ms,
            audio_bytes=int(samples.nbytes),
            audio_duration_ms=round((len(samples) / PIPER_SAMPLE_RATE) * 1000),
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000)
        voice_selection = _select_voice(voice=voice, language=language, backend=requested_backend)
        return TTSSynthesisObservability(
            samples=None,
            backend_name=_resolve_observed_backend_name(active_synthesizer),
            voice_id=voice_selection.voice_id,
            output_sample_rate_hz=PIPER_SAMPLE_RATE,
            fallback_used=_resolve_observed_fallback_used(active_synthesizer),
            synthesis_latency_ms=latency_ms,
            audio_bytes=0,
            audio_duration_ms=0,
            failure_reason=_classify_tts_exception(exc),
        )
