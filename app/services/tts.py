"""Text-to-speech service backed by Piper (local, open-source).

Piper is invoked as a subprocess that reads text on stdin and writes raw
16-bit PCM to stdout.  See https://github.com/rhasspy/piper for installation
and voice-model download instructions.
"""

from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

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


def _select_voice(voice: str | None, language: str | None, backend: str) -> VoiceSelection:
    """Resolve requested voice/language to backend-specific voice selection."""
    if backend == "f5_http":
        voice_id = _resolve_f5_voice(language=language, requested_voice=voice)
        return VoiceSelection(
            voice_id=voice_id,
            model_path="",
            sample_rate=settings.f5_tts_sample_rate,
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
) -> np.ndarray:
    """Synthesize *text* to speech, returning float32 PCM at :data:`PIPER_SAMPLE_RATE` Hz."""
    backend = settings.tts_backend.lower()
    if backend == "f5_http":
        try:
            return await _synthesize_f5_http(text, voice=voice, language=language, options=options)
        except (UnsupportedError, EmptyOutputError) as exc:
            if settings.tts_fallback_backend != "piper":
                raise
            logger.warning("F5-TTS failed (%s); falling back to Piper", exc)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        partial(_synthesize_piper, text, voice=voice, language=language, options=options),
    )
