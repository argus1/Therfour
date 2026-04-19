"""Speech-to-text service backed by faster-whisper (local, open-source)."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from app.core.config import settings
from app.models.schemas import TranscriptionResult

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt")
_model: Optional[WhisperModel] = None


@dataclass(frozen=True)
class _DecodeAttempt:
    language: Optional[str]
    beam_size: int
    fallback_used: bool


def _load_model() -> WhisperModel:
    """Load (and cache) the Whisper model.  Called from a thread-pool worker."""
    global _model
    if _model is None:
        logger.info("Loading Whisper model '%s' on %s …", settings.whisper_model, settings.whisper_device)
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("Whisper model ready")
    return _model


def _build_attempts(language: Optional[str]) -> list[_DecodeAttempt]:
    requested_language = language or settings.whisper_language
    attempts = [
        _DecodeAttempt(
            language=requested_language,
            beam_size=max(1, settings.whisper_primary_beam_size),
            fallback_used=False,
        )
    ]
    if settings.whisper_fallback_enabled:
        fallback_language = None if requested_language else settings.whisper_language
        fallback_attempt = _DecodeAttempt(
            language=fallback_language,
            beam_size=max(1, settings.whisper_fallback_beam_size),
            fallback_used=True,
        )
        if fallback_attempt != attempts[0]:
            attempts.append(fallback_attempt)
    return attempts


def _quality_score(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0

    alnum_count = sum(char.isalnum() for char in stripped)
    if alnum_count == 0:
        return 0.0

    score = min(1.0, alnum_count / max(1, settings.stt_min_text_characters * 4))
    if len(stripped.split()) >= 2:
        score = min(1.0, score + 0.25)
    return score


def _result_from_attempt(
    text: str,
    info,
    *,
    fallback_used: bool,
    failure_reason: str = "",
) -> TranscriptionResult:
    language_confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
    return TranscriptionResult(
        text=text,
        language=str(getattr(info, "language", "") or ""),
        confidence=language_confidence,
        language_confidence=language_confidence,
        transcript_quality_score=_quality_score(text),
        backend_name="faster-whisper",
        fallback_used=fallback_used,
        failure_reason=failure_reason,
    )


def _transcribe(audio: np.ndarray, language: Optional[str]) -> TranscriptionResult:
    model = _load_model()
    best_result: Optional[TranscriptionResult] = None
    last_error: Optional[Exception] = None

    for attempt in _build_attempts(language):
        try:
            segments, info = model.transcribe(
                audio,
                language=attempt.language,
                beam_size=attempt.beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
            )
        except Exception as exc:
            last_error = exc
            logger.warning("Whisper decode attempt failed (fallback=%s)", attempt.fallback_used)
            continue

        text = " ".join(seg.text.strip() for seg in segments).strip()
        result = _result_from_attempt(text, info, fallback_used=attempt.fallback_used)
        if best_result is None or result.transcript_quality_score > best_result.transcript_quality_score:
            best_result = result

        if text and len(text) >= settings.stt_min_text_characters and result.transcript_quality_score >= settings.stt_min_quality_score:
            return result

    if best_result is not None:
        failure_reason = "no_speech" if not best_result.text else "low_quality"
        return best_result.model_copy(update={"text": "", "failure_reason": failure_reason})

    if last_error is not None:
        raise last_error

    return TranscriptionResult(failure_reason="no_speech")


async def transcribe(audio: np.ndarray, language: Optional[str] = None) -> TranscriptionResult:
    """Transcribe *audio* (float32, 16 kHz, mono) to text asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, partial(_transcribe, audio, language))
