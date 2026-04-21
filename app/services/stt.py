"""Speech-to-text service backed by faster-whisper (local, open-source)."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from app.core.config import settings
from app.models.schemas import TranscriptionResult
from app.services.observability import emit_stage_event

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt")
_model: Optional[WhisperModel] = None


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


def _transcribe(audio: np.ndarray, language: Optional[str]) -> TranscriptionResult:
    model = _load_model()
    segments, info = model.transcribe(
        audio,
        language=language or settings.whisper_language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return TranscriptionResult(
        text=text,
        language=info.language,
        confidence=info.language_probability,
        language_confidence=info.language_probability,
    )


async def transcribe(audio: np.ndarray, language: Optional[str] = None) -> TranscriptionResult:
    """Transcribe *audio* (float32, 16 kHz, mono) to text asynchronously."""
    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, partial(_transcribe, audio, language))
    except Exception:
        emit_stage_event(
            stage="stt",
            status="failure",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            failure_reason="decode_error",
            language=language,
        )
        raise

    failure_reason = "" if result.text.strip() else "no_speech"
    if failure_reason:
        result.failure_reason = failure_reason

    emit_stage_event(
        stage="stt",
        status="success" if not failure_reason else "dropped",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        failure_reason=failure_reason,
        language=result.language or language,
        backend_name=result.backend_name,
    )
    return result
