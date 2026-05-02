"""Speech-to-text service with pluggable backends (faster-whisper primary, sherpa-onnx fallback)."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from faster_whisper import WhisperModel

from app.core.config import settings
from app.models.schemas import TranscriptionResult

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt")

STTBackendName = Literal["whisper", "sherpa"]


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class STTBackend(ABC):
    """Abstract base for all STT backends."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> TranscriptionResult:
        """Transcribe float32, 16 kHz mono *audio* and return a result."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _DecodeAttempt:
    language: Optional[str]
    beam_size: int
    fallback_used: bool


class _WhisperBackend(STTBackend):
    """faster-whisper backed STT.  Model is lazily loaded on first call."""

    def __init__(self) -> None:
        self._model: Optional[WhisperModel] = None

    def _load(self) -> WhisperModel:
        if self._model is None:
            logger.info(
                "Loading Whisper model '%s' on %s …",
                settings.whisper_model,
                settings.whisper_device,
            )
            self._model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            logger.info("Whisper model ready")
        return self._model

    def _build_attempts(self, language: Optional[str]) -> list[_DecodeAttempt]:
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

    def _make_result(
        self,
        text: str,
        info,
        *,
        fallback_used: bool,
        failure_reason: str = "",
        audio_duration_s: float = 0.0,
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
            audio_duration_s=audio_duration_s,
        )

    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> TranscriptionResult:
        model = self._load()
        best_result: Optional[TranscriptionResult] = None
        last_error: Optional[Exception] = None
        audio_duration_s = len(audio) / settings.audio_sample_rate_whisper

        for attempt in self._build_attempts(language):
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
            result = self._make_result(
                text, info, fallback_used=attempt.fallback_used, audio_duration_s=audio_duration_s
            )
            if best_result is None or result.transcript_quality_score > best_result.transcript_quality_score:
                best_result = result

            if (
                text
                and len(text) >= settings.stt_min_text_characters
                and result.transcript_quality_score >= settings.stt_min_quality_score
            ):
                return result

        if best_result is not None:
            failure_reason = "no_speech" if not best_result.text else "low_quality"
            return best_result.model_copy(update={"text": "", "failure_reason": failure_reason})

        if last_error is not None:
            raise last_error

        return TranscriptionResult(failure_reason="no_speech", audio_duration_s=audio_duration_s)


# ---------------------------------------------------------------------------
# sherpa-onnx backend
# ---------------------------------------------------------------------------

class _SherpaBackend(STTBackend):
    """sherpa-onnx offline recognizer backend.

    Uses a Whisper-compatible ONNX model layout:
      <sherpa_model_dir>/encoder.int8.onnx
      <sherpa_model_dir>/decoder.int8.onnx
      <sherpa_tokens_path>  (tokens.txt)

    The recognizer is lazily loaded on the first call.  If sherpa_onnx is not
    installed or the model files are absent the constructor still succeeds and
    ``transcribe`` will raise ``RuntimeError`` at call-time rather than at
    import-time, so that the primary Whisper path is never blocked.
    """

    def __init__(self) -> None:
        self._recognizer = None  # sherpa_onnx.OfflineRecognizer | None

    def _load(self):
        if self._recognizer is not None:
            return self._recognizer

        try:
            import sherpa_onnx  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "sherpa-onnx is not installed.  Run: pip install sherpa-onnx"
            ) from exc

        model_dir = Path(settings.sherpa_model_dir)
        encoder = str(model_dir / "encoder.int8.onnx")
        decoder = str(model_dir / "decoder.int8.onnx")
        tokens = settings.sherpa_tokens_path

        for path in (encoder, decoder, tokens):
            if not Path(path).exists():
                raise RuntimeError(
                    f"Sherpa-ONNX model file not found: {path}.  "
                    "Set SHERPA_MODEL_DIR / SHERPA_TOKENS_PATH or place model files there."
                )

        logger.info("Loading Sherpa-ONNX model from '%s' …", model_dir)
        whisper_config = sherpa_onnx.OfflineWhisperModelConfig(
            encoder=encoder,
            decoder=decoder,
        )
        model_config = sherpa_onnx.OfflineModelConfig(
            whisper=whisper_config,
            tokens=tokens,
            num_threads=settings.sherpa_num_threads,
        )
        recognizer_config = sherpa_onnx.OfflineRecognizerConfig(model=model_config)
        self._recognizer = sherpa_onnx.OfflineRecognizer(recognizer_config)
        logger.info("Sherpa-ONNX recognizer ready")
        return self._recognizer

    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> TranscriptionResult:
        recognizer = self._load()
        stream = recognizer.create_stream()
        stream.accept_waveform(settings.audio_sample_rate_whisper, audio)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        quality = _quality_score(text)
        failure_reason = ""
        audio_duration_s = len(audio) / settings.audio_sample_rate_whisper
        if not text:
            failure_reason = "no_speech"
        elif (
            len(text) < settings.stt_min_text_characters
            or quality < settings.stt_min_quality_score
        ):
            failure_reason = "low_quality"
            text = ""
        return TranscriptionResult(
            text=text,
            language=language or "",
            confidence=quality,
            language_confidence=0.0,
            transcript_quality_score=quality,
            backend_name="sherpa-onnx",
            fallback_used=True,
            failure_reason=failure_reason,
            audio_duration_s=audio_duration_s,
        )


# ---------------------------------------------------------------------------
# Backend registry & selection
# ---------------------------------------------------------------------------

_whisper_backend: Optional[_WhisperBackend] = None
_sherpa_backend: Optional[_SherpaBackend] = None


def _get_whisper_backend() -> _WhisperBackend:
    global _whisper_backend
    if _whisper_backend is None:
        _whisper_backend = _WhisperBackend()
    return _whisper_backend


def _get_sherpa_backend() -> _SherpaBackend:
    global _sherpa_backend
    if _sherpa_backend is None:
        _sherpa_backend = _SherpaBackend()
    return _sherpa_backend


# ---------------------------------------------------------------------------
# Core dispatch
# ---------------------------------------------------------------------------

def _transcribe(
    audio: np.ndarray,
    language: Optional[str],
    preferred_backend: Optional[STTBackendName],
) -> TranscriptionResult:
    primary = (preferred_backend or settings.stt_primary_backend).lower()

    if primary == "sherpa":
        return _get_sherpa_backend().transcribe(audio, language)

    # Default: Whisper primary, optional Sherpa fallback.
    whisper = _get_whisper_backend()
    try:
        result = whisper.transcribe(audio, language)
    except Exception as exc:
        if settings.stt_sherpa_fallback_enabled:
            logger.info("Whisper failed; attempting Sherpa-ONNX fallback.")
            try:
                return _get_sherpa_backend().transcribe(audio, language)
            except Exception as sherpa_exc:
                logger.warning("Sherpa-ONNX fallback failed after Whisper exception: %s", sherpa_exc)
        raise exc

    if (
        result.failure_reason in ("no_speech", "low_quality")
        and settings.stt_sherpa_fallback_enabled
    ):
        logger.info(
            "Whisper returned '%s'; attempting Sherpa-ONNX fallback.",
            result.failure_reason,
        )
        try:
            sherpa_result = _get_sherpa_backend().transcribe(audio, language)
            return sherpa_result
        except Exception as exc:
            logger.warning("Sherpa-ONNX fallback failed: %s", exc)

    return result


async def transcribe(
    audio: np.ndarray,
    language: Optional[str] = None,
    preferred_backend: Optional[STTBackendName] = None,
) -> TranscriptionResult:
    """Transcribe *audio* (float32, 16 kHz, mono) to text asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, partial(_transcribe, audio, language, preferred_backend))
