"""Streaming VAD helpers backed by Silero VAD."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional, Protocol

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from silero_vad import VADIterator, load_silero_vad
except ImportError:  # pragma: no cover - exercised via runtime fallback
    VADIterator = None  # type: ignore[assignment]
    load_silero_vad = None  # type: ignore[assignment]


class SupportsStreamingVAD(Protocol):
    def __call__(self, x):
        ...

    def reset_states(self) -> None:
        ...


_model = None


def silero_vad_available() -> bool:
    return load_silero_vad is not None and VADIterator is not None


def _load_model():
    global _model
    if _model is None:
        if not silero_vad_available():
            raise RuntimeError("silero-vad is not installed")
        _model = load_silero_vad()
    return _model


def _create_iterator() -> SupportsStreamingVAD:
    model = _load_model()
    return VADIterator(
        model,
        threshold=settings.vad_threshold,
        sampling_rate=settings.audio_sample_rate_whisper,
        min_silence_duration_ms=settings.vad_min_silence_ms,
        speech_pad_ms=settings.vad_speech_pad_ms,
    )


class StreamingSpeechDetector:
    """Detects speech boundaries and yields finalized voiced spans."""

    frame_samples = 512

    def __init__(self, vad_iterator: Optional[SupportsStreamingVAD] = None) -> None:
        self._vad = vad_iterator or _create_iterator()
        self._remainder = np.empty(0, dtype=np.float32)
        self._speech_frames: list[np.ndarray] = []
        self._preroll_frames: deque[np.ndarray] = deque(maxlen=self._preroll_frame_limit())
        self._speech_active = False

    def process_chunk(self, samples: np.ndarray) -> list[np.ndarray]:
        if samples.size == 0:
            return []

        finalized: list[np.ndarray] = []
        chunk = np.asarray(samples, dtype=np.float32)
        if self._remainder.size:
            chunk = np.concatenate((self._remainder, chunk))

        frame_count, remainder = divmod(len(chunk), self.frame_samples)
        if remainder:
            self._remainder = chunk[-remainder:].copy()
            chunk = chunk[: frame_count * self.frame_samples]
        else:
            self._remainder = np.empty(0, dtype=np.float32)

        for offset in range(0, len(chunk), self.frame_samples):
            frame = chunk[offset : offset + self.frame_samples].copy()
            if not self._speech_active:
                self._preroll_frames.append(frame)

            event = self._vad(frame)

            if not self._speech_active and event and "start" in event:
                self._speech_active = True
                self._speech_frames = [cached.copy() for cached in self._preroll_frames]
                self._preroll_frames.clear()
                continue

            if self._speech_active:
                self._speech_frames.append(frame)
                if event and "end" in event:
                    finalized.append(self._finalize_current())

        return finalized

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def flush(self) -> Optional[np.ndarray]:
        if self._remainder.size:
            padded = np.pad(
                self._remainder,
                (0, self.frame_samples - len(self._remainder)),
                mode="constant",
            ).astype(np.float32)
            self._remainder = np.empty(0, dtype=np.float32)
            finalized = self.process_chunk(padded)
            if finalized:
                return finalized[-1]

        if not self._speech_active or not self._speech_frames:
            return None
        return self._finalize_current()

    def reset(self) -> None:
        self._vad.reset_states()
        self._remainder = np.empty(0, dtype=np.float32)
        self._speech_frames.clear()
        self._preroll_frames.clear()
        self._speech_active = False

    def _finalize_current(self) -> np.ndarray:
        speech = np.concatenate(self._speech_frames).astype(np.float32)
        self._speech_frames.clear()
        self._speech_active = False
        self._preroll_frames.clear()
        return speech

    @staticmethod
    def _preroll_frame_limit() -> int:
        frame_ms = (StreamingSpeechDetector.frame_samples * 1000) / settings.audio_sample_rate_whisper
        return max(1, int(math.ceil(settings.vad_preroll_ms / frame_ms)))