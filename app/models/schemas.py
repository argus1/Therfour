"""Shared request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceServiceError(RuntimeError):
    """Base class for voice-pipeline service failures."""


class UnsupportedError(VoiceServiceError):
    """Raised when a backend or feature is unavailable/unsupported."""


class EmptyOutputError(VoiceServiceError):
    """Raised when a voice backend returns no usable output."""


class TranscriptionResult(BaseModel):
    """Output from speech-to-text transcription."""

    text: str = ""
    language: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    language_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    transcript_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    backend_name: str = "faster-whisper"
    fallback_used: bool = False
    failure_reason: str = ""


class HealthResponse(BaseModel):
    """Health-check payload describing current backend services."""

    status: str
    version: str
    services: dict[str, str]