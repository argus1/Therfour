"""Shared request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VoiceServiceError(RuntimeError):
    """Base class for voice-pipeline service failures."""


class UnsupportedError(VoiceServiceError):
    """Raised when a backend or feature is unavailable/unsupported."""


class EmptyOutputError(VoiceServiceError):
    """Raised when a voice backend returns no usable output."""


class DecodingError(VoiceServiceError):
    """Raised when an audio payload cannot be decoded."""


class InvalidResponseError(VoiceServiceError):
    """Raised when a backend response is malformed."""


class NoSpeechDetectedError(VoiceServiceError):
    """Raised when STT cannot detect speech in audio."""


class HTTPError(VoiceServiceError):
    """Raised when an upstream HTTP backend fails."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class ChatMessage(BaseModel):
    """Normalized chat turn used by runtime and tests."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class TranscriptionResult(BaseModel):
    """Output from speech-to-text transcription."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    language: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    language_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    transcript_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    backend_name: str = "faster-whisper"
    fallback_used: bool = False
    failure_reason: str = ""
    audio_duration_s: float = Field(default=0.0, ge=0.0)  # Audio duration in seconds


class TurnProcessingResult(BaseModel):
    """Bundled result for one completed voice turn."""

    model_config = ConfigDict(frozen=True)

    transcription: TranscriptionResult
    reply: str
    audio_payload: bytes


class STTBackendCapabilities(BaseModel):
    """Capability contract for the active STT backend."""

    model_config = ConfigDict(frozen=True)

    supported_languages: set[str] = Field(default_factory=set)
    supports_live_streaming: bool = False
    notes: str = ""

    @classmethod
    def fallback(cls) -> "STTBackendCapabilities":
        return cls(supported_languages=set(), supports_live_streaming=False, notes="")


class TTSBackendCapabilities(BaseModel):
    """Capability contract for the active TTS backend."""

    model_config = ConfigDict(frozen=True)

    supported_languages: set[str] = Field(default_factory=set)
    supports_voice_hints: bool = True
    notes: str = ""

    @classmethod
    def fallback(cls) -> "TTSBackendCapabilities":
        return cls(supported_languages=set(), supports_voice_hints=True, notes="")


class LLMBackendCapabilities(BaseModel):
    """Capability contract for the active LLM backend."""

    model_config = ConfigDict(frozen=True)

    supported_languages: set[str] = Field(default_factory=set)
    supports_json_response_format: bool = True
    notes: str = ""

    @classmethod
    def fallback(cls) -> "LLMBackendCapabilities":
        return cls(supported_languages=set(), supports_json_response_format=True, notes="")


class HealthResponse(BaseModel):
    """Health-check payload describing current backend services."""

    model_config = ConfigDict(frozen=True)

    status: str
    version: str
    services: dict[str, str]


def make_health_response(app_version: str, whisper_model: str, ollama_model: str) -> HealthResponse:
    """Construct the API health payload from app/runtime backend settings."""
    return HealthResponse(
        status="ok",
        version=app_version,
        services={
            "stt": f"faster-whisper/{whisper_model}",
            "tts": "piper",
            "llm": f"ollama/{ollama_model}",
        },
    )