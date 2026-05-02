"""Shared request/response schemas."""

from __future__ import annotations

from typing import Any, Literal, Optional

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


class TransferHarnessRequest(BaseModel):
    """Input payload for transfer integration harness endpoint."""

    model_config = ConfigDict(frozen=True)

    target_kind: Literal["number", "sip"] = "number"
    target: str
    forwarded_by: str = ""
    topic: str = ""
    priority: str = ""
    announcement: str = ""
    execute_live_update: bool = False
    call_sid: Optional[str] = None


class TransferHarnessResponse(BaseModel):
    """Response payload for transfer integration harness endpoint."""

    model_config = ConfigDict(frozen=True)

    target_kind: Literal["number", "sip"]
    target: str
    twiml: str
    executed_live_update: bool
    call_sid: str = ""


class CallSimulationReportRequest(BaseModel):
    """Input payload for simulation report endpoint."""

    model_config = ConfigDict(frozen=True)

    tier: Literal["tier_a", "tier_b"] = "tier_a"
    max_turns: int = Field(default=8, ge=1, le=64)
    frustration_hangup_threshold: int = Field(default=6, ge=1, le=64)
    force_low_confidence_every_n_turns: int = Field(default=0, ge=0, le=32)
    use_live_therfour_llm: bool = False
    opening_message: str = (
        "Hi, this is Terris. I am here with you. What name would you like me to use for you today?"
    )
    caller_provider: Literal["ollama", "lmstudio"] = "ollama"
    caller_base_url: str = ""
    caller_model_name_override: str = ""
    caller_timeout_s: float = Field(default=30.0, ge=1.0, le=300.0)
    output_filename: str = ""


class CallSimulationReportResponse(BaseModel):
    """Response payload for simulation report endpoint."""

    model_config = ConfigDict(frozen=True)

    report_path: str
    written: bool
    report: dict[str, Any]


class CallSimulationReportSummary(BaseModel):
    """Metadata (and optional body) for one saved simulation report."""

    model_config = ConfigDict(frozen=True)

    filename: str
    report_path: str
    size_bytes: int
    modified_at: str
    generated_at: str = ""
    report: Optional[dict[str, Any]] = None


class RecentCallSimulationReportsResponse(BaseModel):
    """Response payload for listing recent simulation reports."""

    model_config = ConfigDict(frozen=True)

    count: int
    reports: list[CallSimulationReportSummary]


class CallSimulationReportFileResponse(BaseModel):
    """Response payload for a single saved simulation report file."""

    model_config = ConfigDict(frozen=True)

    filename: str
    report_path: str
    size_bytes: int
    modified_at: str
    generated_at: str = ""
    report: dict[str, Any]


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