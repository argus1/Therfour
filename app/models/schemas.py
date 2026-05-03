"""Shared request/response schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CanonicalTurnMessageType(str, Enum):
    TURN_REQUEST = "turn.request"
    TURN_RESPONSE = "turn.response"
    TURN_ERROR = "turn.error"
    TURN_EVENT = "turn.event"


class CanonicalTurnState(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"
    DROPPED = "dropped"


class CanonicalTurnErrorClass(str, Enum):
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    PROVIDER = "provider"
    INTERNAL = "internal"
    UPSTREAM_CANCEL = "upstream_cancel"


class CanonicalTurnSource(str, Enum):
    TELEPHONY = "telephony"
    API = "api"
    SWIFT_BACKEND = "swift-backend"
    WORKER = "worker"


class CanonicalTurnEnvelope(BaseModel):
    """Transport envelope for canonical turn payloads."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    message_type: CanonicalTurnMessageType
    trace_id: str
    turn_id: str
    session_id: str
    created_at: datetime
    source: CanonicalTurnSource = CanonicalTurnSource.TELEPHONY
    idempotency_key: Optional[str] = None


class CanonicalTurnInputAudio(BaseModel):
    """Audio metadata for inbound turn input."""

    model_config = ConfigDict(frozen=True)

    codec: str
    sample_rate_hz: int = Field(ge=1)
    duration_ms: Optional[int] = Field(default=None, ge=0)


class CanonicalTurnInputText(BaseModel):
    """Text input fields for a turn."""

    model_config = ConfigDict(frozen=True)

    text: str


class CanonicalTurnInputDTMF(BaseModel):
    """DTMF input fields for a turn."""

    model_config = ConfigDict(frozen=True)

    digits: str


class CanonicalTurnInputLanguage(BaseModel):
    """Language metadata for turn input."""

    model_config = ConfigDict(frozen=True)

    code: str
    language_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CanonicalTurnInput(BaseModel):
    """Inbound turn modalities."""

    model_config = ConfigDict(frozen=True)

    audio: Optional[CanonicalTurnInputAudio] = None
    text: Optional[CanonicalTurnInputText] = None
    dtmf: Optional[CanonicalTurnInputDTMF] = None
    language: Optional[CanonicalTurnInputLanguage] = None


class CanonicalTurnProcessingVAD(BaseModel):
    """VAD processing fields for a turn."""

    model_config = ConfigDict(frozen=True)

    backend_name: str = "silero"
    vad_voiced_duration_ms: Optional[int] = Field(default=None, ge=0)


class CanonicalTurnProcessingSTT(BaseModel):
    """STT processing fields for a turn."""

    model_config = ConfigDict(frozen=True)

    backend_name: str
    transcript_text: str = ""
    transcript_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    language_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    transcript_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fallback_used: bool = False
    failure_reason: str = ""

    @model_validator(mode="after")
    def _validate_terminal_content(self) -> "CanonicalTurnProcessingSTT":
        if not self.transcript_text and not self.failure_reason:
            raise ValueError(
                "processing.stt requires transcript_text or failure_reason"
            )
        return self


class CanonicalTurnProcessingRAG(BaseModel):
    """RAG processing fields for a turn."""

    model_config = ConfigDict(frozen=True)

    enabled: bool
    retrieval_relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CanonicalTurnProcessingLLM(BaseModel):
    """LLM processing fields for a turn."""

    model_config = ConfigDict(frozen=True)

    backend_name: str
    strategy: Optional[str] = None


class CanonicalTurnProcessingTTS(BaseModel):
    """TTS processing fields for a turn."""

    model_config = ConfigDict(frozen=True)

    backend_name: str
    voice_id: str
    output_format: str
    output_sample_rate_hz: Optional[int] = Field(default=None, ge=1)
    fallback_used: bool = False
    synthesis_latency_ms: Optional[int] = Field(default=None, ge=0)
    audio_bytes: Optional[int] = Field(default=None, ge=0)
    audio_duration_ms: Optional[int] = Field(default=None, ge=0)
    failure_reason: str = ""


class CanonicalTurnProcessing(BaseModel):
    """Normalized processing metadata for the turn pipeline."""

    model_config = ConfigDict(frozen=True)

    vad: Optional[CanonicalTurnProcessingVAD] = None
    stt: Optional[CanonicalTurnProcessingSTT] = None
    rag: Optional[CanonicalTurnProcessingRAG] = None
    llm: Optional[CanonicalTurnProcessingLLM] = None
    tts: Optional[CanonicalTurnProcessingTTS] = None


class CanonicalTurnOutputAssistantAudio(BaseModel):
    """Assistant audio output metadata."""

    model_config = ConfigDict(frozen=True)

    format: str
    sample_rate_hz: int = Field(ge=1)
    duration_ms: Optional[int] = Field(default=None, ge=0)


class CanonicalTurnOutput(BaseModel):
    """Normalized assistant output fields."""

    model_config = ConfigDict(frozen=True)

    assistant_text: str = ""
    assistant_audio: Optional[CanonicalTurnOutputAssistantAudio] = None
    grounding: dict[str, Any] = Field(default_factory=dict)
    safety: dict[str, Any] = Field(default_factory=dict)


class CanonicalTurnStatus(BaseModel):
    """Lifecycle status for a canonical turn."""

    model_config = ConfigDict(frozen=True)

    state: CanonicalTurnState
    failure_reason: str = ""
    retryable: bool = False
    error_class: Optional[CanonicalTurnErrorClass] = None

    @model_validator(mode="after")
    def _validate_failure_fields(self) -> "CanonicalTurnStatus":
        if self.state == CanonicalTurnState.FAILED and not self.failure_reason:
            raise ValueError("failed state requires failure_reason")
        return self


class CanonicalTurnPayload(BaseModel):
    """Semantic payload for one canonical turn."""

    model_config = ConfigDict(frozen=True)

    input: CanonicalTurnInput = Field(default_factory=CanonicalTurnInput)
    processing: CanonicalTurnProcessing = Field(default_factory=CanonicalTurnProcessing)
    output: CanonicalTurnOutput = Field(default_factory=CanonicalTurnOutput)
    status: CanonicalTurnStatus
    parent_turn_id: Optional[str] = None


class CanonicalTurn(BaseModel):
    """Top-level canonical turn document."""

    model_config = ConfigDict(frozen=True)

    envelope: CanonicalTurnEnvelope
    payload: CanonicalTurnPayload


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
    call_control_protocol: Literal["twilio", "asterisk_ari"] = "twilio"
    call_control_payload: str
    # Backward-compatible alias for existing Twilio harness clients.
    twiml: str = ""
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