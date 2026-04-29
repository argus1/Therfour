"""Tests for the HTTP API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.models.schemas import (
    ChatMessage,
    HealthResponse,
    LLMBackendCapabilities,
    STTBackendCapabilities,
    TTSBackendCapabilities,
    TranscriptionResult,
    TurnProcessingResult,
)


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "services" in body


def test_health_services_names(client: TestClient) -> None:
    resp = client.get("/health")
    services = resp.json()["services"]
    assert "stt" in services
    assert "tts" in services
    assert "llm" in services


def test_inbound_call_returns_xml(client: TestClient) -> None:
    resp = client.post("/calls/inbound")
    assert resp.status_code == 200
    assert "application/xml" in resp.headers["content-type"]


def test_inbound_call_twiml_structure(client: TestClient) -> None:
    resp = client.post("/calls/inbound")
    body = resp.text
    assert "<Response>" in body
    assert "<Stream" in body
    assert "/calls/stream" in body


# Contract-focused tests for data models

def test_transcription_result_model() -> None:
    """Test TranscriptionResult model contract."""
    result = TranscriptionResult(text="hello world", language="en", confidence=0.95)

    assert result.text == "hello world"
    assert result.language == "en"
    assert result.confidence == 0.95

    # Test immutability
    with pytest.raises(ValidationError):
        result.text = "modified"  # type: ignore


def test_chat_message_model() -> None:
    """Test ChatMessage model contract."""
    message = ChatMessage(role="user", content="Hello!")

    assert message.role == "user"
    assert message.content == "Hello!"

    # Test immutability
    with pytest.raises(ValidationError):
        message.role = "assistant"  # type: ignore


def test_health_response_model() -> None:
    """Test HealthResponse model contract."""
    response = HealthResponse(
        status="ok",
        version="1.0.0",
        services={"stt": "whisper", "tts": "piper", "llm": "ollama"}
    )

    assert response.status == "ok"
    assert response.version == "1.0.0"
    assert response.services["stt"] == "whisper"

    # Test immutability
    with pytest.raises(ValidationError):
        response.status = "error"  # type: ignore


def test_turn_processing_result_model() -> None:
    """Test TurnProcessingResult model contract."""
    transcription = TranscriptionResult(text="hello", language="en", confidence=0.9)
    result = TurnProcessingResult(
        transcription=transcription,
        reply="Hi there!",
        audio_payload=b"audio_data"
    )

    assert result.transcription == transcription
    assert result.reply == "Hi there!"
    assert result.audio_payload == b"audio_data"

    # Test immutability
    with pytest.raises(ValidationError):
        result.reply = "modified"  # type: ignore


def test_stt_backend_capabilities_model() -> None:
    """Test STTBackendCapabilities model contract."""
    caps = STTBackendCapabilities(
        supported_languages={"en", "es"},
        supports_live_streaming=True,
        notes="Test capabilities"
    )

    assert "en" in caps.supported_languages
    assert caps.supports_live_streaming is True
    assert caps.notes == "Test capabilities"

    # Test fallback
    fallback = STTBackendCapabilities.fallback()
    assert len(fallback.supported_languages) == 0
    assert fallback.supports_live_streaming is False


def test_tts_backend_capabilities_model() -> None:
    """Test TTSBackendCapabilities model contract."""
    caps = TTSBackendCapabilities(
        supported_languages={"en", "fr"},
        supports_voice_hints=False,
        notes="Test TTS capabilities"
    )

    assert "fr" in caps.supported_languages
    assert caps.supports_voice_hints is False
    assert caps.notes == "Test TTS capabilities"

    # Test fallback
    fallback = TTSBackendCapabilities.fallback()
    assert len(fallback.supported_languages) == 0
    assert fallback.supports_voice_hints is True


def test_llm_backend_capabilities_model() -> None:
    """Test LLMBackendCapabilities model contract."""
    caps = LLMBackendCapabilities(
        supported_languages={"en", "de"},
        supports_json_response_format=True,
        notes="Test LLM capabilities"
    )

    assert "de" in caps.supported_languages
    assert caps.supports_json_response_format is True
    assert caps.notes == "Test LLM capabilities"

    # Test fallback
    fallback = LLMBackendCapabilities.fallback()
    assert len(fallback.supported_languages) == 0
    assert fallback.supports_json_response_format is True


def test_voice_service_error_hierarchy() -> None:
    """Test that VoiceServiceError subclasses work correctly."""
    from app.models.schemas import (
        DecodingError,
        EmptyOutputError,
        HTTPError,
        InvalidResponseError,
        NoSpeechDetectedError,
        UnsupportedError,
        VoiceServiceError,
    )

    # Test that all are subclasses of VoiceServiceError
    assert issubclass(DecodingError, VoiceServiceError)
    assert issubclass(EmptyOutputError, VoiceServiceError)
    assert issubclass(HTTPError, VoiceServiceError)
    assert issubclass(InvalidResponseError, VoiceServiceError)
    assert issubclass(NoSpeechDetectedError, VoiceServiceError)
    assert issubclass(UnsupportedError, VoiceServiceError)

    # Test HTTPError specifics
    error = HTTPError(404, "Not found")
    assert error.status_code == 404
    assert "Not found" in str(error)


def test_make_health_response_utility() -> None:
    """Test the make_health_response utility function."""
    from app.models.schemas import make_health_response

    response = make_health_response(
        app_version="1.2.3",
        whisper_model="medium",
        ollama_model="llama3.2:3b"
    )

    assert response.status == "ok"
    assert response.version == "1.2.3"
    assert response.services["stt"] == "faster-whisper/medium"
    assert response.services["tts"] == "piper"
    assert response.services["llm"] == "ollama/llama3.2:3b"
