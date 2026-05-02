"""Tests for the HTTP API endpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

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
from app.core.config import settings
from app.models.Call_SImulation_Agent import SimulationReport, SimulationTier


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


def test_inbound_call_uses_randomized_opener(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.calls.call_flow_phrases.random_opener",
        lambda: "Custom opener test line.",
    )
    resp = client.post("/calls/inbound")
    assert resp.status_code == 200
    assert "Custom opener test line." in resp.text


def test_transfer_harness_disabled_returns_403(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_harness_enabled", False)
    monkeypatch.setattr(settings, "debug", False)

    resp = client.post(
        "/calls/transfer/harness",
        json={"target_kind": "number", "target": "988"},
    )

    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"].lower()


def test_transfer_harness_dry_run_returns_twiml(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_harness_enabled", True)

    resp = client.post(
        "/calls/transfer/harness",
        json={
            "target_kind": "number",
            "target": "988",
            "announcement": "Connecting you to 988 now.",
            "execute_live_update": False,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["target"] == "988"
    assert body["target_kind"] == "number"
    assert body["executed_live_update"] is False
    assert "<Dial>988</Dial>" in body["twiml"]


def test_transfer_harness_live_update_calls_twilio(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_harness_enabled", True)

    called: list[tuple[str, str]] = []

    def _fake_update(call_sid: str, twiml: str) -> None:
        called.append((call_sid, twiml))

    monkeypatch.setattr("app.api.routes.calls.twilio_transfer_call_update", _fake_update)

    resp = client.post(
        "/calls/transfer/harness",
        json={
            "target_kind": "number",
            "target": "911",
            "announcement": "I am connecting you to emergency services now.",
            "execute_live_update": True,
            "call_sid": "CA123",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["executed_live_update"] is True
    assert body["call_sid"] == "CA123"
    assert called and called[0][0] == "CA123"
    assert "<Dial>911</Dial>" in called[0][1]


def test_transfer_harness_sip_adds_custom_headers(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_harness_enabled", True)
    monkeypatch.setattr(settings, "transfer_allow_custom_targets", True)
    monkeypatch.setattr(settings, "transfer_allowed_sip_domains", "example.com")

    resp = client.post(
        "/calls/transfer/harness",
        json={
            "target_kind": "sip",
            "target": "sip:agent@example.com",
            "forwarded_by": "Terris",
            "topic": "overdose",
            "priority": "high",
        },
    )

    assert resp.status_code == 200
    twiml = resp.json()["twiml"]
    assert "<Sip>sip:agent@example.com?" in twiml
    assert "x-forwarded-by=Terris" in twiml
    assert "x-topic=overdose" in twiml
    assert "x-priority=high" in twiml


def test_transfer_harness_number_metadata_strict_mode_rejected(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "transfer_harness_enabled", True)
    monkeypatch.setattr(settings, "transfer_metadata_mode", "strict")

    resp = client.post(
        "/calls/transfer/harness",
        json={
            "target_kind": "number",
            "target": "988",
            "forwarded_by": "Terris",
        },
    )

    assert resp.status_code == 400
    assert "strict mode" in resp.json()["detail"]


def test_simulation_report_harness_disabled_returns_403(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "simulation_harness_enabled", False)
    monkeypatch.setattr(settings, "debug", False)

    resp = client.post("/calls/simulation/report", json={})

    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"].lower()


def test_simulation_report_writes_json_file(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "simulation_harness_enabled", True)

    async def _fake_run(self):
        return SimulationReport(
            tier=SimulationTier.TIER_A_HEADLESS,
            completed_turns=2,
            frustration_score=1,
            hangup_triggered=False,
            hangup_reason="",
            transfer_target="",
            pleasant_ending_detected=True,
            opening_message="Hello",
            turns=[],
            notes="ok",
        )

    monkeypatch.setattr("app.api.routes.calls.CallSimulationAgent.run", _fake_run)

    filename = "api_sim_report_test.json"
    report_path = Path("app/models/Call_SImulation_Agent/reports") / filename
    if report_path.exists():
        report_path.unlink()

    try:
        resp = client.post(
            "/calls/simulation/report",
            json={
                "tier": "tier_a",
                "output_filename": filename,
                "use_live_therfour_llm": False,
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["written"] is True
        assert body["report"]["completed_turns"] == 2
        assert report_path.exists()
    finally:
        if report_path.exists():
            report_path.unlink()


def test_recent_simulation_reports_disabled_returns_403(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "simulation_harness_enabled", False)
    monkeypatch.setattr(settings, "debug", False)

    resp = client.get("/calls/simulation/reports/recent")

    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"].lower()


def test_recent_simulation_reports_returns_sorted_items(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "simulation_harness_enabled", True)

    reports_dir = Path("app/models/Call_SImulation_Agent/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    older = reports_dir / "recent_test_older.json"
    newer = reports_dir / "recent_test_newer.json"

    try:
        older.write_text(
            json.dumps({"generated_at": "2026-05-01T00:00:00Z", "report": {"id": "older"}}),
            encoding="utf-8",
        )
        newer.write_text(
            json.dumps({"generated_at": "2026-05-01T00:01:00Z", "report": {"id": "newer"}}),
            encoding="utf-8",
        )

        os.utime(older, (1000, 1000))
        os.utime(newer, (2000, 2000))

        resp = client.get("/calls/simulation/reports/recent?limit=2")

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert body["reports"][0]["filename"] == "recent_test_newer.json"
        assert body["reports"][1]["filename"] == "recent_test_older.json"
        assert body["reports"][0]["report"] is None
    finally:
        if older.exists():
            older.unlink()
        if newer.exists():
            newer.unlink()


def test_recent_simulation_reports_can_include_report_payload(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "simulation_harness_enabled", True)

    reports_dir = Path("app/models/Call_SImulation_Agent/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    sample = reports_dir / "recent_test_include.json"

    try:
        sample.write_text(
            json.dumps({"generated_at": "2026-05-01T00:02:00Z", "report": {"tier": "tier_a"}}),
            encoding="utf-8",
        )

        resp = client.get("/calls/simulation/reports/recent?limit=1&include_report=true")

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["reports"][0]["filename"] == "recent_test_include.json"
        assert body["reports"][0]["report"] == {"tier": "tier_a"}
    finally:
        if sample.exists():
            sample.unlink()


def test_get_simulation_report_file_disabled_returns_403(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "simulation_harness_enabled", False)
    monkeypatch.setattr(settings, "debug", False)

    resp = client.get("/calls/simulation/reports/some_report.json")

    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"].lower()


def test_get_simulation_report_file_returns_payload(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "simulation_harness_enabled", True)

    reports_dir = Path("app/models/Call_SImulation_Agent/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    sample = reports_dir / "single_fetch_test.json"

    try:
        sample.write_text(
            json.dumps({"generated_at": "2026-05-01T00:03:00Z", "report": {"turns": 3}}),
            encoding="utf-8",
        )

        resp = client.get("/calls/simulation/reports/single_fetch_test.json")

        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "single_fetch_test.json"
        assert body["report"] == {"turns": 3}
    finally:
        if sample.exists():
            sample.unlink()


def test_get_simulation_report_file_returns_404_when_missing(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "simulation_harness_enabled", True)

    resp = client.get("/calls/simulation/reports/does_not_exist.json")

    assert resp.status_code == 404


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
