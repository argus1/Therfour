"""Tests for the HTTP API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


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
