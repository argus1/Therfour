"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


async def _noop_ensure_model_loaded() -> None:
    """Test-only stub: skips Ollama bootstrap so tests run without a live server."""


@pytest.fixture(scope="session")
def client() -> TestClient:
    # Patch the name as bound inside app.main so the lifespan coroutine skips
    # the real Ollama bootstrap, which requires a running Ollama server.
    with patch("app.main.ensure_model_loaded", _noop_ensure_model_loaded):
        with TestClient(app) as c:
            yield c
