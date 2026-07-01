"""Tests for LLM backend adapters."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.llm_backends import (
    OllamaBackend,
    LMStudioBackend,
    OpenAIBackend,
    get_backend,
    llm_service_label,
    llm_timeout_seconds,
)


class TestOllamaBackend:
    """Test Ollama backend adapter."""

    def test_endpoint_includes_api_chat(self) -> None:
        """Endpoint should use /api/chat path."""
        backend = OllamaBackend()
        endpoint = backend.endpoint()
        assert "/api/chat" in endpoint

    def test_payload_structure(self) -> None:
        """Payload should contain model, messages, and stream."""
        backend = OllamaBackend()
        payload = backend.payload(
            [{"role": "user", "content": "hello"}],
            stream=False
        )
        assert "model" in payload
        assert "messages" in payload
        assert "stream" in payload
        assert payload["stream"] is False

    def test_extract_text_from_response(self) -> None:
        """extract_text should handle Ollama response format."""
        backend = OllamaBackend()
        data = {
            "message": {"content": "Response text"}
        }
        text = backend.extract_text(data)
        assert text == "Response text"

    def test_extract_text_handles_missing_content(self) -> None:
        """extract_text should handle missing content gracefully."""
        backend = OllamaBackend()
        data = {"message": {}}
        text = backend.extract_text(data)
        assert text == ""

    def test_iter_stream_tokens_parses_json_line(self) -> None:
        """iter_stream_tokens should parse streaming JSON lines."""
        backend = OllamaBackend()
        line = json.dumps({
            "message": {"content": "token"},
            "done": False
        })
        tokens, done = backend.iter_stream_tokens(line)
        assert tokens == ["token"]
        assert done is False

    def test_iter_stream_tokens_detects_done(self) -> None:
        """iter_stream_tokens should detect stream completion."""
        backend = OllamaBackend()
        line = json.dumps({
            "message": {"content": ""},
            "done": True
        })
        tokens, done = backend.iter_stream_tokens(line)
        assert done is True

    def test_iter_stream_tokens_handles_empty_line(self) -> None:
        """iter_stream_tokens should handle empty lines."""
        backend = OllamaBackend()
        tokens, done = backend.iter_stream_tokens("")
        assert tokens == []
        assert done is False

    def test_model_name_returns_configured_model(self, monkeypatch) -> None:
        """model_name should return configured Ollama model."""
        monkeypatch.setenv("OLLAMA_MODEL", "test-model:latest")
        from app.core.config import Settings
        test_settings = Settings(ollama_model="test-model:latest")
        with patch("app.services.llm_backends.settings", test_settings):
            backend = OllamaBackend()
            assert backend.model_name() == "test-model:latest"

    def test_headers_returns_empty_dict(self) -> None:
        """Ollama backend should not require special headers."""
        backend = OllamaBackend()
        headers = backend.headers()
        assert headers == {}


class TestLMStudioBackend:
    """Test LM Studio backend adapter."""

    def test_endpoint_uses_chat_completions(self) -> None:
        """Endpoint should use /chat/completions path."""
        backend = LMStudioBackend()
        endpoint = backend.endpoint()
        assert "/chat/completions" in endpoint

    def test_payload_structure(self) -> None:
        """Payload should contain model, messages, stream, and temperature."""
        backend = LMStudioBackend()
        payload = backend.payload(
            [{"role": "user", "content": "hello"}],
            stream=True
        )
        assert "model" in payload
        assert "messages" in payload
        assert "stream" in payload
        assert "temperature" in payload
        assert payload["stream"] is True

    def test_extract_text_from_openai_format(self) -> None:
        """extract_text should handle OpenAI-compatible response format."""
        backend = LMStudioBackend()
        data = {
            "choices": [
                {"message": {"content": "Response text"}}
            ]
        }
        text = backend.extract_text(data)
        assert text == "Response text"

    def test_extract_text_handles_empty_choices(self) -> None:
        """extract_text should handle empty choices array."""
        backend = LMStudioBackend()
        data = {"choices": []}
        text = backend.extract_text(data)
        assert text == ""

    def test_extract_text_handles_missing_content(self) -> None:
        """extract_text should handle missing message content."""
        backend = LMStudioBackend()
        data = {"choices": [{"message": {}}]}
        text = backend.extract_text(data)
        assert text == ""

    def test_iter_stream_tokens_parses_sse_format(self) -> None:
        """iter_stream_tokens should parse SSE data format."""
        backend = LMStudioBackend()
        line = 'data: ' + json.dumps({
            "choices": [{"delta": {"content": "hello"}}]
        })
        tokens, done = backend.iter_stream_tokens(line)
        assert tokens == ["hello"]
        assert done is False

    def test_iter_stream_tokens_detects_done_sentinel(self) -> None:
        """iter_stream_tokens should detect [DONE] sentinel."""
        backend = LMStudioBackend()
        tokens, done = backend.iter_stream_tokens('data: [DONE]')
        assert tokens == []
        assert done is True

    def test_iter_stream_tokens_ignores_non_data_lines(self) -> None:
        """iter_stream_tokens should ignore non-data lines."""
        backend = LMStudioBackend()
        tokens, done = backend.iter_stream_tokens("")
        assert tokens == []
        assert done is False

    def test_iter_stream_tokens_handles_empty_delta(self) -> None:
        """iter_stream_tokens should handle messages without content."""
        backend = LMStudioBackend()
        line = 'data: ' + json.dumps({
            "choices": [{"delta": {}}]
        })
        tokens, done = backend.iter_stream_tokens(line)
        assert tokens == []
        assert done is False

    def test_model_name_returns_configured_model(self, monkeypatch) -> None:
        """model_name should return configured LM Studio model."""
        monkeypatch.setenv("LMSTUDIO_MODEL", "custom-model")
        from app.core.config import Settings
        test_settings = Settings(lmstudio_model="custom-model")
        with patch("app.services.llm_backends.settings", test_settings):
            backend = LMStudioBackend()
            assert backend.model_name() == "custom-model"

    def test_headers_returns_empty_dict(self) -> None:
        """LM Studio backend should not require special headers."""
        backend = LMStudioBackend()
        headers = backend.headers()
        assert headers == {}


class TestOpenAIBackend:
    """Test OpenAI backend adapter."""

    def test_endpoint_uses_chat_completions(self) -> None:
        """Endpoint should use /chat/completions path."""
        backend = OpenAIBackend()
        endpoint = backend.endpoint()
        assert "/chat/completions" in endpoint
        assert "https://api.openai.com" in endpoint or "localhost" in endpoint

    def test_payload_structure(self) -> None:
        """Payload should contain model, messages, stream, and temperature."""
        backend = OpenAIBackend()
        payload = backend.payload(
            [{"role": "user", "content": "hello"}],
            stream=False
        )
        assert "model" in payload
        assert "messages" in payload
        assert "stream" in payload
        assert "temperature" in payload
        assert payload["stream"] is False

    def test_extract_text_from_openai_format(self) -> None:
        """extract_text should handle OpenAI response format."""
        backend = OpenAIBackend()
        data = {
            "choices": [
                {"message": {"content": "Response text"}}
            ]
        }
        text = backend.extract_text(data)
        assert text == "Response text"

    def test_extract_text_handles_empty_choices(self) -> None:
        """extract_text should handle empty choices array."""
        backend = OpenAIBackend()
        data = {"choices": []}
        text = backend.extract_text(data)
        assert text == ""

    def test_extract_text_handles_missing_content(self) -> None:
        """extract_text should handle missing message content."""
        backend = OpenAIBackend()
        data = {"choices": [{"message": {}}]}
        text = backend.extract_text(data)
        assert text == ""

    def test_headers_includes_bearer_token(self, monkeypatch) -> None:
        """headers() should include Authorization Bearer token."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        from app.core.config import Settings
        test_settings = Settings(openai_api_key="sk-test-key-12345")
        with patch("app.services.llm_backends.settings", test_settings):
            backend = OpenAIBackend()
            headers = backend.headers()
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer sk-test-key-12345"

    def test_headers_raises_when_api_key_empty(self, monkeypatch) -> None:
        """headers() should raise ValueError when API key is empty."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        from app.core.config import Settings
        test_settings = Settings(openai_api_key="")
        with patch("app.services.llm_backends.settings", test_settings):
            backend = OpenAIBackend()
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                backend.headers()

    def test_headers_raises_when_api_key_only_whitespace(self, monkeypatch) -> None:
        """headers() should raise ValueError when API key is only whitespace."""
        monkeypatch.setenv("OPENAI_API_KEY", "   \t  ")
        from app.core.config import Settings
        test_settings = Settings(openai_api_key="   \t  ")
        with patch("app.services.llm_backends.settings", test_settings):
            backend = OpenAIBackend()
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                backend.headers()

    def test_iter_stream_tokens_parses_sse_format(self) -> None:
        """iter_stream_tokens should parse SSE data format."""
        backend = OpenAIBackend()
        line = 'data: ' + json.dumps({
            "choices": [{"delta": {"content": "hello"}}]
        })
        tokens, done = backend.iter_stream_tokens(line)
        assert tokens == ["hello"]
        assert done is False

    def test_iter_stream_tokens_detects_done_sentinel(self) -> None:
        """iter_stream_tokens should detect [DONE] sentinel."""
        backend = OpenAIBackend()
        tokens, done = backend.iter_stream_tokens('data: [DONE]')
        assert tokens == []
        assert done is True

    def test_iter_stream_tokens_ignores_non_data_lines(self) -> None:
        """iter_stream_tokens should ignore non-data lines."""
        backend = OpenAIBackend()
        tokens, done = backend.iter_stream_tokens("")
        assert tokens == []
        assert done is False

    def test_iter_stream_tokens_handles_empty_delta(self) -> None:
        """iter_stream_tokens should handle messages without content."""
        backend = OpenAIBackend()
        line = 'data: ' + json.dumps({
            "choices": [{"delta": {}}]
        })
        tokens, done = backend.iter_stream_tokens(line)
        assert tokens == []
        assert done is False

    def test_model_name_returns_configured_model(self, monkeypatch) -> None:
        """model_name should return configured OpenAI model."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        from app.core.config import Settings
        test_settings = Settings(openai_model="gpt-4o")
        with patch("app.services.llm_backends.settings", test_settings):
            backend = OpenAIBackend()
            assert backend.model_name() == "gpt-4o"


class TestGetBackend:
    """Test backend selection logic."""

    def test_get_backend_returns_ollama_by_default(self) -> None:
        """get_backend should return OllamaBackend for unknown providers."""
        backend = get_backend("unknown")
        assert isinstance(backend, OllamaBackend)

    def test_get_backend_returns_ollama_for_ollama(self) -> None:
        """get_backend should return OllamaBackend for 'ollama'."""
        backend = get_backend("ollama")
        assert isinstance(backend, OllamaBackend)

    def test_get_backend_returns_lmstudio(self) -> None:
        """get_backend should return LMStudioBackend for 'lmstudio'."""
        backend = get_backend("lmstudio")
        assert isinstance(backend, LMStudioBackend)

    def test_get_backend_returns_openai(self) -> None:
        """get_backend should return OpenAIBackend for 'openai'."""
        backend = get_backend("openai")
        assert isinstance(backend, OpenAIBackend)

    def test_get_backend_normalizes_case(self) -> None:
        """get_backend should normalize provider case."""
        backend_lower = get_backend("OPENAI")
        backend_mixed = get_backend("OpEnAi")
        assert isinstance(backend_lower, OpenAIBackend)
        assert isinstance(backend_mixed, OpenAIBackend)

    def test_get_backend_handles_whitespace(self) -> None:
        """get_backend should strip whitespace from provider."""
        backend = get_backend("  openai  ")
        assert isinstance(backend, OpenAIBackend)


class TestLLMServiceLabel:
    """Test service label generation."""

    def test_llm_service_label_ollama(self, monkeypatch) -> None:
        """llm_service_label should format Ollama label."""
        monkeypatch.setenv("OLLAMA_MODEL", "test-model")
        from app.core.config import Settings
        test_settings = Settings(ollama_model="test-model")
        with patch("app.services.llm_backends.settings", test_settings):
            label = llm_service_label("ollama")
            assert label.startswith("ollama/")
            assert "test-model" in label

    def test_llm_service_label_lmstudio(self, monkeypatch) -> None:
        """llm_service_label should format LM Studio label."""
        monkeypatch.setenv("LMSTUDIO_MODEL", "lm-model")
        from app.core.config import Settings
        test_settings = Settings(lmstudio_model="lm-model")
        with patch("app.services.llm_backends.settings", test_settings):
            label = llm_service_label("lmstudio")
            assert label.startswith("lmstudio/")
            assert "lm-model" in label

    def test_llm_service_label_openai(self, monkeypatch) -> None:
        """llm_service_label should format OpenAI label."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        from app.core.config import Settings
        test_settings = Settings(openai_model="gpt-4o-mini")
        with patch("app.services.llm_backends.settings", test_settings):
            label = llm_service_label("openai")
            assert label.startswith("openai/")
            assert "gpt-4o-mini" in label


class TestLLMTimeoutSeconds:
    """Test timeout calculation logic."""

    def test_llm_timeout_seconds_returns_configured_timeout(self, monkeypatch) -> None:
        """llm_timeout_seconds should return configured timeout when > 0."""
        monkeypatch.setenv("LLM_TIMEOUT", "45.0")
        from app.core.config import Settings
        test_settings = Settings(llm_timeout=45.0, ollama_timeout=60.0)
        with patch("app.services.llm_backends.settings", test_settings):
            timeout = llm_timeout_seconds()
            assert timeout == 45.0

    def test_llm_timeout_seconds_falls_back_to_ollama_timeout(self, monkeypatch) -> None:
        """llm_timeout_seconds should fallback to ollama_timeout when <= 0."""
        monkeypatch.setenv("LLM_TIMEOUT", "0")
        from app.core.config import Settings
        test_settings = Settings(llm_timeout=0.0, ollama_timeout=120.0)
        with patch("app.services.llm_backends.settings", test_settings):
            timeout = llm_timeout_seconds()
            assert timeout == 120.0

    def test_llm_timeout_seconds_handles_negative_timeout(self, monkeypatch) -> None:
        """llm_timeout_seconds should fallback when timeout is negative."""
        monkeypatch.setenv("LLM_TIMEOUT", "-1")
        from app.core.config import Settings
        test_settings = Settings(llm_timeout=-1.0, ollama_timeout=90.0)
        with patch("app.services.llm_backends.settings", test_settings):
            timeout = llm_timeout_seconds()
            assert timeout == 90.0
