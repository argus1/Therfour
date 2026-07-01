"""LLM backend adapters used by the orchestration layer."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from app.core.config import settings


class LLMBackend(ABC):
    """Provider-agnostic interface for chat backends."""

    @abstractmethod
    def endpoint(self) -> str:
        """Return the full HTTP endpoint for chat requests."""

    @abstractmethod
    def payload(self, messages: list[dict[str, str]], *, stream: bool) -> dict:
        """Build provider-specific request payload."""

    @abstractmethod
    def extract_text(self, data: dict) -> str:
        """Extract final assistant text from a non-streaming response."""

    @abstractmethod
    def iter_stream_tokens(self, line: str) -> tuple[list[str], bool]:
        """Parse a streaming line into tokens and a done flag."""

    def headers(self) -> dict[str, str]:
        """Return provider-specific HTTP headers."""
        return {}

    @abstractmethod
    def model_name(self) -> str:
        """Return provider-configured model name for reporting."""


class OllamaBackend(LLMBackend):
    """Ollama adapter using /api/chat."""

    def endpoint(self) -> str:
        return f"{settings.ollama_base_url}/api/chat"

    def payload(self, messages: list[dict[str, str]], *, stream: bool) -> dict:
        return {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": settings.llm_temperature},
        }

    def extract_text(self, data: dict) -> str:
        return str(data.get("message", {}).get("content", ""))

    def iter_stream_tokens(self, line: str) -> tuple[list[str], bool]:
        if not line:
            return [], False
        data = json.loads(line)
        tokens: list[str] = []
        token = data.get("message", {}).get("content")
        if token:
            tokens.append(str(token))
        return tokens, bool(data.get("done"))

    def model_name(self) -> str:
        return settings.ollama_model


class LMStudioBackend(LLMBackend):
    """LM Studio adapter using OpenAI-compatible /chat/completions."""

    def endpoint(self) -> str:
        return f"{settings.lmstudio_base_url.rstrip('/')}/chat/completions"

    def payload(self, messages: list[dict[str, str]], *, stream: bool) -> dict:
        return {
            "model": settings.lmstudio_model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "stream": stream,
        }

    def extract_text(self, data: dict) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", ""))

    def iter_stream_tokens(self, line: str) -> tuple[list[str], bool]:
        if not line or not line.startswith("data:"):
            return [], False

        payload_line = line[len("data:") :].strip()
        if payload_line == "[DONE]":
            return [], True

        data = json.loads(payload_line)
        delta = data.get("choices", [{}])[0].get("delta", {})
        token = delta.get("content")
        if token:
            return [str(token)], False
        return [], False

    def model_name(self) -> str:
        return settings.lmstudio_model


class OpenAIBackend(LLMBackend):
    """OpenAI adapter using /chat/completions."""

    def endpoint(self) -> str:
        return f"{settings.openai_base_url.rstrip('/')}/chat/completions"

    def payload(self, messages: list[dict[str, str]], *, stream: bool) -> dict:
        return {
            "model": settings.openai_model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "stream": stream,
        }

    def headers(self) -> dict[str, str]:
        api_key = settings.openai_api_key.strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set when llm_provider=openai")
        return {"Authorization": f"Bearer {api_key}"}

    def extract_text(self, data: dict) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", ""))

    def iter_stream_tokens(self, line: str) -> tuple[list[str], bool]:
        if not line or not line.startswith("data:"):
            return [], False

        payload_line = line[len("data:") :].strip()
        if payload_line == "[DONE]":
            return [], True

        data = json.loads(payload_line)
        delta = data.get("choices", [{}])[0].get("delta", {})
        token = delta.get("content")
        if token:
            return [str(token)], False
        return [], False

    def model_name(self) -> str:
        return settings.openai_model


def get_backend(provider: str) -> LLMBackend:
    """Return a backend adapter for the configured provider."""
    normalized = provider.strip().lower()
    if normalized == "lmstudio":
        return LMStudioBackend()
    if normalized == "openai":
        return OpenAIBackend()
    return OllamaBackend()


def llm_service_label(provider: str) -> str:
    """Return stable '<provider>/<model>' service label for health reporting."""
    backend = get_backend(provider)
    normalized = provider.strip().lower() or "ollama"
    return f"{normalized}/{backend.model_name()}"


def llm_timeout_seconds() -> float:
    """Return the effective provider request timeout in seconds."""
    configured = float(settings.llm_timeout)
    if configured > 0:
        return configured
    # Backward-compat fallback for existing deployments.
    return float(settings.ollama_timeout)