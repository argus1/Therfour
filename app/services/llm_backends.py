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


def get_backend(provider: str) -> LLMBackend:
    """Return a backend adapter for the configured provider."""
    if provider.strip().lower() == "lmstudio":
        return LMStudioBackend()
    return OllamaBackend()