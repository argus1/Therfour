"""LLM service backed by Ollama for harm-reduction response generation.

Ollama exposes a local OpenAI-compatible API.  All inference runs on-device;
no data is sent to external servers.  See https://ollama.com for setup.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

import httpx

from app.core.config import settings
from app.services.observability import emit_stage_event

logger = logging.getLogger(__name__)

HARM_REDUCTION_SYSTEM_PROMPT = (
    "You are a compassionate harm reduction specialist working on a telephone "
    "helpline. Your role is to provide accurate, non-judgmental, evidence-based "
    "information to people who use substances or are affected by substance use.\n\n"
    "Guidelines:\n"
    "- Always prioritise caller safety above everything else.\n"
    "- Provide accurate information about safer use practices, overdose prevention, "
    "and naloxone.\n"
    "- Never shame or judge callers; use supportive, person-first language.\n"
    "- Share information about available resources (treatment, testing services, "
    "shelters).\n"
    "- If someone is in immediate danger, encourage them to call emergency services "
    "(911 in North America).\n"
    "- Respect caller autonomy while providing honest safety information.\n"
    "- Keep responses concise (2–3 sentences) – this is a real-time phone call.\n"
    "- You are multilingual; always respond in the language the caller uses."
)


async def generate(messages: list[dict]) -> str:
    """Send *messages* to Ollama and return the assistant reply as a string."""
    start = time.perf_counter()
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": HARM_REDUCTION_SYSTEM_PROMPT},
            *messages,
        ],
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        emit_stage_event(
            stage="rag",
            status="failure",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            failure_reason="generation_error",
            message_count=len(messages),
            backend_name="ollama",
            model=settings.ollama_model,
        )
        raise

    content = data.get("message", {}).get("content", "")
    failure_reason = "" if content.strip() else "empty_response"
    emit_stage_event(
        stage="rag",
        status="success" if not failure_reason else "dropped",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        failure_reason=failure_reason,
        message_count=len(messages),
        response_chars=len(content),
        backend_name="ollama",
        model=settings.ollama_model,
    )
    return content


async def generate_stream(messages: list[dict]) -> AsyncIterator[str]:
    """Stream response tokens from Ollama one chunk at a time."""
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": HARM_REDUCTION_SYSTEM_PROMPT},
            *messages,
        ],
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if token := data.get("message", {}).get("content"):
                    yield token
                if data.get("done"):
                    break
