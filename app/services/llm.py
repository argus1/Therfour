"""LLM service backed by Ollama for harm-reduction response generation.

Ollama exposes a local OpenAI-compatible API.  All inference runs on-device;
no data is sent to external servers.  See https://ollama.com for setup.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from app.core.config import settings
from app.models.schemas import EmptyOutputError, HTTPError, InvalidResponseError

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
            if not (200 <= resp.status_code < 300):
                raise HTTPError(resp.status_code, resp.text)

            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            if not content:
                raise EmptyOutputError("LLM returned empty response")

            return content
    except httpx.RequestError as e:
        raise InvalidResponseError(f"Failed to communicate with Ollama: {e}") from e
    except (KeyError, TypeError) as e:
        raise InvalidResponseError(f"Invalid response format from Ollama: {e}") from e


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
