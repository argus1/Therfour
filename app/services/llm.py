"""LLM service backed by Ollama for harm-reduction response generation.

Ollama exposes a local OpenAI-compatible API.  All inference runs on-device;
no data is sent to external servers.  See https://ollama.com for setup.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

import httpx

from app.core.config import settings
from app.services import rag

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

RAG_GROUNDING_RULES = (
    "\n\nRAG grounding policy:\n"
    "- Use retrieved context only when it is directly relevant to the caller question.\n"
    "- Ignore off-topic or duplicate retrieved snippets.\n"
    "- If context is missing or insufficient, answer briefly without mentioning retrieval internals.\n"
    "- Do not expose source labels, scores, or retrieval scaffolding in final caller responses."
)


async def generate(messages: list[dict]) -> str:
    """Send *messages* to Ollama and return the assistant reply as a string."""
    system_prompt = await _build_system_prompt(messages)
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


async def generate_stream(messages: list[dict]) -> AsyncIterator[str]:
    """Stream response tokens from Ollama one chunk at a time."""
    system_prompt = await _build_system_prompt(messages)
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
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


async def _build_system_prompt(messages: list[dict]) -> str:
    prompt = HARM_REDUCTION_SYSTEM_PROMPT
    if not settings.rag_enabled:
        return prompt

    prompt = f"{prompt}{RAG_GROUNDING_RULES}"
    query = _latest_user_message(messages)
    if not query:
        return prompt

    result = await asyncio.to_thread(rag.retrieve, query)
    context_block = rag.build_context_block(result.contexts)
    if not context_block:
        return prompt
    return f"{prompt}\n\n{context_block}"


def _latest_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""
