"""Tests for the LLM service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import llm as llm_service


@pytest.mark.asyncio
async def test_generate_returns_string() -> None:
    """generate() should return the assistant reply as a plain string."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "Please stay safe and call 911 if needed."}
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("app.services.llm.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(llm_service.settings, "rag_enabled", False):
            result = await llm_service.generate(
                [{"role": "user", "content": "What is naloxone?"}]
            )

    assert isinstance(result, str)
    assert "safe" in result.lower()


def test_system_prompt_contains_harm_reduction_keywords() -> None:
    """The system prompt must reference harm reduction and safety."""
    prompt = llm_service.HARM_REDUCTION_SYSTEM_PROMPT.lower()
    assert "harm reduction" in prompt
    assert "safety" in prompt or "safe" in prompt
    assert "non-judgmental" in prompt or "non-judgement" in prompt or "compassionate" in prompt


def test_system_prompt_mentions_multilingual() -> None:
    """The system prompt must instruct the model to be multilingual."""
    assert "multilingual" in llm_service.HARM_REDUCTION_SYSTEM_PROMPT.lower()


@pytest.mark.asyncio
async def test_generate_includes_rag_context_when_enabled() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"message": {"content": "Use naloxone and call emergency services."}}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("app.services.llm.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(llm_service.settings, "rag_enabled", True):
            with patch("app.services.llm.rag.retrieve") as mock_retrieve:
                with patch("app.services.llm.rag.build_context_block") as mock_block:
                    mock_retrieve.return_value = MagicMock(contexts=[MagicMock()])
                    mock_block.return_value = "Retrieved context (use only when directly relevant):\n\n[1 | source: guide | score: 0.91]\nUse naloxone quickly."

                    await llm_service.generate([
                        {"role": "user", "content": "What should I do in an overdose?"}
                    ])

    sent_payload = mock_client.post.await_args.kwargs["json"]
    system_prompt = sent_payload["messages"][0]["content"]
    assert "RAG grounding policy" in system_prompt
    assert "Retrieved context" in system_prompt


def test_latest_user_message_returns_most_recent_user_content() -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
    ]
    assert llm_service._latest_user_message(messages) == "second"
