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
