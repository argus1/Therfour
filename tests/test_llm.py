"""Tests for the LLM service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import llm as llm_service
from app.services.turn_strategy import TurnStrategy


def _system_content_blob(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    )


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


@pytest.mark.asyncio
async def test_generate_applies_post_generation_sanitizer(monkeypatch) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "raw assistant output"}
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    sanitizer_calls: list[tuple[str, bool]] = []

    def _fake_sanitizer(text: str, *, stream_mode: bool) -> str:
        sanitizer_calls.append((text, stream_mode))
        return "sanitized assistant output"

    monkeypatch.setattr(llm_service, "_sanitize_post_generation_output", _fake_sanitizer)

    with patch("app.services.llm.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(llm_service.settings, "rag_enabled", False):
            result = await llm_service.generate(
                [{"role": "user", "content": "hello"}]
            )

    assert result == "sanitized assistant output"
    assert sanitizer_calls == [("raw assistant output", False)]


def test_system_prompt_contains_harm_reduction_keywords() -> None:
    """The system prompt must reference harm reduction and safety."""
    prompt = llm_service.HARM_REDUCTION_SYSTEM_PROMPT.lower()
    assert "harm reduction" in prompt
    assert "safety" in prompt or "safe" in prompt
    assert "non-judgmental" in prompt or "non-judgement" in prompt or "compassionate" in prompt


def test_system_prompt_mentions_multilingual() -> None:
    """The system prompt must instruct the model to be multilingual."""
    assert "multilingual" in llm_service.HARM_REDUCTION_SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_scope_boundary_and_crisis_routing() -> None:
    """The system prompt should define scope boundaries and crisis routing details."""
    prompt = llm_service.HARM_REDUCTION_SYSTEM_PROMPT.lower()
    assert "outside harm reduction support" in prompt
    assert "988" in llm_service.HARM_REDUCTION_SYSTEM_PROMPT
    assert "do not ask for transfer permission" in prompt
    assert "are you okay with this" in prompt


def test_safety_guardrails_include_anti_fabrication_rule() -> None:
    """Safety guardrails should explicitly forbid fabricated claims."""
    guardrails = llm_service.SAFETY_GUARDRAILS.lower()
    assert "never invent facts" in guardrails
    assert "uncertain" in guardrails


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
    system_prompt = _system_content_blob(sent_payload["messages"])
    assert "RAG grounding policy" in system_prompt
    assert "Retrieved context" in system_prompt


def test_latest_user_message_returns_most_recent_user_content() -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
    ]
    assert llm_service._latest_user_message(messages) == "second"


def test_normalize_history_filters_and_limits(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_max_history_messages", 2)
    messages = [
        {"role": "system", "content": "ignore me"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "user", "content": "   "},
    ]

    assert llm_service._normalize_history(messages) == [
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]


@pytest.mark.asyncio
async def test_construct_turn_includes_guardrails_and_retrieval_metadata(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "rag_enabled", True)
    monkeypatch.setattr(llm_service.settings, "llm_max_history_messages", 4)

    fake_result = MagicMock(
        strategy_used="hierarchical",
        selected_category="opioids",
        candidate_count=5,
        filtered_count=2,
        contexts=[MagicMock()],
    )

    with patch("app.services.llm.rag.retrieve", return_value=fake_result):
        with patch("app.services.llm.rag.build_context_block", return_value="Retrieved context block"):
            turn = await llm_service._construct_turn(
                [{"role": "user", "content": "How to reduce overdose risk?"}]
            )

    system_prompt = _system_content_blob(turn.messages)
    assert "Safety guardrails" in system_prompt
    assert "Deterministic turn policy" in system_prompt
    assert "Retrieval metadata" in system_prompt
    assert "selected_category: opioids" in system_prompt
    assert turn.rag_used is True


@pytest.mark.asyncio
async def test_construct_turn_includes_transfer_services_catalog_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "rag_enabled", False)
    monkeypatch.setattr(llm_service.settings, "transfer_services_enabled", True)
    with patch("app.services.llm.transfer_services.build_prompt_block") as mock_block:
        mock_block.return_value = "Transfer services catalog (non-emergency custom transfers):\n- sample"
        turn = await llm_service._construct_turn(
            [{"role": "user", "content": "Can you transfer me to counseling appointments?"}]
        )

    system_prompt = _system_content_blob(turn.messages)
    assert "Transfer services catalog" in system_prompt


@pytest.mark.asyncio
async def test_generate_uses_lmstudio_endpoint_and_response_shape(monkeypatch) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Let's focus on your immediate safety and next supportive step."
                }
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr(llm_service.settings, "llm_provider", "lmstudio")
    monkeypatch.setattr(llm_service.settings, "rag_enabled", False)

    with patch("app.services.llm.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await llm_service.generate([
            {"role": "user", "content": "I feel overwhelmed"}
        ])

    called_url = mock_client.post.await_args.args[0]
    payload = mock_client.post.await_args.kwargs["json"]

    assert called_url.endswith("/chat/completions")
    assert payload["model"] == llm_service.settings.lmstudio_model
    assert payload["stream"] is False
    assert isinstance(result, str)
    assert "safety" in result.lower()


@pytest.mark.asyncio
async def test_construct_turn_suppresses_rag_for_rapport_strategy(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "rag_enabled", True)

    with patch("app.services.llm.rag.retrieve") as mock_retrieve:
        turn = await llm_service._construct_turn(
            [{"role": "user", "content": "I feel overwhelmed and alone."}],
            strategy=TurnStrategy.RAPPORT_BUILDING.value,
            rag_allowed=False,
        )

    assert turn.rag_used is False
    assert "Turn strategy: rapport_building" in _system_content_blob(turn.messages)
    mock_retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_construct_turn_suppresses_rag_for_info_gathering_strategy(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "rag_enabled", True)

    with patch("app.services.llm.rag.retrieve") as mock_retrieve:
        turn = await llm_service._construct_turn(
            [{"role": "user", "content": "I do not feel safe tonight."}],
            strategy=TurnStrategy.INFO_GATHERING_NO_RAG.value,
            rag_allowed=False,
        )

    assert turn.rag_used is False
    assert "Turn strategy: info_gathering_no_rag" in _system_content_blob(turn.messages)
    mock_retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_construct_turn_suppresses_rag_for_understanding_check_strategy(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "rag_enabled", True)

    with patch("app.services.llm.rag.retrieve") as mock_retrieve:
        turn = await llm_service._construct_turn(
            [{"role": "user", "content": "Okay, I guess."}],
            strategy=TurnStrategy.UNDERSTANDING_CHECK_NO_RAG.value,
            rag_allowed=False,
        )

    assert turn.rag_used is False
    assert "Turn strategy: understanding_check_no_rag" in _system_content_blob(turn.messages)
    mock_retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_construct_turn_allows_rag_for_explanation_strategy(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "rag_enabled", True)

    fake_result = MagicMock(
        strategy_used="hierarchical",
        selected_category="general",
        candidate_count=2,
        filtered_count=1,
        contexts=[MagicMock()],
    )

    with patch("app.services.llm.rag.retrieve", return_value=fake_result) as mock_retrieve:
        with patch("app.services.llm.rag.build_context_block", return_value="Retrieved context block"):
            turn = await llm_service._construct_turn(
                [{"role": "user", "content": "I do not understand, can you explain?"}],
                strategy=TurnStrategy.EXPLANATION_RAG_OPTIONAL.value,
                rag_allowed=True,
            )

    assert turn.rag_used is True
    assert "Turn strategy: explanation_rag_optional" in _system_content_blob(turn.messages)
    mock_retrieve.assert_called_once()
