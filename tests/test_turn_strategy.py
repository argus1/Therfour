from __future__ import annotations

from app.services.turn_strategy import TurnStrategy, classify_turn


def test_classify_turn_rapport_for_emotional_disclosure() -> None:
    decision = classify_turn("I feel overwhelmed and alone.", history=[])
    assert decision.strategy == TurnStrategy.RAPPORT_BUILDING


def test_classify_turn_info_gathering_for_ambiguous_safety() -> None:
    decision = classify_turn("I do not know how to keep myself safe tonight.", history=[])
    assert decision.strategy == TurnStrategy.INFO_GATHERING_NO_RAG


def test_classify_turn_rag_eligible_for_factual_request() -> None:
    decision = classify_turn("What is naloxone and how do I use it?", history=[])
    assert decision.strategy == TurnStrategy.TASK_OR_KNOWLEDGE_RAG_ELIGIBLE


def test_classify_turn_understanding_check_after_clause_dense_assistant_reply() -> None:
    history = [
        {
            "role": "assistant",
            "content": (
                "Here are steps: 1. Keep naloxone nearby. 2. Do not use alone. "
                "3. Test in small amounts. 4. Call 911 if breathing slows. "
                "Please review each step and ask questions if anything is unclear."
            ),
        }
    ]
    decision = classify_turn("Okay, I guess.", history=history)
    assert decision.strategy == TurnStrategy.UNDERSTANDING_CHECK_NO_RAG


def test_classify_turn_explanation_when_caller_signals_confusion() -> None:
    decision = classify_turn(
        "I do not understand what you meant. Can you explain that again?",
        history=[],
    )
    assert decision.strategy == TurnStrategy.EXPLANATION_RAG_OPTIONAL
