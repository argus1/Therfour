"""Turn-strategy routing for rapport, info gathering, and RAG-eligible turns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class TurnStrategy(str, Enum):
    RAPPORT_BUILDING = "rapport_building"
    INFO_GATHERING_NO_RAG = "info_gathering_no_rag"
    UNDERSTANDING_CHECK_NO_RAG = "understanding_check_no_rag"
    EXPLANATION_RAG_OPTIONAL = "explanation_rag_optional"
    TASK_OR_KNOWLEDGE_RAG_ELIGIBLE = "task_or_knowledge_rag_eligible"


@dataclass(frozen=True)
class TurnStrategyDecision:
    strategy: TurnStrategy
    reason: str


def classify_turn(user_text: str, history: list[dict]) -> TurnStrategyDecision:
    text = (user_text or "").strip().lower()
    if not text:
        return TurnStrategyDecision(
            strategy=TurnStrategy.INFO_GATHERING_NO_RAG,
            reason="empty_or_minimal_text",
        )

    if _shows_understanding_gap(text):
        return TurnStrategyDecision(
            strategy=TurnStrategy.EXPLANATION_RAG_OPTIONAL,
            reason="caller_signaled_understanding_gap",
        )

    if _should_run_understanding_check(text, history):
        return TurnStrategyDecision(
            strategy=TurnStrategy.UNDERSTANDING_CHECK_NO_RAG,
            reason="followup_after_complex_prior_assistant_turn",
        )

    distress_markers = {
        "i feel",
        "i'm feeling",
        "im feeling",
        "overwhelmed",
        "anxious",
        "scared",
        "afraid",
        "alone",
        "hopeless",
        "stressed",
        "depressed",
        "hurt",
        "it feels like too much",
        "i do not know what to do",
        "i don't know what to do",
    }
    ambiguous_safety_markers = {
        "not safe",
        "keep myself safe",
        "might hurt myself",
        "i could hurt myself",
        "i might use",
        "relapse",
        "tonight",
        "right now",
    }

    has_distress = any(marker in text for marker in distress_markers)
    has_ambiguous_safety = any(marker in text for marker in ambiguous_safety_markers)

    # If the caller sounds distressed and safety context is unclear, prioritize
    # concise clarifying questions over retrieval-heavy informational responses.
    if has_ambiguous_safety:
        return TurnStrategyDecision(
            strategy=TurnStrategy.INFO_GATHERING_NO_RAG,
            reason="ambiguous_safety_context",
        )

    factual_markers = {
        "what is",
        "how do i",
        "how to",
        "where can i",
        "explain",
        "difference between",
        "dose",
        "naloxone",
        "overdose prevention",
        "safer use",
        "needle",
        "syringe",
        "fentanyl",
        "withdrawal",
        "infection",
        "testing",
        "treatment",
    }
    if any(marker in text for marker in factual_markers):
        return TurnStrategyDecision(
            strategy=TurnStrategy.TASK_OR_KNOWLEDGE_RAG_ELIGIBLE,
            reason="factual_or_procedural_request",
        )

    if has_distress:
        return TurnStrategyDecision(
            strategy=TurnStrategy.RAPPORT_BUILDING,
            reason="emotional_disclosure",
        )

    # If the previous assistant turn ended with a question, this turn is often
    # additional context from the caller and should continue no-RAG probing.
    for item in reversed(history[-4:]):
        if str(item.get("role", "")).lower() != "assistant":
            continue
        content = str(item.get("content", "")).strip()
        if content.endswith("?"):
            return TurnStrategyDecision(
                strategy=TurnStrategy.INFO_GATHERING_NO_RAG,
                reason="followup_to_assistant_question",
            )
        break

    return TurnStrategyDecision(
        strategy=TurnStrategy.RAPPORT_BUILDING,
        reason="default_supportive_mode",
    )


def rag_allowed_for_strategy(strategy: TurnStrategy) -> bool:
    return strategy in {
        TurnStrategy.TASK_OR_KNOWLEDGE_RAG_ELIGIBLE,
        TurnStrategy.EXPLANATION_RAG_OPTIONAL,
    }


def _shows_understanding_gap(text: str) -> bool:
    markers = {
        "i do not understand",
        "i don't understand",
        "not understanding",
        "what does that mean",
        "what do you mean",
        "can you explain",
        "explain that",
        "say that again",
        "can you repeat",
        "too much information",
        "too many steps",
        "i am confused",
        "i'm confused",
    }
    return any(marker in text for marker in markers)


def _should_run_understanding_check(text: str, history: list[dict]) -> bool:
    if not history:
        return False

    prior_assistant = ""
    for item in reversed(history[-6:]):
        if str(item.get("role", "")).lower() == "assistant":
            prior_assistant = str(item.get("content", "")).strip()
            break

    if not prior_assistant:
        return False

    if not _looks_clause_dense_or_numbered(prior_assistant):
        return False

    # Trigger understanding checks when the caller gives brief acknowledgment
    # after a dense assistant response, which often masks comprehension gaps.
    light_ack_markers = {
        "ok",
        "okay",
        "i guess",
        "maybe",
        "not sure",
        "i think so",
        "uh huh",
        "alright",
        "got it",
    }
    return any(marker in text for marker in light_ack_markers)


def _looks_clause_dense_or_numbered(text: str) -> bool:
    lowered = text.lower()
    numbered = bool(re.search(r"(?:^|\s)(?:1\.|2\.|3\.|1\)|2\)|3\)|step\s+1)", lowered))
    long_message = len(lowered) >= 320
    clause_like = lowered.count(",") >= 4 or lowered.count(";") >= 2
    return numbered or (long_message and clause_like)
