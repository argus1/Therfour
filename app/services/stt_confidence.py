"""Low-confidence STT handling with confirmation flow."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.core.config import settings
from app.models.schemas import TranscriptionResult
from app.services.llm_backends import get_backend

logger = logging.getLogger(__name__)

PARAPHRASE_SYSTEM_PROMPT = (
    "You are a concise speech-to-text quality assistant. "
    "Your only job is to paraphrase a low-confidence speech transcription "
    "in a natural, conversational way that a user might recognize and confirm. "
    "Keep the paraphrase 2-3 sentences maximum. "
    "Preserve the core meaning and intent. "
    "Do not add information not in the original transcript."
)


@dataclass(frozen=True)
class ConfirmationPrompt:
    """A confirmation prompt for a low-confidence transcription."""

    original_transcript: str
    confirmation_text: str
    should_paraphrase: bool
    prompt: str


class LowConfidenceHandler:
    """Handles low-confidence STT results with user confirmation flow."""

    @staticmethod
    def is_low_confidence(result: TranscriptionResult) -> bool:
        """Check if a transcription result is below confidence threshold."""
        if not result.text or result.failure_reason:
            return False
        return result.language_confidence < settings.stt_low_confidence_threshold

    @staticmethod
    async def generate_confirmation_prompt(
        result: TranscriptionResult,
    ) -> ConfirmationPrompt:
        """Generate a confirmation prompt for low-confidence transcript.
        
        For short audio (<5s), uses verbatim transcript.
        For longer audio (≥5s), generates a paraphrase for better confirmation.
        """
        should_paraphrase = result.audio_duration_s >= 5.0
        
        if should_paraphrase:
            confirmation_text = await _paraphrase_transcript(result.text)
        else:
            confirmation_text = result.text
        
        prompt = (
            f"I'm not sure if I heard you correctly. I think you said, {confirmation_text}. "
            "Am I correct? Tell me yes or no."
        )
        
        return ConfirmationPrompt(
            original_transcript=result.text,
            confirmation_text=confirmation_text,
            should_paraphrase=should_paraphrase,
            prompt=prompt,
        )

    @staticmethod
    def get_retry_prompt(retry_count: int) -> str:
        """Get the appropriate prompt for a retry."""
        if retry_count < settings.stt_max_retries:
            return "I see. I'm sorry, can you please say it again slowly and clearly?"
        else:
            return (
                "I'm having a lot of difficulty understanding what you're trying to say. "
                "I hope you don't mind if we talk about something else. Tell me what else is on your mind."
            )

    @staticmethod
    def should_change_topic(retry_count: int) -> bool:
        """Check if we should allow the user to change topics after max retries."""
        return retry_count >= settings.stt_max_retries


async def _paraphrase_transcript(text: str) -> str:
    """Use LLM to paraphrase a low-confidence transcript for confirmation.
    
    Returns the paraphrase, or the original text if paraphrasing fails.
    """
    try:
        backend = get_backend(settings.llm_provider)
        messages = [
            {
                "role": "system",
                "content": PARAPHRASE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Please paraphrase this for user confirmation: {text}",
            },
        ]
        payload = backend.payload(messages, stream=False)

        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            resp = await client.post(
                backend.endpoint(),
                json=payload,
            )
            resp.raise_for_status()
            paraphrase = backend.extract_text(resp.json())
            logger.info("Generated paraphrase for confirmation: %s", paraphrase)
            return paraphrase
    except Exception:
        logger.exception("Paraphrase generation failed; using original transcript")
        return text
