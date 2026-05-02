"""Call opener/terminator phrase catalog helpers."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_OPENERS = [
    "You have reached the harm reduction helpline. How can I help you today?",
    "Hi, this is Terris with the harm reduction line. What would be most helpful right now?",
    "Thanks for calling. I'm here with you - what do you need support with today?",
    "Welcome to the harm reduction helpline. Tell me what is going on and we'll take it step by step.",
    "Hello, I'm Terris. How can I support you right now?",
]

_DEFAULT_TERMINATORS = [
    "Thank you for calling today. I'm glad we could talk, and you can call back anytime.",
    "I appreciate you reaching out. Take care, and call us again if you need more support.",
    "I'm glad we connected today. If anything changes, please call back and we can continue.",
    "Thank you for speaking with me. You're not alone, and this line is here whenever you need it.",
    "Thanks for the call. Wishing you safety and support - reach back out anytime.",
]


def _load_phrase_catalog() -> tuple[list[str], list[str]]:
    if not settings.call_flow_phrases_enabled:
        return _DEFAULT_OPENERS, _DEFAULT_TERMINATORS

    path = Path(settings.call_flow_phrases_config_path)
    if not path.exists():
        logger.warning("Call flow phrases config missing: %s", path)
        return _DEFAULT_OPENERS, _DEFAULT_TERMINATORS

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load call flow phrases config: %s", path)
        return _DEFAULT_OPENERS, _DEFAULT_TERMINATORS

    openers = [str(item).strip() for item in payload.get("openers", []) if str(item).strip()]
    terminators = [
        str(item).strip() for item in payload.get("terminators", []) if str(item).strip()
    ]
    return (openers or _DEFAULT_OPENERS), (terminators or _DEFAULT_TERMINATORS)


def random_opener() -> str:
    openers, _ = _load_phrase_catalog()
    return random.choice(openers)


def random_terminator() -> str:
    _, terminators = _load_phrase_catalog()
    return random.choice(terminators)
