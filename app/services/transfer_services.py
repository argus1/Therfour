"""Transfer services catalog utilities.

Loads and validates custom-transfer services from JSON configuration so Terris
can only offer configured service destinations for non-emergency transfers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransferService:
    service_id: str
    name: str
    description: str
    target_kind: str
    target: str
    keywords: tuple[str, ...]


def load_services() -> list[TransferService]:
    """Load transfer services from configured JSON file."""
    if not settings.transfer_services_enabled:
        return []

    path = Path(settings.transfer_services_config_path)
    if not path.exists():
        logger.warning("Transfer services config not found: %s", path)
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load transfer services config: %s", path)
        return []

    services_raw = raw.get("services", []) if isinstance(raw, dict) else []
    services: list[TransferService] = []
    for item in services_raw:
        if not isinstance(item, dict):
            continue
        target_kind = str(item.get("target_kind", "")).strip().lower()
        if target_kind not in {"number", "sip"}:
            continue
        target = str(item.get("target", "")).strip()
        if not target:
            continue

        service = TransferService(
            service_id=str(item.get("id", "")).strip() or target,
            name=str(item.get("name", "")).strip() or target,
            description=str(item.get("description", "")).strip(),
            target_kind=target_kind,
            target=target,
            keywords=tuple(
                str(k).strip() for k in (item.get("keywords") or []) if str(k).strip()
            ),
        )
        services.append(service)

    return services


def is_configured_target(target_kind: str, target: str) -> bool:
    """Return True if target exists in transfer-services catalog."""
    normalized_kind = target_kind.strip().lower()
    normalized_target = target.strip()
    for service in load_services():
        if service.target_kind == normalized_kind and service.target == normalized_target:
            return True
    return False


def build_prompt_block() -> str:
    """Build model-facing instruction block for configured service transfers."""
    services = load_services()
    if not services:
        return (
            "Transfer services catalog:\n"
            "- No custom transfer services are currently configured.\n"
            "- Do not emit non-emergency custom TRANSFER:number or TRANSFER:sip directives."
        )

    lines = [
        "Transfer services catalog (non-emergency custom transfers):",
        "- Use custom transfers only when the service clearly benefits the caller's stated goal.",
        "- Only use the exact configured targets below for custom transfers.",
        "- Prefer a direct offer before transfer (e.g., voicemail, appointment line, support queue).",
    ]
    for svc in services:
        keywords = ", ".join(svc.keywords) if svc.keywords else "none"
        lines.append(
            f"- {svc.name}: TRANSFER:{svc.target_kind}:{svc.target}; "
            f"description={svc.description or 'n/a'}; keywords={keywords}"
        )

    return "\n".join(lines)
