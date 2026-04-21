"""Utilities for structured observability events across service stages."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("app.observability")


def emit_stage_event(
    stage: str,
    status: str,
    latency_ms: float,
    failure_reason: str = "",
    **metadata: Any,
) -> None:
    """Emit a single structured log line for a service stage outcome."""
    fields: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "latency_ms": round(latency_ms, 2),
    }
    if failure_reason:
        fields["failure_reason"] = failure_reason
    for key, value in metadata.items():
        if value is not None:
            fields[key] = value

    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("observability %s", payload)
