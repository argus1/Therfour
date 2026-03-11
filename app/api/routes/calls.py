"""Twilio call-control webhooks and Media Stream WebSocket handler."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import Response

from app.core.config import settings
from app.services.telephony import CallSession

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/calls/inbound")
async def inbound_call(request: Request) -> Response:
    """Twilio calls this webhook when a new inbound call is received.

    Returns TwiML that greets the caller and connects a Media Stream so that
    audio can be processed in real time by the voice-agent pipeline.
    """
    host = request.headers.get("host") or settings.public_host
    scheme = "wss" if request.url.scheme == "https" else "ws"
    stream_url = f"{scheme}://{host}/calls/stream"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Say>You have reached the harm reduction helpline. "
        "How can I help you today?</Say>"
        f'<Connect><Stream url="{stream_url}" /></Connect>'
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@router.websocket("/calls/stream")
async def media_stream(websocket: WebSocket) -> None:
    """Twilio Media Stream WebSocket endpoint.

    Each connected call gets its own :class:`~app.services.telephony.CallSession`
    which orchestrates the full STT → LLM → TTS pipeline.
    """
    await websocket.accept()
    logger.info("New media-stream connection accepted")
    session = CallSession(websocket)
    try:
        await session.handle()
    except Exception:
        logger.exception("Unhandled error in media stream session")
    finally:
        logger.info("Media-stream connection closed")
