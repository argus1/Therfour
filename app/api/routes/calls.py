"""Twilio call-control webhooks and Media Stream WebSocket handler."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import Response

from app.core.config import settings
from app.models.schemas import TransferHarnessRequest, TransferHarnessResponse
from app.services.telephony import CallSession, build_transfer_twiml, twilio_transfer_call_update

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


@router.post("/calls/transfer/harness", response_model=TransferHarnessResponse)
async def transfer_harness(payload: TransferHarnessRequest) -> TransferHarnessResponse:
    """Integration harness for transfer behavior.

    Default mode is dry-run and only returns generated TwiML.
    Set ``execute_live_update=true`` with a valid ``call_sid`` to perform a
    real Twilio call update.
    """
    if not settings.transfer_harness_enabled and not settings.debug:
        raise HTTPException(status_code=403, detail="Transfer harness is disabled")

    metadata = {
        "forwarded-by": payload.forwarded_by,
        "topic": payload.topic,
        "priority": payload.priority,
    }
    compact_metadata = {key: value for key, value in metadata.items() if value}

    try:
        twiml = build_transfer_twiml(
            payload.target_kind,
            payload.target,
            payload.announcement,
            metadata=compact_metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    executed_live_update = False

    if payload.execute_live_update:
        if not payload.call_sid:
            raise HTTPException(status_code=400, detail="call_sid is required for live update")
        try:
            await asyncio.to_thread(twilio_transfer_call_update, payload.call_sid, twiml)
        except Exception as exc:
            logger.exception("Transfer harness live update failed")
            raise HTTPException(status_code=502, detail=f"Twilio update failed: {exc}") from exc
        executed_live_update = True

    return TransferHarnessResponse(
        target_kind=payload.target_kind,
        target=payload.target,
        twiml=twiml,
        executed_live_update=executed_live_update,
        call_sid=payload.call_sid or "",
    )
