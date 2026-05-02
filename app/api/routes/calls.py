"""Twilio call-control webhooks and Media Stream WebSocket handler."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket
from fastapi.responses import Response

from app.core.config import settings
from app.models.Call_SImulation_Agent import (
    CallSimulationAgent,
    CallerModelConfig,
    SimulationConfig,
    SimulationTier,
)
from app.models.schemas import (
    CallSimulationReportFileResponse,
    CallSimulationReportRequest,
    CallSimulationReportResponse,
    CallSimulationReportSummary,
    RecentCallSimulationReportsResponse,
    TransferHarnessRequest,
    TransferHarnessResponse,
)
from app.services import call_flow_phrases
from app.services.telephony import (
    CallSession,
    build_transfer_twiml,
    get_custom_transfer_post_call_reopen_mode,
    get_transfer_post_call_reopen_mode,
    twilio_transfer_call_update,
)

logger = logging.getLogger(__name__)

router = APIRouter()
_SIMULATION_REPORTS_DIR = Path("app/models/Call_SImulation_Agent/reports")
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _normalize_for_json(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {k: _normalize_for_json(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_for_json(v) for v in value]
    return value


def _resolve_report_filename(raw_name: str) -> str:
    if raw_name:
        name = raw_name.strip()
        if not _SAFE_FILENAME.fullmatch(name):
            raise HTTPException(status_code=400, detail="output_filename contains invalid characters")
        if not name.endswith(".json"):
            name = f"{name}.json"
        return name

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"call_simulation_report_{stamp}.json"


def _validate_report_lookup_filename(filename: str) -> str:
    name = filename.strip()
    if not _SAFE_FILENAME.fullmatch(name):
        raise HTTPException(status_code=400, detail="filename contains invalid characters")
    if not name.endswith(".json"):
        raise HTTPException(status_code=400, detail="filename must end with .json")
    return name


def _iso_utc_from_mtime(mtime_seconds: float) -> str:
    return datetime.fromtimestamp(mtime_seconds, tz=timezone.utc).isoformat()


@router.post("/calls/inbound")
async def inbound_call(request: Request) -> Response:
    """Twilio calls this webhook when a new inbound call is received.

    Returns TwiML that greets the caller and connects a Media Stream so that
    audio can be processed in real time by the voice-agent pipeline.
    """
    host = request.headers.get("host") or settings.public_host
    scheme = "wss" if request.url.scheme == "https" else "ws"
    stream_url = f"{scheme}://{host}/calls/stream"

    opener = call_flow_phrases.random_opener()
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Say>{opener}</Say>"
        f'<Connect><Stream url="{stream_url}" /></Connect>'
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@router.post("/calls/transfer-completed")
async def transfer_completed(request: Request) -> Response:
    """Twilio fires this after a transferred 911/988 operator ends the call.

    Only reached when post-call reopen mode is ``auto`` or caller-approved via
    ``prompt`` mode. Returns TwiML
    that re-opens a Media Stream so Terris can check in with the caller and let
    them confirm it is safe to disconnect.

    Note: full three-party monitoring *during* the 911/988 call is not possible
    via Twilio Media Streams – the bridge is direct between the caller and PSAP.
    This endpoint handles the post-operator-hangup re-engagement only.
    """
    if (
        get_transfer_post_call_reopen_mode() == "off"
        and get_custom_transfer_post_call_reopen_mode() == "off"
    ):
        # Safety valve: if the feature is off, just hang up gracefully.
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            "<Say>The emergency call has ended. Take care and call us back if you need support.</Say>"
            "<Hangup/>"
            "</Response>"
        )
        return Response(content=twiml, media_type="application/xml")

    host = request.headers.get("host") or settings.public_host
    scheme = "wss" if request.url.scheme == "https" else "ws"
    stream_url = f"{scheme}://{host}/calls/stream"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Say>The operator has ended the call. "
        "I'm still here if you need support. "
        "Say \"you can go\" whenever you're ready for me to hang up.</Say>"
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


@router.post("/calls/simulation/report", response_model=CallSimulationReportResponse)
async def write_simulation_report(payload: CallSimulationReportRequest) -> CallSimulationReportResponse:
    """Run the call simulator and persist a JSON report to disk."""
    if not settings.simulation_harness_enabled and not settings.debug:
        raise HTTPException(status_code=403, detail="Simulation report harness is disabled")

    tier = (
        SimulationTier.TIER_A_HEADLESS
        if payload.tier == "tier_a"
        else SimulationTier.TIER_B_AUDIO_LOOPBACK
    )

    sim_config = SimulationConfig(
        tier=tier,
        max_turns=payload.max_turns,
        frustration_hangup_threshold=payload.frustration_hangup_threshold,
        force_low_confidence_every_n_turns=payload.force_low_confidence_every_n_turns,
        use_live_therfour_llm=payload.use_live_therfour_llm,
        opening_message=payload.opening_message,
    )
    caller_config = CallerModelConfig(
        provider=payload.caller_provider,
        base_url=payload.caller_base_url,
        model_name_override=payload.caller_model_name_override,
        timeout_s=payload.caller_timeout_s,
    )

    agent = CallSimulationAgent(config=sim_config, caller_model=caller_config)
    report = await agent.run()
    report_dict = _normalize_for_json(report)

    filename = _resolve_report_filename(payload.output_filename)
    _SIMULATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _SIMULATION_REPORTS_DIR / filename

    write_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report": report_dict,
    }
    report_path.write_text(json.dumps(write_payload, indent=2), encoding="utf-8")

    return CallSimulationReportResponse(
        report_path=str(report_path),
        written=True,
        report=report_dict,
    )


@router.get(
    "/calls/simulation/reports/recent",
    response_model=RecentCallSimulationReportsResponse,
)
async def get_recent_simulation_reports(
    limit: int = Query(default=10, ge=1, le=50),
    include_report: bool = Query(default=False),
) -> RecentCallSimulationReportsResponse:
    """Fetch recently generated simulation report files."""
    if not settings.simulation_harness_enabled and not settings.debug:
        raise HTTPException(status_code=403, detail="Simulation report harness is disabled")

    if not _SIMULATION_REPORTS_DIR.exists():
        return RecentCallSimulationReportsResponse(count=0, reports=[])

    candidates = sorted(
        _SIMULATION_REPORTS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    reports: list[CallSimulationReportSummary] = []
    for path in candidates:
        stat = path.stat()
        report_payload = None
        generated_at = ""

        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            generated_at = str(parsed.get("generated_at", ""))
            if include_report:
                candidate_report = parsed.get("report")
                if isinstance(candidate_report, dict):
                    report_payload = candidate_report
        except Exception:
            logger.warning("Failed to parse simulation report file: %s", path)

        reports.append(
            CallSimulationReportSummary(
                filename=path.name,
                report_path=str(path),
                size_bytes=stat.st_size,
                modified_at=_iso_utc_from_mtime(stat.st_mtime),
                generated_at=generated_at,
                report=report_payload,
            )
        )

    return RecentCallSimulationReportsResponse(count=len(reports), reports=reports)


@router.get(
    "/calls/simulation/reports/{filename}",
    response_model=CallSimulationReportFileResponse,
)
async def get_simulation_report_file(filename: str) -> CallSimulationReportFileResponse:
    """Fetch one saved simulation report by filename."""
    if not settings.simulation_harness_enabled and not settings.debug:
        raise HTTPException(status_code=403, detail="Simulation report harness is disabled")

    safe_name = _validate_report_lookup_filename(filename)
    path = _SIMULATION_REPORTS_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Simulation report not found")

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse simulation report: {exc}") from exc

    report_payload = parsed.get("report")
    if not isinstance(report_payload, dict):
        raise HTTPException(status_code=500, detail="Simulation report payload is missing or malformed")

    stat = path.stat()
    return CallSimulationReportFileResponse(
        filename=path.name,
        report_path=str(path),
        size_bytes=stat.st_size,
        modified_at=_iso_utc_from_mtime(stat.st_mtime),
        generated_at=str(parsed.get("generated_at", "")),
        report=report_payload,
    )
