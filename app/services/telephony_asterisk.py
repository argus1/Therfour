"""Asterisk compatibility stubs for transport, call control, and dial-plan templates.

These stubs define the minimal interfaces needed to layer ARI/ExternalMedia
support without changing turn-processing semantics in CallSession.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AsteriskSessionContext:
    """Minimal channel/bridge identifiers for ARI session wiring."""

    channel_id: str
    bridge_id: str = ""
    external_media_id: str = ""


class AsteriskExternalMediaTransportStub:
    """Stub transport for ARI ExternalMedia audio streaming.

    Expected production behavior:
    - Inbound: receive raw binary frames (mulaw/slin) from ARI external media WS.
    - Outbound: send raw binary frames back to ARI media socket.
    """

    def __init__(self, websocket, *, sample_rate_hz: int = 8000, codec: str = "mulaw") -> None:
        self._websocket = websocket
        self.sample_rate_hz = sample_rate_hz
        self.codec = codec

    async def receive_audio_frame(self) -> bytes:
        """Receive one inbound audio frame from Asterisk ExternalMedia."""
        raise NotImplementedError("Asterisk ExternalMedia transport is a stub")

    async def send_audio_frame(self, frame: bytes) -> None:
        """Send one outbound audio frame to Asterisk ExternalMedia."""
        raise NotImplementedError("Asterisk ExternalMedia transport is a stub")

    async def clear_outbound(self) -> None:
        """Clear/interrupt outbound media path.

        ARI has no direct Twilio-like `clear` event, so production implementation
        should map to bridge/channel control strategy.
        """
        raise NotImplementedError("Asterisk outbound-clear behavior is a stub")


class AsteriskCallControlStub:
    """Stub call-control adapter for Asterisk ARI APIs."""

    async def transfer(self, session: AsteriskSessionContext, target: str) -> None:
        """Transfer call via ARI redirect/bridge orchestration (stub)."""
        raise NotImplementedError("Asterisk transfer call-control is a stub")

    async def hangup(self, session: AsteriskSessionContext) -> None:
        """End call via ARI channel delete (stub)."""
        raise NotImplementedError("Asterisk hangup call-control is a stub")


def build_asterisk_dialplan_stub(
    *,
    ws_host: str,
    ws_path: str = "/calls/asterisk/stream",
    context_name: str = "therfour-inbound",
    extension: str = "s",
) -> str:
    """Return an Asterisk dial-plan template for ExternalMedia handoff.

    The dial-plan intentionally remains a stub and requires ARI app/channel IDs,
    auth, and deployment-specific values before production use.
    """
    ws_target = f"wss://{ws_host}{ws_path}"
    return "\n".join(
        [
            "; TherFour Asterisk/FreePBX dial-plan stub",
            "; Replace placeholders and ARI args before production use.",
            f"[{context_name}]",
            f"exten => {extension},1,NoOp(TherFour inbound call)",
            " same => n,Answer()",
            " same => n,Set(CHANNEL(language)=en)",
            " same => n,NoOp(Bridge caller audio to TherFour via ExternalMedia)",
            " same => n,ExternalMedia("
            "wss,none,mulaw,8000,"
            f"{ws_target}"
            ")",
            " same => n,Hangup()",
            "",
            "; Optional transfer completion continuation stub",
            "[therfour-transfer-return]",
            "exten => s,1,NoOp(Transfer leg ended; optionally re-open AI session)",
            " same => n,Hangup()",
        ]
    )
