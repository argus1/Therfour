"""Tiered call simulation agent for validating Therfour turn behavior.

Tier A runs a headless simulation against ``CallSession._run_turn`` by patching
STT input and capturing generated assistant speech/transfer actions.
Tier B is reserved for full audio loopback and is intentionally stubbed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np

from app.core.config import settings
from app.models.schemas import TranscriptionResult
from app.services.telephony import CallSession

logger = logging.getLogger(__name__)


class SimulationTier(str, Enum):
    """Execution mode for the simulation harness."""

    TIER_A_HEADLESS = "tier_a"
    TIER_B_AUDIO_LOOPBACK = "tier_b"


@dataclass(frozen=True)
class CallerModelConfig:
    """Configuration for the simulated caller model."""

    stub_path: str = "models/stubs/llm/Kunoichi-DPO-v2-7B-Q4_K_S-imatrix.gguf.stub.json"
    provider: str = "ollama"
    base_url: str = ""
    model_name_override: str = ""
    timeout_s: float = 30.0

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.provider == "lmstudio":
            return settings.lmstudio_base_url.rstrip("/")
        if self.provider == "openai":
            return settings.openai_base_url.rstrip("/")
        return settings.ollama_base_url.rstrip("/")

    def resolved_model_name(self) -> str:
        if self.model_name_override:
            return self.model_name_override

        stub = Path(self.stub_path)
        if not stub.exists():
            raise FileNotFoundError(
                f"Caller model stub not found: {stub}. Hydrate with scripts/fetch_stub.py first."
            )

        payload = json.loads(stub.read_text(encoding="utf-8"))
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError(f"Model stub {stub} has no 'name' field.")
        return name


@dataclass(frozen=True)
class SimulationConfig:
    """Behavior knobs for one simulation run."""

    tier: SimulationTier = SimulationTier.TIER_A_HEADLESS
    max_turns: int = 8
    frustration_hangup_threshold: int = 6
    force_low_confidence_every_n_turns: int = 0
    use_live_therfour_llm: bool = True
    per_turn_timeout_s: float = 25.0
    scripted_caller_turns: tuple[str, ...] = ()
    caller_persona_path: str = ""
    opening_message: str = (
        "Hi, this is Terris. I am here with you. What name would you like me to use for you today?"
    )


@dataclass(frozen=True)
class SimulationTurn:
    """Structured event for each completed simulated turn."""

    turn_index: int
    caller_text: str
    assistant_text: str
    low_confidence_triggered: bool
    transfer_target: str = ""
    frustration_score: int = 0


@dataclass(frozen=True)
class SimulationReport:
    """Final result from a simulation run."""

    tier: SimulationTier
    completed_turns: int
    frustration_score: int
    hangup_triggered: bool
    hangup_reason: str
    transfer_target: str
    pleasant_ending_detected: bool
    opening_message: str
    turns: list[SimulationTurn] = field(default_factory=list)
    notes: str = ""


class _DummyWebSocket:
    async def iter_text(self):  # pragma: no cover - simulation bypasses websocket stream
        if False:
            yield ""


class CallSimulationAgent:
    """Simulates a distressed caller against Therfour turn orchestration."""

    def __init__(
        self,
        config: SimulationConfig,
        caller_model: Optional[CallerModelConfig] = None,
    ) -> None:
        self.config = config
        self.caller_model = caller_model or CallerModelConfig()
        self._caller_history: list[dict[str, str]] = []
        self._scripted_turn_index = 0
        self._caller_persona = self._load_caller_persona(config.caller_persona_path)

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def _load_caller_persona(self, persona_path: str) -> str:
        if not persona_path:
            return ""

        path = Path(persona_path)
        if not path.is_absolute():
            path = self._repo_root() / persona_path

        try:
            if not path.exists():
                logger.warning("Caller persona file not found: %s", path)
                return ""
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                logger.warning("Caller persona file is empty: %s", path)
            return text
        except Exception:
            logger.exception("Failed to load caller persona file: %s", path)
            return ""

    async def run(self) -> SimulationReport:
        if self.config.tier == SimulationTier.TIER_B_AUDIO_LOOPBACK:
            return SimulationReport(
                tier=self.config.tier,
                completed_turns=0,
                frustration_score=0,
                hangup_triggered=False,
                hangup_reason="",
                transfer_target="",
                pleasant_ending_detected=False,
                opening_message=self.config.opening_message,
                notes=(
                    "Tier B is intentionally stubbed. Use Tier A for current validation "
                    "while audio loopback instrumentation is stabilized."
                ),
            )

        return await self._run_tier_a_headless()

    async def _run_tier_a_headless(self) -> SimulationReport:
        session = CallSession(_DummyWebSocket())
        session._call_sid = "CA_SIMULATED"

        min_samples = int(settings.min_audio_duration_s * settings.audio_sample_rate_whisper)
        simulated_audio = np.zeros(max(min_samples + 64, 4096), dtype=np.float32)

        frustration = 0
        transfer_target = ""
        pleasant_end = False
        turns: list[SimulationTurn] = []
        assistant_utterances: list[str] = []
        transfer_events: list[str] = []

        async with self._patched_session_runtime(
            session=session,
            assistant_utterances=assistant_utterances,
            transfer_events=transfer_events,
        ):
            previous_assistant = self.config.opening_message

            for turn_idx in range(1, self.config.max_turns + 1):
                caller_text = await self._next_caller_utterance(previous_assistant)
                if not caller_text:
                    break

                low_conf = self._is_forced_low_confidence_turn(turn_idx)
                session._simulated_stt_result = self._make_stt_result(caller_text, low_conf=low_conf)

                before_tts = len(assistant_utterances)
                before_transfer = len(transfer_events)
                turn_timed_out = False
                try:
                    await asyncio.wait_for(
                        session._run_turn(simulated_audio),
                        timeout=max(1.0, float(self.config.per_turn_timeout_s)),
                    )
                except asyncio.TimeoutError:
                    turn_timed_out = True
                    logger.warning(
                        "Simulation turn timed out (turn=%d, timeout_s=%.2f)",
                        turn_idx,
                        self.config.per_turn_timeout_s,
                    )
                    assistant_utterances.append(
                        "I am still with you. Let us focus on one immediate next safe step."
                    )

                assistant_text = ""
                if len(assistant_utterances) > before_tts:
                    assistant_text = assistant_utterances[-1]

                if len(transfer_events) > before_transfer:
                    transfer_target = transfer_events[-1]

                frustration += self._frustration_delta(
                    assistant_text,
                    low_conf,
                    transfer_happened=bool(transfer_target),
                )
                if turn_timed_out:
                    frustration += 1

                if self._looks_like_pleasant_ending(assistant_text):
                    pleasant_end = True

                turns.append(
                    SimulationTurn(
                        turn_index=turn_idx,
                        caller_text=caller_text,
                        assistant_text=assistant_text,
                        low_confidence_triggered=low_conf,
                        transfer_target=transfer_target,
                        frustration_score=frustration,
                    )
                )
                previous_assistant = assistant_text or previous_assistant

                if frustration >= self.config.frustration_hangup_threshold:
                    return SimulationReport(
                        tier=self.config.tier,
                        completed_turns=turn_idx,
                        frustration_score=frustration,
                        hangup_triggered=True,
                        hangup_reason="frustration_threshold_reached",
                        transfer_target=transfer_target,
                        pleasant_ending_detected=pleasant_end,
                        opening_message=self.config.opening_message,
                        turns=turns,
                    )

                if transfer_target:
                    break

                if pleasant_end:
                    break

        return SimulationReport(
            tier=self.config.tier,
            completed_turns=len(turns),
            frustration_score=frustration,
            hangup_triggered=False,
            hangup_reason="",
            transfer_target=transfer_target,
            pleasant_ending_detected=pleasant_end,
            opening_message=self.config.opening_message,
            turns=turns,
        )

    @asynccontextmanager
    async def _patched_session_runtime(
        self,
        *,
        session: CallSession,
        assistant_utterances: list[str],
        transfer_events: list[str],
    ):
        from app.services import telephony

        original_transcribe = telephony.stt.transcribe
        original_tts = telephony.tts.synthesize
        original_llm_generate = telephony.llm.generate
        original_transfer = session._transfer_call
        original_send = session._send_audio

        async def _fake_transcribe(_audio, language=None, preferred_backend=None):
            result = getattr(session, "_simulated_stt_result", None)
            if result is None:
                return TranscriptionResult(
                    text="",
                    language="en",
                    confidence=0.0,
                    language_confidence=0.0,
                    transcript_quality_score=0.0,
                    backend_name="faster-whisper",
                    fallback_used=False,
                    failure_reason="no_speech",
                )
            return result

        async def _fake_synthesize(text: str, *, language=None):
            assistant_utterances.append(text)
            return np.zeros(1600, dtype=np.float32)

        async def _fake_transfer(transfer, announcement: str, *, enable_post_call_reopen: bool = False):
            transfer_events.append(f"{transfer.target_kind}:{transfer.target}")
            assistant_utterances.append(announcement)
            return True

        async def _fake_llm_generate(
            conversation: list[dict[str, str]],
            strategy: str | None = None,
            rag_allowed: bool | None = None,
        ) -> str:
            latest = ""
            for message in reversed(conversation):
                if str(message.get("role", "")).lower() == "user":
                    latest = str(message.get("content", "")).lower()
                    break

            if "911" in latest or "emergency" in latest:
                return "TRANSFER:911\nI am connecting you to emergency services now."
            if "988" in latest or "suicidal" in latest or "kill myself" in latest:
                return "TRANSFER:988\nI am connecting you to crisis support now."
            if strategy == "understanding_check_no_rag":
                return (
                    "To check I was clear: step one is to keep naloxone nearby, and step two is to avoid using alone. "
                    "Can you tell me in your words what step one means?"
                )
            if strategy == "explanation_rag_optional":
                return (
                    'EMPHASIS-HINTS: "naloxone" | "call 911"\n'
                    "I mean this: keep naloxone close and call 911 right away if breathing slows. "
                    "Does that make sense now?"
                )
            if "all the help" in latest or "i'm okay now" in latest or "i am okay now" in latest:
                return "I am glad this helped. If anything changes, you can call back anytime. Take care."
            return "I hear you. Let us focus on one safe next step together right now."

        async def _fake_send_audio(_samples):
            return None

        telephony.stt.transcribe = _fake_transcribe
        telephony.tts.synthesize = _fake_synthesize
        if not self.config.use_live_therfour_llm:
            telephony.llm.generate = _fake_llm_generate
        session._transfer_call = _fake_transfer
        session._send_audio = _fake_send_audio

        try:
            yield
        finally:
            telephony.stt.transcribe = original_transcribe
            telephony.tts.synthesize = original_tts
            telephony.llm.generate = original_llm_generate
            session._transfer_call = original_transfer
            session._send_audio = original_send

    def _is_forced_low_confidence_turn(self, turn_idx: int) -> bool:
        n = self.config.force_low_confidence_every_n_turns
        return n > 0 and turn_idx % n == 0

    @staticmethod
    def _make_stt_result(text: str, *, low_conf: bool) -> TranscriptionResult:
        confidence = 0.35 if low_conf else 0.93
        return TranscriptionResult(
            text=text,
            language="en",
            confidence=confidence,
            language_confidence=confidence,
            transcript_quality_score=confidence,
            backend_name="faster-whisper",
            fallback_used=False,
            failure_reason="",
        )

    def _frustration_delta(self, assistant_text: str, low_conf: bool, transfer_happened: bool) -> int:
        delta = 1
        if low_conf:
            delta += 1

        msg = assistant_text.lower()
        if transfer_happened:
            return max(0, delta - 2)

        if "i could not complete the transfer" in msg:
            delta += 2

        if "please say yes or no" in msg or "did i hear" in msg or "can you repeat" in msg:
            delta += 1

        if "glad" in msg or "take care" in msg or "i am here with you" in msg:
            delta = max(0, delta - 1)

        return delta

    @staticmethod
    def _looks_like_pleasant_ending(assistant_text: str) -> bool:
        msg = assistant_text.lower()
        return (
            "glad" in msg and "help" in msg
        ) or "anything else i can help" in msg or "take care" in msg

    async def _next_caller_utterance(self, latest_assistant_message: str) -> str:
        self._caller_history.append({"role": "assistant", "content": latest_assistant_message})

        if self._scripted_turn_index < len(self.config.scripted_caller_turns):
            scripted = self.config.scripted_caller_turns[self._scripted_turn_index].strip()
            self._scripted_turn_index += 1
            if scripted:
                self._caller_history.append({"role": "user", "content": scripted})
                return scripted

        system_prompt = (
            "You are role-playing only the caller in a phone conversation with Terris. "
            "Stay in first person as the caller and keep each response to one short sentence. "
            "Never output markdown headers, never output TRANSFER directives, never output XML tags, and never "
            "speak as Terris or as a policy narrator. If asked yes/no questions, answer directly in natural speech."
        )
        if self._caller_persona:
            system_prompt = (
                f"{system_prompt}\n\n"
                "Caller persona details:\n"
                f"{self._caller_persona}"
            )

        messages = [{"role": "system", "content": system_prompt}, *self._caller_history[-10:]]

        try:
            caller_text = await self._generate_caller_text(messages)
            caller_text = self._sanitize_caller_text(caller_text)
            if caller_text:
                self._caller_history.append({"role": "user", "content": caller_text})
                return caller_text
        except Exception:
            logger.exception("Caller model generation failed; using deterministic fallback line")

        fallback = "I feel overwhelmed and I do not know how to keep myself safe tonight."
        self._caller_history.append({"role": "user", "content": fallback})
        return fallback

    @staticmethod
    def _sanitize_caller_text(raw: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
        cleaned_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith("###"):
                continue
            if "transfer:" in lower:
                continue
            if lower.startswith("role:") or lower.startswith("assistant:"):
                continue
            cleaned_lines.append(stripped)

        collapsed = " ".join(cleaned_lines).strip()
        if not collapsed:
            return ""

        first_sentence = re.split(r"(?<=[.!?])\s+", collapsed, maxsplit=1)[0].strip()
        candidate = first_sentence or collapsed
        return candidate[:280].strip()

    async def _generate_caller_text(self, messages: list[dict[str, str]]) -> str:
        model_name = self.caller_model.resolved_model_name()
        base_url = self.caller_model.resolved_base_url()

        if self.caller_model.provider in {"lmstudio", "openai"}:
            payload: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.7,
                "stream": False,
            }
            endpoint = f"{base_url}/chat/completions"
            headers: dict[str, str] = {}
            if self.caller_model.provider == "openai":
                api_key = settings.openai_api_key.strip()
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY must be set when caller_provider=openai"
                    )
                headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=self.caller_model.timeout_s) as client:
                resp = await client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    return ""
                return str(choices[0].get("message", {}).get("content", ""))

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7},
        }
        endpoint = f"{base_url}/api/chat"

        async with httpx.AsyncClient(timeout=self.caller_model.timeout_s) as client:
            resp = await client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("message", {}).get("content", ""))


async def run_default_simulation() -> SimulationReport:
    """Convenience entry point for manual experiments."""
    agent = CallSimulationAgent(
        config=SimulationConfig(tier=SimulationTier.TIER_A_HEADLESS),
        caller_model=CallerModelConfig(),
    )
    return await agent.run()


if __name__ == "__main__":  # pragma: no cover - manual utility mode
    report = asyncio.run(run_default_simulation())
    print(json.dumps(report.__dict__, default=lambda x: x.__dict__, indent=2))
