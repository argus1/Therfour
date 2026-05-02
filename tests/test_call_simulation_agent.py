from __future__ import annotations

import pytest

from app.models.Call_SImulation_Agent import (
    CallSimulationAgent,
    SimulationConfig,
    SimulationTier,
)


@pytest.mark.asyncio
async def test_tier_b_returns_stubbed_report() -> None:
    agent = CallSimulationAgent(
        config=SimulationConfig(tier=SimulationTier.TIER_B_AUDIO_LOOPBACK)
    )

    report = await agent.run()

    assert report.tier == SimulationTier.TIER_B_AUDIO_LOOPBACK
    assert report.completed_turns == 0
    assert "stubbed" in report.notes.lower()


@pytest.mark.asyncio
async def test_tier_a_hangup_after_frustration_threshold(monkeypatch) -> None:
    agent = CallSimulationAgent(
        config=SimulationConfig(
            tier=SimulationTier.TIER_A_HEADLESS,
            max_turns=3,
            frustration_hangup_threshold=1,
            use_live_therfour_llm=False,
        )
    )

    async def _fake_caller(_latest_assistant_message: str) -> str:
        return "I am not okay right now"

    monkeypatch.setattr(agent, "_next_caller_utterance", _fake_caller)

    report = await agent.run()

    assert report.hangup_triggered is True
    assert report.hangup_reason == "frustration_threshold_reached"
    assert report.completed_turns >= 1
