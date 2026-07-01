# Call Simulation Agent

This package provides a tiered simulation agent for validating Therfour end-to-end turn handling.

## Location

- `app/models/Call_SImulation_Agent/simulator.py`

## Tiers

- `tier_a`: Headless simulation against `CallSession._run_turn`.
  - Exercises opening message context, STT turn ingestion, low-confidence confirmation flow, transfer outcomes, and pleasant-ending outcomes.
  - Uses a simulated caller model loaded from the Kunoichi stub metadata by default.
- `tier_b`: Full audio loopback (stubbed for now).
  - Reserved for future implementation where caller text is synthesized and fed through full audio encode/decode and websocket loopback.

## Model Source

Default caller model configuration points to:

- `models/stubs/llm/Kunoichi-DPO-v2-7B-Q4_K_S-imatrix.gguf.stub.json`

The simulator reads the `name` field from the stub manifest and uses it as the runtime model name for provider requests.

Hydrate the model artifact with:

```bash
python scripts/fetch_stub.py models/stubs/llm/Kunoichi-DPO-v2-7B-Q4_K_S-imatrix.gguf.stub.json
```

## Quick Usage

```python
import asyncio

from app.models.Call_SImulation_Agent import (
    CallSimulationAgent,
    CallerModelConfig,
    SimulationConfig,
    SimulationTier,
)

async def main() -> None:
    agent = CallSimulationAgent(
        config=SimulationConfig(
            tier=SimulationTier.TIER_A_HEADLESS,
            max_turns=8,
            frustration_hangup_threshold=6,
            force_low_confidence_every_n_turns=3,
            use_live_therfour_llm=False,
        ),
        caller_model=CallerModelConfig(
            provider="ollama",  # or "lmstudio" or "openai"
        ),
    )
    report = await agent.run()
    print(report)

asyncio.run(main())
```

## Frustration Hangup

Each turn updates a frustration score. The simulation is terminated early when score reaches `frustration_hangup_threshold`.

Score increments for repeated confirmation prompts and failed transfer messaging, and is reduced by successful de-escalation or transfer behavior.

## Current Limitation

Tier B currently returns a stub report with implementation note. This is intentional so Tier A can be stabilized before audio loopback complexity is introduced.

## Stability Toggle

Tier A can run in two modes:

- `use_live_therfour_llm=True` (default): uses Therfour's configured runtime LLM endpoint.
- `use_live_therfour_llm=False`: uses a deterministic in-process reply policy for stable offline simulation and testing.
