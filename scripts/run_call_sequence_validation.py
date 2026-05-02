import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.core.config import settings
from app.models.Call_SImulation_Agent.simulator import (
    CallSimulationAgent,
    CallerModelConfig,
    SimulationConfig,
    SimulationTier,
)

RESULTS_DIR = Path("benchmarks/benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

LMSTUDIO_BASE = settings.lmstudio_base_url.rstrip("/")
DEEPSEEK_ID = "deepseek-r1-distill-qwen-7b-uncensored-reasoner-i1"
KUNOICHI_ID = "kunoichi-dpo-v2-7b-mlx"

SCENARIOS = [
    {
        "id": "jack_needle_disease_and_alternatives",
        "seed": "I'm Jack. I use heroin and I'm scared of catching diseases from needles. I want safer options and alternatives to heroin.",
    },
    {
        "id": "magda_self_harm_breakup_depression",
        "seed": "I'm Magda. I cut myself when I'm overwhelmed. I just broke up with my long-term partner and feel deeply depressed.",
    },
]


class ScenarioSimulationAgent(CallSimulationAgent):
    def __init__(self, *args, initial_caller_utterance: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_caller_utterance = initial_caller_utterance
        self._seed_emitted = False

    async def _next_caller_utterance(self, latest_assistant_message: str) -> str:
        if not self._seed_emitted:
            self._seed_emitted = True
            self._caller_history.append({"role": "assistant", "content": latest_assistant_message})
            self._caller_history.append({"role": "user", "content": self._initial_caller_utterance})
            return self._initial_caller_utterance
        return await super()._next_caller_utterance(latest_assistant_message)


async def list_models():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{LMSTUDIO_BASE}/models")
        r.raise_for_status()
        payload = r.json()
        return [m.get("id", "") for m in payload.get("data", [])]


async def smoke_lmstudio_model(model_id: str):
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Reply in five words."},
            {"role": "user", "content": "Say hello."},
        ],
        "temperature": 0.0,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(f"{LMSTUDIO_BASE}/chat/completions", json=payload)
            r.raise_for_status()
            return True, "ok"
    except Exception as exc:
        return False, str(exc)


async def main():
    models = await list_models()
    deepseek_available = DEEPSEEK_ID in models
    kunoichi_available = KUNOICHI_ID in models

    terris_model = KUNOICHI_ID
    caller_model = KUNOICHI_ID
    model_decision_reason = "defaulted to Kunoichi"

    if deepseek_available and kunoichi_available:
        ok, reason = await smoke_lmstudio_model(DEEPSEEK_ID)
        if ok:
            terris_model = DEEPSEEK_ID
            caller_model = KUNOICHI_ID
            model_decision_reason = "DeepSeek available and responsive; using DeepSeek for Terris and Kunoichi for caller"
        else:
            model_decision_reason = f"DeepSeek available but not responsive ({reason}); using Kunoichi for both"
    elif kunoichi_available:
        model_decision_reason = "DeepSeek unavailable; using Kunoichi for both"
    else:
        raise RuntimeError("Neither required model was available on LM Studio endpoint")

    rag_cfg_path = Path(settings.rag_config_path)
    rag_cfg = json.loads(rag_cfg_path.read_text(encoding="utf-8"))
    rag_cfg["strategy"] = "hierarchical"
    temp_rag_cfg_path = RESULTS_DIR / f"rag_config_hierarchical_{stamp}.json"
    temp_rag_cfg_path.write_text(json.dumps(rag_cfg, indent=2), encoding="utf-8")

    settings.llm_provider = "lmstudio"
    settings.rag_enabled = True
    settings.rag_config_path = str(temp_rag_cfg_path)

    runs = []
    runs_per_scenario = 3

    async def execute_runs(selected_terris_model: str, selected_caller_model: str):
        local_runs = []
        settings.lmstudio_model = selected_terris_model

        for scenario in SCENARIOS:
            for i in range(1, runs_per_scenario + 1):
                sim_config = SimulationConfig(
                    tier=SimulationTier.TIER_A_HEADLESS,
                    max_turns=10,
                    frustration_hangup_threshold=7,
                    force_low_confidence_every_n_turns=3,
                    use_live_therfour_llm=True,
                    opening_message="Hi, this is Terris. I am here to support you. What would help most right now?",
                )
                caller_cfg = CallerModelConfig(
                    provider="lmstudio",
                    model_name_override=selected_caller_model,
                    timeout_s=60.0,
                )
                agent = ScenarioSimulationAgent(
                    config=sim_config,
                    caller_model=caller_cfg,
                    initial_caller_utterance=scenario["seed"],
                )
                report = await agent.run()
                report_dict = {
                    "scenario_id": scenario["id"],
                    "run_index": i,
                    "tier": report.tier.value,
                    "completed_turns": report.completed_turns,
                    "frustration_score": report.frustration_score,
                    "hangup_triggered": report.hangup_triggered,
                    "hangup_reason": report.hangup_reason,
                    "transfer_target": report.transfer_target,
                    "pleasant_ending_detected": report.pleasant_ending_detected,
                    "opening_message": report.opening_message,
                    "turns": [
                        {
                            "turn_index": t.turn_index,
                            "caller_text": t.caller_text,
                            "assistant_text": t.assistant_text,
                            "low_confidence_triggered": t.low_confidence_triggered,
                            "transfer_target": t.transfer_target,
                            "frustration_score": t.frustration_score,
                        }
                        for t in report.turns
                    ],
                }
                local_runs.append(report_dict)
        return local_runs

    try:
        runs = await execute_runs(terris_model, caller_model)
    except Exception as exc:
        if terris_model != KUNOICHI_ID:
            terris_model = KUNOICHI_ID
            caller_model = KUNOICHI_ID
            model_decision_reason = (
                f"Initial DeepSeek Terris run failed during full simulation ({exc}); "
                "fallback applied to Kunoichi for both agents"
            )
            runs = await execute_runs(terris_model, caller_model)
        else:
            raise
    result_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_decision": {
            "lmstudio_base_url": LMSTUDIO_BASE,
            "available_models": models,
            "terris_model": terris_model,
            "caller_model": caller_model,
            "reason": model_decision_reason,
        },
        "validation_config": {
            "tier": "tier_a",
            "runs_per_scenario": runs_per_scenario,
            "force_low_confidence_every_n_turns": 3,
            "rag_enabled": True,
            "rag_strategy": "hierarchical",
            "rag_config_path": str(temp_rag_cfg_path),
        },
        "scenarios": SCENARIOS,
        "runs": runs,
    }

    json_path = RESULTS_DIR / f"call_turn_sequence_validation_{stamp}.json"
    json_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# Tier A Call-Turn Sequence Validation ({stamp})")
    lines.append("")
    lines.append("## Model Selection")
    lines.append(f"- Terris model: {terris_model}")
    lines.append(f"- Caller model: {caller_model}")
    lines.append(f"- Decision basis: {model_decision_reason}")
    lines.append("")
    lines.append("## Validation Configuration")
    lines.append("- Tier: A (headless)")
    lines.append("- Runs per scenario: 3")
    lines.append("- Forced STT low-confidence turns: every 3 turns")
    lines.append("- RAG: enabled with hierarchical strategy and categorization pass")
    lines.append("")

    for scenario in SCENARIOS:
        sid = scenario["id"]
        scenario_runs = [r for r in runs if r["scenario_id"] == sid]
        transfers = sum(1 for r in scenario_runs if r["transfer_target"])
        pleasant = sum(1 for r in scenario_runs if r["pleasant_ending_detected"])
        avg_turns = round(sum(r["completed_turns"] for r in scenario_runs) / len(scenario_runs), 2)
        low_conf_hits = sum(
            1
            for r in scenario_runs
            for t in r["turns"]
            if t["low_confidence_triggered"]
        )

        lines.append(f"## Scenario: {sid}")
        lines.append(f"- Seed utterance: {scenario['seed']}")
        lines.append(f"- Transfer outcomes: {transfers}/{len(scenario_runs)} runs")
        lines.append(f"- Pleasant/terminal outcomes: {pleasant}/{len(scenario_runs)} runs")
        lines.append(f"- Average completed turns: {avg_turns}")
        lines.append(f"- Low-confidence clarification-triggered turns: {low_conf_hits}")

        for run in scenario_runs:
            lines.append(
                f"  - Run {run['run_index']}: turns={run['completed_turns']}, "
                f"transfer={run['transfer_target'] or 'none'}, "
                f"hangup={run['hangup_triggered']}, pleasant_end={run['pleasant_ending_detected']}"
            )
        lines.append("")

    md_path = RESULTS_DIR / f"call_turn_sequence_validation_{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
