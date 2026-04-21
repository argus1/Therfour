# TherFour x HealthCoacher Parity Matrix (Branch Snapshot)

Date: 2026-04-20
Source branch context: argus-branch after merge of origin/Rae-branch
Method: documentation-based assessment only (no additional code verification in this artifact)

## Evidence Used

- Documentation/alignment_plan/Plan.md
- Documentation/alignment_plan/deliverables/STT_Parity_Therfour_vs_HealthCoacher_2026-04-15.md
- Documentation/alignment_plan/deliverables/STT_Input_Normalization_Plan_2026-04-18.md
- Documentation/alignment_plan/deliverables/STT_Task_Checklist_2026-04-18.md
- Documentation/alignment_plan/deliverables/STT_Whisper_Model_Recommendations_2026-04-18.md
- Documentation/alignment_plan/deliverables/RAG_Parity_Therfour_vs_HealthCoacher_2026-04-15.md
- Documentation/alignment_plan/deliverables/Prompt_TurnPolicy_Parity_Therfour_vs_HealthCoacher_2026-04-15.md
- Documentation/alignment_plan/deliverables/Baseline_Contract_Checklist_2026-04-19.md
- Documentation/delieverables/TTS_Parity_Analysis.md
- Documentation/delieverables/TTS_implementation: response_metadata_normalization.md
- Documentation/delieverables/ADR_Package_TTS_Engine_Decision.md

## Consolidated Parity Matrix

| Workstream                               | TherFour documented current state (this branch)                                                                                                                                                          | HealthCoacher target model                                                      | Gap severity now | Owner (from plan) | Estimated effort | Documentation status |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------- | ----------------- | ---------------- | -------------------- |
| STT retry and quality gating             | STT parity analysis initially flagged high gap, but STT checklist now marks fallback decode + low-quality rejection complete in phase 1                                                                  | Multi-strategy decode with quality gates and resilient fallback behavior        | Medium           | Developer A       | S                | Strong               |
| STT VAD-based endpointing                | STT normalization plan defines Silero-first design; checklist reports VAD integration complete with remaining observability and session-level tests open                                                 | Robust endpointing and graceful turn handling under noisy audio                 | Medium           | Developer A       | M                | Strong               |
| STT backend fallback (Sherpa path)       | Documented as phase 2 and not yet complete; feature-flag path planned                                                                                                                                    | Adaptive fallback backend with session-sticky behavior                          | High             | Developer A       | M                | Strong               |
| STT metrics and observability            | Structured STT fields documented; checklist still open for attempt count and decode/fallback metrics                                                                                                     | Backend status visibility and detailed telemetry                                | Medium           | Developer A       | S                | Strong               |
| TTS architecture and fallback            | TTS parity analysis identifies major gap; metadata normalization plan + ADR define target architecture and fallback chain, but implementation completion is not documented in checklist form             | Protocol-driven multi-backend TTS with sticky fallback and typed behavior       | High             | Developer A       | M                | Medium               |
| TTS metadata and typed failure taxonomy  | Detailed plan exists for TTSSynthesisResult and TTSFailureReason; intended parity direction is clear                                                                                                     | Typed audio-service errors and per-turn TTS telemetry                           | Medium-High      | Developer A       | S                | Medium               |
| RAG retrieval and grounding flow         | RAG parity analysis shows no explicit runtime retrieval yet; recommended retrieve-then-generate contract is documented                                                                                   | Explicit retrieval stage, chunk provenance, grounding-aware prompt construction | High             | Developer B       | L                | Strong               |
| Prompt layering and turn policy envelope | Prompt/turn parity analysis shows current monolithic prompt and simpler turn loop; layered roles and sanitizer are proposed                                                                              | Role-layered prompt assembly, sanitizer, multi-phase orchestration              | High             | Developer B       | M                | Strong               |
| Python/Swift contract alignment          | Baseline contract checklist indicates STT contract expectations and Swift decoding behavior are covered; phase-1 schema expansion documented as complete                                                 | Shared explicit cross-stack contracts and failure semantics                     | Low-Medium       | Developer B       | S                | Strong               |
| ADR coverage (required in Plan.md)       | TTS engine ADR is present; ADRs for model runtime target (GGUF vs CoreML) and vector store strategy (Chroma vs WAX) are discussed in plan but not present as standalone ADR docs in this branch snapshot | ADRs exist for TTS engine, model runtime target, and vector store strategy      | Medium           | Developer C       | S                | Partial              |
| Onboarding and runbook readiness         | Plan and deliverables exist, but no dedicated under-2-hour onboarding guide is identified as complete                                                                                                    | New developer can run aligned flow quickly with clear docs and troubleshooting  | Medium           | Developer C       | S-M              | Partial              |

Effort scale: S (<= 2 days), M (3-5 days), L (>= 1 week).

## Sprint-1 Success Criteria Check (Documentation Evidence)

| Plan success criterion                                                 | Evidence in this branch                                                                                                               | Status              |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| Documented parity matrix versus HCA for STT/TTS/RAG                    | This consolidated matrix plus per-domain parity analyses exist                                                                        | Met (documentation) |
| Shared prompt and conversation contract implemented and test-covered   | Prompt parity analysis and baseline contract checklist exist; full prompt-layer implementation evidence is not documented as complete | Partial             |
| STT and TTS pipelines normalized (config/error/latency/fallback)       | STT checklist indicates phase-1 substantial completion; TTS has strong plans/ADR but no equivalent completion checklist artifact      | Partial             |
| RAG follows reproducible retrieval + grounding flow with eval examples | RAG analysis identifies this as missing implementation area                                                                           | Not met             |
| New developer onboarding path under 2 hours                            | Not evidenced by a dedicated onboarding guide artifact in this snapshot                                                               | Not met             |
| ADRs for TTS engine, runtime target, vector store strategy             | TTS ADR present; runtime target and vector store ADR artifacts not found as standalone docs                                           | Partial             |

## Priority Actions to Reach Parity Closure

1. Close high-severity architecture gaps first: RAG retrieval flow, prompt/turn orchestration, and TTS fallback architecture.
2. Publish missing ADR artifacts for runtime target and vector store strategy to satisfy Plan.md criterion.
3. Add TTS implementation checklist artifact (parallel to STT task checklist) to convert TTS from design parity to implementation parity tracking.
4. Complete open STT phase-1 observability and session-level tests to reduce residual operational risk.
5. Add a concise onboarding runbook (setup, test, troubleshoot) to satisfy the under-2-hour onboarding criterion.

## Notes

- This matrix is intentionally grounded in branch documentation state as of 2026-04-20.
- Folder naming inconsistency exists in this branch: both Documentation/alignment_plan/deliverables and Documentation/delieverables are used.
- Recommended cleanup: consolidate to one canonical deliverables path to prevent artifact drift.
