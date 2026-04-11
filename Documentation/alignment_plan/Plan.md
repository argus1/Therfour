# TherFour x HCA Alignment Plan

## Objective

Align TherFour's STT, TTS, RAG, and response-generation behavior with the patterns and quality bar established in `~/Documents/HCA/`, while upskilling a team of 1-3 developers in the process.

## Sprint Constraint

- Primary implementation window: 1 sprint (2 weeks)
- Team size: 1-3 developers
- Optional sprint 2: only if production-hardening items are not complete by end of sprint 1

## Success Criteria (End of Sprint 1)

1. TherFour has a documented parity matrix versus HCA for STT, TTS, and RAG.
2. A shared prompt and conversation contract is implemented and test-covered in TherFour.
3. STT and TTS pipelines are normalized (config, error handling, latency metrics, and fallback behavior).
4. RAG behavior follows a reproducible retrieval + grounding flow with evaluation examples.
5. A short onboarding path exists so a new developer can understand and run the aligned flow in under 2 hours.
6. Architecture decision records (ADRs) exist for TTS engine, model runtime target, and vector store strategy.

## Scope

### In Scope

- Compare architecture and runtime behavior against `~/Documents/HCA/`.
- Align APIs/contracts for:
  - STT request/response shape and confidence handling
  - TTS voice/options mapping and output constraints
  - RAG retrieval inputs, context window strategy, and citation/grounding style
  - Prompt template structure and turn orchestration
- Add or update tests in TherFour (`tests/` and `swift-backend/Tests/`) for high-risk paths.
- Add basic observability for latency and failure reasons across STT/TTS/RAG.

### Out of Scope (Sprint 1)

- Full UI redesign or mobile-native integration work.
- Large model migration unless required for parity.
- Multi-language expansion beyond current production language assumptions.

## Technology Decision Track (Required During Sprint 1)

### 1) Piper TTS vs F5-TTS

#### Keep Piper (Arguments For)

- Lower operational complexity and easier deployment footprint.
- Predictable CPU-first runtime, useful for constrained server configurations.
- Faster team onboarding because the existing path already works.

#### Move to F5-TTS (Arguments For)

- Potentially more natural prosody and voice quality for coaching-like interaction.
- Better long-form expressiveness may improve conversational UX.
- Closer alignment if HCA quality expectations require higher naturalness.

#### Arguments Against Migration to F5-TTS in Sprint 1

- Higher integration and infra complexity (model serving, GPU tuning, voice consistency QA).
- Increased latency variance risk without dedicated performance hardening.
- Small team may lose sprint capacity needed for STT/RAG parity.

#### Decision Guidance

- Default sprint 1 position: keep Piper as production default, run F5-TTS as an A/B experimental path.
- Promote F5-TTS only if it meets all gates:
  - P95 latency within agreed call-turn budget
  - no increase in synthesis failure rate
  - clear MOS/listener preference improvement in internal evals

### 2) GGUF on GPU Web Server vs CoreML Runtime

#### GGUF on GPU Web Server (Arguments For)

- Unified server-side inference path that is easier to share between TherFour and HCA backends.
- Better throughput scaling and centralized model lifecycle management.
- Simpler observability and rollout controls in one deployment surface.

#### CoreML (Arguments For)

- Strong on-device privacy profile and lower network dependency.
- Excellent local latency on Apple hardware when models are optimized.
- Useful fallback mode for offline or degraded network conditions.

#### Arguments Against Full Runtime Convergence in Sprint 1

- Platform goals differ: TherFour is server-first; HCA includes iOS constraints.
- Premature full convergence can force lowest-common-denominator design.
- Benchmark effort is non-trivial for a 1-3 person team.

#### Decision Guidance

- Sprint 1 recommendation: prioritize GGUF GPU server path for TherFour parity and ops simplicity.
- Keep CoreML compatibility as optional follow-up for edge/offline scenarios.
- Required benchmark dimensions before final commitment:
  - P50/P95 latency
  - token throughput
  - cost per 1k tokens
  - failure/retry rate under concurrent load

### 3) ChromaDB Direct Use vs Conversion to WAX

#### ChromaDB Direct (Arguments For)

- Fewer moving parts and less ETL complexity.
- Faster iteration for chunking/index tuning and retrieval experiments.
- Better debugging ergonomics during alignment and evaluation.

#### Conversion to WAX (Arguments For)

- May enable portability or downstream compatibility with existing HCA workflows.
- Can support precomputed optimized retrieval artifacts if already standardized elsewhere.

#### Arguments Against Conversion in Sprint 1

- Conversion adds synchronization risk (stale or mismatched embeddings/index state).
- More tooling to maintain for little immediate parity value.
- Harder root-cause analysis when retrieval quality drops.

#### Decision Guidance

- Sprint 1 recommendation: use ChromaDB directly for online retrieval path.
- Treat WAX conversion as optional export/offline artifact, not primary runtime dependency.
- Revisit only if HCA integration explicitly requires WAX-first workflows.

### 4) Optional Adoption: Sandwiched Translation Schema (HCA)

Definition (working): maintain canonical internal reasoning/response in base language, with deterministic pre-translation on input and post-translation on output, preserving source and translated fields in the turn payload.

#### Why Adopt (Optional)

- Improves consistency of prompts and grounding when retrieval corpus is language-skewed.
- Supports bilingual auditing and easier QA for translation drift.
- Aligns well with cross-platform contract-first design.

#### Why Defer

- Adds token and latency overhead to each turn.
- Introduces another failure surface (translation quality and timeout handling).
- Not mandatory if sprint 1 remains single-language and parity-focused.

#### Adoption Recommendation

- Keep optional in sprint 1 as a feature flag and schema extension only.
- Implement full rollout in follow-up sprint if multilingual requirements are confirmed.

## Team Plan (1-3 Developers)

- Developer A: STT/TTS alignment owner (Python services + tests).
- Developer B: RAG/prompt alignment owner (Python + Swift turn-processing contract).
- Developer C (if available): test harness, metrics, docs, and integration validation.

If only 1 developer is available, execute in the same sequence with reduced parallelism.

## Work Breakdown (2-Week Sprint)

### Week 1 - Baseline + Contracts

#### Day 1-2: Discovery and Parity Matrix

- Review HCA components and extract:
  - STT provider behavior, retries, confidence treatment
  - TTS voice model defaults and output handling
  - RAG retrieval, chunking assumptions, and grounding style
  - Prompt templates and turn policies
- Record baseline ADR notes for:
  - Piper vs F5-TTS
  - GGUF GPU server vs CoreML
  - Chroma direct vs WAX conversion
  - optional sandwiched translation schema
- Produce a parity matrix table:
  - Current TherFour behavior
  - Desired HCA-aligned behavior
  - Gap severity (High/Medium/Low)
  - Owner and effort

#### Day 3-4: Contract Alignment

- Define/update shared contracts for turn processing and service boundaries:
  - Python schemas in `app/models/schemas.py`
  - Swift contracts in `swift-backend/Sources/swift-backend/VoiceContracts.swift`
- Ensure naming, optionality, and error envelopes are consistent.
- Add contract-focused tests before implementation changes where feasible.

#### Day 5: Prompt + RAG Strategy Lock

- Align prompt templates with HCA conventions (role boundaries, system constraints, safety rails).
- Finalize RAG pipeline decisions:
  - retrieval query construction
  - top-k/threshold defaults
  - context packing strategy
  - grounded answer format expectations
- Finalize decision gates and experiment design for technology choices.

### Week 2 - Implementation + Validation

#### Day 6-8: STT/TTS/RAG Implementation

- STT: normalize preprocessing, confidence thresholds, and fallback behavior in `app/services/stt.py`.
- TTS: standardize voice mapping and response metadata in `app/services/tts.py`.
- RAG/LLM: align context assembly and prompt injection strategy in `app/services/llm.py`.
- Telephony turn integration updates in `app/services/telephony.py` as needed.

#### Day 9: Integration and Performance Checks

- Run end-to-end call-turn scenarios across API and Swift backend integration points.
- Validate baseline metrics:
  - STT latency
  - TTS latency
  - retrieval + generation latency
  - failure rate by component
- Execute targeted tradeoff benchmarks:
  - Piper baseline vs F5-TTS experimental path
  - GGUF GPU runtime under expected concurrency
  - Chroma direct retrieval stability and relevance quality

#### Day 10: Hardening + Knowledge Transfer

- Close high-severity parity gaps.
- Document runbooks and architecture notes.
- Conduct a 60-minute internal walkthrough:
  - How aligned STT/TTS/RAG flow works
  - How to run tests
  - How to troubleshoot common failures

## Deliverables

1. Updated TherFour services implementing aligned behavior.
2. Parity matrix document checked into `Documentation/alignment_plan/`.
3. Updated contracts and tests for STT/TTS/RAG/turn-processing.
4. Minimal observability guidance and troubleshooting notes.
5. ADR summary with go/no-go decisions for TTS engine, runtime target, and vector store strategy.

## Acceptance Checklist

- [ ] STT path parity validated on agreed scenarios.
- [ ] TTS output and metadata parity validated.
- [ ] RAG answers show grounded context usage in evaluation samples.
- [ ] Contract tests pass in both Python and Swift backend.
- [ ] README/additional docs explain aligned flow and local test steps.

## Optional Follow-up Sprint (Only If Needed)

Trigger this sprint only if one or more remain: unresolved high-severity gaps, unstable latency, or insufficient test reliability.

### Focus Areas

- Expand automated regression set with realistic call transcripts.
- Add deeper offline evaluation set for RAG quality.
- Improve caching/streaming to reduce P95 latency.
- Add stronger failure recovery and alerting hooks.
- Productionize any deferred decisions:
  - F5-TTS promotion
  - CoreML fallback track
  - WAX export pipeline (if required)
  - full sandwiched translation schema rollout

### Exit Criteria

- No high-severity parity gaps.
- P95 latency within agreed threshold.
- Stable test suite in CI with clear ownership.

## Risks and Mitigations

- Risk: Hidden architectural differences between iOS and server runtime.
  - Mitigation: align contracts/behavior, not platform-specific implementation details.
- Risk: Overfitting to current HCA internals.
  - Mitigation: codify principles and interfaces, not provider-specific hacks.
- Risk: Small team bandwidth constraints.
  - Mitigation: prioritize High severity parity gaps first and defer Medium/Low items.
