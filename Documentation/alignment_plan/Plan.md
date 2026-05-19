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

### Canonical Turn Model (Sprint 1 Contract Baseline)

Purpose: define one cross-runtime contract for a single conversational turn so Python (`app/models/schemas.py`) and Swift (`swift-backend/Sources/swift-backend/VoiceContracts.swift`) can serialize/deserialize without behavioral drift.

#### Shared Payload Structure

Use a two-layer shape for each message exchanged across service boundaries.

- Layer 1: envelope for transport and observability metadata
- Layer 2: payload for turn semantics (what actually happened in the turn)

Canonical high-level shape:

```json
{
  "envelope": {
    "schema_version": "1.0",
    "message_type": "turn.request|turn.response|turn.error|turn.event",
    "trace_id": "uuid",
    "turn_id": "uuid",
    "session_id": "uuid",
    "created_at": "ISO-8601",
    "source": "telephony|api|swift-backend|worker",
    "idempotency_key": "string-optional"
  },
  "payload": {
    "input": {
      "audio": {},
      "text": {},
      "dtmf": {},
      "language": {}
    },
    "processing": {
      "vad": {},
      "stt": {},
      "rag": {},
      "llm": {},
      "tts": {}
    },
    "output": {
      "assistant_text": "string",
      "assistant_audio": {},
      "grounding": {},
      "safety": {}
    },
    "status": {
      "state": "ok|partial|failed|dropped",
      "failure_reason": "string-optional",
      "retryable": false
    }
  }
}
```

#### Field Naming Rules

Adopt strict `snake_case` for all serialized fields in both Python and Swift wire formats.

- IDs: suffix with `_id` (`turn_id`, `session_id`, `trace_id`)
- Timestamps: suffix with `_at` for wall time and `_ms` for durations/offsets
- Booleans: prefix with `is_`, `has_`, or explicit state flags (`fallback_used`)
- Enums: lowercase string values only (`ok`, `partial`, `failed`, `dropped`)
- Confidence-like fields: never generic `confidence` when ambiguous
- Prefer explicit names:
  - `transcript_confidence`
  - `language_confidence`
  - `retrieval_relevance_score`

Cross-runtime mapping rule:

- Swift internal `camelCase` is allowed in code, but encoded JSON keys must remain `snake_case`.

#### Optionality Rules

Define optionality by behavior, not convenience.

- Required always:
  - `envelope.schema_version`
  - `envelope.message_type`
  - `envelope.trace_id`
  - `envelope.turn_id`
  - `envelope.session_id`
  - `envelope.created_at`
  - `payload.status.state`
- Required when present by modality:
  - If audio input exists, require `payload.input.audio.codec` and `payload.input.audio.sample_rate_hz`
  - If STT executed, require `payload.processing.stt.backend_name` and transcript text or explicit `failure_reason`
  - If TTS executed, require `payload.processing.tts.voice_id` and output format
- Optional but recommended:
  - `idempotency_key`, `backend_name`, `fallback_used`, `transcript_quality_score`, `vad_voiced_duration_ms`
- Prohibited:
  - Null for required fields
  - Empty string sentinel values for missing data
  - Mixed-type fields across turns (for example score as string in one turn and number in another)

Missing optional fields should be omitted, not emitted as null, unless a consumer explicitly requires nullable semantics.

#### Envelope Rules

Envelope is transport-safe and stable across message types.

- `schema_version` is mandatory and semantic (`major.minor`)
- `message_type` controls payload validation profile:
  - `turn.request`: caller input + pre-processing metadata
  - `turn.response`: assistant output + synthesis metadata
  - `turn.error`: normalized failure envelope
  - `turn.event`: intermediate state/progress events
- `trace_id` is stable across all turns in one call/session chain when possible
- `turn_id` is unique per turn attempt
- Retries keep `turn_id` only if idempotent replay semantics are guaranteed; otherwise create a new `turn_id` and include `parent_turn_id` in payload
- Every `turn.error` must include:
  - `payload.status.state = failed`
  - `payload.status.failure_reason`
  - `payload.status.retryable`
  - error classification (`timeout|validation|provider|internal|upstream_cancel`)

#### Error and Partial-Turn Normalization

Avoid ad hoc error shapes.

- A dropped no-speech turn should be `state = dropped`, not `failed`
- Partial pipeline success (for example STT success, TTS failure) should be `state = partial`
- Failures from providers should preserve raw details in a debug-only subfield, with sanitized top-level reason for stable contract behavior

#### Action Point Stubs (Implementation Checklist)

<!-- for agents: coordination-only stubs live here. Keep edits in this section scoped to ownership notes and checklist state updates. -->
<!-- for agents: Nicole owns app/models/schemas.py in current branch window. Avoid direct edits there from parallel branches unless explicitly reassigned. -->
<!-- for agents: Prefer additive updates in this document and JSON examples to minimize cross-file contention during schema implementation. -->

- [ ] Define `CanonicalTurnEnvelope` and `CanonicalTurnPayload` models in `app/models/schemas.py`
- [ ] Define matching Codable structs in `swift-backend/Sources/swift-backend/VoiceContracts.swift`
- [ ] Add versioned JSON schema example files in `Documentation/alignment_plan/deliverables/` (`turn.request.v1.json`, `turn.response.v1.json`, `turn.error.v1.json`)
- [ ] Add Python contract tests for required/optional field validation in `tests/test_api.py` and `tests/test_telephony.py`
- [ ] Add Swift decoding/encoding parity tests in `swift-backend/Tests/swift-backendTests/`
- [ ] Add migration notes mapping current TherFour fields to canonical names
- [ ] Add `schema_version` compatibility policy (minor backward-compatible, major breaking)
- [ ] Add observability mapping for `trace_id`, `turn_id`, status, and component latency fields

<!-- for agents: action stub A - If touching app/models/schemas.py becomes necessary, claim ownership in PR/task thread before code edits. -->
<!-- for agents: action stub B - Prefer implementing turn.request/turn.response/turn.error JSON examples first; wire schemas second. -->
<!-- for agents: action stub C - Keep field names snake_case on wire; Swift camelCase mapping only in code-level property names. -->

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
  - [ ] **(Medium) Tune RAG similarity threshold to avoid empty-context turns (REG-001)**
  - Change: lower `similarity_threshold` from **0.35 → 0.20** (then re-run retrieval eval/benchmarks).
  - Rationale: Chroma benchmark findings show threshold=0.35 suppresses context for most query categories (hit rate ~0.20), even though the retrieval engine is stable/error-free.
  - Acceptance: retrieval hit-rate improves materially across categories without introducing off-topic grounding; record updated numbers in `chroma_benchmark_regression_notes.md`.
  - References: `deliverables/RAG_lock.md`, `deliverables/RAG_Parity_Therfour_vs_HealthCoacher_2026-04-15.md`, `deliverables/Parity_Matrix_Therfour_vs_HealthCoacher_2026-04-20.md`.

- Telephony turn integration updates in `app/services/telephony.py` as needed.

### STT Input Normalization Plan

#### Design Goal

Build an open-source, low-latency STT path for conversational hotline calls that:

- reduces false turn triggers from silence/noise
- improves transcript reliability under telephony audio constraints
- keeps deployment practical for a wide user base
- uses GPU acceleration when available without making GPU mandatory

#### Recommended Architecture

Default recommendation: use all three components, but not in one serial blocking path.

- Silero VAD as the always-on speech gate and utterance boundary detector
- Whisper as the final transcript engine for quality and multilingual robustness
- Sherpa-ONNX as optional low-latency streaming/CPU-safe fallback, not a mandatory inline step for every finalized turn

This means the preferred production pattern is:

1. ingest Twilio audio and normalize codec/sample rate
2. run Silero VAD on short rolling frames
3. buffer only voiced spans and finalize turns with VAD hangover logic
4. send finalized speech spans to Whisper for the canonical transcript
5. optionally use Sherpa-ONNX for live partials, quick confirmation, or fallback when Whisper is degraded/unavailable

#### Feasibility Judgment

Using all three is feasible if responsibilities are separated.

- Feasible: Silero VAD + Whisper + Sherpa-ONNX as layered components
- Not recommended: running Sherpa-ONNX and Whisper serially for every turn by default, because this adds latency and operational complexity without guaranteed user-visible benefit

If sprint or complexity limits force prioritization, the order should be:

1. Silero VAD
2. Whisper hardening
3. Sherpa-ONNX fallback/streaming path

Rationale:

- Silero VAD directly improves latency and turn quality by shrinking silence timeout dependence.
- Whisper is already integrated and is the fastest route to better transcript quality with the least architectural churn.
- Sherpa-ONNX is valuable, especially for streaming partials and CPU fallback, but is a second integration surface and should not block sprint 1 parity work.

#### Component Roles

##### Silero VAD

Use Silero VAD as the primary input-normalization control plane.

- Detect speech onset and end on 20-30 ms frames
- Replace or greatly reduce fixed `silence_timeout_s` dependence
- Reject no-speech turns before STT decode
- Trim leading/trailing silence before transcription
- Emit structured VAD metadata for observability:
  - speech_started_at_ms
  - speech_ended_at_ms
  - voiced_duration_ms
  - dropped_as_no_speech

Why first:

- highest latency win for hotline interaction
- open-source and widely used
- model-light compared with full STT backends

##### Whisper

Keep Whisper as the canonical final-transcript backend.

- Continue using faster-whisper for server deployment
- Prefer CUDA execution on GPU-enabled servers
- Keep CPU mode available for broad OSS adoption
- Add decode policy rather than single-pass transcription:
  - primary decode with pinned/default language policy
  - fallback decode with relaxed thresholds or auto-language mode
  - transcript quality gate before LLM handoff

Recommended server defaults:

- GPU servers: faster-whisper with a latency-focused model class such as `small`, `distil-large-v3`, or other benchmarked low-latency option
- CPU-only installs: smaller Whisper class with explicit documentation that accuracy is lower but supported

Whisper should own:

- final transcript text
- language detection metadata
- transcript quality heuristics

##### Sherpa-ONNX

Adopt Sherpa-ONNX as an optional path with two valid roles.

Role A: streaming partial transcript engine

- Produce low-latency partial text while audio is still arriving
- Improve responsiveness if the product later supports interruption, barge-in, or live agent assistance

Role B: fallback backend

- Provide a CPU-friendly open-source fallback when Whisper GPU is unavailable, overloaded, or failing repeatedly
- Offer session-sticky fallback behavior after repeated Whisper failures

Do not make Sherpa-ONNX required for sprint 1 final-turn transcription unless benchmarks prove it beats the existing Whisper path on both latency and transcript quality for hotline audio.

#### Latency Strategy for Hotline Use

Because the agent is conversational and phone-based, the plan should optimize end-of-utterance latency, not only raw model decode speed.

Primary latency actions:

- move from fixed silence timeout toward VAD-based endpointing
- keep audio in rolling buffers rather than waiting for coarse silence windows
- avoid double-decoding every finalized turn unless fallback is needed
- use smaller or distilled Whisper variants by default on server
- keep model warm and reuse process-local workers

Target operational budgets for sprint validation:

- end-of-speech to transcript start: under 300 ms with VAD finalization
- finalized transcript ready: under 900 ms P50 and under 1500 ms P95 on target GPU servers
- no-speech false turn rate reduced materially versus fixed-timeout baseline

#### Implementation Order

##### Phase 1 - Ship This First

- Add Silero VAD-based speech segmentation in `app/services/telephony.py`
- Pass trimmed voiced audio into `app/services/stt.py`
- Add Whisper decode retry/fallback policy in `app/services/stt.py`
- Update schemas/contracts for explicit transcript quality and failure reason fields
- Add tests for no-speech rejection, trimmed input handling, and fallback decode behavior

Phase 1 outcome:

- lowest-risk, highest-value improvement
- no new primary STT backend required

##### Phase 2 - Add Optional Sherpa-ONNX

- Introduce Sherpa-ONNX backend abstraction behind a shared STT interface
- Use it first as:
  - session fallback backend, or
  - optional partial/live transcript backend
- Add config flags so deployments can enable or disable Sherpa independently

Phase 2 outcome:

- broader hardware compatibility
- lower-latency partials if product direction requires them

##### Phase 3 - Benchmark and Promote by Policy

- Compare:
  - VAD + Whisper
  - VAD + Sherpa only
  - VAD + Sherpa partials + Whisper final
- Benchmark on telephony-quality audio and noisy hotline samples
- Choose production defaults based on measured latency, transcript quality, and failure rate

#### Required Contract Changes

Python and Swift contracts should make STT state explicit.

- Rename ambiguous confidence semantics if needed:
  - `confidence` should not mean language ID confidence if downstream users assume transcript confidence
- Add optional metadata fields for:
  - `language_confidence`
  - `transcript_quality_score`
  - `backend_name`
  - `fallback_used`
  - `failure_reason`
  - `vad_voiced_duration_ms`

Files to update:

- `app/models/schemas.py`
- `swift-backend/Sources/swift-backend/VoiceContracts.swift`

#### Implementation Targets in This Repo

- `app/core/config.py`
  - add feature flags and backend selection settings for Silero VAD and Sherpa-ONNX
- `app/services/telephony.py`
  - add frame buffering, VAD gating, and endpoint logic
- `app/services/stt.py`
  - add backend abstraction, Whisper retry policy, and transcript quality gating
- `tests/test_stt.py`
  - add fallback and quality-gate tests
- `tests/test_telephony.py`
  - add VAD segmentation and no-speech drop coverage

#### Decision Recommendation

Sprint 1 production recommendation:

- Ship Silero VAD + Whisper as the default normalized STT path
- Keep Sherpa-ONNX behind a feature flag for fallback and streaming experiments

Why this is the right default for TherFour:

- preserves open-source deployability
- matches the current architecture and codebase maturity
- improves latency where the user actually feels it
- avoids unnecessary double-decode overhead on every call turn
- leaves room for Sherpa-ONNX to add value where it is strongest

#### Acceptance Criteria

- VAD replaces most fixed silence-only turn detection logic
- no-speech turns are rejected before LLM invocation
- Whisper supports at least one fallback decode strategy
- STT result includes explicit backend and failure metadata
- Sherpa-ONNX can be enabled as fallback or partial-transcript path without changing public API shape
- tests cover:
  - speech/no-speech segmentation
  - final-turn trimming
  - Whisper fallback decode
  - session-sticky fallback activation
  - structured STT failure reporting

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
