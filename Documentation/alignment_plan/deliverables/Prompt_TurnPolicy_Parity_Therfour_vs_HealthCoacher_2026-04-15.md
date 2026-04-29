# Prompt and Turn-Policy Parity Analysis: Therfour (Current) vs HealthCoacher (Target Model)

Date: 2026-04-15
Author: Engineering analysis

## Scope

This analysis compares prompt architecture and turn-policy behavior between:

- Therfour current implementation
- HealthCoacher target implementation pattern

Focus areas:

- Role boundaries
- System constraints
- Turn orchestration

Primary evidence reviewed:

- Therfour: app/services/llm.py
- Therfour: app/services/telephony.py
- Therfour: swift-backend/Sources/swift-backend/PromptTemplates.swift
- Therfour: swift-backend/Sources/swift-backend/CallTurnProcessor.swift
- Therfour: swift-backend/Sources/swift-backend/OllamaChatService.swift
- Therfour: swift-backend/Sources/swift-backend/VoiceContracts.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/Chat/PromptTemplates.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/Chat/ChatViewModel.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/Chat/TranslationPipeline.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/LLM/CoreMLLLMClient.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/LLM/LMStudioClient.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/LLM/LLMTooling.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/LLM/ReliableLLMClient.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/App/AppContainer.swift

## Executive Summary

Therfour has a clear and useful single-prompt harm-reduction policy with a straightforward STT -> LLM -> TTS turn loop, but it remains monolithic in prompt layering and lightweight in turn governance.

HealthCoacher models prompts and turn policy as a multi-layer control surface: role-scoped system messages, retrieval/tool constraints, language bridge handling, sanitizer enforcement, and a stateful multi-phase orchestration path with fallback behavior.

Net: parity gap is high in orchestration sophistication and medium-high in prompt boundary rigor.

## Parity Matrix

| Dimension          | Therfour current                                                                               | HealthCoacher model                                                                                                                                                                | Gap severity |
| ------------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| Role boundaries    | Single system prompt + user/assistant history; minimal role stratification                     | Layered system-role scaffolding for policy, language guidance, retrieved context, tool transcript, and direct-answer constraints; tool-call role support in compatible backend     | High         |
| System constraints | Harm-reduction constraints are explicit and domain-strong, but mostly content-style directives | Content-style directives plus anti-chain-of-thought, anti-process narration, retrieval relevance policy, and sanitization of scaffold leakage                                      | Medium-High  |
| Turn orchestration | Simple serial turn: buffer audio -> STT -> LLM -> TTS; broad exception catch at turn level     | Stateful multi-phase pipeline with translation prep/finalize, retrieval, generation streaming, tool round-trips, avatar action stage, TTS branching, and cancellation/error policy | High         |

## Deep Dive

### 1) Role Boundaries

Therfour current behavior:

- Python runtime sends one fixed system prompt and conversation messages to Ollama.
- Runtime conversation is a list of user and assistant entries appended per turn.
- No explicit tool role, no structured policy layering, and no distinct system channels for retrieval/tool metadata.
- Swift backend mirrors this pattern: system prompt prepended once per LLM call with user and assistant message history.

HealthCoacher model behavior:

- Prompt assembly is role-layered and compositional:
  - base system policy prompt
  - language guidance
  - optional extra system prompts
  - optional retrieved context as system message
  - optional tool transcript as system message
  - optional direct-answer system guard
  - bounded history + user turn
- Tool planning path uses explicit JSON decision policy and supports tool messages in OpenAI-compatible backend flows.
- Response sanitization removes leaked scaffolding, planning text, tool-process text, and prompt echoes before user-facing output.

Parity implication:

- Therfour role boundaries are simple and readable but do not isolate operational instructions from conversational history.
- HealthCoacher treats role boundaries as a reliability and safety mechanism, not only a formatting preference.

### 2) System Constraints

Therfour current behavior:

- Strong domain safety intent:
  - prioritize caller safety
  - non-judgmental and person-first language
  - concise response length
  - caller-language response
- Constraints are primarily content-policy directives for harm-reduction coaching.
- No explicit constraints against chain-of-thought leakage, tool narration, retrieval-process narration, or policy echo.

HealthCoacher model behavior:

- Includes concise-assistant policy and retrieval-usage constraints.
- Explicitly forbids disclosing internal planning, retrieval strategy, tool-use decisions, and chain-of-thought.
- Explicitly requires user-facing answer only unless user asks for detailed reasoning.
- Language guidance is formalized per supported language.
- Sanitizer enforces constraints post-generation by stripping prompt scaffolding and planning prelude.

Parity implication:

- Therfour has solid domain policy but limited system-level containment against meta-output leakage.
- HealthCoacher has both pre-generation constraints and post-generation enforcement.

### 3) Turn Orchestration

Therfour current behavior:

- Python telephony orchestration:
  - silence-timer based turn trigger
  - STT
  - append user turn
  - LLM response
  - append assistant turn
  - TTS response and stream back
- Turn loop is compact and pragmatic for real-time voice.
- Error handling is broad at turn level, with exception logging and turn termination.
- Conversation history in telephony path currently grows without explicit trim in that class.
- Swift CallTurnProcessor adds clearer policy gates:
  - rejects empty transcription
  - rejects empty LLM output
  - trims history to configured max

HealthCoacher model behavior:

- Turn orchestration is explicit multi-stage policy:
  - optional interruption of active turn on user barge-in
  - translation sandwich prepare step (including history translation where needed)
  - retrieval stage
  - generation stage with streaming and optional tool execution loop
  - output translation finalize step
  - user-facing sanitization and dedupe
  - avatar action inference stage
  - streaming TTS path with synthesis fallback path
  - cancellation and error state routing
- Includes capability probing and backend summaries that influence behavior.
- Includes memory preflight resource eviction before generation.

Parity implication:

- Therfour currently optimizes for simplicity and low orchestration overhead.
- HealthCoacher optimizes for predictable behavior across multiple runtime modes and failure surfaces.

## What Therfour Should Borrow to Reach HealthCoacher-Level Modeling

### A) Prompt layering model (high priority)

Add composable prompt assembly with explicit layers:

- base system policy
- language directive
- optional retrieval context block
- optional operational constraints block
- bounded conversation window

### B) Role boundary hardening (high priority)

Introduce clear role channels for operational data so it does not blend with normal chat history:

- preserve user/assistant history as conversation only
- inject retrieval/tool metadata in dedicated system sections
- support optional tool-role semantics in future-compatible clients

### C) System constraint expansion (medium-high priority)

Add explicit constraints to prevent unintended meta-output:

- do not expose internal reasoning or planning
- do not narrate retrieval/tool process
- return user-facing response only

### D) Post-generation sanitizer (medium priority)

Implement lightweight sanitization for:

- prompt echo
- planning prelude
- system/tool transcript leakage

### E) Turn-policy envelope and state hooks (high priority)

Introduce explicit turn phases and structured failure handling:

- input accepted
- pre-LLM checks
- retrieval (when enabled)
- generation
- output synthesis
- completion/failure state

This can remain lightweight while enabling better observability and safer future growth.

## Proposed Acceptance Criteria for Prompt and Turn-Policy Parity Work Item

1. Therfour prompt builder supports layered system messages instead of a single fixed string.
2. Role boundaries are explicit for conversation content vs operational control metadata.
3. System constraints include no-chain-of-thought and no-process-narration directives.
4. Turn processor has explicit phase transitions with structured errors per phase.
5. Conversation history is bounded consistently across active runtime paths.
6. Response sanitizer removes scaffold/meta leakage from user-visible output.
7. Tests cover:
   - prompt layering order and invariants
   - history bounding behavior
   - empty-turn rejection and failure routing
   - sanitizer behavior for scaffold/planning leakage

## Risks if No Change is Made

- Prompt and operational metadata may become tightly coupled as features grow.
- Increased risk of policy leakage or process narration in user-facing output.
- Harder to reason about turn failures as additional stages are added.
- Larger refactor cost when introducing retrieval/tooling and multilingual turn policies later.

## Recommendation

Treat this parity area as policy architecture work, not only prompt wording changes.

HealthCoacher demonstrates the target pattern:

- explicit role-layered prompt construction
- stronger system constraints with post-output enforcement
- multi-phase turn orchestration with controlled fallback paths

Therfour should adopt this incrementally, starting with layered prompt assembly and explicit turn-phase structure while preserving current telephony responsiveness.
