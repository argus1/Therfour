# RAG Parity Analysis: Therfour (Current) vs HealthCoacher (Target Model)

Date: 2026-04-15
Author: Engineering analysis

## Scope

This analysis compares retrieval-augmented generation behavior between:

- Therfour current implementation
- HealthCoacher target implementation pattern

Focus areas:

- Retrieval Flow
- Chunking Assumptions
- Grounding and Citation Behavior
- Answer Style

Primary evidence reviewed:

- Therfour: app/services/llm.py
- Therfour: app/services/telephony.py
- Therfour: tests/test_llm.py
- Therfour: swift-backend/Sources/swift-backend/PromptTemplates.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/RAG/RAGEngine.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/RAG/RAGWaxAssetBootstrapper.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/RAG/RAGCoreMLAssetBootstrapper.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/Chat/ChatViewModel.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/Chat/PromptTemplates.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/Chat/TranslationPipeline.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/LLM/CoreMLLLMClient.swift
- HealthCoacher: ios-avatar-rag-prototype/iOSApp/Sources/LLM/LMStudioClient.swift
- HealthCoacher: ios-avatar-rag-prototype/tools/ingest_doclib_docling_chroma.py
- HealthCoacher: ios-avatar-rag-prototype/tools/export_chroma_for_wax.py
- HealthCoacher: ios-avatar-rag-prototype/Docs/CHROMA_TO_WAX_REINDEX_WORKFLOW.md

## Executive Summary

Therfour currently does not implement an explicit RAG pipeline in runtime. It is prompt-only generation with conversational history, without retrieval, chunk indexing, retrieval scoring, or citation scaffolding.

HealthCoacher implements a concrete RAG architecture with retrieval limits, chunk-level scoring, ingestion assumptions, and explicit prompt injection of retrieved context with source and score metadata.

Net: parity gap is high and architectural. Therfour is currently at pre-RAG baseline, while HealthCoacher is already at a bounded and instrumented RAG pattern.

## Parity Matrix

| Dimension               | Therfour current                                                                         | HealthCoacher model                                                                                                               | Gap severity |
| ----------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| Retrieval Flow          | No retrieval step; LLM consumes system prompt plus conversation only                     | Explicit retrieve step before generation using query embeddings and top-k context selection                                       | High         |
| Chunking Assumptions    | No runtime or ingest chunking assumptions defined for RAG                                | Ingest supports Docling intelligent chunking, with fallback windowed chunking and preserved chunk boundaries from Chroma into Wax | High         |
| Grounding and Citations | No retrieved context block, no source metadata injection, no citation policy             | Retrieved snippets are formatted as indexed blocks with source and score, injected as system context with relevance rules         | High         |
| Answer Style            | Harm-reduction style prompt (concise, empathetic) but no retrieval-grounding constraints | Concise style plus explicit anti-hallucination and relevance instructions tied to retrieved context usage                         | Medium-High  |

## Deep Dive

### 1) Retrieval Flow

Therfour current behavior:

- app/services/telephony.py calls STT then directly calls llm.generate with conversation history.
- app/services/llm.py sends system prompt plus messages to Ollama.
- No retrieval service, vector index, embedding call, top-k filtering, or retrieval telemetry in runtime.

HealthCoacher model behavior:

- ChatViewModel executes an explicit retrieval stage before LLM generation.
- RAGEngine retrieves contexts via embedding similarity plus lexical coverage blending.
- Retrieval is bounded by top-k and candidate filtering thresholds.
- Contexts are passed to the LLM client for prompt assembly.
- Retrieval latency and retrieved-count telemetry are recorded.

Parity implication:

- Therfour currently has no retrieval-grounding chain.
- HealthCoacher has a deterministic retrieve-then-generate flow with bounded context payload.

### 2) Chunking Assumptions

Therfour current behavior:

- No active document chunking strategy in runtime because RAG is not implemented.
- Alignment plan references Chroma direct path as desired direction, but implementation is not present yet.

HealthCoacher model behavior:

- Runtime prototype chunking in RAGEngine.addDocument is coarse sentence split by period.
- Production ingest tooling uses Docling intelligent chunking where available.
- Fallback ingest path uses normalized text windows with explicit max length and overlap.
- Export flow preserves chunk text and embeddings from Chroma into Wax payloads without rechunking.
- Wax frame map preserves traceability to original chunk ids and sources.

Parity implication:

- Therfour has no enforced chunk contract.
- HealthCoacher already has chunk provenance, chunker metadata, and conversion workflow assumptions.

### 3) Grounding and Citation Behavior

Therfour current behavior:

- No retrieved context payload is injected into prompt.
- No source labels or retrieval scores are provided to the model.
- No citation behavior policy exists in runtime prompt.

HealthCoacher model behavior:

- LLM prompt includes retrieved context as structured blocks:
  - index number
  - source name
  - relevance score
  - chunk text
- Prompt rules constrain grounding behavior:
  - use retrieved context only when relevant
  - ignore off-topic or duplicated snippets
  - avoid verbatim repetition unless asked
  - if context is insufficient, say so briefly and answer from general knowledge
- Output sanitization strips scaffold and planning artifacts from user-facing responses.

Parity implication:

- Therfour cannot provide grounded traceability today because retrieval artifacts are absent.
- HealthCoacher provides grounding metadata in prompt, but does not require strict user-visible citation syntax in final answer.

### 4) Answer Style

Therfour current behavior:

- Strong domain style guidance exists for harm-reduction phone support.
- Constraints include concise response length and caller-language response.
- Style is independent of retrieval because no RAG context is present.

HealthCoacher model behavior:

- Prompt style is concise and language-aware.
- Additional answer-style controls are retrieval-aware:
  - no internal planning narration
  - no retrieval process narration
  - direct user-facing answer only
- Sanitizer removes planning prelude, prompt echo, tool transcript echoes, and internal scaffolding.

Parity implication:

- Therfour has domain style but lacks retrieval-aware answer governance.
- HealthCoacher combines style guidance with retrieval and sanitization controls.

## What Therfour Should Model from HealthCoacher

### A) Introduce explicit retrieval stage and context contract (high priority)

Implement a retrieval step before generation:

- compute query embedding
- retrieve top-k contexts
- include source and score with each context
- pass contexts as dedicated prompt section

### B) Define chunking and provenance contract (high priority)

Adopt chunk metadata fields now, even before full production index:

- chunk id
- source
- chunk index
- chunker strategy
- embedding model

This avoids migration friction later when moving from prototype chunks to Chroma or Wax-backed assets.

### C) Add grounding guardrails in system prompt (high priority)

Extend Therfour system instructions to include:

- use retrieved context only when directly relevant
- ignore duplicated or off-topic retrieved snippets
- declare insufficient context briefly when needed
- avoid verbatim copying unless user requests quote

### D) Add answer sanitization and anti-scaffold cleanup (medium priority)

Implement lightweight reply sanitation for:

- prompt echoes
- planning text
- system/tool scaffolding leakage

### E) Add retrieval observability (medium priority)

Capture retrieval metrics per turn:

- retrieval latency
- top-k returned
- source diversity
- low-relevance or empty-retrieval counts

## Proposed Acceptance Criteria for RAG Parity Work Item

1. Therfour runtime includes retrieve-then-generate flow with explicit context objects.
2. Retrieved context includes source and relevance score in prompt assembly.
3. Top-k retrieval defaults and thresholds are configurable.
4. Chunk metadata contract exists and is test-covered.
5. Prompt includes grounding rules for context relevance and insufficiency handling.
6. Response sanitization prevents retrieval/process scaffolding from leaking to callers.
7. Tests cover:
   - retrieval hit path
   - empty retrieval path
   - low-relevance filtering path
   - prompt assembly with context metadata
   - grounded answer behavior under mixed relevant/off-topic contexts

## Risks if No Change is Made

- Answers remain ungrounded relative to knowledge corpus.
- No citation or provenance path for QA and clinical review.
- Harder to control hallucination under domain-sensitive harm-reduction scenarios.
- Larger future migration cost when introducing retrieval late.

## Recommendation

Treat RAG parity as an architecture addition, not a prompt tweak.

HealthCoacher demonstrates the target model:

- explicit retrieval orchestration
- chunking and index provenance assumptions
- grounding-aware prompt construction
- answer-style controls that prevent scaffolding leakage

Therfour should first establish the retrieval contract and context injection path, then iterate on chunking quality and grounding evaluation sets.
