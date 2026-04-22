# RAG Strategy Lock (Sprint 1)

Date: 2026-04-21  
Scope: Query construction, retrieval defaults, and grounding behavior for TherFour parity work.

## Decision Summary

### 1) Retrieval Flow

- Lock: Implement explicit retrieve-then-generate before LLM response creation.
- Rationale: TherFour is currently prompt-only; parity target requires deterministic context retrieval and bounded grounding.

### 2) Query Construction Standard

- Lock: Build embedding query from current caller utterance (`STT text`) as default.
- Exception: If the utterance is referential (for example: "that", "tell me more"), prepend one prior assistant turn for minimal disambiguation.
- Normalization (pre-embedding): lowercase, trim whitespace/filler cleanup, max-length cap.
- Rationale: Keeps retrieval topical to current intent while avoiding drift from full-history embedding.

### 3) Retrieval Defaults

- `retrieval_top_k`: 5 candidates (initial recall window)
- `top_k_final`: 3 contexts after filtering (prompt packing set)
- `similarity_threshold`: 0.35 cosine minimum relevance
- Rationale:
  - `top_k=5` improves recall/diversity before filtering.
  - `top_k_final=3` balances grounding quality with turn latency and prompt budget.
  - `threshold=0.35` filters weak/off-topic matches while preserving paraphrase-level recall in a domain corpus.

### 4) Context Packing Contract

- Lock: Inject retrieved context as indexed blocks with `source` and `score` metadata.
- Order: Descending by score.
- Budget: Pack only filtered `top_k_final` contexts.
- Rationale: Reproducible assembly and better QA traceability without overloading short phone-turn responses.

### 5) Grounding Rules (Prompt Policy)

- Use retrieved context only when directly relevant.
- Ignore duplicated/off-topic snippets.
- If retrieval is insufficient, give a brief direct answer without process narration.
- Do not echo retrieval scaffolding (`source`, `score`, or internal formatting) in user-facing output.
- Rationale: Prevents retrieval leakage/hallucinated provenance and preserves concise telephony style.

### 6) Runtime Store Direction

- Lock: ChromaDB direct for Sprint 1 online retrieval.
- Defer: WAX conversion as optional export/offline artifact, not runtime dependency.
- Rationale: Lowest complexity path for parity delivery and retrieval tuning.

## Locked Defaults (Configurable)

| Setting                          | Locked Default | Notes                                  |
| -------------------------------- | -------------: | -------------------------------------- |
| `retrieval_top_k`                |              5 | Candidate pool before threshold filter |
| `top_k_final`                    |              3 | Max contexts packed into prompt        |
| `similarity_threshold`           |           0.35 | Cosine floor for inclusion             |
| `query_max_tokens`               |            512 | Pre-embedding guardrail                |
| `context_per_chunk_token_target` |            200 | Ingest/packing target                  |

## Required Data Contract (Chunk/Context Metadata)

- `chunk_id` (stable id)
- `source` (document/source path or label)
- `chunk_index` (position within source)
- `chunker_strategy` (`docling` or `windowed`)
- `embedding_model` (name/version)
- `score` (query-time similarity)

## Retrieval Outcome Policies

- Hit path: include filtered contexts and apply grounding rules.
- Empty path: no context block, answer directly and concisely.
- Low-relevance path: treat as empty when all scores are below threshold.

## Action Point Stubs

### AP-01: Add Retrieval Config Defaults

- Owner: TBD
- Status: Not Started
- Target Date: TBD
- Scope:
  - Add `retrieval_top_k`, `top_k_final`, `similarity_threshold`, `query_max_tokens` to app config.
  - Wire environment-variable overrides.
- Definition of Done:
  - Defaults active at runtime.
  - Unit tests cover config load and override behavior.

### AP-02: Implement Query Builder

- Owner: TBD
- Status: Not Started
- Target Date: TBD
- Scope:
  - Build normalized query from STT text.
  - Add one-turn disambiguation fallback for referential utterances.
  - Enforce max token cap.
- Definition of Done:
  - Query builder invoked in retrieval path.
  - Tests cover normal, referential, and truncation cases.

### AP-03: Implement Retrieval Orchestrator

- Owner: TBD
- Status: Not Started
- Target Date: TBD
- Scope:
  - Execute vector search with `top_k`.
  - Apply threshold filter.
  - Keep top `top_k_final` contexts.
- Definition of Done:
  - Deterministic retrieve-then-generate sequence.
  - Tests cover hit, empty, and low-relevance paths.

### AP-04: Add Prompt Context Packing

- Owner: TBD
- Status: Not Started
- Target Date: TBD
- Scope:
  - Inject indexed context blocks with source/score metadata.
  - Sort by score descending.
  - Respect prompt budget.
- Definition of Done:
  - Prompt assembly includes contexts only when valid.
  - Tests verify format and max packed contexts.

### AP-05: Add Grounding Guardrails

- Owner: TBD
- Status: Not Started
- Target Date: TBD
- Scope:
  - Extend system prompt rules for relevance, insufficiency, and no-scaffold leakage.
- Definition of Done:
  - Prompt policy updated and covered by snapshot/assertion tests.

### AP-06: Add Response Sanitization

- Owner: TBD
- Status: Not Started
- Target Date: TBD
- Scope:
  - Strip retrieval/process scaffolding leakage from user-facing response.
- Definition of Done:
  - Sanitizer active in response pipeline.
  - Tests include scaffold leakage samples.

### AP-07: Define Metadata Contract Types

- Owner: TBD
- Status: Not Started
- Target Date: TBD
- Scope:
  - Add schema types for retrieved context and metadata fields.
- Definition of Done:
  - Contract shared across retrieval and prompt assembly.
  - Schema validation tests added.

### AP-08: Add Retrieval Observability

- Owner: TBD
- Status: Not Started
- Target Date: TBD
- Scope:
  - Log/metric capture for retrieval latency, returned count, filtered count, source diversity.
- Definition of Done:
  - Metrics visible in existing observability path.
  - Smoke validation confirms events emitted.

## Validation Checklist

- Retrieve-then-generate path exists.
- Context includes source + score metadata.
- Defaults are configurable and tested.
- Grounding behavior validated on mixed relevant/off-topic contexts.
- Empty and low-relevance retrieval behaviors are deterministic.
- No retrieval scaffold leaks in final caller response.
