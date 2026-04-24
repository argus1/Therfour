# RAG Strategy Lock (Sprint 1)

Date: 2026-04-21 (revised 2026-04-23)
Scope: Query construction, retrieval defaults, context packing rules, context window budget, grounded response format, and multi-pass hierarchical structure for TherFour parity work.

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

### 7) Context Packing Rules (Locked)

These rules govern how retrieved chunks are assembled into the prompt section before generation.

**Pack order**

- Sort filtered contexts descending by `score`.
- Never reorder by insertion index or source — score order is canonical.

**Token budget**

- The total context section (all packed chunks combined, including headers) must not exceed `context_section_token_budget` tokens.
- Default lock: **600 tokens** for the context section. This reserves ≥ 1 k tokens for the system prompt base and ≥ 2 k tokens for conversation history + generation headroom in a 4 k effective window.
- Per-chunk ingest target: **200 tokens per chunk** (`context_per_chunk_token_target`). Chunks above 300 tokens at ingest time should be re-split.

**Inclusion gate**

- Include a context only when all of these hold:
  1. `score >= similarity_threshold` (default 0.35)
  2. Chunk is not a near-duplicate of a higher-ranked chunk already packed (cosine > 0.92 after score-ordering is a dedup signal; discard lower-ranked duplicate)
  3. Budget headroom remains before appending this chunk

**Format**

- Each packed context block renders as:
  ```
  [<index> | source: <source> | score: <score>]
  <chunk text>
  ```
- Indexes are 1-based and stable within the assembled block.
- Do not emit empty blocks — omit the entire context section when zero chunks survive the inclusion gate.

**What is not packed**

- Retrieval metadata beyond `source` and `score` (e.g. `chunk_id`, `embedding_model`) must not appear in the prompt context section; those fields are observability-only.
- Full conversation history is never re-embedded and never injected into the context block (history lives in the `messages` list, not the system prompt).

### 8) Context Window Usage Budget (Locked)

Effective context window budget for a Qwen3.5-35B-A3B at q2_k_xl quantization, targeting **4 096 token effective window** (conservative; actual model max is larger but telephony latency penalizes long contexts).

| Region                       | Token budget | Notes                                            |
| ---------------------------- | -----------: | ------------------------------------------------ |
| System prompt base           |          400 | `HARM_REDUCTION_SYSTEM_PROMPT` + grounding rules |
| RAG context section          |          600 | Max packed chunks (§7 above)                     |
| Conversation history (turns) |         1500 | Rolling window; trim oldest if over budget       |
| Current user utterance       |          512 | `query_max_tokens` cap from query builder        |
| Generation headroom (output) |         1084 | Remaining space for response tokens              |
| **Total**                    |    **4 096** |                                                  |

Rules:

- If conversation history would push total input over 4 096 − 1 084 = 3 012 tokens, trim the oldest user+assistant turn pairs until it fits.
- If RAG context section exceeds 600 tokens after packing, truncate the lowest-scored chunk partially (at sentence boundary) or drop it to respect the budget.
- These budgets are expressed as config fields so they can be tuned without code changes (see §Locked Defaults table).

### 9) Grounded Response Format (Locked)

These rules govern the shape of the assistant response when retrieved context is present.

**Required behaviors**

- State conclusions derived from retrieved context in plain language; do not restate chunk text verbatim unless the caller explicitly requests a quote.
- When context directly answers the query, lead with the answer, not with a preamble about what was retrieved.
- When context partially answers the query, answer what is covered and say briefly what is not covered — do not fabricate to fill gaps.
- When context is present but off-topic, ignore it silently; respond from general harm-reduction knowledge without mentioning retrieval.

**Prohibited in caller-facing output**

- Source labels (`source: ...`), retrieval scores, `chunk_id`, index numbers from the context block.
- Process narration: phrases like "Based on the retrieved context…", "According to document X…", "The search returned…".
- Internal planning text, prompt echoes, or tool-call transcripts.
- Uncertainty preambles unless genuinely warranted (e.g. "I think maybe…" when answer is retrievable).

**Sanitization gate (AP-06)**

- A response sanitizer must be applied before audio synthesis.
- Sanitizer checks for scaffold leakage patterns from the prohibited list above and removes them from the response string.
- Sanitizer should preserve natural sentence boundaries; do not truncate mid-sentence.

**Telephony style constraints (carry-forward)**

- Keep responses to 2–3 sentences for most turns. Safety-critical information (overdose, emergency escalation) may extend to 4 sentences.
- Respond in the caller's language — this applies equally when retrieval context is present.

### 10) Multi-Pass Retrieval and Hierarchical Vector DB Structure (Locked)

This section locks behavior for the `hierarchical` strategy and multi-pass execution path described in `RAG_options.md`.

#### Pass 1: Category Routing

- **Purpose:** Map the normalized query to a corpus partition before retrieval.
- **Method progression:** `keyword` → `keyword_then_llm` → `llm` (from least to most expensive; select based on corpus complexity and latency budget).
- **Default lock for Sprint 1:** `keyword_then_llm` for any corpus with ≥ 4 defined categories; `keyword` when ≤ 3 categories.
- **LLM categorizer constraints:**
  - Model: `qwen3.5-35b-a3b:q2-k-xl` (same runtime model as generation — no additional model deployment required).
  - Timeout: 10 s hard cap (`llm.timeout_s`). On timeout, fall back to `default_category` immediately.
  - `strict: true` is required in production — unrecognized model output must default to `default_category`, not raise.
  - Categorizer token budget: keep the routing prompt under 200 tokens (query + category list). Do not embed full conversation history.
- **Category boundary rules:**
  - Each category partition must map to a single topically-bounded corpus; no cross-partition document overlap.
  - Locked corpus partitions for this domain: `overdose`, `opioids`, `stimulants`, `general`.
  - `general` is the mandatory catch-all; it must always be present in the config.

#### Pass 2: Category-Scoped Retrieval

- Execute vector search against the category-specific `chroma_path` + `collection_name`.
- Apply standard retrieval parameters (`top_k`, `similarity_threshold`, `top_k_final`) as defined in §3.
- If the category-scoped query returns zero results and `fallback_to_standard_on_miss=true`, execute a second query against the standard (`general`) collection.
- Log the fallback event (`selected_category` will show as `<original_category>->standard` in observability).

#### Pass 3: Standard Fallback (Conditional)

- Runs only when Pass 2 returns an empty filtered set and `fallback_to_standard_on_miss=true`.
- Use the same normalized query; do not re-normalize.
- Apply the same threshold and budget filters.
- The context block assembled from fallback results is treated identically to a direct hit — no special annotation in the prompt.

#### Multi-Pass Latency Contract

- Pass 1 (keyword): < 2 ms (in-process).
- Pass 1 (LLM categorizer): < 10 s hard cap, target < 3 s.
- Pass 2 (vector search): target < 300 ms; alert threshold > 800 ms.
- Pass 3 (fallback search, if triggered): target < 300 ms additional.
- Total pre-generation overhead target: **< 500 ms** for keyword routing; **< 4 s** for LLM-routed turns.

#### Hierarchical DB Maintenance Rules

- Each partition's `chroma_path` must be provisioned and validated before enabling `strategy: hierarchical`.
- Validation checklist (per partition): path exists, collection named in config is present, at least one document indexed, sample query returns a result.
- Adding a new category requires: corpus ingested + `rag_config.json` category entry + restart — no dynamic registration at runtime.
- `rag_config.json` changes require a process restart (config is cached via `lru_cache`; cache is not invalidated on file change).

## Locked Defaults (Configurable)

| Setting                          |     Locked Default | Notes                                                     |
| -------------------------------- | -----------------: | --------------------------------------------------------- |
| `retrieval_top_k`                |                  5 | Candidate pool before threshold filter                    |
| `top_k_final`                    |                  3 | Max contexts packed into prompt                           |
| `similarity_threshold`           |               0.35 | Cosine floor for inclusion                                |
| `query_max_tokens`               |                512 | Pre-embedding guardrail                                   |
| `context_per_chunk_token_target` |                200 | Ingest/packing target per chunk                           |
| `context_section_token_budget`   |                600 | Max total tokens for packed RAG context in prompt         |
| `conversation_history_token_cap` |               1500 | Rolling history window before oldest-turn trimming        |
| `effective_context_window`       |               4096 | Total input window assumed for budget math                |
| `dedup_cosine_threshold`         |               0.92 | Cosine above which a lower-ranked chunk is treated as dup |
| `categorizer_method`             | `keyword_then_llm` | For corpora with ≥ 4 categories; `keyword` for ≤ 3        |
| `categorizer_llm_timeout_s`      |                 10 | Hard timeout for LLM categorizer; falls back to default   |

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
- Context section total tokens stay within `context_section_token_budget` (default 600).
- Conversation history trimming activates when history exceeds `conversation_history_token_cap`.
- Near-duplicate chunks are deduplicated before packing (cosine gate).
- Sanitizer strips source labels, scores, and process narration before TTS synthesis.
- Hierarchical: Pass 1 keyword routing resolves in < 2 ms; LLM categorizer respects 10 s timeout with default-category fallback.
- Hierarchical: Pass 2 fallback to standard collection is logged with `<category>->standard` marker.
- Hierarchical: All category partitions validated (path exists, collection present, sample query returns result) before enabling `strategy: hierarchical`.
- Multi-pass latency stays within pre-generation overhead targets (< 500 ms keyword; < 4 s LLM-routed).
- `rag_config.json` changes require restart (cached config behavior documented and tested).
