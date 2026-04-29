# RAG Options Configuration Guide

This project supports two RAG strategies selected by configuration:

- `standard`: query a single Chroma database.
- `hierarchical`: run a categorization pass, then query a category-specific Chroma database from a partitioned corpus.

Inside `hierarchical`, categorization now supports three modes:

- `keyword`: keyword-only category routing.
- `llm`: LLM-only category routing.
- `keyword_then_llm`: keyword first, then LLM fallback if keyword matching has no hits.

The runtime config file is:

- `app/core/rag_config.json`

The app-level enable/switch env fields are in:

- `app/core/config.py`

## 1) Enable RAG Runtime

RAG is controlled by `RAG_ENABLED`.

Example `.env` values:

```env
RAG_ENABLED=true
RAG_CONFIG_PATH=app/core/rag_config.json
```

If `RAG_ENABLED=false`, the app stays prompt-only (current baseline behavior).

## 2) Config File Structure

```json
{
  "version": 1,
  "strategy": "standard",
  "retrieval": {
    "top_k": 5,
    "top_k_final": 3,
    "similarity_threshold": 0.35,
    "min_query_chars": 3
  },
  "standard": {
    "chroma_path": "data/chroma/default",
    "collection_name": "therfour_docs"
  },
  "hierarchical": {
    "fallback_to_standard_on_miss": true,
    "categorizer": {
      "method": "keyword",
      "default_category": "general",
      "llm": {
        "model": "qwen3.5-35b-a3b:q2-k-xl",
        "timeout_s": 10,
        "strict": true
      },
      "categories": [
        {
          "name": "opioids",
          "keywords": ["fentanyl", "opioid", "heroin"],
          "chroma_path": "data/chroma/opioids",
          "collection_name": "opioid_docs"
        }
      ]
    }
  },
  "prompt": {
    "include_scores": true,
    "include_sources": true,
    "section_title": "Retrieved context"
  },
  "observability": {
    "log_selected_category": true,
    "log_candidate_count": true,
    "log_filtered_count": true,
    "log_retrieval_latency_ms": true
  }
}
```

## 3) How Strategy Selection Works

- `strategy = "standard"`
  - Query goes directly to `standard.chroma_path` + `standard.collection_name`.

- `strategy = "hierarchical"`
  - First pass categorizes query using `hierarchical.categorizer`.
  - Categorizer `method` can be:
    - `keyword` for deterministic keyword routing.
    - `llm` for model-based routing.
    - `keyword_then_llm` as a third-pass option: use keyword first and invoke LLM only when keyword hits are zero.
  - The selected category maps to its own `chroma_path` and `collection_name`.
  - If no documents are returned and `fallback_to_standard_on_miss=true`, runtime retries the standard DB.

### LLM Categorizer Settings

- `hierarchical.categorizer.llm.model`: model used for category selection.
- `hierarchical.categorizer.llm.timeout_s`: timeout for the categorizer call.
- `hierarchical.categorizer.llm.strict`:
  - `true`: unknown model output falls back to default category.
  - `false`: allows non-listed output (not recommended for production routing).

## 4) Category Design for Partitioned Corpora

Each category should map to a corpus partition with consistent topical boundaries.

Required fields per category:

- `name`
- `keywords` (for keyword categorizer)
- `chroma_path`
- `collection_name`

Recommended partitions for this domain:

- `overdose`
- `opioids`
- `stimulants`
- `general`

## 5) Retrieval Tuning Defaults

- `top_k`: candidate pool size before thresholding.
- `top_k_final`: max contexts sent to prompt after filtering.
- `similarity_threshold`: minimum relevance score.
- `min_query_chars`: skip retrieval for too-short queries.

Initial defaults (aligned with lock guidance):

- `top_k=5`
- `top_k_final=3`
- `similarity_threshold=0.35`

## 6) Prompt Packing and Grounding

Retrieved chunks are packed as indexed blocks and can include source and score metadata. This is controlled by `prompt.include_sources` and `prompt.include_scores`.

### Packing Rules (locked)

- Chunks are sorted descending by `score` before packing.
- Near-duplicate chunks (cosine > `dedup_cosine_threshold`, default 0.92 after score-ordering) are dropped — only the highest-scored copy is retained.
- The packed context section must not exceed `context_section_token_budget` (default 600 tokens). If the last chunk would push past the budget, truncate it at a sentence boundary or drop it.
- Only `top_k_final` chunks (default 3) survive threshold filtering and are packed.
- The context section is omitted entirely when zero chunks survive the inclusion gate — the LLM receives no context block.

### Context Window Usage

A fixed per-turn token budget applies across all regions (see lock document §8):

| Region                         | Budget (tokens) |
| ------------------------------ | --------------: |
| System prompt base             |             400 |
| RAG context section            |             600 |
| Conversation history (rolling) |            1500 |
| Current utterance              |             512 |
| Generation headroom            |            1084 |
| **Total**                      |        **4096** |

Conversation history is trimmed (oldest pairs first) when it would exceed `conversation_history_token_cap` (default 1500).

### Grounding Behavior (locked)

- Use retrieved context only when it is directly relevant to the caller question.
- Ignore off-topic or duplicate retrieved snippets.
- If context is missing or insufficient, answer briefly without mentioning retrieval internals.
- Do not expose source labels, scores, or retrieval scaffolding (`[1 | source: … | score: …]`) in final caller responses.
- Lead with the answer, not with narration about retrieval.
- A response sanitizer (AP-06) must strip scaffold leakage before audio synthesis.

Full lock rationale and acceptance criteria are in `Documentation/alignment_plan/deliverables/RAG_lock.md` §§7–9.

## 7) Operational Gaps the Config Now Covers

The new config fills key gaps that were previously implicit or missing:

- Explicit strategy switch (`standard` vs `hierarchical`).
- Partition routing map from category to Chroma path and collection.
- Categorizer configuration and fallback behavior.
- Retrieval threshold and result-limiting controls.
- Prompt metadata inclusion controls.
- Retrieval observability toggles.

## 8) Example: Switch to Hierarchical Mode

1. Set strategy:

```json
"strategy": "hierarchical"
```

Optional: choose categorizer mode:

```json
"method": "keyword_then_llm"
```

2. Confirm each category has a valid `chroma_path` and `collection_name`.

3. Enable runtime:

```env
RAG_ENABLED=true
```

4. Restart API process.

## 9) Validation Checklist

- `RAG_ENABLED=true` in env.
- `RAG_CONFIG_PATH` points to a readable JSON file.
- Each configured Chroma path exists and has the collection named in config.
- Logs show selected strategy/category and retrieval counts.
- Responses do not leak context scaffolding to callers.

## 10) Multi-Pass Behavior and Hierarchical Latency Contracts

When `strategy = "hierarchical"`, retrieval runs in up to three sequential passes. Each pass has a locked behavior and latency target.

### Pass 1: Category Routing

| Categorizer method | Expected latency                             | Fallback behavior on failure     |
| ------------------ | -------------------------------------------- | -------------------------------- |
| `keyword`          | < 2 ms                                       | none — deterministic             |
| `llm`              | < 10 s (hard cap)                            | timeout → `default_category`     |
| `keyword_then_llm` | < 2 ms if keyword hits; < 10 s if LLM needed | LLM timeout → `default_category` |

- **Locked default method for ≥ 4 categories:** `keyword_then_llm`
- **Locked default method for ≤ 3 categories:** `keyword`
- The LLM categorizer uses a compact routing prompt (< 200 tokens). Full conversation history is never included.
- `strict: true` is required — unrecognized LLM output falls back to `default_category` silently.

### Pass 2: Category-Scoped Retrieval

- Executes standard vector search against the routed partition.
- Applies `top_k`, `similarity_threshold`, `top_k_final` as configured.
- Target latency: < 300 ms. Alert threshold: > 800 ms.

### Pass 3: Standard Fallback (Conditional)

- Runs only when Pass 2 filtered set is empty AND `fallback_to_standard_on_miss: true`.
- Logs `selected_category` as `<original_category>->standard`.
- Same query, same threshold, same budget — no special treatment in prompt assembly.
- Target additional latency: < 300 ms.

### Total Pre-Generation Overhead Targets

| Routing method | Target total |
| -------------- | ------------ |
| Keyword only   | < 500 ms     |
| LLM-routed     | < 4 s        |

### Hierarchical DB Provisioning Rules

- All category partitions must pass the validation checklist before enabling `strategy: hierarchical`.
- Adding a new category requires: corpus ingested + `rag_config.json` updated + process restart.
- `rag_config.json` is cached via `lru_cache`; file edits are not hot-reloaded — restart required.
- See lock document §10 for full provisioning rules.
