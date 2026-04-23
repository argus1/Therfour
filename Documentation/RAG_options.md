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

Grounding behavior should remain:
- use retrieved context only when relevant,
- ignore off-topic/duplicate snippets,
- do not expose retrieval scaffolding to callers.

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
