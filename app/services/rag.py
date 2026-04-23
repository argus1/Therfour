"""RAG retrieval service supporting standard and hierarchical Chroma routing."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    text: str
    source: str
    score: float
    category: str


@dataclass
class RetrievalResult:
    contexts: list[RetrievedContext]
    strategy_used: str
    selected_category: str
    candidate_count: int
    filtered_count: int
    latency_ms: int


def retrieve(query: str) -> RetrievalResult:
    """Run retrieval for *query* according to rag_config.json strategy settings."""
    start = time.perf_counter()
    config = _load_config(settings.rag_config_path)
    retrieval_cfg = config.get("retrieval", {})
    top_k = int(retrieval_cfg.get("top_k", 5))
    top_k_final = int(retrieval_cfg.get("top_k_final", 3))
    similarity_threshold = float(retrieval_cfg.get("similarity_threshold", 0.35))
    min_query_chars = int(retrieval_cfg.get("min_query_chars", 3))

    if len(query.strip()) < min_query_chars:
        return _empty_result("none", "none", start)

    strategy = str(config.get("strategy", "standard")).lower()
    normalized = _normalize_query(query)

    contexts: list[RetrievedContext]
    selected_category = "none"
    if strategy == "hierarchical":
        category, route = _select_hierarchical_route(normalized, config)
        selected_category = category
        contexts = _query_route(normalized, route, category, top_k)
        if not contexts and config.get("hierarchical", {}).get("fallback_to_standard_on_miss", True):
            std = config.get("standard", {})
            contexts = _query_route(normalized, std, "standard", top_k)
            selected_category = f"{selected_category}->standard"
    else:
        strategy = "standard"
        std = config.get("standard", {})
        selected_category = "standard"
        contexts = _query_route(normalized, std, "standard", top_k)

    candidate_count = len(contexts)
    filtered = [c for c in contexts if c.score >= similarity_threshold]
    filtered.sort(key=lambda c: c.score, reverse=True)
    final_contexts = filtered[:top_k_final]

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    _log_metrics(config, strategy, selected_category, candidate_count, len(final_contexts), elapsed_ms)

    return RetrievalResult(
        contexts=final_contexts,
        strategy_used=strategy,
        selected_category=selected_category,
        candidate_count=candidate_count,
        filtered_count=len(final_contexts),
        latency_ms=elapsed_ms,
    )


def build_context_block(contexts: list[RetrievedContext]) -> str:
    """Render retrieved contexts into a deterministic prompt section."""
    if not contexts:
        return ""

    config = _load_config(settings.rag_config_path)
    prompt_cfg = config.get("prompt", {})
    section_title = str(prompt_cfg.get("section_title", "Retrieved context"))
    include_scores = bool(prompt_cfg.get("include_scores", True))
    include_sources = bool(prompt_cfg.get("include_sources", True))

    lines = [f"{section_title} (use only when directly relevant):", ""]
    for idx, ctx in enumerate(contexts, start=1):
        parts = [str(idx)]
        if include_sources:
            parts.append(f"source: {ctx.source}")
        if include_scores:
            parts.append(f"score: {ctx.score:.2f}")
        lines.append(f"[{ ' | '.join(parts) }]")
        lines.append(ctx.text)
        lines.append("")
    return "\n".join(lines).strip()


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


@lru_cache(maxsize=4)
def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_hierarchical_route(query: str, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    hierarchical = config.get("hierarchical", {})
    categorizer = hierarchical.get("categorizer", {})
    method = str(categorizer.get("method", "keyword")).lower()
    categories = categorizer.get("categories", [])
    default_category = str(categorizer.get("default_category", "general"))

    if method == "keyword":
        best_name, _ = _select_keyword_category(query, categories, default_category)
        return _route_for_category(best_name, categories)

    if method == "llm":
        llm_name = _select_llm_category(query, categories, default_category, categorizer)
        return _route_for_category(llm_name, categories)

    if method == "keyword_then_llm":
        best_name, best_hits = _select_keyword_category(query, categories, default_category)
        if best_hits > 0:
            return _route_for_category(best_name, categories)
        llm_name = _select_llm_category(query, categories, default_category, categorizer)
        return _route_for_category(llm_name, categories)

    logger.warning("Unknown categorizer method '%s'; using default category", method)
    return _route_for_category(default_category, categories)


def _select_keyword_category(
    query: str,
    categories: list[dict[str, Any]],
    default_category: str,
) -> tuple[str, int]:
    best_name = default_category
    best_score = -1
    for category in categories:
        keywords = [str(k).lower() for k in category.get("keywords", [])]
        hits = sum(1 for kw in keywords if kw and kw in query)
        if hits > best_score:
            best_score = hits
            best_name = str(category.get("name", default_category))
    return best_name, best_score


def _select_llm_category(
    query: str,
    categories: list[dict[str, Any]],
    default_category: str,
    categorizer_cfg: dict[str, Any],
) -> str:
    category_names = [str(c.get("name", "")).strip() for c in categories if str(c.get("name", "")).strip()]
    if not category_names:
        return default_category

    llm_cfg = categorizer_cfg.get("llm", {})
    timeout_s = float(llm_cfg.get("timeout_s", settings.ollama_timeout))
    model = str(llm_cfg.get("model", settings.ollama_model))
    strict = bool(llm_cfg.get("strict", True))

    selection_prompt = (
        "You are a routing classifier for a retrieval system. "
        "Return exactly one category name from the allowed list, and nothing else.\n\n"
        f"Allowed categories: {', '.join(category_names)}\n"
        f"Default category: {default_category}\n"
        f"User query: {query}\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Classify the user query into one allowed category."},
            {"role": "user", "content": selection_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.exception("LLM categorizer failed; using default category")
        return default_category

    content = str(data.get("message", {}).get("content", "")).strip()
    normalized_map = {name.lower(): name for name in category_names}
    candidate = content.lower().strip()

    if candidate in normalized_map:
        return normalized_map[candidate]

    for key, original in normalized_map.items():
        if key in candidate:
            return original

    if strict:
        logger.warning("LLM categorizer returned unknown category '%s'; using default", content)
        return default_category

    return content or default_category


def _route_for_category(name: str, categories: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    for category in categories:
        if str(category.get("name", "")).lower() == name.lower():
            return name, category
    return name, {}


def _query_route(query: str, route: dict[str, Any], category_name: str, top_k: int) -> list[RetrievedContext]:
    chroma_path = route.get("chroma_path")
    collection_name = route.get("collection_name")
    if not chroma_path or not collection_name:
        return []

    try:
        import chromadb
    except ImportError:
        logger.warning("chromadb is not installed; retrieval is disabled")
        return []

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(str(collection_name))
        response = collection.query(query_texts=[query], n_results=top_k)
    except Exception:
        logger.exception("RAG query failed for category '%s'", category_name)
        return []

    documents = _first_row(response.get("documents", []))
    metadatas = _first_row(response.get("metadatas", []))
    distances = _first_row(response.get("distances", []))

    contexts: list[RetrievedContext] = []
    for idx, doc in enumerate(documents):
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        distance = float(distances[idx]) if idx < len(distances) else 1.0
        score = max(0.0, min(1.0, 1.0 - distance))
        source = str(metadata.get("source", metadata.get("chunk_id", "unknown")))
        contexts.append(
            RetrievedContext(
                text=str(doc),
                source=source,
                score=score,
                category=category_name,
            )
        )
    return contexts


def _first_row(values: list[Any]) -> list[Any]:
    if not values:
        return []
    first = values[0]
    if isinstance(first, list):
        return first
    return values


def _empty_result(strategy: str, category: str, start: float) -> RetrievalResult:
    return RetrievalResult(
        contexts=[],
        strategy_used=strategy,
        selected_category=category,
        candidate_count=0,
        filtered_count=0,
        latency_ms=int((time.perf_counter() - start) * 1000),
    )


def _log_metrics(
    config: dict[str, Any],
    strategy: str,
    category: str,
    candidate_count: int,
    filtered_count: int,
    latency_ms: int,
) -> None:
    obs = config.get("observability", {})
    details: dict[str, Any] = {"strategy": strategy}
    if obs.get("log_selected_category", True):
        details["category"] = category
    if obs.get("log_candidate_count", True):
        details["candidate_count"] = candidate_count
    if obs.get("log_filtered_count", True):
        details["filtered_count"] = filtered_count
    if obs.get("log_retrieval_latency_ms", True):
        details["latency_ms"] = latency_ms
    logger.info("RAG retrieval: %s", details)
