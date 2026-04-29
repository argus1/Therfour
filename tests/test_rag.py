"""Tests for RAG strategy routing and filtering."""

from __future__ import annotations

import json
from pathlib import Path

from app.services import rag


def _write_config(path: Path, strategy: str) -> None:
    payload = {
        "strategy": strategy,
        "retrieval": {"top_k": 5, "top_k_final": 2, "similarity_threshold": 0.35, "min_query_chars": 1},
        "standard": {"chroma_path": "std", "collection_name": "std_coll"},
        "hierarchical": {
            "fallback_to_standard_on_miss": True,
            "categorizer": {
                "method": "keyword",
                "default_category": "general",
                "llm": {"model": "test-model", "timeout_s": 1, "strict": True},
                "categories": [
                    {
                        "name": "opioids",
                        "keywords": ["fentanyl", "opioid"],
                        "chroma_path": "opioids",
                        "collection_name": "opioid_coll",
                    },
                    {
                        "name": "general",
                        "keywords": [],
                        "chroma_path": "general",
                        "collection_name": "general_coll",
                    },
                ],
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_hierarchical_selects_partition(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "rag_config.json"
    _write_config(config_path, "hierarchical")

    monkeypatch.setattr(rag.settings, "rag_config_path", str(config_path))
    rag._load_config.cache_clear()

    calls: list[tuple[str, str]] = []

    def fake_query_route(query: str, route: dict, category_name: str, top_k: int):
        calls.append((route.get("chroma_path", ""), category_name))
        return [
            rag.RetrievedContext("doc1", "src1", 0.9, category_name),
            rag.RetrievedContext("doc2", "src2", 0.4, category_name),
            rag.RetrievedContext("doc3", "src3", 0.1, category_name),
        ]

    monkeypatch.setattr(rag, "_query_route", fake_query_route)

    result = rag.retrieve("Need fentanyl overdose safety steps")

    assert result.strategy_used == "hierarchical"
    assert result.selected_category.startswith("opioids")
    assert calls[0][0] == "opioids"
    assert len(result.contexts) == 2
    assert all(ctx.score >= 0.35 for ctx in result.contexts)


def test_standard_uses_standard_route(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "rag_config.json"
    _write_config(config_path, "standard")

    monkeypatch.setattr(rag.settings, "rag_config_path", str(config_path))
    rag._load_config.cache_clear()

    calls: list[str] = []

    def fake_query_route(query: str, route: dict, category_name: str, top_k: int):
        calls.append(route.get("chroma_path", ""))
        return [rag.RetrievedContext("doc", "src", 0.8, category_name)]

    monkeypatch.setattr(rag, "_query_route", fake_query_route)

    result = rag.retrieve("general safer use advice")

    assert result.strategy_used == "standard"
    assert calls == ["std"]
    assert len(result.contexts) == 1


def test_context_block_format(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "rag_config.json"
    _write_config(config_path, "standard")

    monkeypatch.setattr(rag.settings, "rag_config_path", str(config_path))
    rag._load_config.cache_clear()

    block = rag.build_context_block(
        [rag.RetrievedContext(text="Use naloxone quickly.", source="guide", score=0.88, category="opioids")]
    )

    assert "Retrieved context" in block
    assert "source: guide" in block
    assert "score: 0.88" in block
    assert "Use naloxone quickly." in block


def test_keyword_then_llm_uses_llm_when_no_keyword_hits(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "rag_config.json"
    _write_config(config_path, "hierarchical")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["hierarchical"]["categorizer"]["method"] = "keyword_then_llm"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(rag.settings, "rag_config_path", str(config_path))
    rag._load_config.cache_clear()

    monkeypatch.setattr(rag, "_select_llm_category", lambda *_args, **_kwargs: "general")

    calls: list[tuple[str, str]] = []

    def fake_query_route(query: str, route: dict, category_name: str, top_k: int):
        calls.append((route.get("chroma_path", ""), category_name))
        return [rag.RetrievedContext("doc", "src", 0.8, category_name)]

    monkeypatch.setattr(rag, "_query_route", fake_query_route)

    result = rag.retrieve("Need support resources near me")

    assert result.selected_category.startswith("general")
    assert calls[0][0] == "general"


def test_llm_method_selects_route_from_llm(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "rag_config.json"
    _write_config(config_path, "hierarchical")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["hierarchical"]["categorizer"]["method"] = "llm"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(rag.settings, "rag_config_path", str(config_path))
    rag._load_config.cache_clear()

    monkeypatch.setattr(rag, "_select_llm_category", lambda *_args, **_kwargs: "opioids")

    calls: list[str] = []

    def fake_query_route(query: str, route: dict, category_name: str, top_k: int):
        calls.append(route.get("chroma_path", ""))
        return [rag.RetrievedContext("doc", "src", 0.8, category_name)]

    monkeypatch.setattr(rag, "_query_route", fake_query_route)

    result = rag.retrieve("Any guidance?")

    assert result.selected_category.startswith("opioids")
    assert calls[0] == "opioids"
