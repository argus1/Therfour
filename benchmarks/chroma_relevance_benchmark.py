"""Chroma relevance/stability benchmark focused on threshold-based relevance checks."""

from __future__ import annotations

import json
import math
import statistics
import tempfile
import time
from collections import Counter, defaultdict

import chromadb

THRESHOLD = 0.35
TOP_K = 5
TOP_K_FINAL = 3
RUNS = 30

DOCS: list[tuple[str, str]] = [
    (
        "d1",
        "Naloxone reverses opioid overdose by restoring breathing. Give first dose and call emergency services.",
    ),
    (
        "d2",
        "If someone is not breathing and lips are blue after opioid use, start rescue breaths and administer naloxone.",
    ),
    (
        "d3",
        "For methamphetamine crash, prioritize hydration, sleep, and monitor chest pain or severe agitation.",
    ),
    (
        "d4",
        "Mixing benzodiazepines with opioids raises overdose risk because both suppress breathing.",
    ),
    (
        "d5",
        "Safer injection includes sterile needles, cleaning skin, and avoiding sharing equipment.",
    ),
    (
        "d6",
        "Opioid withdrawal can include muscle aches, diarrhea, anxiety, and sweating; supportive care helps.",
    ),
    (
        "d7",
        "Cocaine chest pain may indicate cardiac risk and needs urgent medical assessment.",
    ),
    (
        "d8",
        "Fentanyl test strips can reduce risk by checking drug supply for fentanyl contamination.",
    ),
    (
        "d9",
        "General support options include local harm-reduction centers and confidential helplines.",
    ),
    (
        "d10",
        "Alcohol withdrawal with seizures or hallucinations is a medical emergency requiring supervised care.",
    ),
]

QUERIES: list[dict[str, object]] = [
    {
        "id": "q1",
        "text": "what should i do for opioid overdose and naloxone",
        "expected": {"d1", "d2", "d4", "d8"},
    },
    {
        "id": "q2",
        "text": "advice for meth crash after stimulant use",
        "expected": {"d3", "d7"},
    },
    {
        "id": "q3",
        "text": "how to reduce infection risk while injecting",
        "expected": {"d5"},
    },
    {
        "id": "q4",
        "text": "symptoms and support for opioid withdrawal",
        "expected": {"d6"},
    },
    {
        "id": "q5",
        "text": "where can i find nonjudgmental harm reduction help",
        "expected": {"d9"},
    },
]


def _tokenize(text: str) -> list[str]:
    chars: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")
    return [w for w in "".join(chars).split() if w]


def _build_embedder(corpus: list[str], queries: list[str]):
    vocab = sorted({tok for item in corpus for tok in _tokenize(item)} | {tok for q in queries for tok in _tokenize(q)})
    index = {term: i for i, term in enumerate(vocab)}

    def embed(text: str) -> list[float]:
        vec = [0.0] * len(vocab)
        counts = Counter(_tokenize(text))
        for token, count in counts.items():
            vec[index[token]] = float(count)
        norm = math.sqrt(sum(value * value for value in vec))
        if norm > 0:
            vec = [value / norm for value in vec]
        return vec

    return embed


def _score(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(distance)))


def run_benchmark() -> dict:
    embed = _build_embedder([text for _, text in DOCS], [str(q["text"]) for q in QUERIES])

    with tempfile.TemporaryDirectory(prefix="chroma_bench_") as tempdir:
        client = chromadb.PersistentClient(path=tempdir)
        collection = client.create_collection("relevance_bench", metadata={"hnsw:space": "cosine"})

        ids = [doc_id for doc_id, _ in DOCS]
        documents = [text for _, text in DOCS]
        embeddings = [embed(text) for text in documents]
        collection.add(ids=ids, documents=documents, embeddings=embeddings)

        aggregate_precision: list[float] = []
        aggregate_recall: list[float] = []
        aggregate_hit: list[float] = []
        aggregate_filtered_count: list[int] = []
        aggregate_latency_ms: list[float] = []

        per_query_top1: dict[str, list[str | None]] = defaultdict(list)
        per_query_precision: dict[str, list[float]] = defaultdict(list)
        per_query_recall: dict[str, list[float]] = defaultdict(list)
        per_query_hit: dict[str, list[float]] = defaultdict(list)

        for _ in range(RUNS):
            for query in QUERIES:
                query_id = str(query["id"])
                query_text = str(query["text"])
                expected = set(query["expected"])

                start = time.perf_counter()
                response = collection.query(query_embeddings=[embed(query_text)], n_results=TOP_K)
                latency_ms = (time.perf_counter() - start) * 1000.0

                scored: list[tuple[str, float]] = []
                for doc_id, distance in zip(response["ids"][0], response["distances"][0]):
                    scored.append((doc_id, _score(float(distance))))

                filtered = [item for item in scored if item[1] >= THRESHOLD]
                filtered.sort(key=lambda item: item[1], reverse=True)
                filtered = filtered[:TOP_K_FINAL]

                retrieved = [doc_id for doc_id, _ in filtered]
                relevant_count = sum(1 for doc_id in retrieved if doc_id in expected)

                precision = (relevant_count / len(retrieved)) if retrieved else 0.0
                recall = relevant_count / len(expected)
                hit = 1.0 if relevant_count > 0 else 0.0

                aggregate_precision.append(precision)
                aggregate_recall.append(recall)
                aggregate_hit.append(hit)
                aggregate_filtered_count.append(len(retrieved))
                aggregate_latency_ms.append(latency_ms)

                per_query_precision[query_id].append(precision)
                per_query_recall[query_id].append(recall)
                per_query_hit[query_id].append(hit)
                per_query_top1[query_id].append(retrieved[0] if retrieved else None)

    top1_consistency: dict[str, dict[str, float | str | int | None]] = {}
    for query_id, top1_values in per_query_top1.items():
        mode_top1, mode_count = Counter(top1_values).most_common(1)[0]
        top1_consistency[query_id] = {
            "mode_top1": mode_top1,
            "consistency": mode_count / len(top1_values),
            "unique_top1_count": len(set(top1_values)),
        }

    results = {
        "config": {
            "threshold": THRESHOLD,
            "top_k": TOP_K,
            "top_k_final": TOP_K_FINAL,
            "runs": RUNS,
            "queries": len(QUERIES),
            "docs": len(DOCS),
            "metric": "score=clamp(1-distance) with cosine space",
        },
        "overall": {
            "mean_precision_at_filtered_k": round(statistics.mean(aggregate_precision), 4),
            "mean_recall_at_filtered_k": round(statistics.mean(aggregate_recall), 4),
            "hit_rate": round(statistics.mean(aggregate_hit), 4),
            "mean_filtered_count": round(statistics.mean(aggregate_filtered_count), 4),
            "stdev_filtered_count": round(statistics.pstdev(aggregate_filtered_count), 4),
            "mean_latency_ms": round(statistics.mean(aggregate_latency_ms), 4),
            "p95_latency_ms": round(
                sorted(aggregate_latency_ms)[int(len(aggregate_latency_ms) * 0.95) - 1],
                4,
            ),
            "stdev_latency_ms": round(statistics.pstdev(aggregate_latency_ms), 4),
        },
        "per_query": {},
        "stability": {
            "top1_consistency": top1_consistency,
            "all_queries_strict_top1_stable": all(
                value["consistency"] == 1.0 for value in top1_consistency.values()
            ),
        },
    }

    for query in QUERIES:
        query_id = str(query["id"])
        results["per_query"][query_id] = {
            "text": query["text"],
            "expected_count": len(query["expected"]),
            "mean_precision": round(statistics.mean(per_query_precision[query_id]), 4),
            "mean_recall": round(statistics.mean(per_query_recall[query_id]), 4),
            "hit_rate": round(statistics.mean(per_query_hit[query_id]), 4),
            "top1_mode": top1_consistency[query_id]["mode_top1"],
            "top1_consistency": round(float(top1_consistency[query_id]["consistency"]), 4),
        }

    return results


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
