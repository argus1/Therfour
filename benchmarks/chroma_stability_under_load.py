"""Chroma stability-under-load benchmark.

Simulates concurrent callers querying a shared PersistentClient collection
at increasing concurrency levels.  Measures:
  - P50/P95/P99 latency per worker count
  - Latency degradation ratio vs serial baseline
  - Top-1 ranking consistency under concurrency
  - Error rate (query exceptions)
  - Throughput (queries per second)
"""

from __future__ import annotations

import json
import math
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb

TOP_K = 5
ROUNDS_PER_WORKER = 20
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]

DOCS: list[tuple[str, str]] = [
    ("d1", "Naloxone reverses opioid overdose by restoring breathing. Give first dose and call emergency services."),
    ("d2", "If someone is not breathing and lips are blue after opioid use, start rescue breaths and administer naloxone."),
    ("d3", "For methamphetamine crash, prioritize hydration, sleep, and monitor chest pain or severe agitation."),
    ("d4", "Mixing benzodiazepines with opioids raises overdose risk because both suppress breathing."),
    ("d5", "Safer injection includes sterile needles, cleaning skin, and avoiding sharing equipment."),
    ("d6", "Opioid withdrawal can include muscle aches, diarrhea, anxiety, and sweating; supportive care helps."),
    ("d7", "Cocaine chest pain may indicate cardiac risk and needs urgent medical assessment."),
    ("d8", "Fentanyl test strips can reduce risk by checking drug supply for fentanyl contamination."),
    ("d9", "General support options include local harm-reduction centers and confidential helplines."),
    ("d10", "Alcohol withdrawal with seizures or hallucinations is a medical emergency requiring supervised care."),
]

QUERIES: list[dict[str, object]] = [
    {"id": "q1", "text": "what should i do for opioid overdose and naloxone", "expected": {"d1", "d2", "d4", "d8"}},
    {"id": "q2", "text": "advice for meth crash after stimulant use", "expected": {"d3", "d7"}},
    {"id": "q3", "text": "how to reduce infection risk while injecting", "expected": {"d5"}},
    {"id": "q4", "text": "symptoms and support for opioid withdrawal", "expected": {"d6"}},
    {"id": "q5", "text": "where can i find nonjudgmental harm reduction help", "expected": {"d9"}},
]


def _tokenize(text: str) -> list[str]:
    buf: list[str] = []
    for ch in text.lower():
        buf.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return [w for w in "".join(buf).split() if w]


def _build_embedder(all_texts: list[str]):
    vocab = sorted({tok for txt in all_texts for tok in _tokenize(txt)})
    idx = {t: i for i, t in enumerate(vocab)}

    def embed(text: str) -> list[float]:
        vec = [0.0] * len(vocab)
        for tok, cnt in Counter(_tokenize(text)).items():
            vec[idx[tok]] = float(cnt)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    return embed


def _percentile(data: list[float], p: float) -> float:
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _worker_task(collection, query: dict, embed, rounds: int) -> list[dict]:
    """Run `rounds` queries against the collection; return per-query result dicts."""
    rows: list[dict] = []
    query_id = str(query["id"])
    query_text = str(query["text"])
    q_emb = embed(query_text)

    for _ in range(rounds):
        error: str | None = None
        top1: str | None = None
        latency_ms: float = 0.0
        try:
            t0 = time.perf_counter()
            resp = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            ids = resp["ids"][0]
            top1 = ids[0] if ids else None
        except Exception as exc:
            error = str(exc)
        rows.append({"query_id": query_id, "latency_ms": latency_ms, "top1": top1, "error": error})
    return rows


def run_benchmark() -> dict:
    all_texts = [txt for _, txt in DOCS] + [str(q["text"]) for q in QUERIES]
    embed = _build_embedder(all_texts)

    with tempfile.TemporaryDirectory(prefix="chroma_load_bench_") as tempdir:
        client = chromadb.PersistentClient(path=tempdir)
        collection = client.create_collection("load_bench", metadata={"hnsw:space": "cosine"})
        collection.add(
            ids=[doc_id for doc_id, _ in DOCS],
            documents=[txt for _, txt in DOCS],
            embeddings=[embed(txt) for _, txt in DOCS],
        )

        level_results: dict[str, dict] = {}

        for workers in CONCURRENCY_LEVELS:
            all_rows: list[dict] = []
            wall_start = time.perf_counter()

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = []
                for _ in range(workers):
                    for query in QUERIES:
                        futures.append(
                            pool.submit(_worker_task, collection, query, embed, ROUNDS_PER_WORKER)
                        )
                for future in as_completed(futures):
                    all_rows.extend(future.result())

            wall_elapsed_s = time.perf_counter() - wall_start

            latencies = [r["latency_ms"] for r in all_rows if r["error"] is None]
            errors = [r for r in all_rows if r["error"] is not None]
            error_rate = len(errors) / len(all_rows) if all_rows else 0.0
            throughput_qps = len(all_rows) / wall_elapsed_s if wall_elapsed_s > 0 else 0.0

            per_query_top1: dict[str, list[str | None]] = defaultdict(list)
            for row in all_rows:
                per_query_top1[row["query_id"]].append(row["top1"])

            top1_consistency: dict[str, float] = {}
            for qid, tops in per_query_top1.items():
                mode_count = Counter(tops).most_common(1)[0][1]
                top1_consistency[qid] = mode_count / len(tops)

            level_results[str(workers)] = {
                "workers": workers,
                "total_queries": len(all_rows),
                "wall_elapsed_s": round(wall_elapsed_s, 4),
                "throughput_qps": round(throughput_qps, 2),
                "error_rate": round(error_rate, 4),
                "error_count": len(errors),
                "latency_p50_ms": round(_percentile(latencies, 50), 4) if latencies else None,
                "latency_p95_ms": round(_percentile(latencies, 95), 4) if latencies else None,
                "latency_p99_ms": round(_percentile(latencies, 99), 4) if latencies else None,
                "latency_mean_ms": round(statistics.mean(latencies), 4) if latencies else None,
                "latency_stdev_ms": round(statistics.pstdev(latencies), 4) if latencies else None,
                "top1_consistency_per_query": {qid: round(v, 4) for qid, v in top1_consistency.items()},
                "top1_consistency_min": round(min(top1_consistency.values()), 4) if top1_consistency else None,
                "top1_consistency_all_stable": all(v == 1.0 for v in top1_consistency.values()),
            }

        # Compute latency degradation ratio relative to serial (workers=1)
        baseline_p95 = level_results["1"]["latency_p95_ms"]
        for key, level in level_results.items():
            p95 = level["latency_p95_ms"]
            level["p95_degradation_ratio"] = round(p95 / baseline_p95, 4) if baseline_p95 and p95 is not None else None

        return {
            "config": {
                "top_k": TOP_K,
                "rounds_per_worker": ROUNDS_PER_WORKER,
                "concurrency_levels": CONCURRENCY_LEVELS,
                "docs": len(DOCS),
                "queries_per_level": len(QUERIES),
                "metric": "score=clamp(1-distance) with cosine space",
            },
            "levels": level_results,
        }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
