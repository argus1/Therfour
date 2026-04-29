# Chroma Retrieval Stability Under Load Benchmark

Date: 2026-04-28

## Objective

Evaluate Chroma retrieval stability and latency behaviour under concurrent load, simulating
multiple simultaneous callers querying a shared collection. Measured dimensions:

- P50/P95/P99 per-query latency at each concurrency level
- Latency degradation ratio relative to single-worker baseline
- Top-1 ranking consistency across concurrent runs
- Error rate (query exceptions under load)
- Throughput (queries per second at wall-clock level)

## Context from Alignment Plan

Supports the sprint goal to validate ChromaDB direct use as the online retrieval path
(ADR track 3 – ChromaDB Direct vs WAX). Key question: does ranking stay stable and latency
stay within call-turn budget when concurrent RAG requests arrive?

## Benchmark Design

- Engine: chromadb PersistentClient (temporary local path, cosine space)
- Corpus: 10 harm-reduction snippets (same as relevance-check benchmark)
- Query set: 5 representative prompts, run by every worker
- Concurrency levels tested: 1, 2, 4, 8, 16 threads
- Rounds per worker per query: 20
- Total queries per level: workers × queries × rounds
- Embedding: TF-IDF cosine (synthetic; isolates Chroma internals from embedding model)

## Results

### Latency and Throughput by Concurrency Level

| Workers | Total Queries | Wall (s) | Throughput (QPS) | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stddev (ms) |
| ------: | ------------: | -------: | ---------------: | -------: | -------: | -------: | --------: | ----------: |
|       1 |           100 |   0.0283 |             3539 |   0.2606 |   0.3505 |   0.6039 |    0.2780 |      0.1150 |
|       2 |           200 |   0.0402 |             4969 |   0.3743 |   0.5661 |   0.8830 |    0.3969 |      0.1044 |
|       4 |           400 |   0.0798 |             5010 |   0.7680 |   1.0076 |   1.3322 |    0.7894 |      0.1101 |
|       8 |           800 |   0.1609 |             4973 |   1.5561 |   2.0612 |   2.2290 |    1.5938 |      0.1917 |
|      16 |          1600 |   0.3193 |             5010 |   3.0977 |   3.7207 |   3.8052 |    3.1638 |      0.2952 |

### P95 Latency Degradation vs Serial Baseline

| Workers | P95 (ms) | Degradation Ratio |
| ------: | -------: | ----------------: |
|       1 |   0.3505 |             1.00× |
|       2 |   0.5661 |             1.62× |
|       4 |   1.0076 |             2.87× |
|       8 |   2.0612 |             5.88× |
|      16 |   3.7207 |            10.62× |

### Ranking Stability Under Load

| Workers | Top-1 Consistent (all queries) | Min Consistency |
| ------: | :----------------------------: | --------------: |
|       1 |              Yes               |          1.0000 |
|       2 |              Yes               |          1.0000 |
|       4 |              Yes               |          1.0000 |
|       8 |              Yes               |          1.0000 |
|      16 |              Yes               |          1.0000 |

### Error Rate

- 0.0000 across all concurrency levels (0 exceptions in 3100 total queries)

## Interpretation

**Ranking stability:** Chroma's cosine-space HNSW index returns identical top-1 results
across all concurrency levels in every run. Concurrent reads do not cause ranking drift.

**Latency:** P95 latency scales roughly linearly with worker count, rising from 0.35 ms at
1 worker to 3.72 ms at 16 workers — a 10.6× degradation ratio. For a real-time telephone
call-turn budget (typically 150–300 ms end-to-end), even 16 concurrent retrievals remain
well within acceptable headroom when retrieval is the only bottleneck.

**Throughput:** Throughput plateaus near 5000 QPS from 2 workers onward, indicating the
embedding computation and thread-pool overhead dominate over Chroma query time at this
corpus size.

**Error resilience:** No retrieval exceptions observed under any tested load level.

## Concurrency Floor Recommendation

Based on the degradation curve, a safe operating target for TherFour (which handles one
caller at a time per process) is up to 4 concurrent RAG queries per process before P95
latency exceeds 1 ms. For multi-tenant deployment, profile against the target corpus size
and real embedding model before raising the concurrency floor further.

## Conclusions

1. Chroma PersistentClient is read-safe under concurrent thread access with no ranking
   drift observed up to 16 simultaneous queries.
2. Latency grows approximately linearly with concurrency; at 16 workers P95 is 3.72 ms,
   still well within call-turn budget.
3. Error rate is 0.0000 across all levels, confirming operational reliability.
4. Throughput saturates near 5000 QPS regardless of worker count, suggesting the bottleneck
   is outside Chroma (embedding or thread scheduling) at this scale.

## Recommended Next Steps

1. Re-run with real sentence-transformer embeddings (chromadb default embedding function)
   to include embedding latency in the per-query numbers.
2. Profile at corpus sizes representative of production (1 k–50 k chunks) to find the
   point where latency degrades non-linearly.
3. Add a load stability probe to CI that asserts error_rate == 0 and P95 < 10 ms at
   workers = 4 as a regression gate.

## Repro Command

```
/Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/chroma_stability_under_load.py
```
