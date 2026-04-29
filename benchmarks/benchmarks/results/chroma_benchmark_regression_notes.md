# Chroma Benchmark — Regression Notes

Date: 2026-04-28  
Session scope: argus-branch, benchmarks run 2026-04-28

## Purpose

Track regressions and quality defects identified during this benchmarking session.
Each entry records the failing dimension, observed value, expected/acceptable value,
severity, and the owning fix action.

---

## REG-001 — Threshold 0.35 suppresses context for 4 of 5 query categories

**Source:** chroma_benchmark_relevance_check.md  
**Dimension:** Hit rate / filtered context count  
**Severity:** High — directly degrades answer grounding in production

**Observed:**

| Metric                                |  Value |
| ------------------------------------- | -----: |
| Overall hit rate at threshold 0.35    | 0.2000 |
| Mean filtered count at threshold 0.35 | 0.2000 |
| Queries with zero relevant contexts   | 4 of 5 |

Only q1 (opioid overdose / naloxone) passed the threshold filter.  
Queries covering meth crash, injection safety, withdrawal, and support resources returned
an empty context block in every run.

**Root cause:** The RAG scoring rule `score = clamp(1 − distance)` produces lower scores
than expected for the configured threshold because Chroma reports cosine distances whose
complement does not map linearly to semantic relatedness when using the current embedding
model. The `similarity_threshold = 0.35` in `app/core/rag_config.json` was likely derived
from a different embedding setup and has not been re-calibrated.

**Acceptable baseline (from sweep):**

| Threshold      | Hit Rate | Mean Filtered Count | Mean Precision |
| -------------- | -------: | ------------------: | -------------: |
| 0.20           |   0.8000 |                 1.4 |         0.6667 |
| 0.35 (current) |   0.2000 |                 0.2 |         0.2000 |

**Required fix:** Lower `similarity_threshold` in `app/core/rag_config.json` from `0.35`
to `0.20` pending re-validation against the production embedding model. Do not ship a
threshold change without running the relevance benchmark against the live corpus first.

**Fix owner:** RAG alignment owner (Developer B per Plan.md)  
**Files affected:** `app/core/rag_config.json`  
**Blocked by:** No populated runtime Chroma index in repo; benchmark used synthetic vectors.
Re-run required with real embeddings before threshold change is committed.

---

## REG-002 — recall capped at 0.25 for highest-relevance query category

**Source:** chroma_benchmark_relevance_check.md  
**Dimension:** Recall at top_k_final  
**Severity:** Medium — expected documents are retrieved by the index but dropped by the
final-k cap before reaching the LLM prompt

**Observed:**  
q1 (opioid overdose and naloxone) has 4 expected documents. With `top_k_final = 3`, recall
is capped at 0.25 even when all 4 expected documents score above threshold.

**Root cause:** `top_k_final = 3` in `rag_config.json` truncates the context window before
all relevant results from a multi-document category can be included.

**Acceptable fix options:**

- Raise `top_k_final` to 4 or 5 for categories with known high expected-document counts
  (e.g. overdose).
- Introduce per-category `top_k_final` overrides in the hierarchical config.

**Fix owner:** RAG alignment owner (Developer B)  
**Files affected:** `app/core/rag_config.json`  
**Note:** Raising `top_k_final` increases prompt token usage; validate against
`context_section_token_budget = 600` before applying.

---

## REG-003 — P95 latency degrades 10.6× at 16 concurrent workers

**Source:** chroma_benchmark_stability_under_load.md  
**Dimension:** P95 latency degradation under concurrency  
**Severity:** Low for current single-caller process model; Medium for any multi-tenant path

**Observed:**

| Workers | P95 (ms) | Degradation |
| ------: | -------: | ----------: |
|       1 |     0.35 |       1.00× |
|       4 |     1.01 |       2.87× |
|      16 |     3.72 |      10.62× |

**Root cause:** Chroma PersistentClient serialises HNSW index access under a GIL-bound
thread pool. Latency scales roughly linearly with worker count; there is no observed
sub-linear scaling benefit from concurrency.

**Mitigating factor:** At 16 workers, P95 is still 3.72 ms, which is within the 150–300 ms
call-turn budget. This is a latency-growth regression risk, not a current breach.

**Acceptable guardrail:** P95 < 10 ms at workers = 4. Exceeded only at workers > 14 in
this run.

**Required action:** Add a CI probe asserting P95 < 10 ms at 4 concurrent workers.
If corpus size grows beyond ~1 k chunks, re-run this benchmark to check whether the
degradation curve changes slope.

**Fix owner:** Infrastructure / Developer C (metrics and integration)  
**Files affected:** None currently; CI pipeline config when probe is added.

---

## Summary Table

| ID      | Dimension                                      | Severity      | Status | Fix Owner   |
| ------- | ---------------------------------------------- | ------------- | ------ | ----------- |
| REG-001 | Threshold 0.35 suppresses 80% of context       | High          | Open   | Developer B |
| REG-002 | Recall capped at 0.25 for multi-doc categories | Medium        | Open   | Developer B |
| REG-003 | P95 degradation 10.6× at 16 workers            | Low (current) | Open   | Developer C |

## Non-Regressions Confirmed This Session

- Top-1 ranking: deterministic and stable across 30 serial runs and 16 concurrent workers.
- Error rate: 0.0000 under all concurrency levels (3100 queries, zero exceptions).
- Latency at single-worker: P95 0.35 ms, within call-turn budget.
- Throughput: saturates near 5000 QPS; no collapse under tested load.
