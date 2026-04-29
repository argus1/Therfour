# Chroma Retrieval Quality and Stability Benchmark (Relevance Checks)

Date: 2026-04-28

## Objective

Evaluate Chroma retrieval quality and run-to-run stability, with emphasis on relevance checks based on the current RAG scoring rule:

- score = clamp(1 - distance)
- relevant if score >= threshold
- final contexts capped at top_k_final = 3

## Context from Alignment Plan

This benchmark supports the sprint objective to make RAG behavior reproducible with evaluation examples and to validate vector-store strategy decisions for Chroma direct usage.

## Benchmark Design

Because no populated runtime Chroma index was available in the repository, this benchmark used a controlled, self-contained corpus to isolate relevance-check behavior.

- Engine: chromadb PersistentClient (temporary local path)
- Distance space: cosine (hnsw:space = cosine)
- Corpus size: 10 harm-reduction snippets
- Query set: 5 representative prompts
- Repeats: 30 runs per query (150 retrievals total)
- Retrieval parameters:
  - top_k = 5
  - top_k_final = 3
  - baseline threshold = 0.35

## Baseline Results (Threshold = 0.35)

### Overall

- Mean precision@filtered_k: 0.2000
- Mean recall@filtered_k: 0.0500
- Hit rate: 0.2000
- Mean filtered count: 0.2000
- Stddev filtered count: 0.4000
- Mean latency: 0.3215 ms
- P95 latency: 0.4140 ms
- Latency stddev: 0.1295 ms

### Per-query quality

- q1 (opioid overdose and naloxone):
  - Precision 1.0000, recall 0.2500, hit rate 1.0000
- q2 (meth crash):
  - Precision 0.0000, recall 0.0000, hit rate 0.0000
- q3 (injection infection risk):
  - Precision 0.0000, recall 0.0000, hit rate 0.0000
- q4 (opioid withdrawal):
  - Precision 0.0000, recall 0.0000, hit rate 0.0000
- q5 (harm-reduction support resources):
  - Precision 0.0000, recall 0.0000, hit rate 0.0000

### Stability

- Top-1 consistency per query: 1.0000 across all queries
- All queries strictly top-1 stable: true

Interpretation:

- Retrieval ranking is stable run-to-run.
- Relevance filtering at threshold 0.35 is too strict for this corpus and scoring setup, causing near-total context suppression outside q1.

## Relevance Check Sensitivity Sweep

| Threshold | Mean Precision | Mean Recall | Hit Rate | Mean Filtered Count | P95 Latency (ms) |
| --------- | -------------: | ----------: | -------: | ------------------: | ---------------: |
| 0.00      |         0.3333 |      0.6000 |   0.8000 |                 3.0 |           0.3924 |
| 0.20      |         0.6667 |      0.6000 |   0.8000 |                 1.4 |           0.3235 |
| 0.35      |         0.2000 |      0.0500 |   0.2000 |                 0.2 |           0.3329 |
| 0.50      |         0.0000 |      0.0000 |   0.0000 |                 0.0 |           0.3824 |

Interpretation:

- The relevance check threshold has the strongest impact on quality/coverage tradeoff.
- In this setup, threshold 0.20 provides materially better precision than 0.00 while preserving recall and hit rate.
- Threshold >= 0.35 over-filters and removes most useful context.

## Conclusions

1. Chroma retrieval is operationally stable in repeated runs (no ranking drift observed).
2. Current relevance check strictness (0.35) is likely miscalibrated for the scoring rule and can degrade answer grounding by returning too few contexts.
3. For this benchmark profile, 0.20 is a better starting threshold than 0.35.

## Recommended Next Steps

1. Add this benchmark to CI as a non-blocking regression check with pass/fail guardrails for hit rate and filtered-count floor.
2. Re-run against production-like embedded corpus and real embedding model outputs (instead of synthetic vectors) before changing default threshold.
3. Track per-category thresholds if hierarchical routes show different score distributions.

## Repro Command

/Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/chroma_relevance_benchmark.py
