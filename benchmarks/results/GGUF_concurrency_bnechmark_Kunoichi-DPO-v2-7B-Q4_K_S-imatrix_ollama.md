# GGUF Concurrency Benchmark: Kunoichi-DPO-v2-7B-Q4_K_S-imatrix on ollama

- Date (UTC): 2026-04-29T07:28:15.124760+00:00
- Platform: ollama
- Host: Darwin arm64 (macOS-26.3.1-arm64-arm-64bit)
- Ollama model tag: kunoichi-dpo-v2-7b-q4-k-s-imatrix-gguf:bench0429
- Load profile: concurrency=[1, 2, 4, 8], requests_per_worker=1, num_predict=4
- Warmup status: success
- Warmup latency: 17.90s

## Load Profile Runs

| Concurrency | Requests/Worker | Total Requests | Success | Failures | Failure Rate | Throughput (req/s) | P50 (s) | P95 (s) | P99 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 1 | 0 | 0.00% | 0.423 | 2.361 | 2.361 | 2.361 |
| 2 | 1 | 2 | 2 | 0 | 0.00% | 0.410 | 3.633 | 4.749 | 4.848 |
| 4 | 1 | 4 | 4 | 0 | 0.00% | 0.453 | 5.581 | 8.499 | 8.759 |
| 8 | 1 | 8 | 8 | 0 | 0.00% | 0.437 | 10.445 | 17.514 | 18.149 |

## Throughput/Failure Observations

- Peak throughput observed at concurrency 4: 0.453 req/s.
- Lowest p95 latency observed at concurrency 1: 2.361s.
- Aggregate failure rate across all profiles: 0/15 (0.00%).
- No request errors were observed in this benchmark run.

## Runtime Decision Evidence

- Decision rule: failure_rate <= 1% and p95 <= max(10s, 2x baseline p95 at concurrency=1).
- Decision: Select concurrency=4 for production default: highest throughput among guardrail-compliant profiles (failure_rate=0.00%, p95=8.50s).

## Repro

- Command: /Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/ollama_concurrency_benchmark.py
