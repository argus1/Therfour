# GGUF Concurrency Benchmark: Kunoichi-DPO-v2-7B-Q4_K_S-imatrix on ollama

- Date (UTC): 2026-04-29T08:22:33.648238+00:00
- Platform: ollama
- Host: Darwin arm64 (macOS-26.3.1-arm64-arm-64bit)
- Ollama model tag: kunoichi-dpo-v2-7b-q4-k-s-imatrix-gguf:bench0429
- Load profile: concurrency=[1, 2, 4, 8, 12], requests_per_worker=2, num_predict=32
- Warmup status: success
- Warmup latency: 22.54s

## Load Profile Runs

| Concurrency | Requests/Worker | Total Requests | Success | Failures | Failure Rate | Throughput (req/s) | P50 (s) | P95 (s) | P99 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 2 | 2 | 0 | 0.00% | 0.087 | 11.457 | 11.471 | 11.472 |
| 2 | 2 | 4 | 4 | 0 | 0.00% | 0.088 | 22.814 | 23.233 | 23.260 |
| 4 | 2 | 8 | 8 | 0 | 0.00% | 0.089 | 44.601 | 46.271 | 46.411 |
| 8 | 2 | 16 | 16 | 0 | 0.00% | 0.082 | 96.679 | 103.796 | 104.815 |
| 12 | 2 | 24 | 24 | 0 | 0.00% | 0.094 | 125.946 | 127.416 | 127.441 |

## Throughput/Failure Observations

- Peak throughput observed at concurrency 12: 0.094 req/s.
- Lowest p95 latency observed at concurrency 1: 11.471s.
- Aggregate failure rate across all profiles: 0/54 (0.00%).
- No request errors were observed in this benchmark run.

## Runtime Decision Evidence

- Decision rule: failure_rate <= 1% and p95 <= max(10s, 2x baseline p95 at concurrency=1).
- Decision: Select concurrency=1 for production default: highest throughput among guardrail-compliant profiles (failure_rate=0.00%, p95=11.47s).

## Repro

- Command: /Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/ollama_concurrency_benchmark.py
