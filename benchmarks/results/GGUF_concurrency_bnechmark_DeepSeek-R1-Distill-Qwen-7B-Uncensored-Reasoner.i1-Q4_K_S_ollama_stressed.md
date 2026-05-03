# GGUF Concurrency Benchmark: DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner.i1-Q4_K_S on ollama

- Date (UTC): 2026-04-29T08:12:02.172014+00:00
- Platform: ollama
- Host: Darwin arm64 (macOS-26.3.1-arm64-arm-64bit)
- Ollama model tag: deepseek-r1-distill-qwen-7b-uncensored-reasoner-gguf:bench0428
- Load profile: concurrency=[1, 2, 4, 8, 12], requests_per_worker=2, num_predict=32
- Warmup status: success
- Warmup latency: 13.82s

## Load Profile Runs

| Concurrency | Requests/Worker | Total Requests | Success | Failures | Failure Rate | Throughput (req/s) | P50 (s) | P95 (s) | P99 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 2 | 2 | 0 | 0.00% | 0.096 | 10.438 | 10.537 | 10.546 |
| 2 | 2 | 4 | 4 | 0 | 0.00% | 0.093 | 21.355 | 21.570 | 21.595 |
| 4 | 2 | 8 | 8 | 0 | 0.00% | 0.093 | 42.047 | 43.147 | 43.259 |
| 8 | 2 | 16 | 16 | 0 | 0.00% | 0.092 | 85.771 | 88.022 | 88.330 |
| 12 | 2 | 24 | 24 | 0 | 0.00% | 0.091 | 128.963 | 132.428 | 133.185 |

## Throughput/Failure Observations

- Peak throughput observed at concurrency 1: 0.096 req/s.
- Lowest p95 latency observed at concurrency 1: 10.537s.
- Aggregate failure rate across all profiles: 0/54 (0.00%).
- No request errors were observed in this benchmark run.

## Runtime Decision Evidence

- Decision rule: failure_rate <= 1% and p95 <= max(10s, 2x baseline p95 at concurrency=1).
- Decision: Select concurrency=1 for production default: highest throughput among guardrail-compliant profiles (failure_rate=0.00%, p95=10.54s).

## Repro

- Command: /Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/ollama_concurrency_benchmark.py
