# GGUF Concurrency Benchmark: DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner.i1-Q4_K_S on ollama

- Date (UTC): 2026-04-29T08:01:45.460137+00:00
- Platform: ollama
- Host: Darwin arm64 (macOS-26.3.1-arm64-arm-64bit)
- Ollama model tag: deepseek-r1-distill-qwen-7b-uncensored-reasoner-gguf:bench0428
- Load profile: concurrency=[1, 2, 4, 8], requests_per_worker=4, num_predict=64
- Warmup status: success
- Warmup latency: 32.68s

## Load Profile Runs

| Concurrency | Requests/Worker | Total Requests | Success | Failures | Failure Rate | Throughput (req/s) | P50 (s) | P95 (s) | P99 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 4 | 4 | 0 | 0.00% | 0.047 | 21.155 | 21.399 | 21.426 |
| 2 | 4 | 8 | 8 | 0 | 0.00% | 0.049 | 40.686 | 41.363 | 41.383 |
| 4 | 4 | 16 | 16 | 0 | 0.00% | 0.048 | 80.619 | 90.986 | 91.677 |
| 8 | 4 | 32 | 32 | 0 | 0.00% | 0.050 | 159.430 | 164.001 | 164.853 |

## Throughput/Failure Observations

- Peak throughput observed at concurrency 8: 0.050 req/s.
- Lowest p95 latency observed at concurrency 1: 21.399s.
- Aggregate failure rate across all profiles: 0/60 (0.00%).
- No request errors were observed in this benchmark run.

## Runtime Decision Evidence

- Decision rule: failure_rate <= 1% and p95 <= max(10s, 2x baseline p95 at concurrency=1).
- Decision: Select concurrency=2 for production default: highest throughput among guardrail-compliant profiles (failure_rate=0.00%, p95=41.36s).

## Repro

- Command: /Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/ollama_concurrency_benchmark.py
