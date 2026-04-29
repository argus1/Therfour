#!/usr/bin/env python3
import argparse
import json
import platform
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


DEFAULT_MODEL = "deepseek-r1-distill-qwen-7b-uncensored-reasoner:i1-q4-k-s"
DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_PROMPT = "In one concise sentence, define harm reduction."


@dataclass
class RunMetrics:
    concurrency: int
    requests_per_worker: int
    total_requests: int
    success: int
    failures: int
    failure_rate: float
    elapsed_s: float
    throughput_rps: float
    latency_min_s: float
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    latency_max_s: float
    latency_mean_s: float


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _request_once(base_url: str, payload: Dict, timeout_s: float) -> Dict:
    start = time.perf_counter()
    req = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
        elapsed = time.perf_counter() - start
        parsed = json.loads(body.decode("utf-8"))
        return {
            "ok": True,
            "latency_s": elapsed,
            "response_len": len((parsed.get("response") or "").strip()),
            "error": "",
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "latency_s": elapsed,
            "response_len": 0,
            "error": repr(exc),
        }


def _run_profile(
    base_url: str,
    payload: Dict,
    timeout_s: float,
    concurrency: int,
    requests_per_worker: int,
) -> Dict:
    total_requests = concurrency * requests_per_worker
    latencies: List[float] = []
    failures = 0
    errors: Dict[str, int] = {}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_request_once, base_url, payload, timeout_s) for _ in range(total_requests)]
        for fut in as_completed(futures):
            out = fut.result()
            latencies.append(out["latency_s"])
            if not out["ok"]:
                failures += 1
                errors[out["error"]] = errors.get(out["error"], 0) + 1
    elapsed_s = time.perf_counter() - t0

    latencies.sort()
    success = total_requests - failures
    metrics = RunMetrics(
        concurrency=concurrency,
        requests_per_worker=requests_per_worker,
        total_requests=total_requests,
        success=success,
        failures=failures,
        failure_rate=(failures / total_requests) if total_requests else 0.0,
        elapsed_s=elapsed_s,
        throughput_rps=(total_requests / elapsed_s) if elapsed_s > 0 else 0.0,
        latency_min_s=latencies[0] if latencies else 0.0,
        latency_p50_s=_percentile(latencies, 0.50),
        latency_p95_s=_percentile(latencies, 0.95),
        latency_p99_s=_percentile(latencies, 0.99),
        latency_max_s=latencies[-1] if latencies else 0.0,
        latency_mean_s=statistics.mean(latencies) if latencies else 0.0,
    )
    return {"metrics": metrics, "errors": errors}


def _choose_runtime_decision(rows: List[RunMetrics]) -> str:
    if not rows:
        return "Insufficient data."
    baseline = rows[0]
    acceptable: List[RunMetrics] = []
    for row in rows:
        latency_guard = row.latency_p95_s <= max(10.0, baseline.latency_p95_s * 2.0)
        failure_guard = row.failure_rate <= 0.01
        if latency_guard and failure_guard:
            acceptable.append(row)
    if not acceptable:
        return "No profile met failure/latency guardrails; keep serial fallback and tune model/runtime."
    best = max(acceptable, key=lambda r: r.throughput_rps)
    return (
        f"Select concurrency={best.concurrency} for production default: highest throughput among guardrail-"
        f"compliant profiles (failure_rate={best.failure_rate:.2%}, p95={best.latency_p95_s:.2f}s)."
    )


def _render_markdown(
    output_file: Path,
    model_name_slug: str,
    model_tag: str,
    platform_name: str,
    host_desc: str,
    load_profile: str,
    warmup_ok: bool,
    warmup_latency_s: float,
    rows: List[RunMetrics],
    error_map: Dict[int, Dict[str, int]],
    decision: str,
) -> None:
    lines: List[str] = []
    lines.append(f"# GGUF Concurrency Benchmark: {model_name_slug} on {platform_name}")
    lines.append("")
    lines.append(f"- Date (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Platform: {platform_name}")
    lines.append(f"- Host: {host_desc}")
    lines.append(f"- Ollama model tag: {model_tag}")
    lines.append(f"- Load profile: {load_profile}")
    lines.append(f"- Warmup status: {'success' if warmup_ok else 'failure'}")
    lines.append(f"- Warmup latency: {warmup_latency_s:.2f}s")
    lines.append("")
    lines.append("## Load Profile Runs")
    lines.append("")
    lines.append("| Concurrency | Requests/Worker | Total Requests | Success | Failures | Failure Rate | Throughput (req/s) | P50 (s) | P95 (s) | P99 (s) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row.concurrency} | {row.requests_per_worker} | {row.total_requests} | {row.success} | "
            f"{row.failures} | {row.failure_rate:.2%} | {row.throughput_rps:.3f} | {row.latency_p50_s:.3f} | "
            f"{row.latency_p95_s:.3f} | {row.latency_p99_s:.3f} |"
        )
    lines.append("")
    lines.append("## Throughput/Failure Observations")
    lines.append("")
    if rows:
        fastest = max(rows, key=lambda r: r.throughput_rps)
        lowest_p95 = min(rows, key=lambda r: r.latency_p95_s)
        total_failures = sum(r.failures for r in rows)
        total_requests = sum(r.total_requests for r in rows)
        lines.append(
            f"- Peak throughput observed at concurrency {fastest.concurrency}: {fastest.throughput_rps:.3f} req/s."
        )
        lines.append(
            f"- Lowest p95 latency observed at concurrency {lowest_p95.concurrency}: {lowest_p95.latency_p95_s:.3f}s."
        )
        lines.append(
            f"- Aggregate failure rate across all profiles: {total_failures}/{total_requests} "
            f"({(total_failures / total_requests if total_requests else 0):.2%})."
        )
    for conc, errs in error_map.items():
        if errs:
            lines.append(f"- Concurrency {conc} errors: {json.dumps(errs, sort_keys=True)}")
    if all(not err for err in error_map.values()):
        lines.append("- No request errors were observed in this benchmark run.")
    lines.append("")
    lines.append("## Runtime Decision Evidence")
    lines.append("")
    lines.append(f"- Decision rule: failure_rate <= 1% and p95 <= max(10s, 2x baseline p95 at concurrency=1).")
    lines.append(f"- Decision: {decision}")
    lines.append("")
    lines.append("## Repro")
    lines.append("")
    lines.append(
        "- Command: /Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/ollama_concurrency_benchmark.py"
    )

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Ollama concurrency behavior for a GGUF-backed model")
    parser.add_argument("--model-tag", default=DEFAULT_MODEL)
    parser.add_argument("--model-slug", default="DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner.i1-Q4_K_S")
    parser.add_argument("--platform", default="ollama")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--requests-per-worker", type=int, default=4)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    payload = {
        "model": args.model_tag,
        "prompt": args.prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": args.num_predict,
        },
    }

    warm = _request_once(args.url, payload, args.timeout)
    concurrency_levels = [int(part.strip()) for part in args.concurrency.split(",") if part.strip()]

    rows: List[RunMetrics] = []
    error_map: Dict[int, Dict[str, int]] = {}
    for conc in concurrency_levels:
        out = _run_profile(args.url, payload, args.timeout, conc, args.requests_per_worker)
        rows.append(out["metrics"])
        error_map[conc] = out["errors"]

    decision = _choose_runtime_decision(rows)

    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix.strip()}" if args.output_suffix.strip() else ""
    filename = f"GGUF_concurrency_bnechmark_{args.model_slug}_{args.platform}{suffix}.md"
    output_file = output_dir / filename

    _render_markdown(
        output_file=output_file,
        model_name_slug=args.model_slug,
        model_tag=args.model_tag,
        platform_name=args.platform,
        host_desc=f"{platform.system()} {platform.machine()} ({platform.platform()})",
        load_profile=f"concurrency={concurrency_levels}, requests_per_worker={args.requests_per_worker}, num_predict={args.num_predict}",
        warmup_ok=warm["ok"],
        warmup_latency_s=warm["latency_s"],
        rows=rows,
        error_map=error_map,
        decision=decision,
    )
    print(output_file)


if __name__ == "__main__":
    main()
