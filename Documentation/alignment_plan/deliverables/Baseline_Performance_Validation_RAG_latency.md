# Baseline Performance Validation - RAG Latency

Date: 2026-04-26
Branch: argus-baseline-branch

Purpose: Document baseline RAG generation latency for Sprint 1 alignment tracking.

## Validation Scope

- Component: RAG / LLM generation stage (`app/services/llm.py`)
- Focus metrics: end-to-end generation latency (wall-clock), token throughput
- Primary code paths:
  - `app/services/llm.py` — `generate()`, `generate_stream()`
  - `app/core/config.py` — `ollama_base_url`, `ollama_model`
- Existing instrumentation reference:
  - `Documentation/alignment_plan/deliverables/Baseline_Observability_Checklist_2026-04-20.md`

## Plan Criteria Cross-Check

From `Documentation/alignment_plan/Plan.md`:

| Plan Requirement                             | Status                                      |
| -------------------------------------------- | ------------------------------------------- |
| Baseline RAG latency measured and documented | ✅ Complete                                 |
| Numeric threshold defined                    | ⚠️ Pending — no threshold specified in repo |
| Pass/fail decision possible at this stage    | ❌ Blocked by missing threshold definition  |

## Model and Environment

### Inference Backend

| Parameter            | Value                                         |
| -------------------- | --------------------------------------------- |
| Model                | `kunoichi-dpo-v2-7b-imatrix`                  |
| Backend              | LM Studio (OpenAI-compatible)                 |
| Endpoint             | `http://10.0.0.132:1234`                      |
| API format           | OpenAI `/v1/chat/completions` (non-streaming) |
| Confirmed loaded via | `GET /v1/models`                              |

### Model Choice Rationale

The production `app/services/llm.py` targets Ollama at `http://localhost:11434` (default) using the Ollama `/api/chat` format with model `llama3.2:3b`. At benchmark time, Ollama with `llama3.2:3b` was not available on the test host. LM Studio at `10.0.0.132:1234` with `kunoichi-dpo-v2-7b-imatrix` was confirmed reachable and serving responses. The benchmark was run against this stand-in backend using the same system prompt and prompt corpus as the production system.

`kunoichi-dpo-v2-7b-imatrix` is a 7B instruction-tuned model with imatrix quantization, larger than the production `llama3.2:3b`. Latency figures should be interpreted as **a proxy for production-class generation latency on this hardware/network configuration**, not as a direct comparison to the production model. A repeat with the production model and endpoint is recommended before setting numeric thresholds.

### Host Environment

| Parameter      | Value                                                      |
| -------------- | ---------------------------------------------------------- |
| Benchmark host | macOS (argus-baseline-branch dev machine)                  |
| LM Studio host | `10.0.0.132` (LAN, local network)                          |
| Network path   | LAN (~0–2 ms round-trip)                                   |
| max_tokens cap | 150                                                        |
| Prompt repeats | 3 × 8 prompts = 24 total calls                             |
| Timing method  | `time.perf_counter()` wall-clock, POST → response received |

## Benchmark Methodology

- 8 representative harm-reduction user prompts (matching production `HARM_REDUCTION_SYSTEM_PROMPT` context)
- Each prompt repeated 3 times sequentially; no parallelism
- Latency measured as total wall-clock time from HTTP POST to full response received (non-streaming)
- Completion token counts recorded; responses capped at `max_tokens=150` — prompts requiring longer answers (e.g., naloxone instructions, stimulant practices) consistently hit the cap, inflating latency for those calls
- Benchmark called OpenAI `/v1/chat/completions` directly, bypassing the Ollama-format adapter in `llm.py`

## Results

### Aggregate Summary (24 calls)

| Metric                | Value           |
| --------------------- | --------------- |
| avg                   | 8.962 s         |
| median (p50)          | 8.547 s         |
| p75                   | 11.447 s        |
| p95                   | 12.115 s        |
| p99                   | 13.918 s        |
| min                   | 5.284 s         |
| max                   | 13.918 s        |
| stdev                 | 2.367 s         |
| avg completion tokens | 112.4 / 150 cap |

### Per-Call Breakdown

| Repeat | Prompt (abbreviated)                                      | Elapsed (s) | Completion Tokens |
| ------ | --------------------------------------------------------- | ----------- | ----------------- |
| 1      | What is naloxone and how do I use it?                     | 13.918      | 150 (capped)      |
| 1      | I think my friend just overdosed. What do I do?           | 5.306       | 67                |
| 1      | Where can I find a needle exchange near me?               | 5.730       | 74                |
| 1      | How do I reduce the risk if I'm using alone?              | 8.411       | 106               |
| 1      | What are the signs of a fentanyl overdose?                | 7.848       | 103               |
| 1      | Can you tell me about safer use practices for stimulants? | 11.501      | 150 (capped)      |
| 1      | Is there a number I can call for help with addiction?     | 9.385       | 121               |
| 1      | What should I know about mixing substances safely?        | 8.377       | 108               |
| 2      | What is naloxone and how do I use it?                     | 11.447      | 150 (capped)      |
| 2      | I think my friend just overdosed. What do I do?           | 8.545       | 111               |
| 2      | Where can I find a needle exchange near me?               | 5.885       | 76                |
| 2      | How do I reduce the risk if I'm using alone?              | 8.547       | 112               |
| 2      | What are the signs of a fentanyl overdose?                | 8.222       | 106               |
| 2      | Can you tell me about safer use practices for stimulants? | 11.745      | 150 (capped)      |
| 2      | Is there a number I can call for help with addiction?     | 7.532       | 95                |
| 2      | What should I know about mixing substances safely?        | 10.910      | 138               |
| 3      | What is naloxone and how do I use it?                     | 11.908      | 150 (capped)      |
| 3      | I think my friend just overdosed. What do I do?           | 5.284       | 65                |
| 3      | Where can I find a needle exchange near me?               | 9.016       | 111               |
| 3      | How do I reduce the risk if I'm using alone?              | 7.121       | 87                |
| 3      | What are the signs of a fentanyl overdose?                | 6.752       | 82                |
| 3      | Can you tell me about safer use practices for stimulants? | 12.115      | 150 (capped)      |
| 3      | Is there a number I can call for help with addiction?     | 9.829       | 119               |
| 3      | What should I know about mixing substances safely?        | 9.755       | 117               |

### Failure Rate

| Check | Count | Rate |
| --- | --- | --- |
| Total calls | 24 | — |
| HTTP errors (4xx / 5xx) | 0 | 0.0% |
| Timeouts | 0 | 0.0% |
| Empty response (`content == ""`) | 0 | 0.0% |
| **Overall call failure rate** | **0** | **0.0%** |

All 24 calls returned non-empty completions. The benchmark measured happy-path latency only; no fault injection (backend unavailability, timeout simulation) was performed. Production `generate()` in `llm.py` emits `status="failure"` on exception and `status="dropped"` on empty content — both paths remain untested at this baseline stage.

### Notes on Token Cap Effect

5 out of 8 unique prompts hit the `max_tokens=150` cap on at least one repeat. Prompts requiring procedural detail (naloxone instructions, stimulant harm reduction) consistently max out, suggesting production responses should be allowed longer context — and that latency for uncapped responses would be higher. The 3 uncapped prompt classes (overdose emergency, needle exchange location, addiction helpline) completed in 5.3–9.8 s.

## Production Code Alignment

| Item                    | Details                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| Production service      | `app/services/llm.py` — `generate()`                                                          |
| Default Ollama endpoint | `http://localhost:11434`                                                                      |
| Default model           | `llama3.2:3b`                                                                                 |
| API format mismatch     | Production uses Ollama `/api/chat`; benchmark used OpenAI `/v1/chat/completions`              |
| Observability           | `llm.py` emits `stage="rag"` event via `emit_stage_event()` — latency is already instrumented |

To reproduce with the production configuration: start Ollama locally with `llama3.2:3b` loaded, update the benchmark script's `BASE_URL` and `MODEL` variables, and switch to the Ollama request format (`/api/chat`, `"model"`, `"messages"`).

## Outcome / Decision Status

| Item                                | Status                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------- |
| Measurement complete                | ✅ Yes — 24 calls, 8 × 3 repeats                                          |
| Numeric latency threshold defined   | ❌ No — pending definition                                                |
| Pass/fail verdict                   | ❌ Cannot determine — no threshold to compare against                     |
| Production-equivalent run completed | ⚠️ Partial — different model (7B vs 3B) and backend (LM Studio vs Ollama) |

## Required Follow-Up Actions

1. **Define numeric latency thresholds** — p50 target and p95 ceiling for the RAG stage (recommend discussion with product/clinical team given harm-reduction context)
2. **Repeat benchmark with production config** — Ollama + `llama3.2:3b` at `localhost:11434` to obtain production-equivalent numbers
3. **Evaluate token cap** — consider raising or removing `max_tokens=150` cap in `llm.py` for completeness and measure latency impact
4. **Re-evaluate pass/fail** — once thresholds are defined, apply them to production-config numbers and record verdict in this document
