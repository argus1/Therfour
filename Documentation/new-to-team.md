# New Developer Onboarding & Troubleshooting Guide

Welcome to **Therfour** – a modular, open-source backend for telephone-based harm-reduction helplines. All AI inference runs locally; no caller data leaves your infrastructure.

---

## Table of Contents

1. [What the Project Does](#1-what-the-project-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Repository Layout](#3-repository-layout)
4. [Prerequisites](#4-prerequisites)
5. [First-Time Setup](#5-first-time-setup)
6. [Running the Server](#6-running-the-server)
7. [Running Tests](#7-running-tests)
8. [Docker / Compose Workflow](#8-docker--compose-workflow)
9. [Key Configuration Reference](#9-key-configuration-reference)
10. [RAG System](#10-rag-system)
11. [Model Storage Conventions](#11-model-storage-conventions)
12. [Twilio Wiring](#12-twilio-wiring)
13. [Troubleshooting](#13-troubleshooting)
14. [Coding Conventions](#14-coding-conventions)

---

## 1. What the Project Does

Therfour accepts inbound phone calls via **Twilio Media Streams**, runs the full voice pipeline locally, and talks back to the caller in real time:

```
Caller ──► Twilio ──► POST /calls/inbound  (TwiML response)
                               │
                               ▼
                    WebSocket /calls/stream
                               │
           ┌───────────────────▼───────────────────┐
           │              CallSession               │
           │  μ-law/8 kHz ──► PCM float32/16 kHz   │
           │          faster-whisper  (STT)         │
           │                  │                     │
           │            Ollama LLM                  │
           │                  │                     │
           │             Piper TTS                  │
           │  float32/22 kHz ──► μ-law/8 kHz        │
           └───────────────────┬───────────────────┘
                               │
                    Audio sent back to caller
```

| Layer       | Library / Tool                                    |
| ----------- | ------------------------------------------------- |
| Web server  | FastAPI + Uvicorn                                 |
| Telephony   | Twilio Media Streams (WebSocket)                  |
| STT         | faster-whisper (primary) + Sherpa-ONNX (fallback) |
| VAD         | silero-vad                                        |
| TTS         | Piper                                             |
| LLM         | Ollama (local, GGUF-backed)                       |
| RAG         | ChromaDB (optional)                               |
| Audio codec | audioop / audioop-lts + SciPy                     |

---

## 2. Architecture Overview

### Request flow

1. Twilio POSTs to `/calls/inbound`. The handler returns TwiML containing a `<Connect><Stream>` directive.
2. Twilio opens a WebSocket to `/calls/stream`. Each call gets its own `CallSession` object.
3. `CallSession` buffers μ-law audio, detects speech end via Silero VAD, then runs **STT → LLM → TTS** sequentially.
4. Synthesised audio is encoded back to μ-law and streamed to Twilio in 20 ms chunks.

### Service modules (`app/services/`)

| File                  | Responsibility                                                           |
| --------------------- | ------------------------------------------------------------------------ |
| `telephony.py`        | `CallSession` orchestration, audio codec helpers, sample-rate conversion |
| `stt.py`              | faster-whisper + Sherpa-ONNX backends, quality scoring, fallback logic   |
| `vad.py`              | Silero VAD wrapper (`StreamingSpeechDetector`)                           |
| `tts.py`              | Piper subprocess wrapper                                                 |
| `llm.py`              | Ollama `/api/chat` client                                                |
| `rag.py`              | ChromaDB retrieval (standard & hierarchical strategies)                  |
| `ollama_bootstrap.py` | On-startup Ollama model registration from a local GGUF                   |

---

## 3. Repository Layout

```
app/
  api/routes/      # FastAPI routers: calls.py, health.py
  core/
    config.py      # All settings (pydantic-settings, loaded from .env)
    rag_config.json
  models/schemas.py
  services/        # STT, VAD, TTS, LLM, RAG, telephony
benchmarks/        # Whisper model benchmark scripts and results
Documentation/     # Architecture docs, alignment plans, this file
models/
  stubs/           # Lightweight Git-tracked manifests for large files
  piper/           # Piper voice .onnx files (Git LFS or fetched via script)
  llm/             # Local GGUF files (Git LFS)
scripts/           # Helper scripts (bootstrap_ollama, fetch_stub, etc.)
tests/             # pytest suite
```

---

## 4. Prerequisites

| Requirement                                               | Version      | Notes                                                              |
| --------------------------------------------------------- | ------------ | ------------------------------------------------------------------ |
| Python                                                    | 3.11 or 3.12 | 3.13 is supported but requires `audioop-lts` (auto-installed)      |
| [Ollama](https://ollama.com)                              | latest       | Must be running before the server starts                           |
| [Piper binary](https://github.com/rhasspy/piper/releases) | any recent   | Must be on your `$PATH` as `piper`                                 |
| Twilio account                                            | —            | Voice-capable phone number required for end-to-end testing         |
| ngrok (or equivalent)                                     | —            | Required to expose localhost to Twilio webhooks during development |

---

## 5. First-Time Setup

### 5.1 Clone and create a virtual environment

```bash
git clone <repo-url>
cd Therfour
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 5.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5.3 Download the Piper voice model

The repository stores lightweight stub manifests in `models/stubs/`. Fetch the real files with:

```bash
python scripts/fetch_gdrive_stub.py models/stubs/piper/en_US-lessac-medium.onnx.stub.json
python scripts/fetch_gdrive_stub.py models/stubs/piper/en_US-lessac-medium.onnx.json.stub.json
```

Alternatively, download directly:

```bash
mkdir -p models/piper
wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
     -O models/piper/en_US-lessac-medium.onnx
wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
     -O models/piper/en_US-lessac-medium.onnx.json
```

### 5.4 Obtain the LLM

**Option A – Pull via Ollama (smaller models, faster start)**

```bash
ollama pull llama3.2:3b
```

Then set `OLLAMA_MODEL=llama3.2:3b` in your `.env`.

**Option B – Use the team's GGUF (default config)**

The default config expects `models/llm/Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf`. Obtain it from the team drive or via Git LFS:

```bash
git lfs pull
```

Ollama will auto-register the GGUF on first server startup via `ollama_bootstrap.py`.

### 5.5 Configure environment

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
PUBLIC_HOST=your-ngrok-subdomain.ngrok.io
```

All available settings are documented in [app/core/config.py](../app/core/config.py). Key defaults:

| Variable           | Default                                 | Description                 |
| ------------------ | --------------------------------------- | --------------------------- |
| `WHISPER_MODEL`    | `small`                                 | faster-whisper model size   |
| `WHISPER_DEVICE`   | `cpu`                                   | `cpu` or `cuda`             |
| `OLLAMA_MODEL`     | `qwen3.5-35b-a3b:q2-k-xl`               | Model name Ollama must know |
| `OLLAMA_BASE_URL`  | `http://localhost:11434`                | Ollama server URL           |
| `PIPER_MODEL_PATH` | `models/piper/en_US-lessac-medium.onnx` | Path to `.onnx` voice file  |
| `RAG_ENABLED`      | `false`                                 | Enable ChromaDB retrieval   |
| `VAD_ENABLED`      | `true`                                  | Enable Silero VAD           |

---

## 6. Running the Server

### Start Ollama first

```bash
ollama serve   # if not already running as a background service
```

### Start the FastAPI server

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

The server starts on `http://localhost:8000`. On startup it:

1. Waits up to `OLLAMA_READY_TIMEOUT` seconds for Ollama to respond.
2. Checks if the configured model is already registered; if not, registers it from the local GGUF.

**Health check:**

```bash
curl http://localhost:8000/health
```

### Expose to Twilio (development)

```bash
ngrok http 8000
```

Set `PUBLIC_HOST` in `.env` to the ngrok hostname (e.g. `abc123.ngrok.io`, without the scheme).

---

## 7. Running Tests

```bash
source .venv/bin/activate
pytest
```

The test suite uses `pytest-asyncio` with `asyncio_mode = auto` (see `pytest.ini`). A shared `TestClient` fixture is provided in `tests/conftest.py`.

Run a single test file:

```bash
pytest tests/test_stt.py -v
```

Run with log output:

```bash
pytest -s --log-cli-level=DEBUG
```

---

## 8. Docker / Compose Workflow

Build and start everything (app + Ollama):

```bash
docker compose up --build
```

The compose file:

- Binds `./models` into both the app container and the Ollama container (so the bootstrap script can find the GGUF).
- Sets `OLLAMA_KEEP_ALIVE=-1` so the model stays resident in memory.
- Persists Ollama's model store in the `ollama_data` named volume.

To rebuild only the app after a code change:

```bash
docker compose up --build app
```

---

## 9. Key Configuration Reference

All configuration is handled by `app/core/config.py` via `pydantic-settings`. Values are loaded from environment variables or a `.env` file in the project root (case-insensitive).

### STT tuning

| Variable                      | Default   | Effect                                                              |
| ----------------------------- | --------- | ------------------------------------------------------------------- |
| `STT_PRIMARY_BACKEND`         | `whisper` | Switch to `sherpa` to use Sherpa-ONNX as primary                    |
| `STT_SHERPA_FALLBACK_ENABLED` | `true`    | Fall back to Sherpa after all Whisper attempts fail                 |
| `WHISPER_FALLBACK_ENABLED`    | `true`    | Run a second Whisper pass with `beam_size=1` on low-quality results |
| `STT_MIN_QUALITY_SCORE`       | `0.25`    | Discard transcriptions below this alphanumeric density threshold    |
| `WHISPER_LANGUAGE`            | `None`    | Force a language code (e.g. `en`) or leave blank for auto-detection |

### Audio pipeline

| Variable               | Default | Effect                                                            |
| ---------------------- | ------- | ----------------------------------------------------------------- |
| `SILENCE_TIMEOUT_S`    | `1.5`   | Seconds of silence before a turn is processed                     |
| `MIN_AUDIO_DURATION_S` | `0.3`   | Discard audio shorter than this (reduces spurious transcriptions) |
| `VAD_THRESHOLD`        | `0.5`   | Silero speech probability threshold                               |
| `VAD_MIN_SILENCE_MS`   | `300`   | Minimum silence length to trigger end-of-speech                   |

---

## 10. RAG System

RAG is **disabled by default** (`RAG_ENABLED=false`). When enabled, it retrieves harm-reduction context from ChromaDB and prepends it to the LLM prompt.

### Enabling RAG

```env
RAG_ENABLED=true
RAG_CONFIG_PATH=app/core/rag_config.json
```

### Strategies

`app/core/rag_config.json` selects the strategy via the `"strategy"` key:

| Strategy       | Description                                                                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `standard`     | Queries a single Chroma collection at `data/chroma/default`                                                                                                   |
| `hierarchical` | Routes to a category-specific collection (overdose, opioids, stimulants, general) based on keyword or LLM categorization, with fallback to standard on a miss |

### Categorizer methods (hierarchical only)

| Method             | Behaviour                                            |
| ------------------ | ---------------------------------------------------- |
| `keyword`          | Fast keyword matching against category keyword lists |
| `llm`              | LLM call to classify the query                       |
| `keyword_then_llm` | Keyword first; LLM fallback if no keyword match      |

Full configuration reference: [Documentation/RAG_options.md](RAG_options.md)

---

## 11. Model Storage Conventions

Large binary files (ONNX voice models, GGUF LLMs) are tracked via **Git LFS** and stored under `models/`. Do not commit large binaries outside this directory.

```
models/
  stubs/         # Small JSON manifests committed to git; used by fetch scripts
  piper/         # Piper .onnx + .onnx.json voice files
  llm/           # GGUF and other LLM weight files
  .downloads/    # Temporary staging only; never commit
```

To populate models from stubs:

```bash
python scripts/fetch_gdrive_stub.py <path-to-stub.json>
```

See [models/README.md](../models/README.md) for more detail.

---

## 12. Twilio Wiring

1. Create a Twilio phone number with Voice capabilities.
2. Set the **"A Call Comes In"** webhook to:
   `https://<PUBLIC_HOST>/calls/inbound` (HTTP POST)
3. Ensure `PUBLIC_HOST` in `.env` matches the hostname exactly (no trailing slash, no scheme).
4. The server returns TwiML instructing Twilio to open a Media Stream WebSocket to `wss://<PUBLIC_HOST>/calls/stream`.

> **TLS is required by Twilio for production.** Use ngrok or a reverse proxy with a valid certificate during development.

---

## 13. Troubleshooting

### Server won't start – "Ollama did not become reachable"

The server polls `GET /api/tags` up to `OLLAMA_READY_TIMEOUT` seconds.

- **Check Ollama is running:** `curl http://localhost:11434/api/tags`
- **Increase the timeout:** `OLLAMA_READY_TIMEOUT=240` in `.env`
- **Docker:** ensure the `ollama` service is healthy before the `app` service starts (`depends_on` is set but health checks require Ollama to be fully loaded).

### Model not found in Ollama after startup

The bootstrap script (`ollama_bootstrap.py`) registers the GGUF only if the model name is absent from `/api/tags`.

- Verify the GGUF exists: `ls -lh models/llm/`
- In Docker, confirm the `models/` volume is mounted into the Ollama container (see `docker-compose.yml`).
- Check logs for `POST /api/create` errors – the GGUF path must be accessible by the Ollama process.

### No audio / silent responses

- Confirm `piper` binary is on `$PATH`: `which piper`
- Confirm the model file exists at `PIPER_MODEL_PATH`.
- Check server logs for `TTS` errors; Piper is invoked as a subprocess and errors appear in stderr.

### STT returns empty or garbled text

- Try a larger Whisper model: `WHISPER_MODEL=medium` (slower, more accurate).
- If audio quality is low, lower `STT_MIN_QUALITY_SCORE` or `STT_MIN_TEXT_CHARACTERS`.
- Force a language to prevent auto-detection errors: `WHISPER_LANGUAGE=en`.
- Enable Sherpa fallback: `STT_SHERPA_FALLBACK_ENABLED=true` (requires Sherpa-ONNX model files in `SHERPA_MODEL_DIR`).

### VAD cuts off speech too early

Tune these settings:

```env
VAD_THRESHOLD=0.4          # Lower = more sensitive; picks up softer voices
VAD_MIN_SILENCE_MS=500     # Longer pause required before end-of-turn
VAD_SPEECH_PAD_MS=150      # Extra ms of audio kept around speech segments
SILENCE_TIMEOUT_S=2.0      # Fallback silence timeout (no VAD event needed)
```

### Twilio returns "Application Error" or 11200

- Confirm the webhook URL is publicly accessible.
- Confirm the server returned valid XML (`Content-Type: application/xml`).
- Check that `PUBLIC_HOST` does not include the scheme (`https://`) – the app prepends `wss://` itself.
- In ngrok, inspect the `/calls/inbound` request/response under `http://127.0.0.1:4040`.

### ChromaDB / RAG errors

- RAG is off by default. If you did not intend to enable it, confirm `RAG_ENABLED=false`.
- The `data/chroma/` directory is not created automatically; you must ingest documents into ChromaDB before querying. Check the relevant ingestion script or documentation.
- `similarity_threshold` in `rag_config.json` defaults to `0.35`; if retrieval returns nothing, lower this value.

### Tests fail with import errors

- Confirm the virtualenv is activated: `which python` should point inside `.venv/`.
- Reinstall dependencies: `pip install -r requirements.txt`.
- Python 3.13 users: `audioop-lts` must be installed (it is listed in `requirements.txt` with a version guard).

### `audioop` ImportError on Python 3.13+

`audioop` was removed from the standard library in Python 3.13. The `audioop-lts` package is the drop-in replacement and is already in `requirements.txt`. If you see this error, run:

```bash
pip install audioop-lts
```

---

## 14. Coding Conventions

- **Python 3.11+** with `from __future__ import annotations` at the top of every module.
- **pydantic-settings** for all configuration – add new settings to `app/core/config.py`, never hardcode values.
- All service functions that do I/O run in a `ThreadPoolExecutor` (`_executor` in `stt.py`) or are `async` – keep the FastAPI event loop unblocked.
- Keep large binary assets out of plain git; use `models/stubs/` + `scripts/fetch_gdrive_stub.py` or Git LFS.
- Test files live in `tests/`; fixtures go in `tests/conftest.py`. Use `pytest-asyncio` for async tests (no `@pytest.mark.asyncio` decorator needed; `asyncio_mode = auto` is set globally).
- Log at `INFO` by default; set `DEBUG=true` in `.env` for verbose output.
