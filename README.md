# Therfour

**Multilingual Voice Agent for Harm Reduction**

Therfour is a modular, open-source backend for telephone-based [harm-reduction
and harm-prevention](https://doi.org/10.1080/13811118.2020.1823916) helplines. It connects to a phone call via
[Twilio Media Streams](https://www.twilio.com/docs/voice/media-streams), runs
all AI inference locally, and returns synthesised speech – no data ever leaves
your infrastructure.

```
Caller ──► Twilio ──► /calls/inbound (TwiML)
                         │
                         ▼
               WebSocket /calls/stream
                         │
          ┌──────────────▼──────────────┐
          │        CallSession          │
          │  μ-law/8 kHz ──► float/16k  │
          │       faster-whisper        │  ◄── STT
          │            │                │
          │         Ollama LLM          │  ◄── Local LLM (harm-reduction prompt)
          │            │                │
          │         Piper TTS           │  ◄── TTS
          │  float/22k ──► μ-law/8 kHz  │
          └──────────────┬──────────────┘
                         │
                         ▼
               Audio sent back to caller
```

## Tech stack

| Layer       | Library / Tool                                                          |
| ----------- | ----------------------------------------------------------------------- |
| Web server  | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn                       |
| Telephony   | [Twilio Media Streams](https://www.twilio.com/docs/voice/media-streams) |
| STT         | [faster-whisper](https://github.com/SYSTRAN/faster-whisper)             |
| VAD         | [silero-vad](https://github.com/snakers4/silero-vad)                    |
| TTS         | [Piper](https://github.com/rhasspy/piper)                               |
| LLM         | [Ollama](https://ollama.com) (local, any model)                         |
| Audio codec | Python `audioop` / `audioop-lts` + SciPy                                |

## Quick start

### Prerequisites

- Python 3.11+
- [Piper binary](https://github.com/rhasspy/piper/releases) on your `$PATH`
- [Ollama](https://ollama.com) running locally with your chosen model pulled
- A Twilio account with a voice-capable phone number

### 1 – Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 – Download a Piper voice model

```bash
mkdir -p models/piper

# Option A: hydrate from a tracked Google Drive stub manifest
python scripts/fetch_gdrive_stub.py models/stubs/piper/en_US-lessac-medium.onnx.stub.json
python scripts/fetch_gdrive_stub.py models/stubs/piper/en_US-lessac-medium.onnx.json.stub.json

# Option B: download manually if you are not using the stub flow
wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
     -O models/piper/en_US-lessac-medium.onnx
wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
     -O models/piper/en_US-lessac-medium.onnx.json
```

### 3 – Pull an Ollama model

```bash
ollama pull llama3.2:3b   # ~2 GB; swap for any model you prefer
```

If you keep local `.gguf`, `.safetensors`, or other LLM packs inside `models/llm/`,
the repo now routes those files through Git LFS instead of plain git blobs.

### 4 – Configure environment

```bash
cp .env.example .env
# Edit .env – at minimum set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, PUBLIC_HOST
```

### 5 – Start the server

```bash
uvicorn app.main:app --reload
```

Expose the server to the internet (e.g. via [ngrok](https://ngrok.com)) and
configure your Twilio phone number to send voice webhooks to
`https://<your-host>/calls/inbound`.

### Docker Compose

```bash
cp .env.example .env   # edit as above
docker compose up --build
```

Ollama and the application container are started together. Pull your model
inside the ollama container after first boot:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

## Project structure

```
app/
├── main.py                  # FastAPI application
├── core/
│   └── config.py            # Pydantic-settings configuration
├── models/
│   └── schemas.py           # Shared Pydantic schemas
├── api/routes/
│   ├── health.py            # GET /health
│   └── calls.py             # POST /calls/inbound  WS /calls/stream
└── services/
    ├── stt.py               # Speech-to-text  (faster-whisper)
    ├── tts.py               # Text-to-speech  (Piper)
    ├── llm.py               # LLM generation  (Ollama)
    └── telephony.py         # Audio pipeline + CallSession orchestrator
tests/
├── test_api.py
├── test_stt.py
├── test_tts.py
├── test_llm.py
└── test_telephony.py
```

## Running tests

```bash
pytest
```

## STT Benchmark Harness

To compare `small` vs `distil-large-v3` on telephony-focused samples, use the benchmark harness in `benchmarks/`.

The harness uses:

- `https://github.com/voxserv/audio_quality_testing_samples`

and writes machine-readable outputs to `benchmarks/results/`.

Run from repo root:

```bash
/Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/compare_whisper_models.py \
     --models small distil-large-v3 \
     --subdir testaudio \
     --device auto \
     --repeats 3
```

For more options and reproducibility guidance, see `benchmarks/README.md`.

## Swift migration (in progress)

To start moving server-side logic from Python to Swift, this repository now
includes a small Swift package at `swift-backend/` that mirrors shared response
models and the `/health` payload contract.

Run Swift tests:

```bash
cd swift-backend
swift test
```

## Configuration reference

All settings can be overridden via environment variables or a `.env` file.
See `.env.example` for the full list with descriptions.

## Model artifact storage

Therfour now follows the same broad pattern used in HealthCoacher for large local
artifacts:

- lightweight stub manifests live in `models/stubs/`
- downloaded or checked-in model binaries live under `models/`
- large binary artifacts under `models/` are tracked with Git LFS

Use the stub flow when you want the repo to carry only metadata for a model stored
in Google Drive. A stub manifest records the target path, provider, Drive file id,
and optional checksum without committing the actual artifact.

Hydrate a stub into a local file:

```bash
python scripts/fetch_gdrive_stub.py models/stubs/piper/en_US-lessac-medium.onnx.stub.json
```

To update a stub for your own Drive-backed asset, copy one of the example manifests
in `models/stubs/`, replace the `file_id`, and optionally add a `sha256`.

If you intentionally commit a large model artifact into `models/`, install Git LFS
first:

```bash
git lfs install
git add .gitattributes models/llm/<your-model>.gguf
```

| Variable                     | Default                                 | Description                                |
| ---------------------------- | --------------------------------------- | ------------------------------------------ |
| `WHISPER_MODEL`              | `small`                                 | faster-whisper model size                  |
| `WHISPER_LANGUAGE`           | _(auto-detect)_                         | Pin transcription language                 |
| `WHISPER_FALLBACK_ENABLED`   | `true`                                  | Enables secondary Whisper decode attempt   |
| `WHISPER_PRIMARY_BEAM_SIZE`  | `5`                                     | Beam size for primary decode attempt       |
| `WHISPER_FALLBACK_BEAM_SIZE` | `1`                                     | Beam size for fallback decode attempt      |
| `STT_MIN_TEXT_CHARACTERS`    | `2`                                     | Minimum transcript length before accept    |
| `STT_MIN_QUALITY_SCORE`      | `0.25`                                  | Minimum heuristic quality score            |
| `VAD_ENABLED`                | `true`                                  | Enables Silero streaming VAD segmentation  |
| `VAD_THRESHOLD`              | `0.5`                                   | Speech probability threshold               |
| `VAD_MIN_SILENCE_MS`         | `300`                                   | Silence duration required to close turn    |
| `VAD_SPEECH_PAD_MS`          | `96`                                    | Speech padding around detected boundaries  |
| `VAD_PREROLL_MS`             | `96`                                    | Audio preroll retained before speech start |
| `PIPER_BINARY`               | `piper`                                 | Path to the Piper executable               |
| `PIPER_MODEL_PATH`           | `models/piper/en_US-lessac-medium.onnx` | Piper voice model                          |
| `OLLAMA_MODEL`               | `llama3.2:3b`                           | Ollama model tag                           |
| `OLLAMA_BASE_URL`            | `http://localhost:11434`                | Ollama API base URL                        |
| `SILENCE_TIMEOUT_S`          | `1.5`                                   | Seconds of silence before turn processing  |
| `PUBLIC_HOST`                | `localhost`                             | Hostname used in the TwiML `<Stream>` URL  |
