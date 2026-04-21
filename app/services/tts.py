"""Text-to-speech service backed by Piper (local, open-source).

Piper is invoked as a subprocess that reads text on stdin and writes raw
16-bit PCM to stdout.  See https://github.com/rhasspy/piper for installation
and voice-model download instructions.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from app.core.config import settings
from app.services.observability import emit_stage_event

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")

# Piper outputs 22 050 Hz by default for the lessac-medium voice.
# Update this constant if you use a different voice model.
PIPER_SAMPLE_RATE = 22050


def _synthesize(text: str) -> np.ndarray:
    """Run piper synchronously and return float32 PCM samples."""
    cmd = [
        settings.piper_binary,
        "--model", settings.piper_model_path,
        "--output_raw",
    ]
    try:
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Piper binary not found at '{settings.piper_binary}'. "
            "Install piper and set PIPER_BINARY in .env."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(f"Piper exited with code {result.returncode}: {result.stderr.decode()}")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


async def synthesize(text: str) -> np.ndarray:
    """Synthesize *text* to speech, returning float32 PCM at :data:`PIPER_SAMPLE_RATE` Hz."""
    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        samples = await loop.run_in_executor(_executor, partial(_synthesize, text))
    except Exception:
        emit_stage_event(
            stage="tts",
            status="failure",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            failure_reason="synthesis_error",
            text_chars=len(text),
            backend_name="piper",
        )
        raise

    failure_reason = "" if len(samples) > 0 else "empty_audio"
    emit_stage_event(
        stage="tts",
        status="success" if not failure_reason else "dropped",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        failure_reason=failure_reason,
        text_chars=len(text),
        sample_rate=PIPER_SAMPLE_RATE,
        output_samples=len(samples),
        backend_name="piper",
    )
    return samples
