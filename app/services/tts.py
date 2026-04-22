"""Text-to-speech service backed by Piper (local, open-source).

Piper is invoked as a subprocess that reads text on stdin and writes raw
16-bit PCM to stdout.  See https://github.com/rhasspy/piper for installation
and voice-model download instructions.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from app.core.config import settings
from app.models.schemas import EmptyOutputError, UnsupportedError

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")

# Piper outputs 22 050 Hz by default for the lessac-medium voice.
# Update this constant if you use a different voice model.
PIPER_SAMPLE_RATE = 22050


def _synthesize(text: str) -> np.ndarray:
    """Run piper synchronously and return float32 PCM samples."""
    if not text.strip():
        raise EmptyOutputError("Cannot synthesize empty text")

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
        raise UnsupportedError(
            f"Piper binary not found at '{settings.piper_binary}'. "
            "Install piper and set PIPER_BINARY in .env."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise UnsupportedError(f"Piper synthesis timed out after 30 seconds") from exc

    if result.returncode != 0:
        raise UnsupportedError(f"Piper exited with code {result.returncode}: {result.stderr.decode()}")

    if not result.stdout:
        raise EmptyOutputError("Piper produced no audio output")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


async def synthesize(text: str) -> np.ndarray:
    """Synthesize *text* to speech, returning float32 PCM at :data:`PIPER_SAMPLE_RATE` Hz."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, partial(_synthesize, text))
