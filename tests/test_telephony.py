"""Tests for the telephony audio-pipeline utilities."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.telephony import downsample, mulaw_to_pcm16, pcm16_to_mulaw, upsample


def _make_pcm16(n_samples: int, value: int = 0) -> bytes:
    return np.full(n_samples, value, dtype=np.int16).tobytes()


def test_mulaw_round_trip() -> None:
    """Encoding then decoding μ-law should reproduce the original PCM closely."""
    # Use a simple sine wave
    t = np.linspace(0, 1, 8000, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    pcm_bytes = samples.tobytes()

    mulaw_bytes = pcm16_to_mulaw(pcm_bytes)
    recovered_bytes = mulaw_to_pcm16(mulaw_bytes)
    recovered = np.frombuffer(recovered_bytes, dtype=np.int16)

    # μ-law is lossy; allow a tolerance of ±512 LSBs
    assert len(recovered) == len(samples)
    max_error = np.abs(recovered.astype(np.int32) - samples.astype(np.int32)).max()
    assert max_error < 512, f"Max error {max_error} exceeds tolerance"


def test_upsample_doubles_length() -> None:
    """Upsampling 8 kHz PCM to 16 kHz should roughly double the sample count."""
    pcm = _make_pcm16(800)  # 100 ms at 8 kHz
    out = upsample(pcm, from_rate=8000, to_rate=16000)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    # resample_poly may produce slightly more or fewer samples than exactly 2×
    assert abs(len(out) - 1600) <= 2


def test_downsample_halves_length() -> None:
    """Downsampling 16 kHz to 8 kHz should roughly halve the sample count."""
    samples = np.zeros(1600, dtype=np.float32)  # 100 ms at 16 kHz
    out = downsample(samples, from_rate=16000, to_rate=8000)
    assert isinstance(out, bytes)
    n_samples = len(out) // 2  # 16-bit = 2 bytes per sample
    assert abs(n_samples - 800) <= 2


def test_upsample_same_rate_is_noop() -> None:
    """upsample with from_rate == to_rate should return the data unchanged."""
    pcm = _make_pcm16(160, value=1000)
    out = upsample(pcm, from_rate=8000, to_rate=8000)
    original = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    np.testing.assert_array_equal(out, original)
