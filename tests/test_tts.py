"""Tests for the TTS service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services import tts as tts_service


@pytest.mark.asyncio
async def test_synthesize_returns_float32_array() -> None:
    """synthesize() should return a float32 numpy array of PCM samples."""
    # Simulate piper producing 1 second of silence at 22 050 Hz
    fake_pcm = np.zeros(tts_service.PIPER_SAMPLE_RATE, dtype=np.int16).tobytes()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = fake_pcm

    with patch("subprocess.run", return_value=mock_result):
        samples = await tts_service.synthesize("test phrase")

    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32
    assert len(samples) == tts_service.PIPER_SAMPLE_RATE


@pytest.mark.asyncio
async def test_synthesize_raises_on_piper_not_found() -> None:
    """synthesize() should raise RuntimeError when the piper binary is missing."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="Piper binary not found"):
            await tts_service.synthesize("hello")


@pytest.mark.asyncio
async def test_synthesize_raises_on_piper_failure() -> None:
    """synthesize() should raise RuntimeError when piper exits non-zero."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = b"model not found"

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="Piper exited"):
            await tts_service.synthesize("hello")


@pytest.mark.asyncio
async def test_synthesize_emits_observability_event_on_success() -> None:
    fake_pcm = np.zeros(tts_service.PIPER_SAMPLE_RATE, dtype=np.int16).tobytes()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = fake_pcm

    with patch("subprocess.run", return_value=mock_result), patch(
        "app.services.tts.emit_stage_event"
    ) as emit_mock:
        await tts_service.synthesize("test phrase")

    emit_mock.assert_called_once()
    assert emit_mock.call_args.kwargs["stage"] == "tts"
    assert emit_mock.call_args.kwargs["status"] == "success"
    assert emit_mock.call_args.kwargs["failure_reason"] == ""


@pytest.mark.asyncio
async def test_synthesize_emits_observability_event_on_failure() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError), patch(
        "app.services.tts.emit_stage_event"
    ) as emit_mock:
        with pytest.raises(RuntimeError, match="Piper binary not found"):
            await tts_service.synthesize("hello")

    emit_mock.assert_called_once()
    assert emit_mock.call_args.kwargs["stage"] == "tts"
    assert emit_mock.call_args.kwargs["status"] == "failure"
    assert emit_mock.call_args.kwargs["failure_reason"] == "synthesis_error"
