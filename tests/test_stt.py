"""Tests for the STT service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.models.schemas import TranscriptionResult
from app.services import stt as stt_service


@pytest.mark.asyncio
async def test_transcribe_returns_transcription_result() -> None:
    """transcribe() should return a TranscriptionResult with the correct text."""
    audio = np.zeros(16000, dtype=np.float32)  # 1 s of silence

    mock_seg = MagicMock()
    mock_seg.text = " hello world"
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg], mock_info)

    with patch("app.services.stt._load_model", return_value=mock_model):
        result = await stt_service.transcribe(audio)

    assert isinstance(result, TranscriptionResult)
    assert result.text == "hello world"
    assert result.language == "en"
    assert result.confidence == pytest.approx(0.99)


@pytest.mark.asyncio
async def test_transcribe_empty_segments_gives_empty_string() -> None:
    """If Whisper returns no segments the result text should be an empty string."""
    audio = np.zeros(16000, dtype=np.float32)

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.5

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], mock_info)

    with patch("app.services.stt._load_model", return_value=mock_model):
        result = await stt_service.transcribe(audio)

    assert result.text == ""
