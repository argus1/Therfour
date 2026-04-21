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
    assert result.language_confidence == pytest.approx(0.99)
    assert result.backend_name == "faster-whisper"
    assert result.failure_reason == ""


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
    assert result.failure_reason == "no_speech"


@pytest.mark.asyncio
async def test_transcribe_uses_fallback_attempt_after_empty_primary() -> None:
    audio = np.zeros(16000, dtype=np.float32)

    primary_info = MagicMock()
    primary_info.language = "en"
    primary_info.language_probability = 0.3

    fallback_seg = MagicMock()
    fallback_seg.text = " fallback transcript"
    fallback_info = MagicMock()
    fallback_info.language = "en"
    fallback_info.language_probability = 0.95

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = [([], primary_info), ([fallback_seg], fallback_info)]

    with patch("app.services.stt._load_model", return_value=mock_model), patch.object(
        stt_service.settings, "whisper_fallback_enabled", True
    ):
        result = await stt_service.transcribe(audio, language="en")

    assert result.text == "fallback transcript"
    assert result.fallback_used is True
    assert result.failure_reason == ""


@pytest.mark.asyncio
async def test_transcribe_rejects_low_quality_text() -> None:
    audio = np.zeros(16000, dtype=np.float32)

    low_quality_seg = MagicMock()
    low_quality_seg.text = "."
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.5

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([low_quality_seg], mock_info)

    with patch("app.services.stt._load_model", return_value=mock_model), patch.object(
        stt_service.settings, "whisper_fallback_enabled", False
    ):
        result = await stt_service.transcribe(audio)

    assert result.text == ""
    assert result.failure_reason == "low_quality"


@pytest.mark.asyncio
async def test_transcribe_uses_single_attempt_when_fallback_disabled() -> None:
    audio = np.zeros(16000, dtype=np.float32)

    mock_seg = MagicMock()
    mock_seg.text = " hello"
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.9

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg], mock_info)

    with patch("app.services.stt._load_model", return_value=mock_model), patch.object(
        stt_service.settings, "whisper_fallback_enabled", False
    ):
        result = await stt_service.transcribe(audio, language="en")

    assert result.text == "hello"
    assert result.fallback_used is False
    assert mock_model.transcribe.call_count == 1


@pytest.mark.asyncio
async def test_transcribe_fallback_attempt_uses_auto_language_after_primary_failure() -> None:
    audio = np.zeros(16000, dtype=np.float32)

    fallback_seg = MagicMock()
    fallback_seg.text = " fallback"
    fallback_info = MagicMock()
    fallback_info.language = "en"
    fallback_info.language_probability = 0.88

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = [RuntimeError("primary failed"), ([fallback_seg], fallback_info)]

    with patch("app.services.stt._load_model", return_value=mock_model), patch.object(
        stt_service.settings, "whisper_fallback_enabled", True
    ):
        result = await stt_service.transcribe(audio, language="en")

    assert result.text == "fallback"
    assert result.fallback_used is True
    assert result.failure_reason == ""
    assert mock_model.transcribe.call_count == 2
    assert mock_model.transcribe.call_args_list[1].kwargs["language"] is None


@pytest.mark.asyncio
async def test_transcribe_raises_when_all_attempts_fail() -> None:
    audio = np.zeros(16000, dtype=np.float32)
    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("all attempts failed")

    with patch("app.services.stt._load_model", return_value=mock_model), patch.object(
        stt_service.settings, "whisper_fallback_enabled", True
    ):
        with pytest.raises(RuntimeError, match="all attempts failed"):
            await stt_service.transcribe(audio, language="en")


@pytest.mark.asyncio
async def test_transcribe_emits_observability_event_on_empty_text() -> None:
    audio = np.zeros(16000, dtype=np.float32)

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.5

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], mock_info)

    with patch("app.services.stt._load_model", return_value=mock_model), patch(
        "app.services.stt.emit_stage_event"
    ) as emit_mock:
        result = await stt_service.transcribe(audio)

    assert result.failure_reason == "no_speech"
    emit_mock.assert_called_once()
    assert emit_mock.call_args.kwargs["stage"] == "stt"
    assert emit_mock.call_args.kwargs["status"] == "dropped"
    assert emit_mock.call_args.kwargs["failure_reason"] == "no_speech"
