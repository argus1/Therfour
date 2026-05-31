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

    with patch.object(tts_service.settings, "tts_backend", "piper"):
        with patch("subprocess.run", return_value=mock_result):
            samples = await tts_service.synthesize("test phrase")

    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32
    assert len(samples) == tts_service.PIPER_SAMPLE_RATE


@pytest.mark.asyncio
async def test_synthesize_raises_on_piper_not_found() -> None:
    """synthesize() should raise RuntimeError when the piper binary is missing."""
    with patch.object(tts_service.settings, "tts_backend", "piper"):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="Piper binary not found"):
                await tts_service.synthesize("hello")


@pytest.mark.asyncio
async def test_synthesize_raises_on_piper_failure() -> None:
    """synthesize() should raise RuntimeError when piper exits non-zero."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = b"model not found"

    with patch.object(tts_service.settings, "tts_backend", "piper"):
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Piper exited"):
                await tts_service.synthesize("hello")


@pytest.mark.asyncio
async def test_synthesize_maps_ava_voice_alias_to_default_model() -> None:
    """Known provider-style aliases should resolve to the default Piper model."""
    fake_pcm = np.zeros(tts_service.PIPER_SAMPLE_RATE, dtype=np.int16).tobytes()
    mock_result = MagicMock(returncode=0, stdout=fake_pcm)

    with patch.object(tts_service.settings, "tts_backend", "piper"):
        with patch("subprocess.run", return_value=mock_result) as run_mock:
            await tts_service.synthesize("hello", voice="en-US-AvaMultilingualNeural")

    cmd = run_mock.call_args.args[0]
    model_index = cmd.index("--model") + 1
    expected_model = tts_service._select_voice(
        voice="en-US-AvaMultilingualNeural",
        language="en-US",
        backend="piper",
    ).model_path
    assert cmd[model_index] == expected_model


def test_select_voice_uses_catalog_for_libritts_and_amy() -> None:
    libritts = tts_service._select_voice(voice="libritts_r", language="en-US", backend="piper")
    amy = tts_service._select_voice(voice="amy", language="en-US", backend="piper")

    assert libritts.voice_id == "en-US-libritts-r-medium"
    assert "en_US-libritts_r-medium.onnx" in libritts.model_path
    assert amy.voice_id == "en-US-amy-medium"
    assert "en_US-amy-medium.onnx" in amy.model_path


def test_select_voice_defaults_to_configured_default_not_lessac() -> None:
    selected = tts_service._select_voice(voice=None, language="en-US", backend="piper")

    assert selected.voice_id == "en-US-libritts-r-medium"


@pytest.mark.asyncio
async def test_synthesize_maps_options_to_piper_flags() -> None:
    """Input options should be normalized into Piper command-line flags."""
    fake_pcm = np.zeros(tts_service.PIPER_SAMPLE_RATE, dtype=np.int16).tobytes()
    mock_result = MagicMock(returncode=0, stdout=fake_pcm)

    with patch.object(tts_service.settings, "tts_backend", "piper"):
        with patch("subprocess.run", return_value=mock_result) as run_mock:
            await tts_service.synthesize(
                "hello",
                options={"rate": 2.0, "noiseScale": 0.5, "noise_w": 0.6, "speaker": 3},
            )

    cmd = run_mock.call_args.args[0]
    length_scale_index = cmd.index("--length_scale") + 1
    noise_scale_index = cmd.index("--noise_scale") + 1
    noise_w_index = cmd.index("--noise_w") + 1
    speaker_index = cmd.index("--speaker") + 1

    assert cmd[length_scale_index] == "0.5"
    assert cmd[noise_scale_index] == "0.5"
    assert cmd[noise_w_index] == "0.6"
    assert cmd[speaker_index] == "3"


@pytest.mark.asyncio
async def test_synthesize_f5_posts_healthcoacher_style_payload() -> None:
    """F5 backend should send text/voice/language payload compatible with HealthCoacher."""

    class _FakeResponse:
        status_code = 200
        headers = {"content-type": "audio/wav"}

        def __init__(self, content: bytes) -> None:
            self.content = content

    class _FakeClient:
        def __init__(self) -> None:
            self.payload = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, _url, json):
            self.payload = json
            # A 24 kHz, mono, 16-bit WAV with 10 ms silence.
            import io
            import wave

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(np.zeros(240, dtype=np.int16).tobytes())
            self.last_payload = json
            return _FakeResponse(buf.getvalue())

    fake_client = _FakeClient()

    with patch.object(tts_service.settings, "tts_backend", "f5_http"):
        with patch.object(tts_service.settings, "tts_fallback_backend", "none"):
            with patch.object(tts_service.settings, "f5_tts_endpoint", "http://localhost:8880/synthesize"):
                with patch("app.services.tts.httpx.AsyncClient", return_value=fake_client):
                    samples = await tts_service.synthesize(
                        "hello",
                        voice="en-US-AvaMultilingualNeural",
                        language="en-US",
                    )

    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32
    assert fake_client.last_payload["text"] == "hello"
    assert fake_client.last_payload["voice"] == "en_default"
    assert fake_client.last_payload["language"] == "en-US"


@pytest.mark.asyncio
async def test_synthesize_f5_falls_back_to_piper() -> None:
    """F5 failures should fall back to Piper when configured."""
    fake_pcm = np.zeros(tts_service.PIPER_SAMPLE_RATE, dtype=np.int16).tobytes()
    mock_result = MagicMock(returncode=0, stdout=fake_pcm)

    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, _url, json):  # pragma: no cover - exercised in async path
            raise tts_service.httpx.ConnectError("unreachable")

    with patch.object(tts_service.settings, "tts_backend", "f5_http"):
        with patch.object(tts_service.settings, "tts_fallback_backend", "piper"):
            with patch("app.services.tts.httpx.AsyncClient", return_value=_FailingClient()):
                with patch("subprocess.run", return_value=mock_result):
                    samples = await tts_service.synthesize("hello")

    assert len(samples) == tts_service.PIPER_SAMPLE_RATE


@pytest.mark.asyncio
async def test_session_sticky_fallback_remains_on_fallback_after_first_failure() -> None:
    class _Primary:
        backend_name = "f5_http"

        def __init__(self) -> None:
            self.calls = 0

        async def synthesize(self, text, *, voice=None, language=None, options=None):
            self.calls += 1
            raise tts_service.UnsupportedError("primary unavailable")

    class _Fallback:
        backend_name = "piper"

        def __init__(self) -> None:
            self.calls = 0

        async def synthesize(self, text, *, voice=None, language=None, options=None):
            self.calls += 1
            return np.zeros(tts_service.PIPER_SAMPLE_RATE, dtype=np.float32)

    primary = _Primary()
    fallback = _Fallback()
    wrapper = tts_service.SessionStickyFallbackSpeechSynthesizer(primary=primary, fallback=fallback)

    first = await wrapper.synthesize("hello")
    second = await wrapper.synthesize("hello again")

    assert wrapper.fallback_active is True
    assert primary.calls == 1
    assert fallback.calls == 2
    assert len(first) == tts_service.PIPER_SAMPLE_RATE
    assert len(second) == tts_service.PIPER_SAMPLE_RATE


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (tts_service.EmptyOutputError("Cannot synthesize empty text"), "empty_text"),
        (tts_service.EmptyOutputError("Piper produced no audio output"), "empty_output"),
        (tts_service.UnsupportedError("Piper binary not found"), "binary_not_found"),
        (tts_service.UnsupportedError("F5-TTS request timed out"), "timeout"),
        (tts_service.UnsupportedError("F5-TTS returned HTTP 503"), "backend_http_error"),
        (tts_service.UnsupportedError("F5-TTS JSON audio payload is not valid base64"), "invalid_response"),
        (tts_service.UnsupportedError("Piper exited with code 1"), "synthesis_failed"),
        (RuntimeError("unexpected"), "unknown"),
    ],
)
def test_classify_tts_exception_maps_to_failure_taxonomy(exc, expected) -> None:
    assert tts_service._classify_tts_exception(exc) == expected


@pytest.mark.asyncio
async def test_synthesize_with_observability_reports_taxonomy_reason(monkeypatch) -> None:
    async def _raise_timeout(*args, **kwargs):
        raise tts_service.UnsupportedError("F5-TTS request timed out after 30 seconds")

    monkeypatch.setattr(tts_service, "synthesize", _raise_timeout)

    obs = await tts_service.synthesize_with_observability("hello", language="en")

    assert obs.ok is False
    assert obs.failure_reason == "timeout"
    assert obs.audio_bytes == 0
    assert obs.audio_duration_ms == 0
