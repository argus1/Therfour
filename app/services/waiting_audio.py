"""Waiting-audio helpers used while RAG-backed response generation is pending."""

from __future__ import annotations

import itertools
import logging
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from app.core.config import settings
from app.services.tts import PIPER_SAMPLE_RATE

logger = logging.getLogger(__name__)

_PHRASE_FILES_BY_LANGUAGE = {
    "en-US": [
        "phrase_en-US_let_me_think_about_that.wav",
        "phrase_en-US_i_need_a_moment_to_come_up_with_a_good_response.wav",
        "phrase_en-US_good_question.wav",
    ],
    "zh-CN": [
        "phrase_zh-CN_let_me_think_about_that.wav",
        "phrase_zh-CN_i_need_a_moment_to_come_up_with_a_good_response.wav",
        "phrase_zh-CN_good_question.wav",
    ],
    "yue-Hant-HK": [
        "phrase_yue-Hant-HK_let_me_think_about_that.wav",
        "phrase_yue-Hant-HK_i_need_a_moment_to_come_up_with_a_good_response.wav",
        "phrase_yue-Hant-HK_good_question.wav",
    ],
    "ja-JP": [
        "phrase_ja-JP_let_me_think_about_that.wav",
        "phrase_ja-JP_i_need_a_moment_to_come_up_with_a_good_response.wav",
        "phrase_ja-JP_good_question.wav",
    ],
}
_AMBIENT_FILES = [
    "ambient_thinking_1.wav",
    "ambient_breath_1.wav",
    "ambient_thinking_2.wav",
    "ambient_breath_2.wav",
]
_phrase_cycles = {
    language: itertools.cycle(files)
    for language, files in _PHRASE_FILES_BY_LANGUAGE.items()
}
_ambient_cycle = itertools.cycle(_AMBIENT_FILES)


def build_waiting_audio(language: str | None = None) -> np.ndarray:
    """Return phrase+ambient waiting audio resampled to ``PIPER_SAMPLE_RATE``.

    Phrase selection supports English, Mandarin, Cantonese, and Japanese assets
    borrowed from HealthCoacher. Unsupported languages fall back to English.
    """
    assets_dir = Path(settings.rag_waiting_audio_assets_dir)
    phrase_language = _resolve_phrase_language(language)
    phrase = _load_asset(assets_dir / next(_phrase_cycles[phrase_language]))
    ambient = _load_asset(assets_dir / next(_ambient_cycle))

    silence = np.zeros(int(PIPER_SAMPLE_RATE * 0.12), dtype=np.float32)
    ambient = ambient[: int(PIPER_SAMPLE_RATE * 1.25)]
    if ambient.size:
        ambient = ambient * 0.55

    if phrase.size and ambient.size:
        return np.concatenate([phrase, silence, ambient]).astype(np.float32)
    if phrase.size:
        return phrase.astype(np.float32)
    if ambient.size:
        return ambient.astype(np.float32)
    return np.zeros(0, dtype=np.float32)


def _resolve_phrase_language(language: str | None) -> str:
    if not language:
        return "en-US"

    normalized = language.strip().lower()
    if normalized.startswith("yue") or "hant-hk" in normalized:
        return "yue-Hant-HK"
    if normalized.startswith("zh"):
        return "zh-CN"
    if normalized.startswith("ja"):
        return "ja-JP"
    return "en-US"


def _load_asset(path: Path) -> np.ndarray:
    if not path.exists():
        logger.warning("Waiting-audio asset not found: %s", path)
        return np.zeros(0, dtype=np.float32)

    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_bytes = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        logger.warning("Unsupported waiting-audio sample width %s for %s", sample_width, path)
        return np.zeros(0, dtype=np.float32)

    pcm = np.frombuffer(frame_bytes, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)

    samples = pcm.astype(np.float32) / 32768.0
    if sample_rate != PIPER_SAMPLE_RATE:
        samples = resample_poly(samples, PIPER_SAMPLE_RATE, sample_rate).astype(np.float32)

    return samples.astype(np.float32)