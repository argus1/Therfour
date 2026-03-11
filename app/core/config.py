"""Application configuration loaded from environment variables or a .env file."""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "Therfour"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Twilio ───────────────────────────────────────────────────────────────
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    # Public hostname used to construct the wss:// URL returned in TwiML.
    public_host: str = "localhost"

    # ── STT – faster-whisper ─────────────────────────────────────────────────
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # None → auto-detect; set to e.g. "en" to force a language
    whisper_language: Optional[str] = None

    # ── TTS – Piper ──────────────────────────────────────────────────────────
    piper_model_path: str = "models/en_US-lessac-medium.onnx"
    piper_binary: str = "piper"

    # ── LLM – Ollama ─────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: float = 30.0

    # ── Audio pipeline ───────────────────────────────────────────────────────
    # Twilio Media Streams deliver μ-law audio at 8 kHz.
    audio_sample_rate_twilio: int = 8000
    # Whisper expects 16 kHz.
    audio_sample_rate_whisper: int = 16000
    # Seconds of silence after the last received chunk before a turn is processed.
    silence_timeout_s: float = 1.5
    # Discard utterances shorter than this to avoid spurious transcriptions.
    min_audio_duration_s: float = 0.3


settings = Settings()
