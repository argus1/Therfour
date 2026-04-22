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
    whisper_fallback_enabled: bool = True
    whisper_primary_beam_size: int = 5
    whisper_fallback_beam_size: int = 1
    stt_min_text_characters: int = 2
    stt_min_quality_score: float = 0.25

    # ── VAD – Silero ─────────────────────────────────────────────────────────
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 96
    vad_preroll_ms: int = 96

    # ── TTS – Piper ──────────────────────────────────────────────────────────
    piper_model_path: str = "models/piper/en_US-lessac-medium.onnx"
    piper_binary: str = "piper"

    # ── LLM – Ollama ─────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    # Name Ollama knows the model by (used in every /api/chat request).
    ollama_model: str = "qwen3.5-35b-a3b:q2-k-xl"
    ollama_timeout: float = 60.0
    # Absolute or repo-relative path to the GGUF to register on first boot.
    # Leave empty to skip auto-registration (e.g. when using a pulled model).
    ollama_model_gguf_path: str = "models/llm/Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf"
    # How long (seconds) to wait for Ollama to become reachable on startup.
    ollama_ready_timeout: float = 120.0


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
