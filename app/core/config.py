"""Application configuration loaded from environment variables or a .env file."""

from __future__ import annotations

from typing import Literal, Optional

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
    # Enables /calls/transfer/harness integration endpoint for transfer simulation.
    transfer_harness_enabled: bool = False
    # Enables /calls/simulation/report endpoint for simulation report generation.
    simulation_harness_enabled: bool = False
    # Allows non-emergency custom transfer targets from directives/harness.
    transfer_allow_custom_targets: bool = False
    # Comma-separated allowlist for custom PSTN transfer targets (E.164 only).
    transfer_allowed_numbers: str = ""
    # Comma-separated allowlist for SIP domains (e.g. help.example.com).
    transfer_allowed_sip_domains: str = ""
    # strict: reject metadata for PSTN targets; compat: keep metadata in logs.
    transfer_metadata_mode: Literal["strict", "compat"] = "compat"
    # Require verbal caller confirmation before executing a 911/988 transfer.
    transfer_confirmation_required: bool = True
    # Post-call re-engagement behavior for 911/988 transfers:
    # - off: never request post-call reopening
    # - auto: always request reopening after operator disconnects
    # - prompt: ask the caller before transfer whether Terris should reopen
    transfer_post_call_reopen_mode: Literal["off", "auto", "prompt"] = "off"
    # Controls whether custom (non-911/988) transfers are selected from the
    # transfer services catalog file.
    transfer_services_enabled: bool = True
    # JSON file describing allowed custom transfer services and their use cases.
    transfer_services_config_path: str = "app/core/transfer_services.json"
    # Post-call re-engagement behavior for custom transfers:
    # - off: never reopen after destination disconnects
    # - auto: always reopen after destination disconnects
    # - prompt: ask caller before transfer whether Terris should reopen
    transfer_custom_post_call_reopen_mode: Literal["off", "auto", "prompt"] = "off"
    # Deprecated compatibility flag. If true and mode is "off", behavior maps to
    # "auto" to preserve previous deployments.
    transfer_stay_on_line_enabled: bool = False
    # Enables randomized call openers/terminators from call flow phrase catalog.
    call_flow_phrases_enabled: bool = True
    # JSON file containing opener/terminator phrase lists.
    call_flow_phrases_config_path: str = "app/core/call_flow_phrases.json"
    # Delay between terminator and each "Are you still there?" round.
    call_end_presence_delay_s: float = 12.0
    # Number of "Are you still there?" rounds before ending call.
    call_end_presence_rounds: int = 2

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
    # STT Confidence thresholding for low-confidence handling
    # Confidence below this threshold triggers confirmation flow (0.0-1.0)
    stt_low_confidence_threshold: float = 0.75
    # Maximum retry attempts for low-confidence confirmations
    stt_max_retries: int = 3
    # "whisper" is the primary backend; set to "sherpa" to use Sherpa-ONNX directly.
    stt_primary_backend: Literal["whisper", "sherpa"] = "whisper"
    # Fall back to Sherpa-ONNX when all Whisper decode attempts are exhausted.
    stt_sherpa_fallback_enabled: bool = True
    # Directory that contains encoder.int8.onnx and decoder.int8.onnx for Sherpa.
    sherpa_model_dir: str = "models/sherpa"
    # Path to the tokens.txt file for the Sherpa model.
    sherpa_tokens_path: str = "models/sherpa/tokens.txt"
    # Number of CPU threads used by the Sherpa-ONNX recognizer.
    sherpa_num_threads: int = 1

    # ── VAD – Silero ─────────────────────────────────────────────────────────
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 96
    vad_preroll_ms: int = 96

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_backend: Literal["piper", "f5_http", "f5_mlx_local"] = "f5_http"
    tts_fallback_backend: Literal["none", "piper"] = "piper"
    # f5-tts-mlx model name (HuggingFace repo id); only used when tts_backend="f5_mlx_local"
    f5_mlx_model: str = "lucasnewman/f5-tts-mlx"
    # F5 HTTP endpoint that accepts {text, voice, language, options}
    f5_tts_endpoint: str = "http://localhost:8880/synthesize"
    f5_tts_voice: str = "en_default"
    f5_tts_timeout_s: float = 30.0
    f5_tts_sample_rate: int = 24000

    # Piper fallback/default backend settings
    # Legacy single-model path used when no voice catalog entry is resolved.
    piper_model_path: str = "models/piper/en_US-libritts_r-medium.onnx"
    # Default Piper voice id selected when voice is omitted or unsupported.
    piper_default_voice_id: str = "en-US-libritts-r-medium"
    # JSON catalog path that maps Piper voice ids/aliases to model paths.
    piper_voices_config_path: str = "app/core/piper_voices.json"
    piper_binary: str = "piper"

    # ── LLM – Ollama ─────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "lmstudio"] = "ollama"
    # Keep this separate so LM Studio can be phased out for a custom interface later.
    llm_temperature: float = 0.0
    llm_max_history_messages: int = 8
    ollama_base_url: str = "http://localhost:11434"
    # Name Ollama knows the model by (used in every /api/chat request).
    ollama_model: str = "qwen3.5-35b-a3b:q2-k-xl"
    ollama_timeout: float = 60.0
    # Absolute or repo-relative path to the GGUF to register on first boot.
    # Leave empty to skip auto-registration (e.g. when using a pulled model).
    ollama_model_gguf_path: str = "models/llm/Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf"
    # How long (seconds) to wait for Ollama to become reachable on startup.
    ollama_ready_timeout: float = 120.0

    # ── LLM – LM Studio (temporary backend target) ─────────────────────────
    lmstudio_base_url: str = "http://10.0.0.132:1234/v1"
    lmstudio_model: str = "deepseek-r1-distill-qwen-7b-uncensored-reasoner-i1"

    # ── RAG ─────────────────────────────────────────────────────────────────
    rag_enabled: bool = False
    rag_config_path: str = "app/core/rag_config.json"
    # Play short filler audio while RAG-backed response generation is in progress.
    rag_waiting_audio_enabled: bool = False
    # Delay before filler audio starts so short retrievals do not get interrupted.
    rag_waiting_audio_delay_s: float = 0.35
    # Directory containing phrase and ambient wav assets used during RAG wait time.
    rag_waiting_audio_assets_dir: str = "app/assets/waiting_audio"
    # Interrupt active assistant generation/playback when caller speech resumes.
    turn_interrupt_enabled: bool = True
    # Enable per-turn routing between rapport, info gathering, and RAG-eligible modes.
    turn_strategy_router_enabled: bool = True
    # Force no-RAG mode for rapport-building turns.
    turn_strategy_no_rag_for_rapport: bool = True
    # Force no-RAG mode for information-gathering turns.
    turn_strategy_no_rag_for_info_gathering: bool = True
    # Force no-RAG mode for understanding-check turns.
    turn_strategy_no_rag_for_understanding_check: bool = True
    # Allow retrieval during explanation turns when understanding gaps are detected.
    turn_strategy_rag_optional_for_explanation: bool = True
    # Emit debug logs for selected turn strategy and RAG usage.
    turn_strategy_debug_logging: bool = True


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
