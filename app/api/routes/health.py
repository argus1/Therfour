"""Health-check endpoint."""

from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service status and the names of the configured AI backends."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        services={
            "stt": f"faster-whisper/{settings.whisper_model}",
            "tts": "piper",
            "llm": f"ollama/{settings.ollama_model}",
        },
    )
