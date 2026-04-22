"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import calls, health
from app.core.config import settings
from app.services.ollama_bootstrap import ensure_model_loaded

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting up – bootstrapping Ollama model …")
    await ensure_model_loaded()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multilingual voice agent backend for harm-reduction telephony.",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(calls.router, tags=["calls"])
