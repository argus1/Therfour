"""FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import calls, health
from app.core.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multilingual voice agent backend for harm-reduction telephony.",
)

app.include_router(health.router, tags=["health"])
app.include_router(calls.router, tags=["calls"])
