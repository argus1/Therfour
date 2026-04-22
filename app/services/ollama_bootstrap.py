"""Bootstrap the Ollama model store from a local GGUF on first boot.

Sequence
--------
1. Poll ``GET /api/tags`` until Ollama is reachable (up to *ready_timeout* s).
2. If the configured model name is already present, return immediately.
3. Otherwise, call ``POST /api/create`` with a minimal Modelfile pointing to
   the local GGUF path and stream progress lines to the logger.

The GGUF path must be readable by the Ollama process.  In Docker this means
the ``./models`` host directory must be mounted inside the Ollama container
(see ``docker-compose.yml``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


async def wait_for_ollama(
    base_url: str,
    timeout: float,
    client: httpx.AsyncClient,
) -> None:
    """Poll /api/tags until Ollama responds or *timeout* seconds elapse."""
    deadline = asyncio.get_event_loop().time() + timeout
    delay = 1.0
    while True:
        try:
            resp = await client.get(f"{base_url}/api/tags", timeout=5.0)
            if resp.status_code < 500:
                logger.info("Ollama is reachable at %s", base_url)
                return
        except (httpx.ConnectError, httpx.TimeoutException):
            pass

        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError(
                f"Ollama did not become reachable within {timeout}s at {base_url}"
            )

        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 1.5, 10.0)  # back off up to 10 s


async def model_is_registered(base_url: str, model_name: str, client: httpx.AsyncClient) -> bool:
    """Return True if *model_name* is already in Ollama's model store."""
    resp = await client.get(f"{base_url}/api/tags", timeout=10.0)
    resp.raise_for_status()
    models = resp.json().get("models", [])
    return any(m.get("name") == model_name for m in models)


async def register_model(
    base_url: str,
    model_name: str,
    gguf_path: Path,
    client: httpx.AsyncClient,
) -> None:
    """Create the model in Ollama by streaming POST /api/create."""
    if not gguf_path.exists():
        raise FileNotFoundError(
            f"GGUF not found at {gguf_path}. "
            "Run: python scripts/fetch_stub.py models/stubs/llm/Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf.stub.json"
        )

    modelfile = f"FROM {gguf_path.resolve()}\n"
    payload = {"name": model_name, "modelfile": modelfile, "stream": True}

    logger.info(
        "Registering model '%s' in Ollama from %s – this may take several minutes …",
        model_name,
        gguf_path,
    )

    async with client.stream(
        "POST",
        f"{base_url}/api/create",
        json=payload,
        timeout=None,  # no timeout – large models can take a while to copy
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            data = json.loads(line)
            status = data.get("status", "")
            if status:
                logger.info("[ollama create] %s", status)
            if data.get("error"):
                raise RuntimeError(f"Ollama create error: {data['error']}")

    logger.info("Model '%s' is ready.", model_name)


async def ensure_model_loaded() -> None:
    """Top-level entry point called from the app lifespan and CLI."""
    base_url = settings.ollama_base_url
    model_name = settings.ollama_model
    gguf_path_str = settings.ollama_model_gguf_path
    ready_timeout = settings.ollama_ready_timeout

    async with httpx.AsyncClient(follow_redirects=True) as client:
        await wait_for_ollama(base_url, ready_timeout, client)

        if await model_is_registered(base_url, model_name, client):
            logger.info("Model '%s' already registered in Ollama – skipping create.", model_name)
            return

        if not gguf_path_str:
            logger.warning(
                "OLLAMA_MODEL_GGUF_PATH is not set and model '%s' is not registered. "
                "Requests will fail until the model is available.",
                model_name,
            )
            return

        gguf_path = Path(gguf_path_str)
        await register_model(base_url, model_name, gguf_path, client)
