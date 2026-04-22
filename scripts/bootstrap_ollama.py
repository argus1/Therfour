#!/usr/bin/env python3
"""CLI wrapper around the Ollama bootstrap service.

Usage (local dev, from repo root):
    python scripts/bootstrap_ollama.py

Options:
    --base-url   Override OLLAMA_BASE_URL (default: from config / env)
    --model      Override OLLAMA_MODEL name
    --gguf       Override OLLAMA_MODEL_GGUF_PATH
    --timeout    Seconds to wait for Ollama to become reachable (default: 120)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)

from app.core.config import settings  # noqa: E402 – after sys.path fix
from app.services.ollama_bootstrap import (  # noqa: E402
    ensure_model_loaded,
    model_is_registered,
    register_model,
    wait_for_ollama,
)

import httpx  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Ollama with a local GGUF model.")
    parser.add_argument("--base-url", default=None, help="Ollama base URL")
    parser.add_argument("--model", default=None, help="Model name to register")
    parser.add_argument("--gguf", type=Path, default=None, help="Path to the GGUF file")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Seconds to wait for Ollama to become reachable",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    base_url = args.base_url or settings.ollama_base_url
    model_name = args.model or settings.ollama_model
    gguf_path_str = str(args.gguf) if args.gguf else settings.ollama_model_gguf_path
    ready_timeout = args.timeout if args.timeout is not None else settings.ollama_ready_timeout

    async with httpx.AsyncClient(follow_redirects=True) as client:
        await wait_for_ollama(base_url, ready_timeout, client)

        if await model_is_registered(base_url, model_name, client):
            print(f"Model '{model_name}' is already registered – nothing to do.")
            return 0

        if not gguf_path_str:
            print(
                f"ERROR: Model '{model_name}' is not registered and no GGUF path is configured.",
                file=sys.stderr,
            )
            print(
                "Run: python scripts/fetch_stub.py models/stubs/llm/Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf.stub.json",
                file=sys.stderr,
            )
            return 1

        gguf_path = Path(gguf_path_str)
        await register_model(base_url, model_name, gguf_path, client)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
