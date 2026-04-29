#!/usr/bin/env python3
"""Download a model artifact described by a stub manifest.

Supports providers:
  - google_drive  (uses Drive export API, handles confirmation tokens)
  - huggingface   (direct HTTPS download from hf.co resolve URL)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx


GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hydrate a stub manifest into a local model file.",
    )
    parser.add_argument("stub", type=Path, help="Path to the stub JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the target path from the stub manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved target path without downloading.",
    )
    return parser.parse_args()


def load_stub(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        stub = json.load(handle)

    source = stub.get("source", {})
    provider = source.get("provider", "")

    if provider == "google_drive":
        file_id = source.get("file_id", "")
        if not file_id or "REPLACE_WITH_DRIVE_FILE_ID" in file_id:
            raise ValueError(
                f"Stub {path} does not have a usable Google Drive file id yet.",
            )
    elif provider == "huggingface":
        repo = source.get("repo", "")
        filename = source.get("filename", "")
        if not repo or not filename:
            raise ValueError(
                f"Stub {path} is missing source.repo or source.filename for huggingface provider.",
            )
    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. Expected 'google_drive' or 'huggingface'.",
        )

    return stub


def build_target_path(stub: dict[str, Any], output_override: Path | None) -> Path:
    if output_override is not None:
        return output_override

    target = stub.get("target_path")
    if not target:
        raise ValueError("Stub manifest is missing target_path.")

    return Path(target)


# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------

def extract_confirm_token(response_text: str) -> str | None:
    hidden_input = re.search(r'name="confirm" value="([^"]+)"', response_text)
    if hidden_input:
        return hidden_input.group(1)

    confirm_link = re.search(r"confirm=([0-9A-Za-z_\-]+)", response_text)
    if confirm_link:
        return confirm_link.group(1)

    return None


def resolve_gdrive_params(client: httpx.Client, file_id: str) -> dict[str, str]:
    params: dict[str, str] = {"export": "download", "id": file_id}
    response = client.get(GOOGLE_DRIVE_DOWNLOAD_URL, params=params)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        return params

    for cookie_name, cookie_value in response.cookies.items():
        if cookie_name.startswith("download_warning"):
            return {**params, "confirm": cookie_value}

    confirm_token = extract_confirm_token(response.text)
    if confirm_token:
        return {**params, "confirm": confirm_token}

    raise RuntimeError("Could not resolve Google Drive confirmation token for download.")


def download_gdrive(client: httpx.Client, file_id: str, target_path: Path) -> None:
    params = resolve_gdrive_params(client, file_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target_path.with_suffix(target_path.suffix + ".partial")

    with client.stream("GET", GOOGLE_DRIVE_DOWNLOAD_URL, params=params) as response:
        response.raise_for_status()
        with partial_path.open("wb") as handle:
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)

    os.replace(partial_path, target_path)


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------

def build_hf_url(source: dict[str, str]) -> str:
    repo = source["repo"]
    filename = source["filename"]
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def download_hf(client: httpx.Client, source: dict[str, str], target_path: Path) -> None:
    url = build_hf_url(source)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target_path.with_suffix(target_path.suffix + ".partial")

    with client.stream("GET", url) as response:
        response.raise_for_status()
        with partial_path.open("wb") as handle:
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)

    os.replace(partial_path, target_path)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def verify_sha256(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    actual_sha256 = digest.hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}",
        )


def main() -> int:
    args = parse_args()
    stub = load_stub(args.stub)
    target_path = build_target_path(stub, args.output)

    if args.dry_run:
        print(target_path)
        return 0

    if target_path.exists() and not args.force:
        raise FileExistsError(
            f"Target already exists: {target_path}. Use --force to overwrite it.",
        )

    provider = stub["source"]["provider"]

    with httpx.Client(follow_redirects=True, timeout=None) as client:
        if provider == "google_drive":
            download_gdrive(client, stub["source"]["file_id"], target_path)
        elif provider == "huggingface":
            download_hf(client, stub["source"], target_path)

    expected_sha256 = stub.get("sha256")
    if expected_sha256:
        verify_sha256(target_path, expected_sha256)

    print(f"Downloaded {stub['name']} -> {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
