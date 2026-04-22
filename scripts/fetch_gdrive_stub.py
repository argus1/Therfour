#!/usr/bin/env python3
"""Download a Google Drive-backed model artifact described by a stub manifest."""

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
        description="Hydrate a tracked Google Drive stub manifest into a local model file.",
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
    if source.get("provider") != "google_drive":
        raise ValueError("Only google_drive stub manifests are supported.")

    file_id = source.get("file_id", "")
    if not file_id or "REPLACE_WITH_DRIVE_FILE_ID" in file_id:
        raise ValueError(
            f"Stub {path} does not have a usable Google Drive file id yet.",
        )

    return stub


def build_target_path(stub: dict[str, Any], output_override: Path | None) -> Path:
    if output_override is not None:
        return output_override

    target = stub.get("target_path")
    if not target:
        raise ValueError("Stub manifest is missing target_path.")

    return Path(target)


def extract_confirm_token(response_text: str) -> str | None:
    hidden_input = re.search(r'name="confirm" value="([^"]+)"', response_text)
    if hidden_input:
        return hidden_input.group(1)

    confirm_link = re.search(r"confirm=([0-9A-Za-z_\-]+)", response_text)
    if confirm_link:
        return confirm_link.group(1)

    return None


def resolve_download_params(client: httpx.Client, file_id: str) -> dict[str, str]:
    params = {"export": "download", "id": file_id}
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


def download_file(client: httpx.Client, params: dict[str, str], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target_path.with_suffix(target_path.suffix + ".partial")

    with client.stream("GET", GOOGLE_DRIVE_DOWNLOAD_URL, params=params) as response:
        response.raise_for_status()
        with partial_path.open("wb") as handle:
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)

    os.replace(partial_path, target_path)


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

    file_id = stub["source"]["file_id"]
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        params = resolve_download_params(client, file_id)
        download_file(client, params, target_path)

    expected_sha256 = stub.get("sha256")
    if expected_sha256:
        verify_sha256(target_path, expected_sha256)

    print(f"Downloaded {stub['name']} -> {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())