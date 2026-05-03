"""Lightweight CUDA compatibility checker for local benchmark setups.

Run directly:
    python benchmarks/cuda_compat_checker.py

The script emits a JSON report suitable for quick triage on new hardware.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def _run_cmd(cmd: list[str], timeout_s: float = 3.0) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout_s)
        return True, out.strip()
    except Exception as exc:
        return False, str(exc)


def _probe_nvidia_smi() -> dict[str, Any]:
    ok, header_or_err = _run_cmd(["nvidia-smi"], timeout_s=3.0)
    if not ok:
        return {
            "present": False,
            "error": header_or_err,
            "driver_version": None,
            "cuda_runtime_version": None,
            "gpus": [],
        }

    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", header_or_err)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", header_or_err)

    ok_q, gpu_query = _run_cmd(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        timeout_s=3.0,
    )
    gpus: list[dict[str, str]] = []
    if ok_q:
        for line in [ln.strip() for ln in gpu_query.splitlines() if ln.strip()]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpus.append({"name": parts[0], "driver_version": parts[1]})

    return {
        "present": True,
        "error": None,
        "driver_version": driver_match.group(1) if driver_match else None,
        "cuda_runtime_version": cuda_match.group(1) if cuda_match else None,
        "gpus": gpus,
    }


def _probe_cuda_driver_api() -> dict[str, Any]:
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return {
            "libcuda_load_ok": False,
            "libcuda_error": str(exc),
            "cuinit_code": None,
            "driver_api_version": None,
        }

    cu_init = lib.cuInit
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int

    cu_driver_get_version = lib.cuDriverGetVersion
    cu_driver_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_driver_get_version.restype = ctypes.c_int

    cuinit_code = int(cu_init(0))
    version = ctypes.c_int()
    version_code = int(cu_driver_get_version(ctypes.byref(version)))

    return {
        "libcuda_load_ok": True,
        "libcuda_error": None,
        "cuinit_code": cuinit_code,
        "driver_api_version": int(version.value) if version_code == 0 else None,
    }


def _probe_torch() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:
        return {
            "installed": False,
            "error": str(exc),
            "version": None,
            "cuda_built_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    device_name = None
    if cuda_available and device_count > 0:
        try:
            device_name = str(torch.cuda.get_device_name(0))
        except Exception:
            device_name = None

    return {
        "installed": True,
        "error": None,
        "version": str(torch.__version__),
        "cuda_built_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_device_name": device_name,
    }


def run_cuda_compatibility_check() -> dict[str, Any]:
    nvidia = _probe_nvidia_smi()
    driver_api = _probe_cuda_driver_api()
    torch_info = _probe_torch()

    notes: list[str] = []
    compatible = True

    if not nvidia["present"]:
        compatible = False
        notes.append("nvidia-smi not available")

    if not driver_api["libcuda_load_ok"]:
        compatible = False
        notes.append("libcuda.so.1 could not be loaded")

    cuinit_code = driver_api.get("cuinit_code")
    if cuinit_code not in (None, 0):
        compatible = False
        notes.append(f"cuInit failed with code {cuinit_code}")

    if not torch_info["installed"]:
        compatible = False
        notes.append("torch is not installed")
    else:
        if not torch_info["cuda_built_version"]:
            compatible = False
            notes.append("torch was not built with CUDA")
        if not torch_info["cuda_available"]:
            compatible = False
            notes.append("torch.cuda.is_available() is False")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "nvidia_smi": nvidia,
        "cuda_driver_api": driver_api,
        "torch": torch_info,
        "compatible_for_local_cuda_f5": compatible,
        "notes": notes,
    }


def main() -> None:
    report = run_cuda_compatibility_check()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
