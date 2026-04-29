#!/usr/bin/env python3
"""Reproducible STT benchmark harness for Therfour.

Compares faster-whisper models on telephony-oriented samples from:
https://github.com/voxserv/audio_quality_testing_samples
"""

from __future__ import annotations

import argparse
import csv
import difflib
import itertools
import json
import math
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from scipy.signal import resample_poly

SAMPLES_REPO_URL = "https://github.com/voxserv/audio_quality_testing_samples"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SAMPLES_DIR = REPO_ROOT / "third_party" / "audio_quality_testing_samples"
DEFAULT_NOISE_DIR = SCRIPT_DIR / "ambient_noise"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _sysctl(key: str) -> str:
    """Return the string value of a sysctl key, or '' on failure."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=2
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _cpu_brand_linux() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[-1].strip()
    except Exception:
        pass
    return ""


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "brand": "",
        "arch": platform.machine(),
        "logical_cores": None,
        "physical_cores": None,
        "max_freq_mhz": None,
    }

    system = platform.system()

    if system == "Darwin":
        info["brand"] = _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model")
        raw_cores = _sysctl("hw.physicalcpu")
        if raw_cores.isdigit():
            info["physical_cores"] = int(raw_cores)
        raw_lcores = _sysctl("hw.logicalcpu")
        if raw_lcores.isdigit():
            info["logical_cores"] = int(raw_lcores)
        # hw.cpufrequency_max not present on Apple Silicon; try anyway
        raw_freq = _sysctl("hw.cpufrequency_max")
        if raw_freq.isdigit():
            info["max_freq_mhz"] = round(int(raw_freq) / 1_000_000, 1)
    elif system == "Linux":
        info["brand"] = _cpu_brand_linux()
        raw_cores = _sysctl("kernel.nproc")  # fallback; may be empty
        # Try os.cpu_count as a last resort for logical cores
        import os
        info["logical_cores"] = os.cpu_count()
    else:
        # Windows / other
        info["brand"] = platform.processor()

    if not info["brand"]:
        info["brand"] = platform.processor() or "unknown"

    return info


def _ram_gb() -> float | None:
    system = platform.system()
    try:
        if system == "Darwin":
            raw = _sysctl("hw.memsize")
            if raw.isdigit():
                return round(int(raw) / (1024 ** 3), 2)
        elif system == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(re.search(r"\d+", line).group())  # type: ignore[union-attr]
                        return round(kb / (1024 ** 2), 2)
    except Exception:
        pass
    return None


def _gpu_info() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    try:
        import torch

        # CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "index": i,
                        "name": props.name,
                        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
                        "backend": "cuda",
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
        # Apple Metal (MPS) — unified memory; report system RAM as "vram"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            ram = _ram_gb()
            gpus.append(
                {
                    "index": 0,
                    "name": platform.machine() + " (Apple Silicon / Metal)",
                    "vram_gb": ram,  # unified memory — same pool as RAM
                    "backend": "mps",
                    "note": "Unified memory architecture; VRAM = system RAM",
                }
            )
    except ImportError:
        pass
    return gpus


def collect_hardware_info() -> dict[str, Any]:
    try:
        import torch
        torch_version: str | None = torch.__version__
    except ImportError:
        torch_version = None

    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch_version": torch_version,
        "cpu": _cpu_info(),
        "ram_gb": _ram_gb(),
        "gpus": _gpu_info(),
    }


@dataclass
class FileBenchmarkResult:
    model: str
    file_path: str
    scenario: str
    noise_file: str
    target_snr_db: float | None
    duration_s: float
    elapsed_s: float
    rtf: float
    text_chars: int
    text_words: int
    language: str
    language_probability: float
    robustness_score: float | None
    robustness_reference_chars: int


@dataclass
class ModelSummary:
    model: str
    files: int
    total_audio_s: float
    total_elapsed_s: float
    avg_rtf: float
    p50_rtf: float
    p95_rtf: float
    p50_latency_s: float
    p95_latency_s: float
    avg_noisy_robustness: float
    p50_noisy_robustness: float
    p10_noisy_robustness: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark small vs distil-large-v3 on telephony samples")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Local path where the sample repository will exist",
    )
    parser.add_argument(
        "--samples-repo-url",
        default=SAMPLES_REPO_URL,
        help="Sample repository URL",
    )
    parser.add_argument(
        "--subdir",
        choices=["testaudio", "orig", "all"],
        default="testaudio",
        help="Which dataset subset to benchmark",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["small", "distil-large-v3"],
        help="Models to compare",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="faster-whisper compute_type. Use auto to pick float16 on cuda, int8 on cpu",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language hint. Use empty string for auto detect",
    )
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vad-filter", action="store_true", default=True)
    parser.add_argument("--no-vad-filter", action="store_false", dest="vad_filter")
    parser.add_argument(
        "--preprocess-profile",
        choices=["telephony", "none"],
        default="telephony",
        help="Audio preprocessing before transcription",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=DEFAULT_NOISE_DIR,
        help="Directory containing background noise wav files",
    )
    parser.add_argument(
        "--noise-snr-db",
        type=float,
        nargs="+",
        default=[0.0, -5.0, -10.0],
        help="Target SNR values for speech-vs-noise stress scenarios",
    )
    parser.add_argument(
        "--noise-start-offset-s",
        type=float,
        default=0.0,
        help="Offset into noise clip before mixing (seconds)",
    )
    parser.add_argument(
        "--noise-combo-max-size",
        type=int,
        default=2,
        help="Maximum number of simultaneous noise files in a combo scenario",
    )
    parser.add_argument(
        "--noise-max-combinations",
        type=int,
        default=12,
        help="Cap the number of generated noise combinations",
    )
    parser.add_argument(
        "--include-clean",
        action="store_true",
        default=True,
        help="Include clean speech scenario",
    )
    parser.add_argument("--no-clean", action="store_false", dest="include_clean")
    parser.add_argument(
        "--edge-cases",
        action="store_true",
        default=True,
        help="Include extra robustness scenarios (very_low_speech, clipped, leading_noise)",
    )
    parser.add_argument("--no-edge-cases", action="store_false", dest="edge_cases")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to run each file")
    parser.add_argument("--max-files", type=int, default=0, help="Limit files for quick test runs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON and CSV outputs are written",
    )
    parser.add_argument(
        "--auto-clone",
        action="store_true",
        default=True,
        help="Clone sample repo if missing",
    )
    parser.add_argument(
        "--no-auto-clone",
        action="store_false",
        dest="auto_clone",
        help="Fail if sample repo is missing",
    )
    return parser.parse_args()


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(raw_device: str) -> str:
    if raw_device != "auto":
        return raw_device
    return "cuda" if _has_cuda() else "cpu"


def resolve_compute_type(raw_compute_type: str, device: str) -> str:
    if raw_compute_type != "auto":
        return raw_compute_type
    return "float16" if device == "cuda" else "int8"


def ensure_samples_repo(samples_dir: Path, repo_url: str, auto_clone: bool) -> None:
    if samples_dir.exists():
        return
    if not auto_clone:
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    samples_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", repo_url, str(samples_dir)]
    subprocess.run(cmd, check=True)


def collect_audio_files(samples_dir: Path, subdir: str, max_files: int) -> list[Path]:
    roots: Sequence[Path]
    if subdir == "all":
        roots = [samples_dir / "testaudio", samples_dir / "orig"]
    else:
        roots = [samples_dir / subdir]

    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.wav")))

    if max_files > 0:
        files = files[:max_files]
    return files


def collect_noise_files(noise_dir: Path) -> list[Path]:
    if not noise_dir.exists():
        return []
    return sorted(noise_dir.rglob("*.wav"))


def _to_pcm16_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _from_pcm16_bytes(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def _mulaw_roundtrip(pcm16: bytes) -> bytes:
    try:
        import audioop  # type: ignore[import]
    except ImportError:
        import audioop_lts as audioop  # type: ignore[no-redef]
    mulaw = audioop.lin2ulaw(pcm16, 2)
    return audioop.ulaw2lin(mulaw, 2)


def preprocess_audio_for_stt(audio_16k: np.ndarray, profile: str) -> np.ndarray:
    """Mirror production path: 16k float -> 8k PCM16 -> mu-law -> 8k PCM16 -> 16k float."""
    if profile == "none":
        return audio_16k.astype(np.float32)

    if profile != "telephony":
        raise ValueError(f"Unknown preprocess profile: {profile}")

    # 16k float -> 8k float
    audio_8k = resample_poly(audio_16k.astype(np.float32), 8000, 16000)
    # 8k float -> PCM16 bytes
    pcm8k = _to_pcm16_bytes(audio_8k)
    # μ-law encode/decode roundtrip to emulate Twilio transport codec
    pcm8k_roundtrip = _mulaw_roundtrip(pcm8k)
    # back to float and upsample to 16k for Whisper
    float8k = _from_pcm16_bytes(pcm8k_roundtrip)
    return resample_poly(float8k, 16000, 8000).astype(np.float32)


def _normalize_rms(signal: np.ndarray, target_rms: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(signal))) + 1e-12)
    if rms <= 0:
        return signal
    return (signal * (target_rms / rms)).astype(np.float32)


def _tile_or_trim(signal: np.ndarray, target_len: int, offset_samples: int = 0) -> np.ndarray:
    if len(signal) == 0 or target_len <= 0:
        return np.zeros(max(0, target_len), dtype=np.float32)

    start = max(0, offset_samples)
    if start >= len(signal):
        start = start % len(signal)

    segment = signal[start:]
    if len(segment) >= target_len:
        return segment[:target_len].astype(np.float32)

    reps = int(math.ceil(target_len / len(segment))) if len(segment) > 0 else 1
    tiled = np.tile(segment, reps)
    return tiled[:target_len].astype(np.float32)


def _mix_noise_sources(
    noise_sources: Sequence[np.ndarray],
    *,
    target_len: int,
    noise_start_offset_s: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    if not noise_sources:
        return np.zeros(target_len, dtype=np.float32)

    offset_samples = int(noise_start_offset_s * sample_rate)
    aligned: list[np.ndarray] = []
    for source in noise_sources:
        fit = _tile_or_trim(source.astype(np.float32), target_len, offset_samples)
        fit_rms = float(np.sqrt(np.mean(np.square(fit))) + 1e-12)
        if fit_rms > 0:
            fit = fit / fit_rms
        aligned.append(fit)

    stacked = np.stack(aligned, axis=0)
    # Average preserves relative contribution and avoids exploding amplitude.
    return np.mean(stacked, axis=0).astype(np.float32)


def _noise_combinations(
    noise_files: Sequence[Path],
    *,
    max_size: int,
    max_combinations: int,
) -> list[tuple[Path, ...]]:
    combos: list[tuple[Path, ...]] = []
    sorted_files = sorted(noise_files)
    max_size = max(1, min(max_size, len(sorted_files)))

    for size in range(1, max_size + 1):
        for combo in itertools.combinations(sorted_files, size):
            combos.append(combo)
            if max_combinations > 0 and len(combos) >= max_combinations:
                return combos
    return combos


def mix_with_noise(
    speech: np.ndarray,
    noise: np.ndarray,
    *,
    target_snr_db: float,
    noise_start_offset_s: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    speech = speech.astype(np.float32)
    offset_samples = int(noise_start_offset_s * sample_rate)
    noise_fit = _tile_or_trim(noise.astype(np.float32), len(speech), offset_samples)

    speech_rms = float(np.sqrt(np.mean(np.square(speech))) + 1e-12)
    # target_snr_db = 20*log10(speech_rms / noise_rms_target)
    noise_rms_target = speech_rms / (10.0 ** (target_snr_db / 20.0))
    noise_scaled = _normalize_rms(noise_fit, noise_rms_target)

    mixed = speech + noise_scaled
    peak = float(np.max(np.abs(mixed)) + 1e-12)
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


@dataclass(frozen=True)
class AudioScenario:
    name: str
    noise_file: str
    target_snr_db: float | None
    audio: np.ndarray


def build_scenarios(
    speech_16k: np.ndarray,
    noise_files: Sequence[Path],
    *,
    snr_values: Sequence[float],
    noise_start_offset_s: float,
    noise_combo_max_size: int,
    noise_max_combinations: int,
    include_clean: bool,
    edge_cases: bool,
) -> list[AudioScenario]:
    scenarios: list[AudioScenario] = []

    if include_clean:
        scenarios.append(
            AudioScenario(
                name="clean",
                noise_file="",
                target_snr_db=None,
                audio=speech_16k.astype(np.float32),
            )
        )

    noise_cache: dict[Path, np.ndarray] = {
        noise_path: decode_audio(str(noise_path), sampling_rate=16000).astype(np.float32)
        for noise_path in noise_files
    }
    for combo in _noise_combinations(
        noise_files,
        max_size=noise_combo_max_size,
        max_combinations=noise_max_combinations,
    ):
        combo_noise = _mix_noise_sources(
            [noise_cache[path] for path in combo],
            target_len=len(speech_16k),
            noise_start_offset_s=noise_start_offset_s,
        )
        combo_label = "+".join(path.stem for path in combo)
        combo_sources = ";".join(str(path) for path in combo)
        for snr in snr_values:
            mixed = mix_with_noise(
                speech_16k,
                combo_noise,
                target_snr_db=snr,
                noise_start_offset_s=0.0,
            )
            scenarios.append(
                AudioScenario(
                    name=f"noise_combo_{combo_label}_snr_{snr:g}dB",
                    noise_file=combo_sources,
                    target_snr_db=float(snr),
                    audio=mixed,
                )
            )

    if edge_cases:
        if include_clean:
            # Force speech quieter than typical case.
            very_low = (speech_16k * 0.2).astype(np.float32)
            scenarios.append(
                AudioScenario(
                    name="edge_very_low_speech",
                    noise_file="",
                    target_snr_db=None,
                    audio=very_low,
                )
            )

            # Simulate clipping distortion from aggressive input gain.
            clipped = np.clip(speech_16k * 2.5, -0.6, 0.6).astype(np.float32)
            scenarios.append(
                AudioScenario(
                    name="edge_clipped",
                    noise_file="",
                    target_snr_db=None,
                    audio=clipped,
                )
            )

        # Leading non-speech energy that can confuse VAD and turn detection.
        if noise_files:
            lead_noise = decode_audio(str(noise_files[0]), sampling_rate=16000).astype(np.float32)
            lead_noise = _tile_or_trim(lead_noise, int(2.0 * 16000))
            padded = np.concatenate([lead_noise, speech_16k.astype(np.float32)])
            scenarios.append(
                AudioScenario(
                    name="edge_leading_noise_2s",
                    noise_file=str(noise_files[0]),
                    target_snr_db=None,
                    audio=padded,
                )
            )

    return scenarios


def percentile(values: Iterable[float], q: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    index = (len(sorted_values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(sorted_values) - 1)
    fraction = index - lo
    return float(sorted_values[lo] * (1 - fraction) + sorted_values[hi] * fraction)


def _normalize_text_for_scoring(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def compute_robustness_score(reference_text: str, hypothesis_text: str) -> float | None:
    ref = _normalize_text_for_scoring(reference_text)
    hyp = _normalize_text_for_scoring(hypothesis_text)
    if not ref:
        return None
    if not hyp:
        return 0.0
    return round(difflib.SequenceMatcher(None, ref, hyp).ratio(), 4)


def benchmark_model(
    model_name: str,
    files: Sequence[Path],
    noise_files: Sequence[Path],
    *,
    preprocess_profile: str,
    noise_snr_db: Sequence[float],
    noise_start_offset_s: float,
    noise_combo_max_size: int,
    noise_max_combinations: int,
    include_clean: bool,
    edge_cases: bool,
    device: str,
    compute_type: str,
    language: str | None,
    beam_size: int,
    temperature: float,
    vad_filter: bool,
    repeats: int,
) -> tuple[list[FileBenchmarkResult], ModelSummary]:
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    rows: list[FileBenchmarkResult] = []
    for path in files:
        file_run_rows: list[dict[str, Any]] = []
        source_audio = decode_audio(str(path), sampling_rate=16000).astype(np.float32)
        scenarios = build_scenarios(
            source_audio,
            noise_files,
            snr_values=noise_snr_db,
            noise_start_offset_s=noise_start_offset_s,
            noise_combo_max_size=noise_combo_max_size,
            noise_max_combinations=noise_max_combinations,
            include_clean=include_clean,
            edge_cases=edge_cases,
        )

        for scenario in scenarios:
            audio = preprocess_audio_for_stt(scenario.audio, preprocess_profile)
            duration_s = len(audio) / 16000.0

            for _ in range(repeats):
                start = time.perf_counter()
                segments, info = model.transcribe(
                    audio,
                    language=language,
                    beam_size=beam_size,
                    temperature=temperature,
                    vad_filter=vad_filter,
                    condition_on_previous_text=False,
                )
                elapsed_s = time.perf_counter() - start
                text = " ".join(segment.text.strip() for segment in segments).strip()
                words = [w for w in text.split() if w]
                rtf = (elapsed_s / duration_s) if duration_s > 0 else 0.0

                file_run_rows.append(
                    {
                        "model": model_name,
                        "file_path": str(path),
                        "scenario": scenario.name,
                        "noise_file": scenario.noise_file,
                        "target_snr_db": scenario.target_snr_db,
                        "duration_s": round(duration_s, 4),
                        "elapsed_s": round(elapsed_s, 4),
                        "rtf": round(rtf, 4),
                        "text": text,
                        "text_chars": len(text),
                        "text_words": len(words),
                        "language": str(getattr(info, "language", "") or ""),
                        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
                    }
                )

        clean_candidates = [item for item in file_run_rows if item["scenario"] == "clean"]
        if clean_candidates:
            best_clean = max(clean_candidates, key=lambda item: item["text_chars"])
            reference_text = best_clean["text"]
            reference_chars = int(best_clean["text_chars"])
        else:
            reference_text = ""
            reference_chars = 0

        for item in file_run_rows:
            robustness_score = compute_robustness_score(reference_text, item["text"])
            rows.append(
                FileBenchmarkResult(
                    model=item["model"],
                    file_path=item["file_path"],
                    scenario=item["scenario"],
                    noise_file=item["noise_file"],
                    target_snr_db=item["target_snr_db"],
                    duration_s=item["duration_s"],
                    elapsed_s=item["elapsed_s"],
                    rtf=item["rtf"],
                    text_chars=item["text_chars"],
                    text_words=item["text_words"],
                    language=item["language"],
                    language_probability=item["language_probability"],
                    robustness_score=robustness_score,
                    robustness_reference_chars=reference_chars,
                )
            )

    total_audio = sum(item.duration_s for item in rows)
    total_elapsed = sum(item.elapsed_s for item in rows)
    rtfs = [item.rtf for item in rows]
    latencies = [item.elapsed_s for item in rows]
    noisy_robustness = [
        item.robustness_score
        for item in rows
        if item.robustness_score is not None and item.scenario != "clean"
    ]

    summary = ModelSummary(
        model=model_name,
        files=len(rows),
        total_audio_s=round(total_audio, 4),
        total_elapsed_s=round(total_elapsed, 4),
        avg_rtf=round((total_elapsed / total_audio) if total_audio > 0 else 0.0, 4),
        p50_rtf=round(percentile(rtfs, 0.50), 4),
        p95_rtf=round(percentile(rtfs, 0.95), 4),
        p50_latency_s=round(percentile(latencies, 0.50), 4),
        p95_latency_s=round(percentile(latencies, 0.95), 4),
        avg_noisy_robustness=round(sum(noisy_robustness) / len(noisy_robustness), 4) if noisy_robustness else 0.0,
        p50_noisy_robustness=round(percentile(noisy_robustness, 0.50), 4),
        p10_noisy_robustness=round(percentile(noisy_robustness, 0.10), 4),
    )
    return rows, summary


def write_outputs(
    output_dir: Path,
    *,
    run_id: str,
    config: dict,
    hardware: dict,
    rows: Sequence[FileBenchmarkResult],
    summaries: Sequence[ModelSummary],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"whisper_benchmark_{run_id}.json"
    csv_path = output_dir / f"whisper_benchmark_{run_id}.csv"

    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "config": config,
        "model_summaries": [asdict(summary) for summary in summaries],
        "file_results": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(asdict(item) for item in rows)

    return json_path, csv_path


def print_summary(summaries: Sequence[ModelSummary]) -> None:
    print("\nModel comparison summary:")
    print(
        "model                files  total_audio_s  total_elapsed_s  avg_rtf  "
        "p95_latency_s  avg_noisy_robustness  p10_noisy_robustness"
    )
    for summary in summaries:
        print(
            f"{summary.model:<20} {summary.files:>5}  {summary.total_audio_s:>13.2f}  "
            f"{summary.total_elapsed_s:>15.2f}  {summary.avg_rtf:>7.3f}  {summary.p95_latency_s:>13.3f}  "
            f"{summary.avg_noisy_robustness:>20.3f}  {summary.p10_noisy_robustness:>20.3f}"
        )


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    compute_type = resolve_compute_type(args.compute_type, device)

    ensure_samples_repo(args.samples_dir, args.samples_repo_url, args.auto_clone)
    files = collect_audio_files(args.samples_dir, args.subdir, args.max_files)
    if not files:
        print("No .wav files found to benchmark.", file=sys.stderr)
        return 2
    noise_files = collect_noise_files(args.noise_dir)
    if args.noise_snr_db and not noise_files:
        print(
            f"Warning: noise SNR scenarios requested but no noise files found in {args.noise_dir}. Running clean/edge-only scenarios.",
            file=sys.stderr,
        )

    language = args.language.strip() or None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("Collecting hardware info...")
    hardware = collect_hardware_info()
    config = {
        "samples_dir": str(args.samples_dir),
        "samples_repo_url": args.samples_repo_url,
        "subdir": args.subdir,
        "models": args.models,
        "device": device,
        "compute_type": compute_type,
        "language": language,
        "beam_size": args.beam_size,
        "temperature": args.temperature,
        "vad_filter": args.vad_filter,
        "preprocess_profile": args.preprocess_profile,
        "noise_dir": str(args.noise_dir),
        "noise_files": [str(path) for path in noise_files],
        "noise_snr_db": list(args.noise_snr_db),
        "noise_start_offset_s": args.noise_start_offset_s,
        "noise_combo_max_size": args.noise_combo_max_size,
        "noise_max_combinations": args.noise_max_combinations,
        "include_clean": args.include_clean,
        "edge_cases": args.edge_cases,
        "repeats": args.repeats,
        "max_files": args.max_files,
        "files": [str(path) for path in files],
    }

    all_rows: list[FileBenchmarkResult] = []
    summaries: list[ModelSummary] = []

    for model_name in args.models:
        print(f"Benchmarking model: {model_name}")
        rows, summary = benchmark_model(
            model_name,
            files,
            noise_files,
            preprocess_profile=args.preprocess_profile,
            noise_snr_db=args.noise_snr_db,
            noise_start_offset_s=args.noise_start_offset_s,
            noise_combo_max_size=args.noise_combo_max_size,
            noise_max_combinations=args.noise_max_combinations,
            include_clean=args.include_clean,
            edge_cases=args.edge_cases,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=args.beam_size,
            temperature=args.temperature,
            vad_filter=args.vad_filter,
            repeats=args.repeats,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    json_path, csv_path = write_outputs(
        args.output_dir,
        run_id=run_id,
        config=config,
        hardware=hardware,
        rows=all_rows,
        summaries=summaries,
    )

    print_summary(summaries)
    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
