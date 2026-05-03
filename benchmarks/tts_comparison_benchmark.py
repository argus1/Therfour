"""TTS latency and failure comparison: Piper vs F5-TTS.

Run from the repo root with the venv active:
    python benchmarks/tts_comparison_benchmark.py

Outputs
-------
- benchmarks/results/TTS_benchmark_raw_<timestamp>.json
- benchmarks/results/MOS_easy.wav
- benchmarks/results/MOS_moderate.wav
- benchmarks/results/MOS_difficult.wav
- benchmarks/results/MOS_combined.wav  (all three with 2 s silence gaps)

F5-TTS is contacted at the endpoint configured in .env (F5_TTS_ENDPOINT,
default http://localhost:8880/synthesize).  When unavailable, synthesis
attempts are recorded as failures with their error details.
"""

from __future__ import annotations

import io
import json
import os
import statistics
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_PIPER_MODEL_CANDIDATES = [
    REPO_ROOT / "models" / "piper" / "en_US-lessac-medium.onnx",
    REPO_ROOT / "models" / "en_US-lessac-medium.onnx",
]
PIPER_MODEL_PATH = str(next((p for p in _PIPER_MODEL_CANDIDATES if p.exists()), _PIPER_MODEL_CANDIDATES[0]))
PIPER_SAMPLE_RATE = 22050

# F5-TTS HTTP endpoint — read from env with fallback
F5_ENDPOINT = os.environ.get("F5_TTS_ENDPOINT", "http://localhost:8880/synthesize")
F5_TIMEOUT_S = float(os.environ.get("F5_TTS_TIMEOUT_S", "15"))
F5_VOICE = os.environ.get("F5_TTS_VOICE", "en_default")

# F5-TTS MLX local
F5_MLX_MODEL = os.environ.get("F5_MLX_MODEL", "lucasnewman/f5-tts-mlx")
F5_MLX_SAMPLE_RATE = 24000
F5_MLX_STEPS = int(os.environ.get("F5_MLX_STEPS", "8"))
F5_MLX_REF_AUDIO_PATH = str(
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "MOS_moderate.wav"
)
# Transcript of MOS_moderate.wav (Piper synthesis of the former moderate passage)
F5_MLX_REF_AUDIO_TEXT = (
    "Research consistently shows that people in crisis benefit most from calm, "
    "non-judgmental responses that validate their experience, "
    "explore available options with them, "
    "and collaboratively identify small, achievable next steps toward safety."
)
_F5_MLX_REF_AUDIO_24K_PATH: str | None = None

SILENCE_DURATION_S = 2.0

# ── passages ──────────────────────────────────────────────────────────────────
PASSAGES: list[dict[str, str]] = [
    {
        "id": "easy",
        "label": "Easy",
        "text": (
            "Take a deep breath. "
            "When you feel overwhelmed, even small steps can help. "
            "Start with something simple: drink a glass of water, step outside, and notice one thing around you."
        ),
    },
    {
        "id": "moderate",
        "label": "Moderate",
        "text": (
            "Cognitive behavioral therapy teaches us to examine the link between our thoughts, feelings, and actions. "
            "By identifying and challenging unhelpful thought patterns, "
            "we can gradually shift our emotional responses and behavioral habits."
        ),
    },
    {
        "id": "difficult",
        "label": "Difficult",
        "text": (
            "Despite the considerable uncertainty surrounding the patient's prognosis "
            "and the competing recommendations from multiple specialists, "
            "the care team reached consensus on a modified treatment protocol "
            "that balanced efficacy, tolerability, and the patient's clearly stated preferences."
        ),
    },
]

WARMUP_TEXT = "Warming up the synthesis engine."
BENCHMARK_RUNS = int(os.environ.get("BENCHMARK_RUNS", "5"))


# ── result types ──────────────────────────────────────────────────────────────
@dataclass
class SynthesisResult:
    backend: str
    passage_id: str
    run_index: int
    latency_ms: float | None
    audio_bytes: int | None
    success: bool
    error: str | None = None


@dataclass
class BenchmarkReport:
    timestamp_utc: str
    piper_model: str
    f5_endpoint: str
    f5_voice: str
    benchmark_runs: int
    results: list[SynthesisResult] = field(default_factory=list)

    # populated by summarise()
    summary: dict[str, Any] = field(default_factory=dict)

    def add(self, r: SynthesisResult) -> None:
        self.results.append(r)

    def summarise(self) -> None:
        for backend in ("piper", "f5", "f5_mlx"):
            for passage in PASSAGES:
                pid = passage["id"]
                runs = [
                    r for r in self.results
                    if r.backend == backend
                    and r.passage_id == pid
                    and r.success
                ]
                failures = [
                    r for r in self.results
                    if r.backend == backend
                    and r.passage_id == pid
                    and not r.success
                ]
                latencies = [r.latency_ms for r in runs if r.latency_ms is not None]
                key = f"{backend}/{pid}"
                self.summary[key] = {
                    "success_count": len(runs),
                    "failure_count": len(failures),
                    "failure_rate": (
                        len(failures) / (len(runs) + len(failures))
                        if (runs or failures) else None
                    ),
                    "p50_ms": statistics.median(latencies) if latencies else None,
                    "p95_ms": (
                        sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2
                        else (latencies[0] if latencies else None)
                    ),
                    "mean_ms": statistics.mean(latencies) if latencies else None,
                    "min_ms": min(latencies) if latencies else None,
                    "max_ms": max(latencies) if latencies else None,
                    "failure_reasons": list({r.error for r in failures if r.error}),
                }


# ── piper synthesis ───────────────────────────────────────────────────────────
def _piper_voice():
    """Load piper voice (cached module-level after first call)."""
    if not hasattr(_piper_voice, "_cache"):
        import piper as piper_lib  # noqa: PLC0415
        _piper_voice._cache = piper_lib.PiperVoice.load(PIPER_MODEL_PATH)
    return _piper_voice._cache


def synthesize_piper(text: str) -> tuple[bytes, float]:
    """Return (wav_bytes, latency_ms). Raises on failure."""
    voice = _piper_voice()
    buf = io.BytesIO()
    wav_file = wave.open(buf, "wb")
    t0 = time.perf_counter()
    voice.synthesize_wav(text, wav_file)
    wav_file.close()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return buf.getvalue(), latency_ms


# ── f5-tts synthesis ──────────────────────────────────────────────────────────
def synthesize_f5(text: str, voice: str = F5_VOICE) -> tuple[bytes, float]:
    """Return (raw_audio_bytes, latency_ms). Raises on failure."""
    payload = {
        "text": text,
        "voice": voice,
        "language": "en-US",
        "options": {},
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=F5_TIMEOUT_S) as client:
        response = client.post(F5_ENDPOINT, json=payload)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"F5-TTS HTTP {response.status_code}: {response.text[:200]}"
        )
    content_type = response.headers.get("content-type", "").lower()
    audio_bytes = response.content
    if "application/json" in content_type:
        import base64  # noqa: PLC0415
        body = response.json()
        encoded = body.get("audio_base64") or body.get("audio") or body.get("data")
        if not encoded:
            raise RuntimeError("F5-TTS JSON response contains no audio field")
        audio_bytes = base64.b64decode(encoded)
    return audio_bytes, latency_ms


# ── audio file helpers ────────────────────────────────────────────────────────
def _wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV to float32 PCM array + sample rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    return pcm, sr


def _silence(duration_s: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    pcm_int16 = np.clip(pcm, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())


def _prepare_f5_mlx_ref_audio_24k() -> str:
    """Return a 24kHz mono WAV path for f5-tts-mlx reference conditioning."""
    global _F5_MLX_REF_AUDIO_24K_PATH
    if _F5_MLX_REF_AUDIO_24K_PATH and os.path.exists(_F5_MLX_REF_AUDIO_24K_PATH):
        return _F5_MLX_REF_AUDIO_24K_PATH

    with wave.open(F5_MLX_REF_AUDIO_PATH, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise RuntimeError(
            f"Unsupported reference WAV sample width: {sample_width * 8} bits"
        )

    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)

    if sample_rate != F5_MLX_SAMPLE_RATE:
        from scipy.signal import resample_poly  # noqa: PLC0415

        pcm = resample_poly(pcm, F5_MLX_SAMPLE_RATE, sample_rate).astype(np.float32)

    pcm_int16 = np.clip(pcm / 32768.0, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767).astype(np.int16)

    tmp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_ref.close()
    with wave.open(tmp_ref.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(F5_MLX_SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())

    _F5_MLX_REF_AUDIO_24K_PATH = tmp_ref.name
    return _F5_MLX_REF_AUDIO_24K_PATH


def _synth_label(text: str, sample_rate: int) -> np.ndarray:
    """Synthesise a spoken label via Piper and return float32 PCM."""
    voice = _piper_voice()
    buf = io.BytesIO()
    wav_f = wave.open(buf, "wb")
    voice.synthesize_wav(text, wav_f)
    wav_f.close()
    pcm, sr = _wav_bytes_to_pcm(buf.getvalue())
    if sr != sample_rate:
        from scipy.signal import resample_poly  # noqa: PLC0415
        pcm = resample_poly(pcm, sample_rate, sr).astype(np.float32)
    return pcm


def generate_mos_files(report: BenchmarkReport) -> dict[str, Path]:
    """
    Generate per-passage WAV files (Piper only) and a combined MOS WAV that
    interleaves Piper and F5-TTS sections for every passage, each preceded by
    a spoken label synthesised via Piper.

    Combined layout (for each passage in order easy → moderate → difficult):
        [spoken: "Piper. <Passage label> passage."]
        [Piper audio]
        [2 s silence]
        [spoken: "F5-TTS. <Passage label> passage."]
        [F5-TTS audio  OR  spoken: "F5-TTS service not available."]
        [2 s silence between passages; no trailing silence after last block]

    Returns {passage_id: piper_wav_path}.
    """
    print("\n── Generating MOS audio files ──")
    sample_rate = PIPER_SAMPLE_RATE
    paths: dict[str, Path] = {}
    combined_segments: list[np.ndarray] = []
    gap = _silence(SILENCE_DURATION_S, sample_rate)
    unavailable_pcm: np.ndarray | None = None  # cached spoken unavailable notice

    for p_idx, passage in enumerate(PASSAGES):
        pid = passage["id"]
        text = passage["text"]
        label = passage["label"]

        # ── Piper ─────────────────────────────────────────────────────────────
        piper_label_text = f"Piper. {label} passage."
        print(f"  [{label}] Piper label …", end=" ", flush=True)
        piper_label_pcm = _synth_label(piper_label_text, sample_rate)
        print(f"ok  |  synthesising passage …", end=" ", flush=True)

        voice = _piper_voice()
        buf = io.BytesIO()
        wav_f = wave.open(buf, "wb")
        voice.synthesize_wav(text, wav_f)
        wav_f.close()
        piper_pcm, sr = _wav_bytes_to_pcm(buf.getvalue())
        if sr != sample_rate:
            from scipy.signal import resample_poly  # noqa: PLC0415
            piper_pcm = resample_poly(piper_pcm, sample_rate, sr).astype(np.float32)

        out_path = RESULTS_DIR / f"MOS_{pid}.wav"
        _write_wav(out_path, piper_pcm, sample_rate)
        paths[pid] = out_path
        print(f"saved → {out_path.name}")

        combined_segments.extend([piper_label_pcm, piper_pcm, gap.copy()])

        # ── F5-TTS MLX ────────────────────────────────────────────────────────
        f5_label_text = f"F5-TTS MLX. {label} passage."
        print(f"  [{label}] F5-TTS MLX label …", end=" ", flush=True)
        f5_label_pcm = _synth_label(f5_label_text, sample_rate)
        print("ok  |  synthesising passage …", end=" ", flush=True)

        # Look for a successful f5_mlx result for this passage in the report
        f5_successful = [
            r for r in report.results
            if r.backend == "f5_mlx" and r.passage_id == pid and r.success
        ]
        if f5_successful:
            # Re-synthesise fresh for the MOS file
            try:
                f5_audio_bytes, _ = synthesize_f5_mlx(text)
                f5_pcm, f5_sr = _wav_bytes_to_pcm(f5_audio_bytes)
                if f5_sr != sample_rate:
                    from scipy.signal import resample_poly  # noqa: PLC0415
                    f5_pcm = resample_poly(f5_pcm, sample_rate, f5_sr).astype(np.float32)
                combined_segments.extend([f5_label_pcm, f5_pcm])
                print(f"ok  ({len(f5_audio_bytes)//1024} KB)")
            except Exception as exc:
                print(f"FAILED ({exc}) — inserting notice")
                f5_successful = []  # fall through to unavailable path

        if not f5_successful:
            if unavailable_pcm is None:
                unavailable_pcm = _synth_label("F5-TTS MLX not available.", sample_rate)
            combined_segments.extend([f5_label_pcm, unavailable_pcm])
            print("not available — spoken notice inserted")

        # 2 s gap after each full passage block (skip after final passage)
        if p_idx < len(PASSAGES) - 1:
            combined_segments.append(gap.copy())

    combined = np.concatenate(combined_segments)
    combined_path = RESULTS_DIR / "MOS_combined.wav"
    _write_wav(combined_path, combined, sample_rate)
    duration_s = len(combined) / sample_rate
    print(f"\n  Combined MOS file → {combined_path.name}  ({duration_s:.1f} s)")
    return paths


# ── F5-TTS MLX local synthesis ───────────────────────────────────────────────
def synthesize_f5_mlx(text: str, speed: float = 1.0) -> tuple[bytes, float]:
    """Synthesise via f5-tts-mlx (Apple Silicon / MLX). Return (wav_bytes, latency_ms)."""
    from f5_tts_mlx.generate import generate  # noqa: PLC0415

    ref_audio_path = _prepare_f5_mlx_ref_audio_24k()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
        t0 = time.perf_counter()
        generate(
            generation_text=text,
            model_name=F5_MLX_MODEL,
            ref_audio_path=ref_audio_path,
            ref_audio_text=F5_MLX_REF_AUDIO_TEXT,
            speed=speed,
            steps=F5_MLX_STEPS,
            output_path=tmp_wav.name,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        tmp_wav.seek(0)
        wav_bytes = tmp_wav.read()

    if not wav_bytes:
        raise RuntimeError("f5-tts-mlx produced empty WAV output")
    return wav_bytes, latency_ms


# ── check f5 availability ─────────────────────────────────────────────────────
def probe_f5_mlx_health() -> tuple[bool, str]:
    """Return (available, note). Checks whether f5-tts-mlx is importable."""
    try:
        import f5_tts_mlx  # noqa: PLC0415, F401
        return True, f"f5-tts-mlx installed (model: {F5_MLX_MODEL})"
    except ImportError:
        return False, "f5-tts-mlx not installed — run: pip install f5-tts-mlx"


def probe_f5_http_health() -> tuple[bool, str]:
    """Return (available, note). Check F5-TTS HTTP endpoint."""
    try:
        r = httpx.post(F5_ENDPOINT, json={"text": "test", "voice": F5_VOICE}, timeout=1.0)
        if r.status_code < 500:
            return True, f"F5-TTS HTTP service at {F5_ENDPOINT}"
    except Exception:
        pass
    return False, f"F5-TTS HTTP not available at {F5_ENDPOINT}"


# ── main benchmark loop ───────────────────────────────────────────────────────
def run_benchmark() -> BenchmarkReport:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = BenchmarkReport(
        timestamp_utc=ts,
        piper_model=PIPER_MODEL_PATH,
        f5_endpoint=F5_ENDPOINT,
        f5_voice=F5_VOICE,
        benchmark_runs=BENCHMARK_RUNS,
    )

    # ── Piper warmup ──────────────────────────────────────────────────────────
    print("── Piper warmup …", end=" ", flush=True)
    try:
        _piper_voice()
        synthesize_piper(WARMUP_TEXT)
        print("ok")
    except Exception as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    # ── F5-TTS availability probes ────────────────────────────────────────────
    print("── Probing f5-tts-mlx …", end=" ", flush=True)
    f5_mlx_available, f5_mlx_probe_note = probe_f5_mlx_health()
    print(f"{'available' if f5_mlx_available else 'UNAVAILABLE'} — {f5_mlx_probe_note}")

    print(f"── Probing F5-TTS HTTP at {F5_ENDPOINT} …", end=" ", flush=True)
    f5_http_available, f5_http_probe_note = probe_f5_http_health()
    print(f"{'available' if f5_http_available else 'UNAVAILABLE'} — {f5_http_probe_note}")

    f5_available = f5_mlx_available or f5_http_available
    f5_probe_note = f5_mlx_probe_note if f5_mlx_available else f5_http_probe_note

    # ── Benchmark runs ────────────────────────────────────────────────────────
    for passage in PASSAGES:
        pid = passage["id"]
        text = passage["text"]
        label = passage["label"]
        print(f"\n── Passage: {label} ({len(text)} chars) ──")

        for run in range(BENCHMARK_RUNS):
            # Piper
            try:
                wav_bytes, latency_ms = synthesize_piper(text)
                report.add(SynthesisResult(
                    backend="piper",
                    passage_id=pid,
                    run_index=run,
                    latency_ms=latency_ms,
                    audio_bytes=len(wav_bytes),
                    success=True,
                ))
                print(f"  piper run {run}: {latency_ms:.1f} ms  {len(wav_bytes)//1024} KB")
            except Exception as exc:
                report.add(SynthesisResult(
                    backend="piper",
                    passage_id=pid,
                    run_index=run,
                    latency_ms=None,
                    audio_bytes=None,
                    success=False,
                    error=str(exc),
                ))
                print(f"  piper run {run}: FAILED — {exc}")

            # F5-TTS MLX local
            if f5_mlx_available:
                try:
                    audio_bytes, latency_ms = synthesize_f5_mlx(text)
                    report.add(SynthesisResult(
                        backend="f5_mlx",
                        passage_id=pid,
                        run_index=run,
                        latency_ms=latency_ms,
                        audio_bytes=len(audio_bytes),
                        success=True,
                    ))
                    print(f"  f5_mlx run {run}: {latency_ms:.1f} ms  {len(audio_bytes)//1024} KB")
                except Exception as exc:
                    report.add(SynthesisResult(
                        backend="f5_mlx",
                        passage_id=pid,
                        run_index=run,
                        latency_ms=None,
                        audio_bytes=None,
                        success=False,
                        error=str(exc),
                    ))
                    print(f"  f5_mlx run {run}: FAILED — {exc}")
            else:
                report.add(SynthesisResult(
                    backend="f5_mlx",
                    passage_id=pid,
                    run_index=run,
                    latency_ms=None,
                    audio_bytes=None,
                    success=False,
                    error=f"f5-tts-mlx not available: {f5_mlx_probe_note}",
                ))

            # F5-TTS HTTP (kept for completeness; usually not available in this env)
            if f5_http_available:
                try:
                    audio_bytes, latency_ms = synthesize_f5(text)
                    report.add(SynthesisResult(
                        backend="f5",
                        passage_id=pid,
                        run_index=run,
                        latency_ms=latency_ms,
                        audio_bytes=len(audio_bytes),
                        success=True,
                    ))
                    print(f"  f5_http run {run}: {latency_ms:.1f} ms  {len(audio_bytes)//1024} KB")
                except Exception as exc:
                    report.add(SynthesisResult(
                        backend="f5",
                        passage_id=pid,
                        run_index=run,
                        latency_ms=None,
                        audio_bytes=None,
                        success=False,
                        error=str(exc),
                    ))
                    print(f"  f5_http run {run}: FAILED — {exc}")

    report.summarise()

    # save raw JSON
    raw_path = RESULTS_DIR / f"TTS_benchmark_raw_{ts}.json"
    with open(raw_path, "w") as fh:
        json.dump(
            {
                "timestamp_utc": report.timestamp_utc,
                "piper_model": report.piper_model,
                "f5_endpoint": report.f5_endpoint,
                "f5_voice": report.f5_voice,
                "benchmark_runs": report.benchmark_runs,
                "f5_mlx_available": f5_mlx_available,
                "f5_mlx_probe_note": f5_mlx_probe_note,
                "f5_http_available": f5_http_available,
                "f5_http_probe_note": f5_http_probe_note,
                "summary": report.summary,
                "results": [asdict(r) for r in report.results],
            },
            fh,
            indent=2,
        )
    print(f"\n── Raw results saved → {raw_path.name}")

    # print summary table
    print("\n── Summary ──")
    fmt = "{:<20} {:<12} {:>10} {:>10} {:>10} {:>10} {:>8}"
    print(fmt.format("backend/passage", "successes", "mean_ms", "p50_ms", "p95_ms", "max_ms", "fail%"))
    print("─" * 90)
    for key, s in report.summary.items():
        fail_pct = f"{s['failure_rate']*100:.0f}%" if s["failure_rate"] is not None else "N/A"
        mean = f"{s['mean_ms']:.1f}" if s["mean_ms"] is not None else "—"
        p50 = f"{s['p50_ms']:.1f}" if s["p50_ms"] is not None else "—"
        p95 = f"{s['p95_ms']:.1f}" if s["p95_ms"] is not None else "—"
        mx = f"{s['max_ms']:.1f}" if s["max_ms"] is not None else "—"
        print(fmt.format(key, str(s["success_count"]), mean, p50, p95, mx, fail_pct))

    return report


if __name__ == "__main__":
    report = run_benchmark()
    generate_mos_files(report)
    print("\nDone.")
