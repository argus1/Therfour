# EC2 Readiness for End-to-End Benchmarking

This document evaluates readiness for running Therfour benchmark workflows (with focus on the current TTS benchmark path: Piper vs local F5 CUDA) on:

- **EC2 G4** (for example `g4dn.*`, x86_64 + NVIDIA T4)
- **EC2 T4g** (for example `t4g.*`, arm64/Graviton, no NVIDIA GPU)

Evaluation date: **2026-05-03**

## Scope and assumptions

- "End-to-end benchmark" in this context means running the repository benchmark harnesses locally on the instance, including the current CUDA-enabled TTS comparison flow in `benchmarks/tts_comparison_benchmark.py`.
- The TTS benchmark now writes artifacts to `benchmarks/benchmarks/results/TTS/CUDA/`.
- Current code expects optional local CUDA F5 inference via `f5_tts.api.F5TTS` and validates CUDA with `benchmarks/cuda_compat_checker.py`.

## Executive summary

- **EC2 G4**: **Conditionally ready** for full local CUDA TTS benchmarking after environment setup (NVIDIA driver/CUDA runtime/PyTorch CUDA wheel + benchmark extras).
- **EC2 T4g**: **Not ready** for local CUDA TTS benchmarking by design (no NVIDIA GPU). Possible only with architecture-aware fallback strategy (CPU-only Piper and/or remote F5 HTTP backend).

## Readiness matrix

| Capability | EC2 G4 (g4dn) | EC2 T4g |
|---|---|---|
| CPU architecture compatibility | ✅ x86_64 | ⚠️ arm64 (some repo defaults assume x86_64) |
| NVIDIA GPU available | ✅ | ❌ |
| Local CUDA F5 benchmark path (`f5_cuda_local`) | ✅ with setup | ❌ not possible locally |
| Piper local benchmark path | ✅ | ⚠️ possible with arm64-appropriate Piper binary/runtime |
| Current Dockerfile compatibility | ✅ | ✅ architecture-aware Piper binary download |
| Full local Piper vs F5 CUDA comparison | ✅ | ❌ |

## EC2 G4 readiness details

### What is already aligned

- Hardware class supports CUDA benchmarking (NVIDIA T4).
- Current benchmark harness includes a dedicated CUDA compatibility checker (`benchmarks/cuda_compat_checker.py`).
- Current local F5 CUDA backend path is implemented in `benchmarks/tts_comparison_benchmark.py` (`f5_cuda_local`).

### What must be in place before running

1. NVIDIA driver and runtime visible to userspace (`nvidia-smi` healthy).
2. CUDA-compatible PyTorch build (for example, torch with a CUDA build matching the driver/runtime).
3. Benchmark dependencies present in the Python environment:
   - `f5-tts` (local CUDA backend)
   - `piper-tts` (Python `piper` module used by benchmark script)
4. Piper model artifact present at one of:
   - `models/piper/en_US-lessac-medium.onnx`
   - `models/en_US-lessac-medium.onnx`
5. Optional but recommended: Hugging Face auth token for faster model downloads and reduced throttling.

### Current repository gaps affecting turnkey G4 setup

- Benchmark-only dependencies are now formalized in `requirements-bench.txt` (includes `f5-tts` and `piper-tts`).
- First-time local F5 runs will download large model/vocoder artifacts, so warm-up time and disk requirements are non-trivial.

### G4 verdict

**Ready with setup steps**. Operational risk is moderate until benchmark-only dependencies are formalized in environment/bootstrap instructions.

## EC2 T4g readiness details

### Hard blockers for local CUDA benchmarking

- T4g instances are Graviton (arm64) and do not provide NVIDIA GPUs.
- Therefore, local CUDA path (`f5_cuda_local`) is not viable.

### Additional architecture caveat

- CUDA remains unavailable on T4g by hardware design, but the Dockerfile now supports arm64 Piper binary selection for non-CUDA paths.

### What is still feasible on T4g

- CPU-only benchmark subsets (for example Piper-only) **if** arm64-compatible Piper runtime is installed.
- Hybrid benchmark mode using:
  - local Piper on T4g
  - remote F5 service (`F5_TTS_ENDPOINT`) hosted on a GPU-capable node (for example G4).

### T4g verdict

**Not ready for full local end-to-end CUDA TTS benchmarking**. Use T4g for CPU/control-plane roles or remote-inference topologies.

## Practical recommendation

- Use **G4** as the primary benchmark execution environment for local CUDA TTS benchmarking.
- Use **T4g** only for:
  - non-CUDA benchmark subsets, or
  - orchestrating/serving workloads that call a remote GPU inference endpoint.

## Suggested next improvements (repo-level)

1. Add explicit EC2 benchmark setup snippets for G4 in docs (driver/runtime, torch CUDA wheel selection, warm-cache step).
2. Add benchmark mode flags to skip unavailable backends cleanly in mixed environments.
3. Consider adding an arm64 CI smoke test for Piper-only benchmark flow.
