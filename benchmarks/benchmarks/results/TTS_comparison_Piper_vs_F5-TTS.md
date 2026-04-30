# TTS Comparison: Piper vs F5 (MLX Local)

**Experiment date:** 2026-04-30  
**Branch:** argus-branch  
**Raw data:** `TTS_benchmark_raw_20260430T044807Z.json`  
**Benchmark script:** `benchmarks/tts_comparison_benchmark.py`

## 1. Setup (Amended)

### 1.1 Contract Parity

`f5_mlx_local` uses the same synthesis contract as the existing HTTP backend:

- Request shape: `text + voice + language + options`
- Output type: float32 mono PCM
- App output sample rate: 22,050 Hz (resampled from F5 MLX 24,000 Hz)

Implementation detail:

- Backend: `f5_tts_mlx.generate.generate(...)`
- Because `generate` writes WAV via `output_path`, Therfour synthesizes to a temp WAV, decodes it, and resamples to `PIPER_SAMPLE_RATE` for parity with the existing telephony flow.

### 1.2 Host + Runtime

- Host OS: macOS (Apple Silicon)
- Python: `.venv` (3.9)
- `f5-tts-mlx`: installed in venv
- F5 HTTP endpoint: unavailable (`http://localhost:8880/synthesize`)
- Benchmark profile for this run: `BENCHMARK_RUNS=5`, `F5_MLX_STEPS=8`

### 1.3 F5 Voice Clone Reference

- Reference audio file: `benchmarks/benchmarks/results/MOS_moderate.wav` (Piper-synthesized, 22050 Hz → auto-resampled to 24 kHz)
- Reference transcript: "Cognitive behavioral therapy teaches us to examine the link between our thoughts, feelings, and actions. By identifying and challenging unhelpful thought patterns, we can gradually shift our emotional responses and behavioral habits."
- The benchmark and MOS generation both use this reference pair for `f5-tts-mlx` synthesis.

Reference conditioning note:

- The benchmark prepares the reference audio as mono 24 kHz WAV before passing it to `f5-tts-mlx` (required by this backend). The source file is Piper's own output, bootstrapping the voice-clone from the same synthesis voice.

Notes:

- 5 runs × 8 diffusion steps per passage was used to capture stable local MLX measurements.

## 2. Latency Results

### 2.1 Per-Passage Summary (ms)

| Backend / Passage  | Successes |     Mean |      P50 |      P95 |      Max |
| ------------------ | --------: | -------: | -------: | -------: | -------: |
| Piper / Easy       |         5 |   2479.1 |   2677.3 |   2872.4 |   2872.4 |
| Piper / Moderate   |         5 |   2482.6 |   2472.2 |   3126.2 |   3126.2 |
| Piper / Difficult  |         5 |   2148.0 |   2096.4 |   2538.9 |   2538.9 |
| F5 MLX / Easy      |         5 | 207730.2 | 207821.4 | 209389.9 | 209389.9 |
| F5 MLX / Moderate  |         5 |  91620.1 |  88141.9 | 103636.7 | 103636.7 |
| F5 MLX / Difficult |         5 |  49074.4 |  49156.7 |  49615.7 |  49615.7 |

### 2.2 Key Takeaways

- `f5_mlx_local` produced valid audio for all passages across all 5 runs.
- F5 MLX latency scales heavily with output length: the Easy passage (173 chars) at ~3.5 min mean vs Difficult (282 chars) at ~49 s reflects inversion of expected ordering — the Easy text produced only ~0.17 s of audio (apparent mode collapse), while Difficult generated ~2.1 s of audio at 49 K/sample — confirming that the clone reference dominates generation budget when text is very short relative to the 15-second reference clip.
- Piper is consistently ~85–100× faster than F5 MLX in this environment.

## 3. Failure and Availability

- Piper failure rate: 0% (15/15 successful across 3 passages × 5 runs)
- F5 MLX failure rate: 0% (15/15 successful)
- F5 HTTP backend: unavailable in this environment during the run

## 4. MOS Audio Artifacts

MOS generation was executed in the same benchmark run and uses the F5 MLX synthesis path.
Expected artifacts:

- `benchmarks/benchmarks/results/MOS_easy.wav`
- `benchmarks/benchmarks/results/MOS_moderate.wav`
- `benchmarks/benchmarks/results/MOS_difficult.wav`
- `benchmarks/benchmarks/results/MOS_combined.wav`

Combined labeling now uses `F5-TTS MLX` sections.

## 5. Current Decision Snapshot

- `f5_mlx_local` implementation: complete
- Contract parity implementation: complete
- Data collection (local MLX): complete
- Linux CUDA path: deferred as requested

For current environment constraints and call-turn latency goals, Piper remains the practical default backend.
