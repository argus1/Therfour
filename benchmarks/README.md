# STT Benchmark Harness

This folder contains a reproducible benchmark harness to compare Whisper model tiers for Therfour.

Primary comparison target:

- `small`
- `distil-large-v3`

Telephony sample source:

- https://github.com/voxserv/audio_quality_testing_samples

## What It Measures

For each model and each audio file, the harness records:

- decode latency (`elapsed_s`)
- audio duration (`duration_s`)
- real-time factor (`rtf = elapsed_s / duration_s`)
- transcript length (`text_chars`, `text_words`)
- detected language metadata
- robustness score (`robustness_score`) against the clean transcript for the same model+file

It writes both JSON and CSV outputs under `benchmarks/results/`.

It also records hardware metadata (`cpu`, `ram_gb`, `gpus`) and scenario metadata (`scenario`, `noise_file`, `target_snr_db`).

Robustness scoring details:

- baseline reference is the best clean transcript (highest `text_chars`) for that model+file
- transcripts are normalized (lowercase, punctuation stripped, spaces collapsed)
- score is `SequenceMatcher` ratio in `[0, 1]` where `1.0` means closest match to clean transcript
- model summary includes `avg_noisy_robustness`, `p50_noisy_robustness`, and `p10_noisy_robustness`

## Quick Start

From repo root:

```bash
/Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/compare_whisper_models.py \
  --models small distil-large-v3 \
  --subdir testaudio \
  --device auto \
  --repeats 1
```

If samples are missing, the harness clones:

- `https://github.com/voxserv/audio_quality_testing_samples`

into:

- `third_party/audio_quality_testing_samples`

## Reproducibility Guidance

Use fixed settings when comparing models:

- same sample subset (`--subdir`)
- same language hint (`--language`)
- same decode parameters (`--beam-size`, `--temperature`, `--vad-filter`)
- same hardware and load conditions
- run at least 3 repeats for stable p95 comparisons

For preprocessing robustness experiments, keep these fixed too:

- same preprocessing profile (`--preprocess-profile`)
- same noise set and combo policy (`--noise-dir`, `--noise-combo-max-size`, `--noise-max-combinations`)
- same SNR sweep (`--noise-snr-db`)

Example with stronger statistical stability:

```bash
/Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/compare_whisper_models.py \
  --models small distil-large-v3 \
  --subdir testaudio \
  --device auto \
  --repeats 3
```

## Useful Flags

- `--samples-dir`: custom location of sample repo
- `--no-auto-clone`: fail instead of cloning samples
- `--subdir {testaudio,orig,all}`: choose telephony-focused or source audio sets
- `--max-files N`: quick smoke-run limit
- `--compute-type`: override `auto` selection
- `--no-vad-filter`: disable Whisper internal VAD for controlled experiments
- `--preprocess-profile {telephony,none}`: run production-like codec/resample path or raw decode
- `--noise-dir`: directory of background noise files to mix with speech
- `--noise-snr-db`: SNR sweep values (supports negative SNR where speech is quieter than noise)
- `--noise-combo-max-size`: max simultaneous noise files in one scenario (default 2)
- `--noise-max-combinations`: cap generated noise combinations (default 12)
- `--no-edge-cases`: disable extra robustness scenarios (very low speech, clipped audio, leading noise)

## Stress-Test Example

Run all single noise files plus some pairwise combinations, with negative SNR sweeps:

```bash
/Users/argussun/Documents/Therfour/.venv/bin/python benchmarks/compare_whisper_models.py \
  --models small distil-large-v3 \
  --subdir testaudio \
  --device auto \
  --repeats 3 \
  --preprocess-profile telephony \
  --noise-dir benchmarks/ambient_noise \
  --noise-snr-db 0 -5 -10 \
  --noise-combo-max-size 2 \
  --noise-max-combinations 12
```

## Output Files

For each run:

- `benchmarks/results/whisper_benchmark_<RUN_ID>.json`
- `benchmarks/results/whisper_benchmark_<RUN_ID>.csv`

`RUN_ID` is UTC timestamp in `YYYYMMDDTHHMMSSZ` format.
