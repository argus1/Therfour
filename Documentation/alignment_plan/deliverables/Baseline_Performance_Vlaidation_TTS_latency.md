# Baseline Performance Validation - TTS Latency

Date: 2026-04-26
Branch: argus-baseline-branch

Purpose: document baseline TTS latency validation status and evidence for Sprint 1 alignment tracking.

## Validation Scope

- Component: TTS (Piper synthesis path)
- Focus metrics: P50 latency, P95 latency, and synthesis failure behavior
- Primary code paths:
  - app/services/tts.py
  - app/core/config.py
- Existing instrumentation reference:
  - Documentation/alignment_plan/deliverables/Baseline_Observability_Checklist_2026-04-20.md

## Plan Criteria Cross-Check

From Documentation/alignment_plan/Plan.md:

- Baseline metrics include TTS latency.
- Exit criteria require P95 latency within an agreed threshold.

Note: as with STT, no explicit numeric TTS P95 threshold is currently defined in-repo.

## Runtime Readiness Check (2026-04-26)

Observed in workspace runtime check:

- Piper binary not found on PATH.
- Default configured model path not present:
  - models/en_US-lessac-medium.onnx

Implication: a production-path TTS latency benchmark cannot be executed yet in this environment.

## Measured Results

No numeric benchmark results are available yet for TTS in the repository artifacts.

| Run ID | Engine | Samples | P50 Latency (s) | P95 Latency (s) | Notes                                               |
| ------ | ------ | ------: | --------------: | --------------: | --------------------------------------------------- |
| N/A    | Piper  |     N/A |             N/A |             N/A | Benchmark blocked: Piper runtime assets unavailable |

## Validation Outcome

- TTS observability instrumentation is implemented and test-covered for latency_ms and failure_reason emission.
- Numeric baseline latency validation is currently blocked by missing Piper runtime dependencies in the checked environment.

## Decision Status

- TTS latency baseline performance: Pending benchmark execution.
- Exit gate status "P95 latency within agreed threshold": Pending benchmark data and explicit numeric threshold.

## Required Actions To Complete Validation

1. Install or configure Piper binary (PIPER_BINARY).
2. Add/download a valid Piper voice model (PIPER_MODEL_PATH), or update .env to an existing model path.
3. Execute a repeatable TTS benchmark run over representative utterance sets.
4. Record measured P50/P95 values in this document and attach raw artifact paths.
5. Define and document the numeric TTS P95 threshold for strict pass/fail gating.

## Suggested Benchmark Artifact Naming

- benchmarks/results/piper*benchmark*<timestamp>.json
- benchmarks/results/piper*benchmark*<timestamp>.csv
