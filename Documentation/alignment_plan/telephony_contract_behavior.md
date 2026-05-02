# Telephony Contract Behavior — Reevaluation

Date: 2026-05-02
Scope: End-to-end Twilio contract behavior reevaluated against current codebase; Asterisk/FreePBX compatibility feasibility assessed.

---

## Part 1 — Twilio End-to-End Contract

### What the Plan specified

The alignment plan called for a fully integrated audio pipeline (μ-law/8 kHz ↔ Twilio, float32/16 kHz ↔ Whisper), STT/TTS normalization, RAG/LLM alignment, and a canonical structured turn contract (`CanonicalTurnEnvelope` + `CanonicalTurnPayload`) with `trace_id`, `turn_id`, and `session_id` at every pipeline boundary.

---

### Contract: what is now implemented

| Contract surface                               | Status       | Notes                                                                                                                                                  |
| ---------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Inbound webhook `POST /calls/inbound`          | ✅ Solid     | Returns `<Say>` + `<Connect><Stream>` TwiML correctly. `streamSid`/`callSid` captured from `start` event.                                              |
| Media Stream WebSocket `/calls/stream`         | ✅ Solid     | `connected`, `start`, `media`, `stop` event dispatch is complete.                                                                                      |
| μ-law decode/encode round-trip                 | ✅ Solid     | `audioop`/`audioop_lts` with `resample_poly` up/down between 8 kHz ↔ 16/22 kHz. Tested.                                                                |
| VAD gating (Silero)                            | ✅ Solid     | `StreamingSpeechDetector` runs as the primary turn gate; silence-timer fallback when unavailable.                                                      |
| STT pipeline                                   | ✅ Solid     | Whisper primary + Sherpa-ONNX fallback, session-sticky on first Sherpa result, minimum duration gate, low-confidence confirmation flow.                |
| Turn strategy router                           | ✅ Solid     | Rapport / info-gathering / RAG-eligible classification, per-strategy RAG gating, debug logging flag.                                                   |
| LLM + RAG integration                          | ✅ Solid     | `_generate_with_optional_waiting_audio` correctly races waiting audio against LLM completion; waiting audio is cancelled on LLM return.                |
| TTS (F5-TTS primary + Piper fallback)          | ✅ Solid     | `tts_backend` defaults to `f5_http`; `tts_fallback_backend` = `piper`. `PIPER_SAMPLE_RATE` is the canonical output rate for `_send_audio`.             |
| Transfer — 911/988                             | ✅ Solid     | Verbal consent required by default (`transfer_confirmation_required=True`). `<Dial>` TwiML generated; Twilio REST call update used for actual handoff. |
| Transfer — SIP/PSTN custom                     | ✅ Solid     | Transfer-services catalog, domain allowlist, E.164 check, SIP header passthrough for metadata.                                                         |
| Post-call reopen (`/calls/transfer-completed`) | ✅ Solid     | off/auto/prompt modes, flag-gated, separate mode for custom vs emergency transfers. Legacy `transfer_stay_on_line_enabled` compat preserved.           |
| Barge-in interruption                          | ✅ Solid     | `turn_interrupt_enabled`, `clear` event sent to Twilio on task cancel.                                                                                 |
| End-call terminator + presence loop            | ✅ Solid     | `_run_end_call_presence_loop` with configurable delay/rounds, REST hangup → WebSocket close fallback.                                                  |
| LLM `<think>` / `EMPHASIS-HINTS` scrubbing     | ✅ Solid     | Regex-stripped before TTS and before conversation history append.                                                                                      |
| Clause-aware TTS hint capture                  | ⚠️ Stub only | `ClauseAwareTTSStubHints` captured and logged, not wired to TTS. Intentional per streaming plan.                                                       |

---

### Contract gaps that matter for production

**1. The canonical turn schema from the Plan is unimplemented.**

`Plan.md` specifies `CanonicalTurnEnvelope` + `CanonicalTurnPayload` with `trace_id`, `turn_id`, `session_id`, `created_at`, and explicit `payload.status.state` fields. `app/models/schemas.py` has `TranscriptionResult` and `TurnProcessingResult` but no envelope layer. `CallSession` carries no session or trace identity at runtime. All observability is in free-form log strings only — there is no structured turn record that could feed a downstream audit log, replay harness, or latency dashboard.

**2. `TransferHarnessResponse.twiml` is Twilio-coupled at the wire level.**

The `twiml` field is exposed directly in the API response schema. Any consumer that reads that field is implicitly coupled to Twilio XML. If a protocol adapter layer is added later (see Part 2), this field name becomes a misleading contract artifact. Consider renaming to `call_control_payload` or adding an explicit `protocol` discriminator field.

**3. No retry semantics or `idempotency_key` at the session layer.**

The Plan required `parent_turn_id` for retry attempts and `retryable` flags on error shapes. Nothing in `CallSession` tracks whether a turn was a retry, and `TranscriptionResult.failure_reason` is a free-form string with no typed enum. This matters for call replay, A/B evaluation, and automated quality scoring.

**4. `_send_audio` sends a `mark` event that is never awaited.**

The `mark` event (`end_of_response`) is sent after every outbound audio sequence but its acknowledgment from Twilio is never awaited. If a slow outbound queue builds (e.g. in a high-latency relay scenario), turns can overlap from the caller's perspective. The barge-in interrupt path mitigates this but does not eliminate it under congestion.

**5. VAD fallback path has a silent degradation.**

When `silero_vad_available()` is false, the session drops to `silence_timeout_s = 1.5s` without any metric or log field that a monitoring system could alert on. Under telephony audio conditions (hold music, background noise, slow callers), this fallback degrades turn quality invisibly.

---

### Overall Twilio contract verdict

The core telephony behavior is production-credible. The audio pipeline, VAD gating, STT/TTS normalization, multi-mode transfer flows, and barge-in behavior were all added cleanly since the plan was written. The main deficit is structural: the canonical turn envelope contract from the Plan was never implemented. What exists instead is a functional but loosely observable runtime — good enough for calls, insufficient for systematic audit, replay, or multi-provider portability without further work.

---

## Part 2 — Asterisk / FreePBX Compatibility Feasibility

### The protocol mismatch

Twilio Media Streams deliver audio over a persistent WebSocket as **JSON-framed base64 μ-law chunks**:

```json
{ "event": "media", "streamSid": "...", "media": { "payload": "<base64>" } }
```

Asterisk ARI External Media bridges a channel to a WebSocket endpoint and delivers audio as **raw binary frames** (μ-law or linear PCM, no JSON wrapper). The ARI also uses completely different session lifecycle signals — no `connected`/`start`/`stop` events; instead, ARI `StasisStart`/`StasisEnd` events are delivered separately over the ARI WebSocket or HTTP callbacks.

FreePBX is a dialplan GUI on top of Asterisk; the API surface for real-time audio is identical to raw Asterisk ARI.

---

### Where the coupling is concentrated

| Coupling point      | File / method                                              | What Asterisk needs instead                                           |
| ------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------- |
| Event dispatch loop | `CallSession.handle()`                                     | ARI sends raw binary audio; no JSON event envelope                    |
| Media decode        | `_on_media()` — base64 decode                              | Direct `bytes` frame, no JSON unwrap                                  |
| Audio send          | `_send_audio()` — JSON + base64 + `mark` event             | Raw binary write; marks not available                                 |
| Outbound clear      | `_clear_outbound_audio()` — Twilio `clear` event           | ARI has no equivalent; requires bridge mute or channel control        |
| Call transfer       | `build_transfer_twiml()` + `twilio_transfer_call_update()` | ARI `POST /channels/{id}/redirect` or `POST /bridges`                 |
| Hangup              | `_hangup_call()` → Twilio REST                             | ARI `DELETE /channels/{id}`                                           |
| Inbound entry point | `POST /calls/inbound` returning TwiML                      | Asterisk dialplan + `ExternalMedia` application, no HTTP webhook call |
| Post-call reopen    | `POST /calls/transfer-completed` → TwiML                   | Asterisk dialplan continuation; no equivalent action URL pattern      |

The good news: everything from `_run_turn` downward (VAD, STT, LLM, TTS, all the conversation flows) is **already transport-agnostic**. The full turn processing state machine — confirmation flows, transfer confirmation, end-call presence loop, turn strategy routing — does not reference any Twilio primitive.

---

### What a compatibility layer would require

#### Tier 1 — Audio bridge only (no transfer/hangup control): Medium effort

Introduce a `TelephonyTransport` abstract protocol with four operations:

```python
async def receive_audio_frame() -> bytes: ...   # one μ-law frame
async def send_audio(samples: np.ndarray) -> None: ...  # synthesized PCM
async def clear_outbound() -> None: ...         # interrupt playback
def session_id() -> str: ...
```

Implement `TwilioTransport` (wraps the existing `handle()` JSON parsing) and `AsteriskARITransport` (reads raw binary frames from an ARI external media WebSocket, writes binary back). `CallSession.handle()` delegates to whichever transport is active.

The Asterisk ARI external media WebSocket supports raw binary audio exchange and is the correct integration point. The codec framing (μ-law vs slin) is negotiated at the ARI `POST /channels/externalMedia` call.

#### Tier 2 — Transfer and hangup control: Higher effort

The transfer path (`_transfer_call` → TwiML → Twilio REST) has no direct Asterisk equivalent. Asterisk ARI call control uses:

- `POST /ari/channels/{id}/redirect` — redirects a channel to a new dialplan context
- `POST /ari/bridges` — bridges two channels together for a transfer
- `DELETE /ari/channels/{id}` — hangs up

This requires a second abstraction: `TelephonyCallControl` with `transfer(target)` and `hangup()` methods. `build_transfer_twiml` would only be invoked by the Twilio implementation of that interface.

#### Tier 3 — Inbound routing and post-call reopen: Asterisk dialplan work

The Twilio `/calls/inbound` webhook has no Asterisk equivalent. In Asterisk you would:

1. Configure a dialplan extension that calls `ExternalMedia(...)` with the application's WebSocket URL.
2. The `ExternalMedia` application opens the raw binary audio bridge.

The post-call reopen callback (`/calls/transfer-completed`) also has no native Asterisk equivalent. It can be approximated by a dialplan continuation after the `Dial()` application returns, but this is entirely server-side dialplan work and the semantics differ from Twilio's action URL callback pattern.

---

### Feasibility verdict

| Layer                                 | Feasibility                       | Effort                                                         |
| ------------------------------------- | --------------------------------- | -------------------------------------------------------------- |
| Core audio pipeline (VAD/STT/LLM/TTS) | ✅ Already transport-agnostic     | None                                                           |
| Audio frame transport adapter         | ✅ Feasible                       | ~1–2 days for `TelephonyTransport` abstraction + ARI adapter   |
| Transfer/hangup abstraction           | ✅ Feasible with scoped refactor  | ~2–3 days — `TelephonyCallControl` layer + ARI REST calls      |
| Inbound routing                       | ✅ Feasible via Asterisk dialplan | Asterisk configuration work, not Python                        |
| Post-call reopen callback             | ⚠️ Approximable, not equivalent   | Dialplan continuation; semantics differ from Twilio action URL |
| Barge-in `clear` outbound             | ⚠️ No direct ARI equivalent       | ARI mute/unmute or bridge hold; behavior will differ subtly    |
| Test coverage                         | ⚠️ All tests are Twilio-framed    | New integration tests needed for Asterisk transport path       |

**Overall:** Adding Asterisk/FreePBX support is feasible without rewriting the application. The core conversation logic is already isolated. The refactoring surface is bounded — primarily `CallSession.handle()`, `_on_media()`, `_send_audio()`, `_clear_outbound_audio()`, and the transfer/hangup methods.

The biggest practical risk is the post-call reopen behavior, which relies on Twilio's action URL callback pattern and has no clean Asterisk counterpart — that feature would need to be re-expressed as a dialplan continuation or dropped for the Asterisk path.

### Recommended prerequisite before starting Asterisk work

Implement the `CanonicalTurnEnvelope` contract specified in `Plan.md` first. That work would naturally introduce the session/trace ID tracking that is currently absent from `CallSession`, and it would make the transport adapter layer significantly easier to instrument and test correctly under either provider.
