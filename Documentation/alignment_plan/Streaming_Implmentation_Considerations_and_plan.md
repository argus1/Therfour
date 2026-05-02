# Streaming Implementation Considerations and Plan

Date: 2026-05-01
Scope: Evaluate whether true streamed assistant speech is appropriate for Therfour telephony, with emphasis on natural cadence, pace, interruption behavior, and implementation fit with the current stack.

## Objective

Determine whether Therfour should adopt true streamed assistant responses for phone calls, and if not, define the most suitable alternative that reduces perceived latency without degrading conversational quality.

## Current Therfour Reality

Therfour currently follows a full-turn response pattern:

1. Receive caller audio.
2. Run STT.
3. Run LLM generation.
4. Parse transfer directives if present.
5. Run TTS on the complete reply.
6. Send synthesized audio to Twilio.

Relevant implementation boundaries:

- LLM supports both full-response and token streaming modes.
- TTS currently synthesizes complete text input rather than incremental chunks.
- Telephony sends complete synthesized PCM after TTS returns.
- Waiting audio and barge-in interruption are now implemented in the turn controller.

This means Therfour can already reduce perceived silence during retrieval or generation, but it does not yet support prosody-safe incremental speech output.

## Primary Question

Is true streamed assistant speech a good fit for telephony?

## Short Answer

Not as the default mode for Therfour in its current architecture.

A token-by-token or very early partial-sentence streaming model is likely to harm call quality more than it helps. For phone conversations, natural cadence matters more than raw response onset latency once the latency is within a tolerable range. A short deliberate pause, especially when covered by a subtle waiting cue, is usually preferable to speech that starts too early and sounds unstable.

## Why True Streaming Is Risky in Telephony

### 1. Prosody arrives late

LLM token streams do not expose final sentence shape early enough to guarantee natural speech rhythm.

Consequences:

- punctuation may arrive after TTS has already started speaking
- emphasis may land in the wrong place
- clauses can sound clipped or prematurely resolved
- final intonation can feel robotic or uncertain

For telephony, this is especially noticeable because the audio channel is narrow and the voice itself carries most of the user experience.

### 2. Partial text often changes direction

When speech starts from unstable partial text, the model may later qualify, narrow, or redirect the sentence.

Consequences:

- awkward self-corrections
- unnatural pauses between fragments
- repeated content
- increased risk of safety-sensitive wording sounding speculative before the full sentence is known

In a harm-reduction or crisis-adjacent setting, that is a quality risk, not just a cosmetic issue.

### 3. Whole-text TTS and stream-first LLM are mismatched today

Therfour has an LLM stream primitive, but its TTS path expects stable text input for synthesis. That means adopting true streaming now would require building chunking and playback behavior on top of a TTS stack that is not currently designed for it.

Consequences:

- more buffering logic in the telephony controller
- more cancellation boundaries
- more edge cases around chunk overlap and voice continuity
- increased likelihood of audible seams between fragments

### 4. Telephony reward function is different from chat UI reward function

In a text chat UI, seeing output appear immediately is almost always a positive.

In a phone call, speech quality matters more than first-token latency.

Good call behavior usually sounds like:

- short pause
- calm onset
- complete thought
- stable cadence
- predictable barge-in behavior

Bad call behavior often sounds like:

- speech begins too fast
- sentence shape changes mid-utterance
- pauses appear in unnatural places
- user interrupts because the pacing feels synthetic

## Why a Hybrid Approach Fits Better

A hybrid model gives Therfour the main benefit people seek from streaming, lower perceived latency, without forcing unstable speech output.

Recommended behavior:

1. Keep waiting audio for long retrieval or generation gaps.
2. Begin assistant speech only once a stable clause or full first sentence is available.
3. Stream later chunks only when they cross punctuation or clause boundaries.
4. Preserve immediate interruption when caller speech resumes.

This approach respects natural phone cadence while still improving responsiveness.

## Evaluation Summary

### True token streaming

Recommendation: Not suitable as Therfour's default telephony mode.

Reason:

- highest risk to naturalness
- highest implementation complexity relative to current stack
- weakest safety margin for phrase stability

### Sentence-first streaming

Recommendation: Potentially suitable after targeted infrastructure work.

Reason:

- allows stable prosody on the first spoken unit
- reduces dead air versus full-turn batching
- simpler than token streaming while preserving barge-in value

### Clause-first streaming

Recommendation: Best long-term target.

Reason:

- faster than sentence-first when the model writes longer responses
- can still sound natural if chunk boundaries are conservative
- works well with interruption-aware telephony control

## Decision

Use a hybrid clause-aware or sentence-aware streaming model as the future direction.

Do not adopt true token-by-token telephony speech as the primary runtime mode.

## Resulting Implementation Plan

### Phase 0: Keep Current Baseline Stable

Status:

- waiting audio during RAG/generation exists
- interruption on caller barge-in exists
- current full-turn speech output remains the production-safe baseline

Exit criteria:

- existing telephony tests stay green
- no regression in transfer or confirmation flows

### Phase 1: Add Stable Chunk Accumulation Layer

Goal:

Introduce a buffering layer between `llm.generate_stream()` and TTS.

Requirements:

- accumulate tokens until a stable boundary is reached
- boundaries should prefer `.`, `?`, `!`, `:`, `;`, or clear clause endings
- require a minimum character count to avoid tiny fragments
- avoid speaking fragments that end in incomplete coordinator patterns such as `and`, `but`, `so`, `because`

Deliverable:

- a chunker utility that converts token stream to speech-safe text units

Exit criteria:

- chunker produces stable first spoken unit for representative responses
- no chunk emitted that sounds obviously incomplete in manual review

### Phase 2: Add Incremental Speech Playback Controller

Goal:

Create a controller that can accept chunked text, synthesize sequentially, and send chunk audio in order.

Requirements:

- queue chunk synthesis work asynchronously
- prevent overlap between chunks
- stop waiting audio immediately when first real chunk starts
- preserve exact barge-in cancellation semantics
- handle empty or superseded chunks safely

Deliverable:

- telephony playback coordinator for incremental assistant speech

Exit criteria:

- first chunk interrupts waiting audio cleanly
- later chunks play without audible overlap
- interruption clears queued and active chunk playback

### Phase 3: Introduce Conservative Sentence-First Mode

Goal:

Deploy a low-risk hybrid mode before clause-first mode.

Behavior:

- wait for the first full sentence
- begin speaking that sentence
- optionally continue buffering the rest of the response as sentence-level chunks

Why this first:

- lower naturalness risk than clause-first
- easier to validate manually
- simpler rollback path if callers perceive it as unnatural

Exit criteria:

- first spoken sentence sounds complete in internal testing
- interruption still works reliably
- no frequent dead gaps between sentence chunks

### Phase 4: Evaluate Clause-First Mode

Goal:

Reduce latency further while retaining natural cadence.

Requirements:

- clause segmentation heuristics based on punctuation and token stability
- short grace buffer after likely chunk boundary to avoid premature emission
- optional merge of very short clauses into the following clause

Exit criteria:

- manual listening review judges cadence comparable to full-turn output
- no noticeable increase in caller interruption caused by awkward pacing

### Phase 5: Production Gating and Metrics

Goal:

Only promote hybrid streaming if it is measurably better than current full-turn output.

Required metrics:

- time to first meaningful audio
- total answer latency
- interruption success rate
- proportion of truncated or cancelled assistant responses
- manual listening score for naturalness
- manual listening score for calmness and pacing

Promotion rule:

Keep full-turn mode as default until hybrid streaming shows better perceived responsiveness without clear degradation in cadence.

## Natural Cadence Requirements

Any streaming-adjacent implementation should satisfy the following:

1. The first spoken unit must sound complete.
2. The assistant must not begin speaking on unstable text.
3. Pause placement must follow natural punctuation or clause boundaries.
4. Interruption must stop playback quickly enough that callers do not talk over stale audio.
5. Waiting audio must never overlap with real assistant speech.
6. Safety-sensitive language should be spoken only after the sentence is semantically stable.

## Risks to Watch

### Acoustic seam risk

Separate TTS chunks can sound like different utterances if the engine resets prosody too strongly between chunks.

### Latency inversion risk

Aggressive chunking can reduce first-audio latency but increase total interaction awkwardness if the system repeatedly pauses between chunks.

### Safety wording instability

If the first chunk is emitted too early, the assistant may sound uncertain or incomplete on sensitive guidance.

### Barge-in race conditions

If queued chunk playback is not cancelled cleanly, callers may hear stale assistant audio after they begin speaking.

## Recommended Near-Term Position

For Therfour production behavior today:

- keep full-turn synthesized speech as the default
- keep waiting audio for long RAG/generation delays
- keep barge-in interruption enabled
- treat hybrid streaming as an experimental feature branch

## Concrete Recommendation

Do not implement true token-by-token streamed telephony speech as the production target.

Instead, build toward:

1. sentence-first incremental speech as the first experimental milestone
2. clause-first incremental speech as the second milestone
3. production rollout only after cadence and interruption quality are validated by listening tests

## File and Component Impact for Future Work

Likely touch points:

- `app/services/llm.py`
  - adapt `generate_stream()` consumer behavior for chunk-safe buffering
- `app/services/tts.py`
  - optionally add chunk-oriented synthesis helpers
- `app/services/telephony.py`
  - own incremental playback queue, waiting-audio stop, and interruption handling
- `tests/test_telephony.py`
  - add chunk playback, cancellation, and boundary tests
- `benchmarks/`
  - add time-to-first-audio and interruption benchmarks

### Current Stub Linkage (Implemented)

The telephony stack now includes temporary clause-aware TTS stubs to prepare future streaming work:

- Optional model control line format: `EMPHASIS-HINTS: "phrase one" | "phrase two"`
- Hint lines are parsed into an internal stub hint container for future clause-aware synthesis.
- Hint lines are removed from spoken output so callers only hear natural speech text.
- Captured hints are logged with source linkage to this plan file.

This keeps production behavior stable while creating integration points for later keyword/phrase emphasis in clause-aware synthesis.

## Final Conclusion

True streaming is technically possible but not currently desirable for Therfour telephony.

A buffered hybrid approach is the right engineering target because it improves perceived responsiveness while protecting natural cadence, calm pacing, and interruption quality on the phone channel.
