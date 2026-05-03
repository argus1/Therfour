# Translation Alignment Plan

## Source Repository

`~/Documents/HealthCoacher/ios-avatar-rag-prototype/`

All translation logic referenced below lives under:

- `iOSApp/Sources/Chat/TranslationPipeline.swift` — core translation types, clients, and sandwich coordinator
- `iOSApp/Sources/Chat/LanguageSupport.swift` — `SupportedLanguage` enum, BCP-47 routing, TTS voice hints
- `iOSApp/Sources/App/AppContainer.swift` — backend selection and wiring
- `tools/materialize_liquid_translation_bundle.sh` — CoreML bundle staging tool
- `tools/convert_liquid_translation_coreml.sh` — CoreML conversion tool

---

## What the HealthCoacher Translation Architecture Does

### The Sandwiched Translation Schema

HealthCoacher implements a deterministic **translation sandwich**: user input in a non-English language is translated to English before hitting RAG and the LLM, the LLM generates its reply internally in English, and the reply is translated back to the user's language before display and TTS synthesis. Conversation history is also translated into English before being sent to the LLM context window.

This keeps the LLM reasoning path, the RAG corpus, and the prompt templates in a single canonical language (English), while presenting the interface in the user's preferred language.

```
[User speech, language L]
        │
        ▼
  WhisperKit STT
  (language-aware decoding, BCP-47 prefill + fallback strategies)
        │
        ▼
  TranslationSandwichCoordinator.prepareTurn()
  • if L is in bridgedLanguages (Farsi, Japanese):
    – translate STT text L → English  (userTurn purpose)
    – translate history window L → English  (conversationHistory purpose)
    – RAG query = English text
    – LLM receives English input
  • else (English, Mandarin, Cantonese):
    – pass through unchanged
        │
        ▼
  RAG retrieval + LLM generation  (internal language = English)
        │
        ▼
  TranslationSandwichCoordinator.finalizeReply()
  • if L in bridgedLanguages:
    – translate English reply → L  (assistantReply purpose)
  • else:
    – pass through unchanged
        │
        ▼
  TTS synthesis in language L
```

### Translation Clients (Three-Tier Stack)

| Class                           | Backend          | Languages                            | Notes                                             |
| ------------------------------- | ---------------- | ------------------------------------ | ------------------------------------------------- |
| `LiquidCoreMLTranslationClient` | On-device CoreML | EN↔JP (LFM2-350M), EN↔FA (Sheikhaei) | Primary; gracefully absent if pack not downloaded |
| `LLMJSONTranslationClient`      | LLM JSON prompt  | Any pair the LLM knows               | Fallback when CoreML assets are missing           |
| `PassthroughTranslationClient`  | No-op            | Identity                             | Dev / English-only mode                           |

The `LiquidCoreMLTranslationClient` loads one or more **asset bundles**, each with a `translation_manifest.json`, a CoreML `.mlpackage`/`.mlmodelc`, and a `tokenizer.json`. Two model families are supported:

- **Liquid-LFM2-350M-ENJP-MT** (`lfm2_350m_enjp_mt_v1` prompt template) — Japanese ↔ English
- **Sheikhaei-Llama3.2-1B** (`sheikhaei_llama32_1b_enfa_v1` prompt template) — Farsi ↔ English

### TranslationSandwichCoordinator

The coordinator holds:

- `translationClient: TranslationClient` — the active client
- `bridgedLanguages: Set<SupportedLanguage>` — languages that require bridging (default: `.farsi`, `.japanese`)
- `historyWindow: Int` — number of history turns to include (default: 10)

Key methods:

- `prepareTurn(userText:history:language:) -> PreparedChatTurn` — translates input and history, returns English-normalized turn context
- `finalizeReply(_:language:) -> FinalizedAssistantReply` — translates LLM output back to target language, preserves the internal English text for auditing

Both methods have explicit error fallback: if translation fails, the coordinator degrades to `direct` mode (raw user language) and records the `fallbackReason`.

### WhisperKit Language-Aware Decoding

`WhisperKitTranscriber` in HealthCoacher runs a strategy cascade per transcription:

1. **Primary** — BCP-47 language prefill, VAD chunking, `usePrefillPrompt: true`
2. **Fallback 1** (non-EN/ZH) — same language code, `usePrefillPrompt: false`
3. **Fallback 2** (non-EN/ZH) — `detectLanguage: true`, no prefill
4. **Fallback 3** (non-EN/ZH) — relax quality thresholds, language prefill off

The best non-empty result is returned. TherFour's faster-whisper STT backend does not currently implement this cascade.

---

## TherFour Current State

| Area          | Current Behavior                                                                             | Gap                                                                                     |
| ------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| STT           | faster-whisper, single language attempt, no cascade                                          | No language-specific fallback strategy; no detected-language output                     |
| Schema        | `CanonicalTurnInput.language` exists; `CanonicalTurnProcessingSTT` has `language_confidence` | No translation fields in `CanonicalTurnProcessing` or `CanonicalTurnPayload`            |
| LLM           | Single-language prompt path                                                                  | No pre-translation of input or post-translation of output                               |
| RAG           | ChromaDB, English corpus                                                                     | Retrieval will degrade silently for non-English queries                                 |
| Turn contract | `app/models/schemas.py` (Python) + `VoiceContracts.swift` (Swift)                            | No `translation` processing block; no `display_text` vs `internal_text` split on output |
| Services      | No translation service module                                                                | `app/services/translation.py` does not exist                                            |
| Feature flag  | None                                                                                         | Translation path is always-on or always-off with no runtime toggle                      |

---

## Adoption Plan

### Phase 1 — Schema Extension (Feature-Flag Safe)

**Goal:** Extend the Python canonical turn schema to carry translation metadata without changing any existing behavior. All new fields are optional and off by default.

**Changes in `app/models/schemas.py`:**

Add a `CanonicalTurnProcessingTranslation` block:

```python
class TranslationSandwichMode(str, Enum):
    DIRECT = "direct"
    ENGLISH_BRIDGE = "english_bridge"

class TranslationPurpose(str, Enum):
    USER_TURN = "user_turn"
    CONVERSATION_HISTORY = "conversation_history"
    ASSISTANT_REPLY = "assistant_reply"

class CanonicalTurnProcessingTranslation(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: TranslationSandwichMode
    backend_name: str
    user_turn_source_language: Optional[str] = None
    user_turn_internal_text: Optional[str] = None   # English-normalized STT text
    assistant_internal_text: Optional[str] = None   # English-generated LLM reply
    fallback_reason: str = ""
```

Add `translation` to `CanonicalTurnProcessing`:

```python
class CanonicalTurnProcessing(BaseModel):
    ...
    translation: Optional[CanonicalTurnProcessingTranslation] = None
```

Split `CanonicalTurnOutput` to carry both display and internal text:

```python
class CanonicalTurnOutput(BaseModel):
    ...
    assistant_text: str = ""            # display text (translated if bridged)
    assistant_internal_text: str = ""   # English-internal text (same as display if direct)
```

**Equivalent extension in `swift-backend/Sources/swift-backend/VoiceContracts.swift`:** mirror the same `translation` block to maintain cross-runtime contract parity as defined in `Plan.md`.

---

### Phase 2 — Translation Service (Python)

Create `app/services/translation.py` modeled on the HealthCoacher client hierarchy.

**`TranslationRequest` dataclass:**

```python
@dataclass(frozen=True)
class TranslationRequest:
    text: str
    source_language: str      # BCP-47 or short code, e.g. "ja", "fa"
    target_language: str
    purpose: TranslationPurpose
```

**`TranslationClient` protocol (ABC):**

```python
class TranslationClient(ABC):
    @property
    @abstractmethod
    def backend_label(self) -> str: ...

    @abstractmethod
    async def translate(self, request: TranslationRequest) -> str: ...
```

**Three concrete implementations to port:**

1. **`OllamaJSONTranslationClient`** — analogous to `LLMJSONTranslationClient`. Sends a structured system + user prompt to the Ollama LLM backend, parses `{ "translated_text": "..." }` JSON response. This is TherFour's primary backend since on-device CoreML models are not used in the server stack.

2. **`LiquidHTTPTranslationClient`** (optional) — thin HTTP wrapper around a separately hosted Liquid translation server (mirrors the F5-TTS HTTP pattern already used in TherFour). Use this if a dedicated translation inference server is set up.

3. **`PassthroughTranslationClient`** — identity; used when feature flag is off.

**`TranslationSandwichCoordinator` class:**

Port the HealthCoacher coordinator directly:

```python
class TranslationSandwichCoordinator:
    DEFAULT_BRIDGED_LANGUAGES = {"ja", "fa"}  # BCP-47 short codes

    def __init__(
        self,
        client: TranslationClient,
        bridged_languages: set[str] = DEFAULT_BRIDGED_LANGUAGES,
        history_window: int = 10,
    ): ...

    async def prepare_turn(
        self, user_text: str, history: list[ChatMessage], language: str
    ) -> PreparedChatTurn: ...

    async def finalize_reply(
        self, internal_reply: str, language: str
    ) -> FinalizedAssistantReply: ...
```

`prepare_turn` returns a `PreparedChatTurn` dataclass mirroring HealthCoacher's Swift struct:

- `mode`: `"direct"` or `"english_bridge"`
- `rag_query`: English-normalized query string for ChromaDB
- `llm_user_text`: English-normalized user text for LLM prompt
- `llm_language`: canonical language code for LLM system prompt
- `llm_history`: history list with content translated to English
- `fallback_reason`: populated if translation failed and degraded to direct

`finalize_reply` returns a `FinalizedAssistantReply`:

- `mode`
- `internal_reply`: English LLM output
- `display_reply`: translated output (or same as `internal_reply` if direct)
- `fallback_reason`

**Error handling rule (same as HealthCoacher):** if the translation call throws, log the exception, populate `fallback_reason`, and degrade to `direct` mode rather than failing the turn.

---

### Phase 3 — STT Language Routing

Update `app/services/stt.py` to pass the active language code from the turn request into the transcription call, so faster-whisper uses the correct language hint instead of defaulting to auto-detect.

Add a `language_code` parameter to `STTBackend.transcribe()`:

```python
def transcribe(
    self, audio: np.ndarray, language: Optional[str]
) -> TranscriptionResult: ...
```

`TranscriptionResult` should surface `detected_language` alongside the existing confidence fields so the translation coordinator can use the actual detected language if the caller passed `None`.

For non-English languages, mirror the HealthCoacher Whisper fallback strategy cascade as a best-effort improvement: attempt transcription with language prefill first; if the result is below the quality threshold (`stt_min_text_characters`), retry with `language=None` (auto-detect) and take the longer/higher-confidence result.

---

### Phase 4 — LLM Prompt Integration

Update the turn processing path (in `app/services/llm.py` and any call-flow orchestration) to:

1. Call `coordinator.prepare_turn()` with the raw STT text and session history.
2. Use `prepared_turn.rag_query` for the ChromaDB retrieval call.
3. Use `prepared_turn.llm_user_text` and `prepared_turn.llm_history` when building the LLM prompt.
4. Pass `prepared_turn.llm_language` to `PromptTemplates` for language-specific system prompt guidance (port `LanguageSupport.llmInstruction` into TherFour's prompt construction).
5. Call `coordinator.finalize_reply()` on the raw LLM output.
6. Set `CanonicalTurnOutput.assistant_text = finalized.display_reply` and `assistant_internal_text = finalized.internal_reply`.
7. Populate `CanonicalTurnProcessingTranslation` with mode, backend label, and any fallback reason.

---

### Phase 5 — TTS Language Routing

Update `app/services/tts.py` to receive the active display language (`SupportedLanguage` equivalent) and:

- Select an appropriate voice or voice hint per language.
- Pass the correct language locale to the TTS backend if supported.

This mirrors the `SupportedLanguage.defaultTTSVoiceHint` mapping in HealthCoacher.

---

### Feature Flag

Control the entire translation path with a single environment variable:

```
THERFOUR_TRANSLATION_ENABLED=1          # master toggle (default: 0)
THERFOUR_TRANSLATION_BACKEND=ollama_json # or: liquid_http, passthrough
THERFOUR_TRANSLATION_BRIDGED_LANGUAGES=ja,fa  # comma-separated BCP-47 codes
```

When `THERFOUR_TRANSLATION_ENABLED=0` (default), the coordinator is wired to `PassthroughTranslationClient` and `mode` is always `direct`. No behavioral change for existing English-only deployments.

---

## Assets to Borrow from HealthCoacher

| Asset                                      | Source Path                                               | Purpose in TherFour                                                              |
| ------------------------------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `TranslationPipeline.swift`                | `iOSApp/Sources/Chat/`                                    | Reference implementation for Swift-backend contract extension                    |
| `LanguageSupport.swift`                    | `iOSApp/Sources/Chat/`                                    | `SupportedLanguage` enum, BCP-47 tags, TTS voice hints, `llmInstruction` strings |
| `translation_manifest.json` schema         | `iOSApp/Resources/CoreMLModels/Liquid-LFM2-350M-ENJP-MT/` | Manifest contract for any future server-side Liquid translator                   |
| `materialize_liquid_translation_bundle.sh` | `tools/`                                                  | Optional: stage Liquid models if a server-side translation endpoint is added     |
| Prompt templates (LFM2 v1, Sheikhaei v1)   | `TranslationPipeline.swift` lines ~1107–1145              | Use verbatim if running Liquid models via Ollama or HTTP                         |
| `LLMJSONTranslationClient` system prompt   | `TranslationPipeline.swift` lines ~93–113                 | Port exactly to `OllamaJSONTranslationClient`                                    |

---

## Language Scope

| Language  | BCP-47   | STT support                   | Translation model        | Mode                                                |
| --------- | -------- | ----------------------------- | ------------------------ | --------------------------------------------------- |
| English   | `en-US`  | faster-whisper                | —                        | `direct`                                            |
| Japanese  | `ja-JP`  | faster-whisper (multilingual) | LFM2-350M or Ollama JSON | `english_bridge`                                    |
| Farsi     | `fa-IR`  | faster-whisper (multilingual) | Sheikhaei / Ollama JSON  | `english_bridge`                                    |
| Mandarin  | `zh-CN`  | faster-whisper (multilingual) | Ollama JSON              | `direct` (LLM handles natively) or `english_bridge` |
| Cantonese | `yue-HK` | faster-whisper (limited)      | Ollama JSON              | `english_bridge`                                    |

Default bridged set for Phase 2: `{ja, fa}` — matching HealthCoacher's `defaultBridgedLanguages`.

---

## Schema Parity Matrix

| HealthCoacher Swift field                | TherFour Python equivalent                                   | Status                     |
| ---------------------------------------- | ------------------------------------------------------------ | -------------------------- |
| `TranslationSandwichMode`                | `TranslationSandwichMode` enum                               | Add to `schemas.py`        |
| `PreparedChatTurn.mode`                  | `CanonicalTurnProcessingTranslation.mode`                    | Add                        |
| `PreparedChatTurn.ragQuery`              | used internally; not stored                                  | —                          |
| `PreparedChatTurn.llmUserText`           | `CanonicalTurnProcessingTranslation.user_turn_internal_text` | Add                        |
| `FinalizedAssistantReply.internalReply`  | `CanonicalTurnOutput.assistant_internal_text`                | Add                        |
| `FinalizedAssistantReply.displayReply`   | `CanonicalTurnOutput.assistant_text`                         | Exists; semantics extended |
| `FinalizedAssistantReply.fallbackReason` | `CanonicalTurnProcessingTranslation.fallback_reason`         | Add                        |
| `TranslationClient.backendLabel`         | `TranslationClient.backend_label`                            | Add                        |

---

## Work Breakdown

### Week 1

| Day | Task                                                                                                                                           | Owner |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| 1   | Extend `app/models/schemas.py` with `CanonicalTurnProcessingTranslation`, split `assistant_text`/`assistant_internal_text`                     | Dev A |
| 1   | Mirror schema extension in `swift-backend/Sources/swift-backend/VoiceContracts.swift`                                                          | Dev A |
| 2   | Create `app/services/translation.py`: `TranslationRequest`, `TranslationClient`, `PassthroughTranslationClient`, `OllamaJSONTranslationClient` | Dev A |
| 2   | Port `LLMJSONTranslationClient` system + user prompt verbatim from HealthCoacher                                                               | Dev A |
| 3   | Implement `TranslationSandwichCoordinator` (`prepare_turn`, `finalize_reply`, `translate_history`)                                             | Dev A |
| 3   | Add feature-flag env var wiring in `app/core/config.py`                                                                                        | Dev A |
| 4   | Update `app/services/stt.py`: add `language` parameter propagation, basic two-attempt fallback                                                 | Dev B |
| 4   | Update LLM orchestration path to call coordinator, populate translation processing fields                                                      | Dev B |
| 5   | Update `app/services/tts.py`: language-aware voice routing                                                                                     | Dev B |

### Week 2

| Day  | Task                                                                                                                                 | Owner         |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------- |
| 6–7  | Write tests: `tests/test_translation.py` — unit tests for each client, coordinator prepare/finalize, fallback degradation            | Dev C / Dev A |
| 6–7  | Integration test: full turn with `ja` user text, verify RAG query is English, verify display reply is Japanese                       | Dev C         |
| 8    | Run existing `tests/` suite with `THERFOUR_TRANSLATION_ENABLED=0`; confirm zero regression                                           | Dev B         |
| 8    | Update `IMPLEMENTATION_SUMMARY.md` and `Documentation/new-to-team.md` with translation toggle instructions                           | Dev C         |
| 9–10 | ADR: document translation backend choice (Ollama JSON vs. Liquid HTTP) analogous to the Piper vs. F5-TTS decision track in `Plan.md` | Dev A         |

---

## Success Criteria

1. `THERFOUR_TRANSLATION_ENABLED=0` produces no behavioral or schema change versus current main.
2. With `THERFOUR_TRANSLATION_ENABLED=1` and a Japanese input turn, `CanonicalTurnProcessingTranslation.mode == "english_bridge"`, the RAG query and LLM input are English, and `CanonicalTurnOutput.assistant_text` is Japanese.
3. If the translation client throws, the turn completes in `direct` mode with `fallback_reason` populated — no turn failure.
4. `TranslationSandwichCoordinator` has unit test coverage for: direct passthrough, bridged success, translation failure → fallback, history window truncation.
5. `CanonicalTurnProcessingTranslation` round-trips through Pydantic serialization without loss.
6. Python schema and Swift contract translation fields are named and typed consistently per the parity matrix above.

---

## Deferred (Post-Sprint)

- On-device Liquid CoreML model running inside TherFour server via a sidecar process (analogous to the F5-TTS docker container in `docker/f5tts_server/`).
- Mandarin and Cantonese as bridged languages (requires evaluation of LLM native handling quality vs. English-bridge overhead).
- Per-language RAG corpus segments or language-tagged chunking.
- Real-time streaming translation for mid-turn partial transcripts.
