# STT Low-Confidence Thresholding Implementation Summary

## Overview

This implementation adds intelligent thresholding for low-confidence STT (Speech-to-Text) results in the Whisper-based telephony system. When a transcript's confidence falls below a configurable threshold, the system enters a confirmation flow where the user is prompted to verify what was heard, with intelligent handling of retries and topic changes.

## Components Implemented

### 1. Configuration Settings (`app/core/config.py`)

Added three new configuration parameters:

- **`stt_low_confidence_threshold`** (default: 0.75): Confidence threshold below which confirmation flow is triggered. Valid range: 0.0-1.0
- **`stt_max_retries`** (default: 3): Maximum number of retry attempts for low-confidence confirmations
- These settings follow 2026 telephony standards targeting 5-10% WER (90-95% accuracy)

### 2. Enhanced Transcription Schema (`app/models/schemas.py`)

Updated `TranscriptionResult` dataclass with:

- **`audio_duration_s`** (float): Duration of the audio in seconds
  - Used to determine whether to use verbatim transcript or paraphrase for confirmation
  - Short clips (<5s): Use original transcript verbatim
  - Long clips (≥5s): Generate paraphrase for better recognition

### 3. STT Backend Updates (`app/services/stt.py`)

- Modified `_WhisperBackend.transcribe()` to calculate and pass audio duration
- Modified `_SherpaBackend.transcribe()` to calculate and pass audio duration
- Duration calculation: `len(audio) / settings.audio_sample_rate_whisper`

### 4. Low-Confidence Handler (`app/services/stt_confidence.py`)

New module providing:

#### `LowConfidenceHandler` class:

- **`is_low_confidence(result)`**: Detects if transcript is below confidence threshold
- **`generate_confirmation_prompt(result)`**: Creates confirmation prompt
  - Paraphrases long transcripts (≥5s) using LLM for better user recognition
  - Uses verbatim text for short transcripts (<5s)
  - Returns `ConfirmationPrompt` with:
    - `original_transcript`: The raw STT output
    - `confirmation_text`: Text to read to user (verbatim or paraphrased)
    - `should_paraphrase`: Boolean indicating if paraphrasing was used
    - `prompt`: Full confirmation message
- **`get_retry_prompt(retry_count)`**: Returns appropriate retry message
  - For retries <3: "I see. I'm sorry, can you please say it again slowly and clearly?"
  - For retries ≥3 (topic change): "I'm having a lot of difficulty understanding what you're trying to say. I hope you don't mind if we talk about something else? Tell me what else is on your mind."
- **`should_change_topic(retry_count)`**: Checks if max retries exceeded

#### Paraphrase Generation:

- Uses LLM (`_paraphrase_transcript()`) to generate natural paraphrases
- Falls back to original text if paraphrasing fails
- Includes timeout and error handling

### 5. CallSession Enhancements (`app/services/telephony.py`)

Added low-confidence confirmation flow state tracking:

#### New instance variables:

- **`_in_confirmation_flow`** (bool): Flag indicating if user is in confirmation mode
- **`_confirmation_retry_count`** (int): Number of retries for current confirmation
- **`_pending_low_confidence_text`** (str): Original transcript waiting for confirmation

#### New methods:

- **`_enter_confirmation_flow(stt_result)`**:
  - Marks entry into confirmation mode
  - Generates confirmation prompt using `LowConfidenceHandler`
  - Synthesizes and sends confirmation prompt to user
  - Logs confidence metrics

- **`_handle_confirmation_response(response_text, response_result)`**:
  - Parses user's yes/no response
  - **If YES (affirmative)**:
    - Exits confirmation flow
    - Adds original (confirmed) transcript to conversation
    - Calls `llm.generate()` for RAG search and text generation (if RAG enabled)
    - Synthesizes and sends LLM reply
  - **If NO (negative)**:
    - Increments retry counter
    - Checks if max retries exceeded
    - If max retries exceeded: Generates topic-change prompt and exits confirmation flow
    - If retries remaining: Generates retry prompt and waits for new audio
  - **If UNCLEAR**: Asks user to clarify with "yes" or "no"

- **`_is_affirmative_response(text)`** (static):
  - Detects affirmative responses: "yes", "yeah", "yep", "sure", "correct", "right", "uh-huh"

- **`_is_negative_response(text)`** (static):
  - Detects negative responses: "no", "nope", "nah", "incorrect", "wrong", "uh-uh"

#### Updated `_run_turn()` flow:

1. **Minimum duration gate**: Discard very short audio
2. **STT transcription**: Get transcript and confidence
3. **Confirmation check**: If in confirmation flow, handle yes/no response (new step 2.5)
4. **Low-confidence detection**: If confidence below threshold, enter confirmation flow (new step 3)
5. **LLM processing**: Generate response (automatic RAG if enabled)
6. **TTS synthesis**: Convert response to audio
7. **Audio sending**: Send audio to caller

## Data Flow

### Low-Confidence Confirmation Sequence:

```
User speaks utterance
    ↓
STT transcribes → Confidence < threshold
    ↓
Generate confirmation prompt
    ├─ If audio < 5s: Use verbatim text
    └─ If audio ≥ 5s: Paraphrase with LLM
    ↓
Send: "I think you said [text], am I correct? Tell me yes or no."
    ↓
User responds
    ├─ YES: Process confirmed text through LLM + RAG → Send LLM response
    ├─ NO (retry < 3): Send "Please say it again slowly and clearly" → Await new audio
    ├─ NO (retry ≥ 3): Send "Topic change prompt" → Await new user input
    └─ UNCLEAR: Send "Tell me yes or no" → Await clarification
```

## Integration with Existing Systems

### LLM Integration:

- Confirmed transcripts are processed through the existing `llm.generate()` pipeline
- RAG (Retrieval-Augmented Generation) is automatically applied if enabled in config
- Full conversation history is maintained for context

### Paraphrasing via LLM:

- Uses the same Ollama backend configured for main LLM operations
- Shares timeout settings (`settings.ollama_timeout`)
- Graceful fallback to original text if paraphrasing fails

### TTS Integration:

- All prompts (confirmation, retry, topic change) use the configured TTS backend
- Language is inferred from the STT result
- Same TTS pipeline as regular responses

## Configuration Tuning

### For Different Use Cases:

**High-Accuracy Scenarios** (e.g., emergency/compliance):

```env
STT_LOW_CONFIDENCE_THRESHOLD=0.85
STT_MAX_RETRIES=5
```

**Standard Production** (default):

```env
STT_LOW_CONFIDENCE_THRESHOLD=0.75
STT_MAX_RETRIES=3
```

**Lenient Scenarios** (e.g., accessibility):

```env
STT_LOW_CONFIDENCE_THRESHOLD=0.60
STT_MAX_RETRIES=2
```

## Testing Considerations

1. **Confidence Threshold Testing**:
   - Test with various confidence scores around the threshold
   - Verify confirmation flow triggers correctly

2. **Paraphrase Testing**:
   - Test with short clips (<5s) to verify verbatim handling
   - Test with long clips (≥5s) to verify paraphrase generation

3. **Response Detection Testing**:
   - Test yes/no detection with various phrasings
   - Test unclear response handling

4. **Retry Flow Testing**:
   - Test retries up to max limit
   - Verify topic change prompt on max retries

5. **Integration Testing**:
   - Verify RAG is called after confirmation
   - Verify full conversation history is maintained
   - Test with different LLM backends (Ollama, LM Studio)

## Standards Compliance

This implementation follows 2026 telephony standards documented in `Documentation/alignment_plan/deliverables/STT_Thresholds.md`:

- **Target WER**: 5-10% (90-95% accuracy for good/production grade)
- **Confidence Threshold**: Calibrated for production voice agents
- **User Experience**: Confirmation flow provides natural conversation recovery
- **Performance**: All prompts synthesized and sent asynchronously to maintain responsiveness

## Files Modified

1. **app/core/config.py**
   - Added `stt_low_confidence_threshold`
   - Added `stt_max_retries`

2. **app/models/schemas.py**
   - Added `audio_duration_s` to `TranscriptionResult`

3. **app/services/stt.py**
   - Updated `_WhisperBackend._make_result()` to accept and pass audio duration
   - Updated `_WhisperBackend.transcribe()` to calculate audio duration
   - Updated `_SherpaBackend.transcribe()` to calculate audio duration

4. **app/services/telephony.py**
   - Added import for `LowConfidenceHandler`
   - Added `_in_confirmation_flow`, `_confirmation_retry_count`, `_pending_low_confidence_text` instance variables
   - Updated `_run_turn()` to include confirmation flow logic
   - Added `_enter_confirmation_flow()` method
   - Added `_handle_confirmation_response()` method
   - Added `_is_affirmative_response()` static method
   - Added `_is_negative_response()` static method

5. **app/services/stt_confidence.py** (new file)
   - `LowConfidenceHandler` class with low-confidence detection, prompt generation, and retry management
   - `_paraphrase_transcript()` async function for LLM-based paraphrasing
   - `ConfirmationPrompt` dataclass for structured prompt information
