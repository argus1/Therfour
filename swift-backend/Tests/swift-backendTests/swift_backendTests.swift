import Foundation
import Testing
@testable import swift_backend

// MARK: - Model Contract Tests

@Test
func transcriptionResultEncodesExpectedKeys() throws {
    let result = TranscriptionResult(text: "hello world", language: "en", confidence: 0.99)
    let payload = try JSONEncoder().encode(result)
    let decoded = try JSONDecoder().decode(TranscriptionResult.self, from: payload)

    #expect(decoded.text == "hello world")
    #expect(decoded.language == "en")
    #expect(decoded.confidence == 0.99)
}

@Test
func transcriptionResultEquatable() {
    let result1 = TranscriptionResult(text: "hello", language: "en", confidence: 0.95)
    let result2 = TranscriptionResult(text: "hello", language: "en", confidence: 0.95)
    let result3 = TranscriptionResult(text: "world", language: "en", confidence: 0.95)

    #expect(result1 == result2)
    #expect(result1 != result3)
}

@Test
func chatMessageEncodesExpectedKeys() throws {
    let message = ChatMessage(role: "user", content: "Hello, world!")
    let payload = try JSONEncoder().encode(message)
    let decoded = try JSONDecoder().decode(ChatMessage.self, from: payload)

    #expect(decoded.role == "user")
    #expect(decoded.content == "Hello, world!")
}

@Test
func chatMessageEquatable() {
    let msg1 = ChatMessage(role: "user", content: "hello")
    let msg2 = ChatMessage(role: "user", content: "hello")
    let msg3 = ChatMessage(role: "assistant", content: "hello")

    #expect(msg1 == msg2)
    #expect(msg1 != msg3)
}

@Test
func turnProcessingResultEquatable() {
    let transcription = TranscriptionResult(text: "hello", language: "en", confidence: 0.9)
    let result1 = TurnProcessingResult(
        transcription: transcription,
        reply: "Hi there!",
        audioPayload: Data([0x01, 0x02])
    )
    let result2 = TurnProcessingResult(
        transcription: transcription,
        reply: "Hi there!",
        audioPayload: Data([0x01, 0x02])
    )
    let result3 = TurnProcessingResult(
        transcription: transcription,
        reply: "Different reply",
        audioPayload: Data([0x01, 0x02])
    )

    #expect(result1 == result2)
    #expect(result1 != result3)
}

@Test
func healthResponseUsesConfiguredModelNames() {
    let response = makeHealthResponse(
        appVersion: "1.2.3",
        whisperModel: "medium",
        ollamaModel: "llama3.2:3b"
    )

    #expect(response.status == "ok")
    #expect(response.version == "1.2.3")
    #expect(response.services["stt"] == "faster-whisper/medium")
    #expect(response.services["tts"] == "piper")
    #expect(response.services["llm"] == "ollama/llama3.2:3b")
}

@Test
func healthResponseEncodesExpectedKeys() throws {
    let response = HealthResponse(
        status: "ok",
        version: "1.0.0",
        services: ["stt": "whisper", "tts": "piper", "llm": "ollama"]
    )
    let payload = try JSONEncoder().encode(response)
    let decoded = try JSONDecoder().decode(HealthResponse.self, from: payload)

    #expect(decoded.status == "ok")
    #expect(decoded.version == "1.0.0")
    #expect(decoded.services == ["stt": "whisper", "tts": "piper", "llm": "ollama"])
}

// MARK: - Error Contract Tests

@Test
func voiceServiceErrorEquatable() {
    #expect(VoiceServiceError.unsupported == VoiceServiceError.unsupported)
    #expect(VoiceServiceError.invalidResponse == VoiceServiceError.invalidResponse)
    #expect(VoiceServiceError.httpError(404) == VoiceServiceError.httpError(404))
    #expect(VoiceServiceError.httpError(404) != VoiceServiceError.httpError(500))
    #expect(VoiceServiceError.decodingFailure == VoiceServiceError.decodingFailure)
    #expect(VoiceServiceError.emptyOutput == VoiceServiceError.emptyOutput)
    #expect(VoiceServiceError.noSpeechDetected == VoiceServiceError.noSpeechDetected)
}

// MARK: - Capability Contract Tests

@Test
func sttBackendCapabilitiesEquatable() {
    let cap1 = STTBackendCapabilities(
        supportedLanguages: ["en", "es"],
        supportsLiveStreaming: true,
        notes: "Test capabilities"
    )
    let cap2 = STTBackendCapabilities(
        supportedLanguages: ["en", "es"],
        supportsLiveStreaming: true,
        notes: "Test capabilities"
    )
    let cap3 = STTBackendCapabilities(
        supportedLanguages: ["en"],
        supportsLiveStreaming: true,
        notes: "Test capabilities"
    )

    #expect(cap1 == cap2)
    #expect(cap1 != cap3)
}

@Test
func sttBackendCapabilitiesFallback() {
    let fallback = STTBackendCapabilities.fallback

    #expect(fallback.supportedLanguages.isEmpty)
    #expect(fallback.supportsLiveStreaming == false)
    #expect(fallback.notes == "Fallback assumptions")
}

@Test
func ttsBackendCapabilitiesEquatable() {
    let cap1 = TTSBackendCapabilities(
        supportedLanguages: ["en", "fr"],
        supportsVoiceHints: true,
        notes: "Test TTS"
    )
    let cap2 = TTSBackendCapabilities(
        supportedLanguages: ["en", "fr"],
        supportsVoiceHints: true,
        notes: "Test TTS"
    )
    let cap3 = TTSBackendCapabilities(
        supportedLanguages: ["en"],
        supportsVoiceHints: true,
        notes: "Test TTS"
    )

    #expect(cap1 == cap2)
    #expect(cap1 != cap3)
}

@Test
func ttsBackendCapabilitiesFallback() {
    let fallback = TTSBackendCapabilities.fallback

    #expect(fallback.supportedLanguages.isEmpty)
    #expect(fallback.supportsVoiceHints == true)
    #expect(fallback.notes == "Fallback assumptions")
}

@Test
func llmBackendCapabilitiesEquatable() {
    let cap1 = LLMBackendCapabilities(
        supportedLanguages: ["en", "de"],
        supportsJSONResponseFormat: true,
        notes: "Test LLM"
    )
    let cap2 = LLMBackendCapabilities(
        supportedLanguages: ["en", "de"],
        supportsJSONResponseFormat: true,
        notes: "Test LLM"
    )
    let cap3 = LLMBackendCapabilities(
        supportedLanguages: ["en"],
        supportsJSONResponseFormat: true,
        notes: "Test LLM"
    )

    #expect(cap1 == cap2)
    #expect(cap1 != cap3)
}

@Test
func llmBackendCapabilitiesFallback() {
    let fallback = LLMBackendCapabilities.fallback

    #expect(fallback.supportedLanguages.isEmpty)
    #expect(fallback.supportsJSONResponseFormat == true)
    #expect(fallback.notes == "Fallback assumptions")
}
