import Foundation

enum VoiceServiceError: Error, Equatable {
    case unsupported
    case invalidResponse
    case httpError(Int)
    case decodingFailure
    case emptyOutput
    case noSpeechDetected
}

protocol SpeechTranscriber: Sendable {
    func transcribe(audioPCM16kMono: Data, languageHint: String?) async throws
        -> TranscriptionResult
}

protocol SpeechSynthesizer: Sendable {
    func synthesize(text: String, language: String?) async throws -> Data
}

protocol ChatResponder: Sendable {
    func generateReply(messages: [ChatMessage], systemPrompt: String) async throws -> String
}

protocol STTCapabilityProbe {
    func probeSTTCapabilities() async -> STTBackendCapabilities
}

protocol TTSCapabilityProbe {
    func probeTTSCapabilities() async -> TTSBackendCapabilities
}

protocol LLMCapabilityProbe {
    func probeLLMCapabilities() async -> LLMBackendCapabilities
}

struct STTBackendCapabilities: Equatable {
    let supportedLanguages: Set<String>
    let supportsLiveStreaming: Bool
    let notes: String

    static let fallback = STTBackendCapabilities(
        supportedLanguages: [],
        supportsLiveStreaming: false,
        notes: "Fallback assumptions"
    )
}

struct TTSBackendCapabilities: Equatable {
    let supportedLanguages: Set<String>
    let supportsVoiceHints: Bool
    let notes: String

    static let fallback = TTSBackendCapabilities(
        supportedLanguages: [],
        supportsVoiceHints: true,
        notes: "Fallback assumptions"
    )
}

struct LLMBackendCapabilities: Equatable {
    let supportedLanguages: Set<String>
    let supportsJSONResponseFormat: Bool
    let notes: String

    static let fallback = LLMBackendCapabilities(
        supportedLanguages: [],
        supportsJSONResponseFormat: true,
        notes: "Fallback assumptions"
    )
}
