import Foundation

// MARK: - Error Types

enum VoiceServiceError: Error, Equatable {
    case unsupported
    case invalidResponse
    case httpError(Int)
    case decodingFailure
    case emptyOutput
    case noSpeechDetected
}

// MARK: - Service Protocols

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

// MARK: - Capability Probing Protocols

protocol STTCapabilityProbe {
    func probeSTTCapabilities() async -> STTBackendCapabilities
}

protocol TTSCapabilityProbe {
    func probeTTSCapabilities() async -> TTSBackendCapabilities
}

protocol LLMCapabilityProbe {
    func probeLLMCapabilities() async -> LLMBackendCapabilities
}

// MARK: - Turn Processing Protocol

protocol TurnProcessor: Sendable {
    func processTurn(audioPCM16kMono: Data, languageHint: String?) async throws
        -> TurnProcessingResult

    func currentConversation() -> [ChatMessage]
    func resetConversation()
}

// MARK: - Capability Structures

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
