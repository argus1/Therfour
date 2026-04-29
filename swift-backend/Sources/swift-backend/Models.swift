import Foundation

struct TranscriptionResult: Codable, Equatable {
    let text: String
    let language: String
    let confidence: Double
    let languageConfidence: Double?
    let transcriptQualityScore: Double?
    let backendName: String?
    let fallbackUsed: Bool?
    let failureReason: String?

    private enum CodingKeys: String, CodingKey {
        case text
        case language
        case confidence
        case languageConfidence = "language_confidence"
        case transcriptQualityScore = "transcript_quality_score"
        case backendName = "backend_name"
        case fallbackUsed = "fallback_used"
        case failureReason = "failure_reason"
    }
}

struct HealthResponse: Codable, Equatable {
    let status: String
    let version: String
    let services: [String: String]
}

struct ChatMessage: Codable, Equatable {
    let role: String
    let content: String
}

struct TurnProcessingResult: Equatable {
    let transcription: TranscriptionResult
    let reply: String
    let audioPayload: Data
}

func makeHealthResponse(
    appVersion: String,
    whisperModel: String,
    ollamaModel: String
) -> HealthResponse {
    HealthResponse(
        status: "ok",
        version: appVersion,
        services: [
            "stt": "faster-whisper/\(whisperModel)",
            "tts": "piper",
            "llm": "ollama/\(ollamaModel)",
        ]
    )
}
