import Foundation

struct TranscriptionResult: Codable, Equatable {
    let text: String
    let language: String
    let confidence: Double
}

struct HealthResponse: Codable, Equatable {
    let status: String
    let version: String
    let services: [String: String]
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

@main
struct TherfourBackend {
    static func main() throws {
        let env = ProcessInfo.processInfo.environment
        let response = makeHealthResponse(
            appVersion: env["APP_VERSION"] ?? "0.1.0",
            whisperModel: env["WHISPER_MODEL"] ?? "small",
            ollamaModel: env["OLLAMA_MODEL"] ?? "llama3.2:3b"
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let payload = try encoder.encode(response)
        if let json = String(data: payload, encoding: .utf8) {
            print(json)
        }
    }
}
