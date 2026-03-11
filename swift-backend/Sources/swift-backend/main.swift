import Foundation

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
