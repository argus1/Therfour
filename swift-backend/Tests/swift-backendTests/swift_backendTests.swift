import Foundation
import Testing
@testable import swift_backend

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
func transcriptionResultEncodesExpectedKeys() throws {
    let result = TranscriptionResult(text: "hello world", language: "en", confidence: 0.99)
    let payload = try JSONEncoder().encode(result)
    let decoded = try JSONDecoder().decode(TranscriptionResult.self, from: payload)

    #expect(decoded.text == "hello world")
    #expect(decoded.language == "en")
    #expect(decoded.confidence == 0.99)
}
