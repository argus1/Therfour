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
    let result = TranscriptionResult(
        text: "hello world",
        language: "en",
        confidence: 0.99,
        languageConfidence: 0.99,
        transcriptQualityScore: 0.9,
        backendName: "faster-whisper",
        fallbackUsed: false,
        failureReason: ""
    )
    let payload = try JSONEncoder().encode(result)
    let decoded = try JSONDecoder().decode(TranscriptionResult.self, from: payload)

    #expect(decoded.text == "hello world")
    #expect(decoded.language == "en")
    #expect(decoded.confidence == 0.99)
    #expect(decoded.languageConfidence == 0.99)
    #expect(decoded.transcriptQualityScore == 0.9)
    #expect(decoded.backendName == "faster-whisper")
    #expect(decoded.fallbackUsed == false)
    #expect(decoded.failureReason == "")
}

@Test
func transcriptionResultDecodesSnakeCaseMetadata() throws {
        let payload = """
        {
            "text": "hello world",
            "language": "en",
            "confidence": 0.91,
            "language_confidence": 0.88,
            "transcript_quality_score": 0.72,
            "backend_name": "faster-whisper",
            "fallback_used": true,
            "failure_reason": ""
        }
        """.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(TranscriptionResult.self, from: payload)

        #expect(decoded.languageConfidence == 0.88)
        #expect(decoded.transcriptQualityScore == 0.72)
        #expect(decoded.backendName == "faster-whisper")
        #expect(decoded.fallbackUsed == true)
        #expect(decoded.failureReason == "")
}

@Test
func transcriptionResultAllowsMissingOptionalMetadata() throws {
        let payload = """
        {
            "text": "ok",
            "language": "en",
            "confidence": 0.5
        }
        """.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(TranscriptionResult.self, from: payload)

        #expect(decoded.languageConfidence == nil)
        #expect(decoded.transcriptQualityScore == nil)
        #expect(decoded.backendName == nil)
        #expect(decoded.fallbackUsed == nil)
        #expect(decoded.failureReason == nil)
}
