import Foundation
import Testing

@testable import swift_backend

private final class StubTranscriber: SpeechTranscriber, @unchecked Sendable {
    var result: TranscriptionResult

    init(result: TranscriptionResult) {
        self.result = result
    }

    func transcribe(audioPCM16kMono: Data, languageHint: String?) async throws
        -> TranscriptionResult
    {
        result
    }
}

private final class StubResponder: ChatResponder, @unchecked Sendable {
    var reply: String
    var receivedMessages: [ChatMessage] = []

    init(reply: String) {
        self.reply = reply
    }

    func generateReply(messages: [ChatMessage], systemPrompt: String) async throws -> String {
        receivedMessages = messages
        return reply
    }
}

private final class StubSynthesizer: SpeechSynthesizer, @unchecked Sendable {
    var audio = Data([0x01, 0x02, 0x03])

    func synthesize(text: String, language: String?) async throws -> Data {
        audio
    }
}

@Test
func processTurnRunsSTTToLLMToTTS() async throws {
    let stt = StubTranscriber(
        result: TranscriptionResult(text: "I need clean needles", language: "en", confidence: 0.98)
    )
    let llm = StubResponder(
        reply: "I can share nearby exchange resources and overdose safety steps.")
    let tts = StubSynthesizer()
    let processor = CallTurnProcessor(stt: stt, llm: llm, tts: tts)

    let result = try await processor.processTurn(audioPCM16kMono: Data([0x10, 0x20]))

    #expect(result.transcription.text == "I need clean needles")
    #expect(result.reply.contains("exchange resources"))
    #expect(result.audioPayload == Data([0x01, 0x02, 0x03]))

    let conversation = await processor.currentConversation()
    #expect(conversation.count == 2)
    #expect(conversation.first?.role == "user")
    #expect(conversation.last?.role == "assistant")
}

@Test
func processTurnRejectsEmptyTranscription() async {
    let stt = StubTranscriber(
        result: TranscriptionResult(text: "   ", language: "en", confidence: 0.0)
    )
    let llm = StubResponder(reply: "unused")
    let tts = StubSynthesizer()
    let processor = CallTurnProcessor(stt: stt, llm: llm, tts: tts)

    do {
        _ = try await processor.processTurn(audioPCM16kMono: Data([0x01]))
        Issue.record("Expected noSpeechDetected error")
    } catch let error as VoiceServiceError {
        #expect(error == .noSpeechDetected)
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test
func processTurnTrimsConversationToLimit() async throws {
    let stt = StubTranscriber(
        result: TranscriptionResult(text: "hello", language: "en", confidence: 1.0)
    )
    let llm = StubResponder(reply: "hi")
    let tts = StubSynthesizer()
    let processor = CallTurnProcessor(stt: stt, llm: llm, tts: tts, maxHistoryMessages: 4)

    _ = try await processor.processTurn(audioPCM16kMono: Data([0x01]))
    _ = try await processor.processTurn(audioPCM16kMono: Data([0x02]))
    _ = try await processor.processTurn(audioPCM16kMono: Data([0x03]))

    let conversation = await processor.currentConversation()
    #expect(conversation.count == 4)
}
