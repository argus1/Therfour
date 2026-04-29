import Foundation

actor CallTurnProcessor: TurnProcessor {
    private let stt: SpeechTranscriber
    private let llm: ChatResponder
    private let tts: SpeechSynthesizer
    private let maxHistoryMessages: Int
    private var conversationHistory: [ChatMessage] = []

    init(
        stt: SpeechTranscriber,
        llm: ChatResponder,
        tts: SpeechSynthesizer,
        maxHistoryMessages: Int = 20
    ) {
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.maxHistoryMessages = maxHistoryMessages
    }

    func processTurn(audioPCM16kMono: Data, languageHint: String? = nil) async throws
        -> TurnProcessingResult
    {
        let transcription = try await stt.transcribe(
            audioPCM16kMono: audioPCM16kMono, languageHint: languageHint)
        let text = transcription.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            throw VoiceServiceError.noSpeechDetected
        }

        conversationHistory.append(ChatMessage(role: "user", content: text))
        trimHistoryIfNeeded()

        let reply = try await llm.generateReply(
            messages: conversationHistory,
            systemPrompt: PromptTemplates.harmReductionSystemPrompt
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        guard !reply.isEmpty else {
            throw VoiceServiceError.emptyOutput
        }

        conversationHistory.append(ChatMessage(role: "assistant", content: reply))
        trimHistoryIfNeeded()

        let audio = try await tts.synthesize(text: reply, language: transcription.language)
        return TurnProcessingResult(transcription: transcription, reply: reply, audioPayload: audio)
    }

    func currentConversation() -> [ChatMessage] {
        conversationHistory
    }

    func resetConversation() {
        conversationHistory.removeAll()
    }

    private func trimHistoryIfNeeded() {
        guard conversationHistory.count > maxHistoryMessages else {
            return
        }

        conversationHistory = Array(conversationHistory.suffix(maxHistoryMessages))
    }
}
