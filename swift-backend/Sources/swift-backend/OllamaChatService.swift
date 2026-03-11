import Foundation

final class OllamaChatService: ChatResponder, LLMCapabilityProbe, @unchecked Sendable {
    private let baseURL: URL
    private let model: String
    private let session: URLSession

    init(baseURL: URL, model: String, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.model = model
        self.session = session
    }

    func generateReply(messages: [ChatMessage], systemPrompt: String) async throws -> String {
        let requestPayload = OllamaChatRequest(
            model: model,
            messages: [OllamaMessage(role: "system", content: systemPrompt)]
                + messages.map {
                    OllamaMessage(role: $0.role, content: $0.content)
                },
            stream: false
        )

        var request = URLRequest(url: baseURL.appendingPathComponent("api/chat"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(requestPayload)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw VoiceServiceError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            throw VoiceServiceError.httpError(http.statusCode)
        }

        let decoded = try JSONDecoder().decode(OllamaChatResponse.self, from: data)
        let content = decoded.message.content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else {
            throw VoiceServiceError.emptyOutput
        }

        return content
    }

    func probeLLMCapabilities() async -> LLMBackendCapabilities {
        let capabilityURL = baseURL.appendingPathComponent("capabilities")

        do {
            let (data, response) = try await session.data(from: capabilityURL)
            guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode)
            else {
                return .fallback
            }

            struct CapabilityResponse: Decodable {
                let supported_languages: [String]?
                let supports_json_response_format: Bool?
                let notes: String?
            }

            let decoded = try JSONDecoder().decode(CapabilityResponse.self, from: data)
            return LLMBackendCapabilities(
                supportedLanguages: Set(decoded.supported_languages ?? []),
                supportsJSONResponseFormat: decoded.supports_json_response_format ?? true,
                notes: decoded.notes ?? "Remote capabilities"
            )
        } catch {
            return .fallback
        }
    }
}

private struct OllamaChatRequest: Encodable {
    let model: String
    let messages: [OllamaMessage]
    let stream: Bool
}

private struct OllamaMessage: Codable {
    let role: String
    let content: String
}

private struct OllamaChatResponse: Decodable {
    let message: OllamaMessage
}
