Includes threshold behavior, low-confidence handling, and fallback response paths.

In 2026, typical speech-to-text (STT) thresholds for telephony agents, which operate in challenging environments with audio compression (8 kHz), background noise, and varying accents, generally target a Word Error Rate (WER) below 10–15%.

For high-performing, real-time voice agents in production, a WER under 10% is considered "good," while under 5% is considered excellent, close to human parity.

Accuracy Thresholds (Word Error Rate - WER)

Excellent (5% WER or lower): Nearly perfect; 95% accuracy. Suitable for high-stakes, automated compliance monitoring.

Good/Production Grade (5–10% WER): Highly usable; 90-95% accuracy. Standard for modern voice agents and agent-assist tools.

Acceptable (10–15% WER): Usable for general analytics, sentiment analysis, and call summarization, but requires monitoring.

Poor (Above 15% WER): Frequently creates misunderstandings. Not recommended for production automated systems.

Key Performance Metrics for Telephony

Because standard WER does not account for critical business information, telephony agents often use specific, more stringent thresholds:

Entity Accuracy: Near-zero error rates on crucial entities like phone numbers, names, and account IDs.

Semantic WER: A metric focusing on preserving meaning rather than exact word matching. Semantic accuracy can remain above 85% even if raw WER reaches 10–12%.

Time to Final Result: For voice agents, final transcripts are expected in under 300ms to provide a natural conversation, though some systems allow 700ms–1.5s for increased accuracy.

Factors Influencing Thresholds

Noise/Environment: Phone conversations typically yield lower accuracy (80–88%) compared to clean studio recordings (95-98%).

Industry/Domain: Specialized models, such as those used for healthcare, are expected to maintain lower WER (e.g., 4.9% medical entity error rate) than general-purpose models.

Agent Assist vs. Automated Agent: Agents helping a human need 85%+ accuracy, whereas fully automated IVRs (Interactive Voice Response) often require 90%+ to prevent, for example, misrouted calls increasing handle times.

Typical 2026 Phone Call Benchmarks

According to 2026 data, 80-88% accuracy (12-20% WER) is typical for phone conversations due to 8kHz audio compression. However, top-tier vendors in 2026 now report 93-94% accuracy (5-6% WER) on similar datasets using advanced models.
