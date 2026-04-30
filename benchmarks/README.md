# ChromaDB Retrieval Benchmark Experiment

This benchmark experiment evaluates ChromaDB vector retrieval quality and stability for the TherFour harm reduction voice agent system.

## Overview

The benchmark suite measures:
- **Relevance**: How well retrieved documents match user queries (precision/recall metrics)
- **Stability**: Performance consistency under concurrent load
- **Regression Tracking**: Historical performance monitoring and quality gates

## Key Metrics

### Relevance Metrics
- **Precision@K**: Fraction of retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents that are retrieved
- Measured at K=[1,3,5,10] for comprehensive evaluation

### Performance Metrics
- **Average Retrieval Time**: Mean query response time
- **P95/P99 Latency**: 95th/99th percentile response times
- **Stability Score**: Consistency of response times (lower variance = higher stability)
- **Error Rate**: Fraction of failed queries

### Load Testing
- **Concurrency Levels**: Tests at 1, 5, 10, 20 concurrent requests
- **Duration-based**: Sustained load testing over time periods

## Usage

### Running Tests
```bash
# Run all benchmark tests
python -m pytest benchmarks/test_chroma_benchmark.py -v

# Run specific test
python -m pytest benchmarks/test_chroma_benchmark.py::test_relevance_precision_at_k -v

# Run with detailed output
python -m pytest benchmarks/test_chroma_benchmark.py -v --tb=short
```

### Standalone Benchmark
```bash
# Run benchmark directly
python benchmarks/test_chroma_benchmark.py
```

### Integration with Real ChromaDB
When ChromaDB is implemented, update the `ChromaBenchmarkSuite` class:
1. Uncomment the `chromadb` imports
2. Initialize `self.client` and `self.collection` with real ChromaDB instances
3. Replace `mock_retrieve()` calls with actual `collection.query()` calls

## Test Data

The benchmark uses mock data when ChromaDB is not available:
- 1000 synthetic harm reduction documents
- 10 topic categories (needle exchange, naloxone, overdose prevention, etc.)
- Normalized embeddings for consistent similarity calculations

## Quality Gates

For production readiness, benchmarks should meet:
- **Precision@5 > 0.3** (at least 30% of top-5 results relevant)
- **Average retrieval time < 1.0s**
- **Error rate < 0.05** (less than 5% failures)
- **Stability score > 1.0** (response time variance acceptable)

## Results Storage

Benchmark results are automatically saved to `./chroma_benchmark_data/`:
- JSON format with full metrics
- Timestamped files for historical tracking
- Regression baseline for comparison

## Future Enhancements

- Real embedding model integration (currently uses random vectors)
- Multi-modal retrieval testing (text + audio queries)
- A/B testing framework for retrieval algorithm improvements
- Automated alerting for performance regressions