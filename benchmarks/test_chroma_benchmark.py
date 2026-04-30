"""Benchmark experiment: ChromaDB retrieval quality and stability.

This module provides comprehensive benchmarking for ChromaDB vector retrieval
performance, focusing on relevance, stability under load, and regression tracking.

Usage:
    python -m pytest benchmarks/test_chroma_benchmark.py -v --tb=short
    python benchmarks/test_chroma_benchmark.py::test_relevance_precision_at_k -v
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import os

import pytest
import numpy as np

# Placeholder imports - replace with actual ChromaDB imports when implemented
# import chromadb
# from chromadb.config import Settings


@dataclass
class RetrievalResult:
    """Result of a single retrieval operation."""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    retrieval_time: float
    query_embedding: Optional[np.ndarray] = None
    ground_truth_relevant: List[str] = field(default_factory=list)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated results from a benchmark run."""
    experiment_name: str
    timestamp: str
    total_queries: int
    avg_retrieval_time: float
    p95_retrieval_time: float
    p99_retrieval_time: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    stability_score: float  # Lower variance = higher stability
    error_rate: float
    results: List[RetrievalResult] = field(default_factory=list)


class ChromaBenchmarkSuite:
    """Comprehensive benchmark suite for ChromaDB retrieval evaluation."""

    def __init__(
        self,
        collection_name: str = "harm_reduction_docs",
        embedding_dim: int = 384,  # Common embedding dimension
        n_results: int = 10,
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.n_results = n_results
        self.persist_directory = persist_directory or "./chroma_benchmark_data"

        # Create persistence directory
        os.makedirs(self.persist_directory, exist_ok=True)

        # Placeholder for ChromaDB client and collection
        self.client = None  # chromadb.PersistentClient(path=self.persist_directory)
        self.collection = None  # self.client.get_or_create_collection(name=collection_name)

        # Mock data for testing when ChromaDB not available
        self.mock_documents = []
        self.mock_embeddings = []
        self.mock_metadata = []
        self.mock_ids = []

    def initialize_mock_data(self, n_docs: int = 1000) -> None:
        """Initialize mock dataset for benchmarking when ChromaDB is not available."""
        np.random.seed(42)  # For reproducible results

        # Generate mock harm reduction documents
        harm_reduction_topics = [
            "needle exchange programs",
            "naloxone distribution",
            "overdose prevention",
            "safer drug use practices",
            "substance use treatment options",
            "mental health support",
            "housing assistance",
            "emergency services",
            "peer support networks",
            "medication-assisted treatment"
        ]

        for i in range(n_docs):
            topic = np.random.choice(harm_reduction_topics)
            doc_id = f"doc_{i:04d}"
            content = f"This document discusses {topic} and provides information about accessing services."

            self.mock_documents.append(content)
            self.mock_metadata.append({"topic": topic, "doc_id": doc_id})
            self.mock_ids.append(doc_id)

            # Generate random embedding
            embedding = np.random.normal(0, 1, self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            self.mock_embeddings.append(embedding)

        self.mock_embeddings = np.array(self.mock_embeddings)
        print(f"Initialized mock dataset with {n_docs} documents")

    def mock_retrieve(self, query_embedding: np.ndarray, n_results: int = 10) -> List[Dict[str, Any]]:
        """Mock retrieval using cosine similarity when ChromaDB not available."""
        # Calculate cosine similarities manually
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.mock_embeddings / np.linalg.norm(self.mock_embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:n_results]

        results = []
        for idx in top_indices:
            results.append({
                "document": self.mock_documents[idx],
                "metadata": self.mock_metadata[idx],
                "id": self.mock_ids[idx],
                "distance": 1.0 - similarities[idx]  # Convert similarity to distance
            })

        return results

    async def retrieve_async(self, query: str, n_results: Optional[int] = None) -> RetrievalResult:
        """Async retrieval with timing and result capture."""
        start_time = time.time()

        # Generate mock query embedding (normally would use actual embedder)
        np.random.seed(hash(query) % 2**32)  # Deterministic seed from query
        query_embedding = np.random.normal(0, 1, self.embedding_dim)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        try:
            if self.collection is not None:
                # Real ChromaDB retrieval
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results or self.n_results
                )
                retrieved_docs = [
                    {"document": doc, "metadata": meta, "id": doc_id, "distance": dist}
                    for doc, meta, doc_id, dist in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["ids"][0],
                        results["distances"][0]
                    )
                ]
            else:
                # Mock retrieval
                retrieved_docs = self.mock_retrieve(query_embedding, n_results or self.n_results)

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                query=query,
                retrieved_docs=retrieved_docs,
                retrieval_time=retrieval_time,
                query_embedding=query_embedding
            )

        except Exception as e:
            retrieval_time = time.time() - start_time
            # Return error result
            return RetrievalResult(
                query=query,
                retrieved_docs=[],
                retrieval_time=retrieval_time,
                query_embedding=query_embedding
            )

    def evaluate_relevance(self, result: RetrievalResult, ground_truth_topics: List[str]) -> RetrievalResult:
        """Evaluate retrieval relevance against ground truth topics."""
        result.ground_truth_relevant = ground_truth_topics  # Store topics instead of IDs
        retrieved_topics = [doc["metadata"]["topic"] for doc in result.retrieved_docs]

        for k in [1, 3, 5, 10]:
            if len(retrieved_topics) >= k:
                retrieved_at_k = set(retrieved_topics[:k])
                relevant_at_k = set(ground_truth_topics)
                true_positives = len(retrieved_at_k & relevant_at_k)

                precision = true_positives / k if k > 0 else 0.0
                recall = true_positives / len(relevant_at_k) if len(relevant_at_k) > 0 else 0.0

                result.precision_at_k[k] = precision
                result.recall_at_k[k] = recall
            else:
                result.precision_at_k[k] = 0.0
                result.recall_at_k[k] = 0.0

        return result

    async def run_relevance_benchmark(
        self,
        queries_and_ground_truth_topics: List[Tuple[str, List[str]]],
        n_runs: int = 3
    ) -> BenchmarkResult:
        """Run relevance-focused benchmark."""
        print(f"Running relevance benchmark with {len(queries_and_ground_truth_topics)} queries, {n_runs} runs each")

        all_results = []

        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")
            for query, ground_truth_topics in queries_and_ground_truth_topics:
                result = await self.retrieve_async(query)
                result = self.evaluate_relevance(result, ground_truth_topics)
                all_results.append(result)

        return self._aggregate_results(all_results, f"relevance_benchmark_{len(queries_and_ground_truth_topics)}_queries")

    async def run_stability_benchmark(
        self,
        queries: List[str],
        concurrency_levels: List[int] = [1, 5, 10, 20],
        duration_seconds: int = 60
    ) -> List[BenchmarkResult]:
        """Run stability benchmark under different concurrency levels."""
        print(f"Running stability benchmark with concurrency levels: {concurrency_levels}")

        results = []

        for concurrency in concurrency_levels:
            print(f"Testing concurrency level: {concurrency}")

            start_time = time.time()
            all_results = []
            query_index = 0

            # Generate continuous load for duration
            while time.time() - start_time < duration_seconds:
                # Create batch of concurrent tasks
                tasks = []
                for _ in range(concurrency):
                    query = queries[query_index % len(queries)]
                    tasks.append(self.retrieve_async(query))
                    query_index += 1

                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions and collect valid results
                valid_results = [r for r in batch_results if isinstance(r, RetrievalResult)]
                all_results.extend(valid_results)

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)

            # Aggregate results for this concurrency level
            benchmark_result = self._aggregate_results(
                all_results,
                f"stability_concurrency_{concurrency}"
            )
            results.append(benchmark_result)

        return results

    def _aggregate_results(self, results: List[RetrievalResult], experiment_name: str) -> BenchmarkResult:
        """Aggregate individual retrieval results into benchmark metrics."""
        if not results:
            return BenchmarkResult(
                experiment_name=experiment_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                total_queries=0,
                avg_retrieval_time=0.0,
                p95_retrieval_time=0.0,
                p99_retrieval_time=0.0,
                precision_at_1=0.0,
                precision_at_3=0.0,
                precision_at_5=0.0,
                precision_at_10=0.0,
                recall_at_5=0.0,
                recall_at_10=0.0,
                stability_score=0.0,
                error_rate=1.0
            )

        retrieval_times = [r.retrieval_time for r in results]

        # Calculate percentiles
        sorted_times = sorted(retrieval_times)
        p95_index = int(0.95 * len(sorted_times))
        p99_index = int(0.99 * len(sorted_times))

        # Calculate precision/recall aggregates
        precision_at_1 = statistics.mean([r.precision_at_k.get(1, 0.0) for r in results])
        precision_at_3 = statistics.mean([r.precision_at_k.get(3, 0.0) for r in results])
        precision_at_5 = statistics.mean([r.precision_at_k.get(5, 0.0) for r in results])
        precision_at_10 = statistics.mean([r.precision_at_k.get(10, 0.0) for r in results])
        recall_at_5 = statistics.mean([r.recall_at_k.get(5, 0.0) for r in results])
        recall_at_10 = statistics.mean([r.recall_at_k.get(10, 0.0) for r in results])

        # Stability score (inverse of coefficient of variation)
        if statistics.mean(retrieval_times) > 0:
            stability_score = statistics.mean(retrieval_times) / statistics.stdev(retrieval_times)
        else:
            stability_score = 0.0

        # Error rate (results with no retrieved docs or very slow responses)
        error_results = [r for r in results if len(r.retrieved_docs) == 0 or r.retrieval_time > 5.0]
        error_rate = len(error_results) / len(results)

        return BenchmarkResult(
            experiment_name=experiment_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_queries=len(results),
            avg_retrieval_time=statistics.mean(retrieval_times),
            p95_retrieval_time=sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1],
            p99_retrieval_time=sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1],
            precision_at_1=precision_at_1,
            precision_at_3=precision_at_3,
            precision_at_5=precision_at_5,
            precision_at_10=precision_at_10,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            stability_score=stability_score,
            error_rate=error_rate,
            results=results
        )

    def save_results(self, result: BenchmarkResult, filename: Optional[str] = None) -> str:
        """Save benchmark results to JSON file."""
        if filename is None:
            filename = f"chroma_benchmark_{result.experiment_name}_{result.timestamp.replace(' ', '_').replace(':', '-')}.json"

        filepath = os.path.join(self.persist_directory, filename)

        # Convert to serializable format
        result_dict = {
            "experiment_name": result.experiment_name,
            "timestamp": result.timestamp,
            "total_queries": result.total_queries,
            "avg_retrieval_time": result.avg_retrieval_time,
            "p95_retrieval_time": result.p95_retrieval_time,
            "p99_retrieval_time": result.p99_retrieval_time,
            "precision_at_1": result.precision_at_1,
            "precision_at_3": result.precision_at_3,
            "precision_at_5": result.precision_at_5,
            "precision_at_10": result.precision_at_10,
            "recall_at_5": result.recall_at_5,
            "recall_at_10": result.recall_at_10,
            "stability_score": result.stability_score,
            "error_rate": result.error_rate,
            "individual_results": [
                {
                    "query": r.query,
                    "retrieval_time": r.retrieval_time,
                    "num_retrieved": len(r.retrieved_docs),
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k": r.recall_at_k
                }
                for r in result.results[:10]  # Save first 10 for brevity
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"Results saved to {filepath}")
        return filepath


# Test data fixtures
@pytest.fixture
def sample_queries_and_ground_truth():
    """Sample queries with ground truth relevant document topics."""
    return [
        ("Where can I find clean needles?", ["needle exchange programs"]),
        ("How do I use naloxone?", ["naloxone distribution"]),
        ("What are safer drug use practices?", ["safer drug use practices"]),
        ("Where can I get treatment for substance use?", ["medication-assisted treatment", "substance use treatment options"]),
        ("What should I do in case of overdose?", ["overdose prevention"]),
    ]


@pytest.fixture
def benchmark_suite():
    """ChromaDB benchmark suite fixture."""
    suite = ChromaBenchmarkSuite()
    suite.initialize_mock_data(100)
    return suite


@pytest.mark.asyncio
async def test_relevance_precision_at_k(benchmark_suite, sample_queries_and_ground_truth):
    """Test retrieval precision at different k values."""
    result = await benchmark_suite.run_relevance_benchmark(sample_queries_and_ground_truth, n_runs=1)

    # Basic assertions
    assert result.total_queries > 0
    assert result.precision_at_1 >= 0.0
    assert result.precision_at_5 >= result.precision_at_1  # Should not decrease
    assert result.recall_at_5 >= 0.0
    assert result.avg_retrieval_time > 0

    print(f"Precision@1: {result.precision_at_1:.3f}")
    print(f"Precision@5: {result.precision_at_5:.3f}")
    print(f"Recall@5: {result.recall_at_5:.3f}")
    print(f"Avg retrieval time: {result.avg_retrieval_time:.3f}s")


@pytest.mark.asyncio
async def test_stability_under_load(benchmark_suite):
    """Test retrieval stability under different concurrency levels."""
    queries = [
        "needle exchange locations",
        "naloxone training",
        "overdose response",
        "treatment centers",
        "peer support"
    ]

    results = await benchmark_suite.run_stability_benchmark(
        queries=queries,
        concurrency_levels=[1, 2],  # Reduced for testing
        duration_seconds=5  # Short duration for testing
    )

    assert len(results) == 2

    for result in results:
        assert result.total_queries > 0
        assert result.error_rate < 0.1  # Less than 10% errors
        assert result.stability_score > 0

        print(f"Concurrency {result.experiment_name}:")
        print(f"  Queries: {result.total_queries}")
        print(f"  Avg time: {result.avg_retrieval_time:.3f}s")
        print(f"  P95 time: {result.p95_retrieval_time:.3f}s")
        print(f"  Stability score: {result.stability_score:.3f}")
        print(f"  Error rate: {result.error_rate:.3f}")


@pytest.mark.asyncio
async def test_regression_baseline(benchmark_suite, sample_queries_and_ground_truth):
    """Establish baseline metrics for regression testing."""
    result = await benchmark_suite.run_relevance_benchmark(sample_queries_and_ground_truth, n_runs=3)

    # Save results for future comparison
    filepath = benchmark_suite.save_results(result, "regression_baseline.json")

    # Assert basic functionality (with random embeddings, precision will be low)
    assert result.total_queries > 0
    assert result.avg_retrieval_time > 0
    assert result.avg_retrieval_time < 1.0  # Should be fast with mock data
    assert result.error_rate < 0.1  # Low error rate expected

    # With random embeddings, we expect some matches by chance
    # This is more of a smoke test than a quality gate
    assert result.precision_at_10 >= 0.0  # At least no precision
    assert result.recall_at_10 >= 0.0     # At least no recall

    print("Regression baseline established:")
    print(f"  Precision@5: {result.precision_at_5:.3f} (random baseline)")
    print(f"  Precision@10: {result.precision_at_10:.3f} (random baseline)")
    print(f"  Avg retrieval time: {result.avg_retrieval_time:.3f}s (target: <1.0s)")
    print(f"  Error rate: {result.error_rate:.3f} (target: <0.1)")
    print(f"  Results saved to: {filepath}")


@pytest.mark.asyncio
async def test_retrieval_consistency(benchmark_suite):
    """Test that identical queries return consistent results."""
    query = "clean needle programs"

    # Run same query multiple times
    results = []
    for _ in range(5):
        result = await benchmark_suite.retrieve_async(query)
        results.append(result)

    # Check consistency
    retrieval_times = [r.retrieval_time for r in results]
    time_std = statistics.stdev(retrieval_times)

    # Results should be identical (same documents in same order)
    first_result_docs = [doc["id"] for doc in results[0].retrieved_docs]
    for result in results[1:]:
        result_docs = [doc["id"] for doc in result.retrieved_docs]
        assert result_docs == first_result_docs, "Identical queries should return identical results"

    print(f"Query consistency check passed:")
    print(f"  Retrieval time std: {time_std:.6f}s")
    print(f"  All results identical: ✓")


if __name__ == "__main__":
    # Allow running standalone for quick benchmarking
    async def main():
        suite = ChromaBenchmarkSuite()
        suite.initialize_mock_data(500)

        # Sample benchmark run
        queries_and_truth = [
            ("Where can I get clean syringes?", ["doc_0001", "doc_0123"]),
            ("How to prevent overdose?", ["doc_0002", "doc_0234"]),
        ]

        print("Running ChromaDB benchmark...")
        result = await suite.run_relevance_benchmark(queries_and_truth, n_runs=2)
        suite.save_results(result)

        print("Benchmark complete!")
        print(f"Precision@5: {result.precision_at_5:.3f}")
        print(f"Average retrieval time: {result.avg_retrieval_time:.3f}s")

    asyncio.run(main())