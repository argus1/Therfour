"""Real ChromaDB RAG system benchmarking with ingested documents.

This module benchmarks actual ChromaDB retrieval performance against real harm
reduction documents, measuring relevance, stability, and regression tracking.

Usage:
    python benchmarks/benchmark_rag_system.py
    python -m pytest benchmarks/test_rag_benchmark.py -v
"""

import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from app.services.rag_retriever import RAGRetriever


@dataclass
class RAGBenchmarkResult:
    """Single RAG query benchmark result."""

    query: str
    retrieval_time: float
    num_results: int
    top_similarity: float
    avg_similarity: float
    retrieved_sources: List[str]


@dataclass
class RAGBenchmarkSummary:
    """Aggregated RAG benchmark results."""

    experiment_name: str
    timestamp: str
    total_queries: int
    avg_retrieval_time: float
    p95_retrieval_time: float
    p99_retrieval_time: float
    avg_top_similarity: float
    avg_similarity_score: float
    unique_sources_retrieved: int
    error_rate: float
    results: List[RAGBenchmarkResult]


class RAGSystemBenchmark:
    """Benchmark suite for real ChromaDB RAG system."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "therfour_docs",
    ):
        """Initialize benchmark with real RAG system.

        Args:
            persist_dir: ChromaDB persistence directory
            collection_name: Collection name in ChromaDB
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Try multiple persistence directories
        possible_dirs = [
            persist_dir,
            "./test_chroma_db",
            "./chroma_db",
        ]

        self.available = False
        for db_dir in possible_dirs:
            try:
                self.retriever = RAGRetriever(
                    persist_dir=db_dir,
                    collection_name=collection_name,
                )
                self.available = True
                info = self.retriever.get_collection_info()
                self.collection_size = info["count"]
                self.persist_dir = db_dir  # Use found directory
                break
            except Exception as e:
                self.init_error = str(e)

        if not self.available:
            self.collection_size = 0

    def run_relevance_benchmark(
        self,
        queries: List[str],
        n_results: int = 5,
    ) -> RAGBenchmarkSummary:
        """Benchmark retrieval quality and relevance.

        Args:
            queries: List of test queries
            n_results: Number of results to retrieve per query

        Returns:
            Benchmark summary with relevance metrics
        """
        if not self.available:
            raise RuntimeError(
                f"RAG system not available: {self.init_error}. "
                "Run: python ingest_doclib_docling_chroma.py --pdf-dir ./doclib"
            )

        print(f"Running relevance benchmark with {len(queries)} queries...")

        results = []
        retrieval_times = []
        top_similarities = []
        all_similarities = []
        all_sources = set()

        for query in queries:
            start_time = time.time()
            try:
                retrieved = self.retriever.retrieve(query=query, n_results=n_results)
                retrieval_time = time.time() - start_time

                if retrieved:
                    similarities = [r["similarity"] for r in retrieved]
                    top_similarity = similarities[0]
                    avg_similarity = statistics.mean(similarities)
                    sources = [r["metadata"]["source"] for r in retrieved]

                    top_similarities.append(top_similarity)
                    all_similarities.extend(similarities)
                    all_sources.update(sources)

                    results.append(
                        RAGBenchmarkResult(
                            query=query,
                            retrieval_time=retrieval_time,
                            num_results=len(retrieved),
                            top_similarity=top_similarity,
                            avg_similarity=avg_similarity,
                            retrieved_sources=sources,
                        )
                    )
                    retrieval_times.append(retrieval_time)

            except Exception as e:
                print(f"Query failed: {query} - {e}")
                continue

        # Compute statistics
        if not retrieval_times:
            raise RuntimeError("No successful queries in benchmark")

        sorted_times = sorted(retrieval_times)
        p95_index = int(0.95 * len(sorted_times))
        p99_index = int(0.99 * len(sorted_times))

        return RAGBenchmarkSummary(
            experiment_name="relevance_benchmark",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_queries=len(results),
            avg_retrieval_time=statistics.mean(retrieval_times),
            p95_retrieval_time=sorted_times[p95_index]
            if p95_index < len(sorted_times)
            else sorted_times[-1],
            p99_retrieval_time=sorted_times[p99_index]
            if p99_index < len(sorted_times)
            else sorted_times[-1],
            avg_top_similarity=statistics.mean(top_similarities)
            if top_similarities
            else 0.0,
            avg_similarity_score=statistics.mean(all_similarities)
            if all_similarities
            else 0.0,
            unique_sources_retrieved=len(all_sources),
            error_rate=1.0 - (len(results) / len(queries)) if queries else 1.0,
            results=results,
        )

    def run_concurrent_benchmark(
        self,
        queries: List[str],
        concurrency_levels: List[int] = [1, 5, 10],
        n_results: int = 5,
    ) -> Dict[int, RAGBenchmarkSummary]:
        """Benchmark stability under concurrent load.

        Args:
            queries: List of test queries to cycle through
            concurrency_levels: Concurrency levels to test
            n_results: Number of results per query

        Returns:
            Dictionary of concurrency level -> benchmark summary
        """
        if not self.available:
            raise RuntimeError(
                f"RAG system not available: {self.init_error}. "
                "Run: python ingest_doclib_docling_chroma.py --pdf-dir ./doclib"
            )

        print(f"Running concurrent load benchmark...")

        results_by_level = {}

        for concurrency in concurrency_levels:
            print(f"Testing concurrency level: {concurrency}")
            results = []
            retrieval_times = []

            # Run each query concurrency times
            for _ in range(concurrency):
                for query in queries:
                    start_time = time.time()
                    try:
                        retrieved = self.retriever.retrieve(query=query, n_results=n_results)
                        retrieval_time = time.time() - start_time

                        if retrieved:
                            results.append(
                                RAGBenchmarkResult(
                                    query=query,
                                    retrieval_time=retrieval_time,
                                    num_results=len(retrieved),
                                    top_similarity=retrieved[0]["similarity"],
                                    avg_similarity=statistics.mean(
                                        [r["similarity"] for r in retrieved]
                                    ),
                                    retrieved_sources=[
                                        r["metadata"]["source"] for r in retrieved
                                    ],
                                )
                            )
                            retrieval_times.append(retrieval_time)
                    except Exception:
                        continue

            if retrieval_times:
                sorted_times = sorted(retrieval_times)
                p95_idx = int(0.95 * len(sorted_times))
                p99_idx = int(0.99 * len(sorted_times))

                results_by_level[concurrency] = RAGBenchmarkSummary(
                    experiment_name=f"concurrent_benchmark_c{concurrency}",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    total_queries=len(results),
                    avg_retrieval_time=statistics.mean(retrieval_times),
                    p95_retrieval_time=sorted_times[p95_idx]
                    if p95_idx < len(sorted_times)
                    else sorted_times[-1],
                    p99_retrieval_time=sorted_times[p99_idx]
                    if p99_idx < len(sorted_times)
                    else sorted_times[-1],
                    avg_top_similarity=statistics.mean(
                        [r.top_similarity for r in results]
                    ),
                    avg_similarity_score=statistics.mean(
                        [r.avg_similarity for r in results]
                    ),
                    unique_sources_retrieved=len(
                        set().union(*[set(r.retrieved_sources) for r in results])
                    ),
                    error_rate=1.0 - (len(results) / (len(queries) * concurrency))
                    if queries
                    else 1.0,
                    results=results,
                )

        return results_by_level

    def save_results(self, summary: RAGBenchmarkSummary, filename: Optional[str] = None):
        """Save benchmark results to JSON.

        Args:
            summary: Benchmark summary to save
            filename: Optional output filename
        """
        if filename is None:
            filename = (
                f"rag_benchmark_{summary.experiment_name}_"
                f"{summary.timestamp.replace(' ', '_').replace(':', '-')}.json"
            )

        output_dir = Path(self.persist_dir).parent / "benchmark_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        # Convert to serializable format
        data = {
            "experiment_name": summary.experiment_name,
            "timestamp": summary.timestamp,
            "total_queries": summary.total_queries,
            "avg_retrieval_time": summary.avg_retrieval_time,
            "p95_retrieval_time": summary.p95_retrieval_time,
            "p99_retrieval_time": summary.p99_retrieval_time,
            "avg_top_similarity": summary.avg_top_similarity,
            "avg_similarity_score": summary.avg_similarity_score,
            "unique_sources_retrieved": summary.unique_sources_retrieved,
            "error_rate": summary.error_rate,
            "collection_size": self.collection_size,
            "individual_results": [
                {
                    "query": r.query,
                    "retrieval_time": r.retrieval_time,
                    "num_results": r.num_results,
                    "top_similarity": r.top_similarity,
                    "avg_similarity": r.avg_similarity,
                    "sources": r.retrieved_sources,
                }
                for r in summary.results[:10]  # Save first 10
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {filepath}")
        return filepath


# Test fixtures
@pytest.fixture
def rag_benchmark():
    """RAG benchmark fixture."""
    return RAGSystemBenchmark()


@pytest.fixture
def harm_reduction_queries():
    """Real harm reduction queries for testing."""
    return [
        "What is harm reduction and how does it work?",
        "Where can I find clean needle programs?",
        "How does naloxone work in an overdose?",
        "What are the benefits of medication-assisted treatment?",
        "How can I find peer support groups?",
        "What should I do if someone is overdosing?",
        "Are there safer ways to use drugs?",
        "What is syringe service programs?",
        "How do I access treatment for substance use?",
        "What are the signs of overdose?",
    ]


# Tests
@pytest.mark.skipif(
    not (Path("./chroma_db").exists() or Path("./test_chroma_db").exists()),
    reason="ChromaDB not initialized. Run: python ingest_doclib_docling_chroma.py",
)
def test_rag_system_available(rag_benchmark):
    """Verify RAG system is available."""
    assert rag_benchmark.available, f"RAG system error: {rag_benchmark.init_error}"
    assert rag_benchmark.collection_size > 0, "Collection is empty"
    print(f"✓ RAG system available with {rag_benchmark.collection_size} documents")


@pytest.mark.skipif(
    not (Path("./chroma_db").exists() or Path("./test_chroma_db").exists()),
    reason="ChromaDB not initialized. Run: python ingest_doclib_docling_chroma.py",
)
def test_relevance_benchmark(rag_benchmark, harm_reduction_queries):
    """Benchmark retrieval relevance."""
    summary = rag_benchmark.run_relevance_benchmark(
        queries=harm_reduction_queries,
        n_results=5,
    )

    print(f"\n{'='*60}")
    print("RELEVANCE BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Queries: {summary.total_queries}")
    print(f"Avg retrieval time: {summary.avg_retrieval_time:.4f}s")
    print(f"P95 retrieval time: {summary.p95_retrieval_time:.4f}s")
    print(f"P99 retrieval time: {summary.p99_retrieval_time:.4f}s")
    print(f"Avg top similarity: {summary.avg_top_similarity:.3f}")
    print(f"Avg all similarities: {summary.avg_similarity_score:.3f}")
    print(f"Unique sources: {summary.unique_sources_retrieved}")
    print(f"Error rate: {summary.error_rate:.1%}")
    print(f"{'='*60}\n")

    assert summary.total_queries > 0
    assert summary.avg_retrieval_time < 2.0  # Should be fast
    assert summary.avg_top_similarity > 0  # Should find relevant docs
    assert summary.error_rate < 0.2  # Less than 20% errors

    rag_benchmark.save_results(summary)


@pytest.mark.skipif(
    not (Path("./chroma_db").exists() or Path("./test_chroma_db").exists()),
    reason="ChromaDB not initialized. Run: python ingest_doclib_docling_chroma.py",
)
def test_concurrent_benchmark(rag_benchmark, harm_reduction_queries):
    """Benchmark stability under concurrent load."""
    results = rag_benchmark.run_concurrent_benchmark(
        queries=harm_reduction_queries[:3],  # Limit for speed
        concurrency_levels=[1, 2, 5],
        n_results=5,
    )

    print(f"\n{'='*60}")
    print("CONCURRENT LOAD BENCHMARK RESULTS")
    print(f"{'='*60}")

    for concurrency, summary in sorted(results.items()):
        print(f"\nConcurrency Level: {concurrency}")
        print(f"  Queries: {summary.total_queries}")
        print(f"  Avg time: {summary.avg_retrieval_time:.4f}s")
        print(f"  P95 time: {summary.p95_retrieval_time:.4f}s")
        print(f"  P99 time: {summary.p99_retrieval_time:.4f}s")
        print(f"  Avg similarity: {summary.avg_similarity_score:.3f}")
        print(f"  Error rate: {summary.error_rate:.1%}")

    print(f"{'='*60}\n")

    assert all(s.error_rate < 0.2 for s in results.values())


if __name__ == "__main__":
    benchmark = RAGSystemBenchmark()

    if not benchmark.available:
        print(f"❌ RAG system not available: {benchmark.init_error}")
        print("Run: python ingest_doclib_docling_chroma.py --pdf-dir ./doclib")
        exit(1)

    print(f"✓ RAG system available with {benchmark.collection_size} documents\n")

    queries = [
        "What is harm reduction?",
        "Where can I get clean needles?",
        "How does naloxone work?",
        "What is medication-assisted treatment?",
        "How do I find peer support?",
    ]

    # Run benchmark
    summary = benchmark.run_relevance_benchmark(queries, n_results=5)
    benchmark.save_results(summary)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Queries: {summary.total_queries}")
    print(f"Avg retrieval time: {summary.avg_retrieval_time:.4f}s")
    print(f"Top similarity: {summary.avg_top_similarity:.3f}")
    print(f"Avg similarity: {summary.avg_similarity_score:.3f}")
    print(f"Unique sources: {summary.unique_sources_retrieved}")
    print(f"Error rate: {summary.error_rate:.1%}")
    print(f"{'='*60}")
