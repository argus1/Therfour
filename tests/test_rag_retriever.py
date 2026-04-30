"""Tests for RAG retriever functionality."""

import pytest
from pathlib import Path
import json
import tempfile
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from app.services.rag_retriever import RAGRetriever


# Create a persistent test database directory
TEST_CHROMA_DIR = "./test_chroma_db"


@pytest.fixture(scope="session", autouse=True)
def setup_test_chroma():
    """Set up a persistent ChromaDB collection for all tests."""
    os.makedirs(TEST_CHROMA_DIR, exist_ok=True)
    
    # Initialize client and collection
    client = chromadb.PersistentClient(
        path=TEST_CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # Create collection
    try:
        collection = client.get_collection(name="therfour_docs")
        # Clear existing data if present
        collection.delete(where={})
    except:
        collection = client.get_or_create_collection(name="therfour_docs")

    # Add test documents
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    test_docs = [
        "Needle exchange programs provide clean needles to reduce disease transmission",
        "Naloxone is an opioid antagonist that can reverse overdose",
        "Harm reduction is healthcare focused on reducing substance use harms",
        "Peer support groups help people maintain recovery and share experiences",
        "Medication-assisted treatment combines medication with behavioral therapy",
    ]

    embeddings = embedder.encode(test_docs, normalize_embeddings=True).tolist()

    ids = [f"doc_{i}" for i in range(len(test_docs))]
    metadatas = [
        {"source": f"document_{i}.pdf", "chunk_index": 0}
        for i in range(len(test_docs))
    ]

    collection.upsert(
        ids=ids,
        documents=test_docs,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    yield

    # Cleanup after all tests
    # (optional: comment out if you want to keep test data)
    # import shutil
    # shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)


def test_rag_retriever_initialization(setup_test_chroma):
    """Test RAGRetriever initialization."""
    retriever = RAGRetriever(persist_dir=TEST_CHROMA_DIR)
    info = retriever.get_collection_info()

    assert info["name"] == "therfour_docs"
    assert info["count"] == 5


def test_retrieve_documents(setup_test_chroma):
    """Test retrieving documents by query."""
    retriever = RAGRetriever(persist_dir=TEST_CHROMA_DIR)

    results = retriever.retrieve(
        query="How can I reduce overdose risk?",
        n_results=3,
    )

    assert len(results) <= 3
    assert all("id" in r for r in results)
    assert all("document" in r for r in results)
    assert all("similarity" in r for r in results)
    assert all(0 <= r["similarity"] <= 1 for r in results)


def test_retrieve_by_source(setup_test_chroma):
    """Test retrieving documents from specific source."""
    retriever = RAGRetriever(persist_dir=TEST_CHROMA_DIR)

    results = retriever.retrieve_by_source(
        query="harm reduction",
        source_pdf="document_0.pdf",
        n_results=5,
    )

    assert all(r["metadata"]["source"] == "document_0.pdf" for r in results)


def test_retrieve_returns_sorted_by_similarity(setup_test_chroma):
    """Test that results are sorted by relevance."""
    retriever = RAGRetriever(persist_dir=TEST_CHROMA_DIR)

    results = retriever.retrieve(
        query="needle exchange programs",
        n_results=5,
    )

    if len(results) > 1:
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)


def test_collection_info(setup_test_chroma):
    """Test getting collection information."""
    retriever = RAGRetriever(persist_dir=TEST_CHROMA_DIR)
    info = retriever.get_collection_info()

    assert "name" in info
    assert "count" in info
    assert "embedding_model" in info
    assert "persist_dir" in info
    assert info["count"] == 5


@pytest.mark.parametrize(
    "query",
    [
        "What is harm reduction?",
        "How do I get clean needles?",
        "How does naloxone work?",
        "Where can I find peer support?",
        "What is medication-assisted treatment?",
    ],
)
def test_retrieve_various_queries(setup_test_chroma, query):
    """Test retrieval with various harm reduction queries."""
    retriever = RAGRetriever(persist_dir=TEST_CHROMA_DIR)
    results = retriever.retrieve(query=query, n_results=3)

    assert len(results) > 0
    assert len(results) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
