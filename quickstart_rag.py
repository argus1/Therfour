#!/usr/bin/env python
"""Quick start example for ChromaDB RAG system.

This script demonstrates how to:
1. Ingest PDFs from the doclib directory
2. Query the RAG system for harm reduction information
3. Use the RAG endpoint in the API
"""

import json
import subprocess
import sys
from pathlib import Path

from app.services.rag_retriever import RAGRetriever


def ingest_doclib():
    """Run the PDF ingestion pipeline."""
    print("=" * 70)
    print("STEP 1: Ingesting PDFs from doclib/")
    print("=" * 70)
    
    cmd = [
        sys.executable,
        "ingest_doclib_docling_chroma.py",
        "--pdf-dir", "./doclib",
        "--persist-dir", "./chroma_db",
        "--collection", "therfour_docs",
        "--output-json", "./app/data/rag_precomputed_index.json",
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print("Ingestion failed!")
        return False
    
    return True


def query_rag():
    """Query the RAG system with example queries."""
    print("\n" + "=" * 70)
    print("STEP 2: Querying the RAG system")
    print("=" * 70)
    
    try:
        retriever = RAGRetriever()
        info = retriever.get_collection_info()
        
        print(f"\nCollection Info:")
        print(f"  Name: {info['name']}")
        print(f"  Documents: {info['count']}")
        print(f"  Model: {info['embedding_model']}")
        
        # Example queries
        queries = [
            "What are harm reduction strategies?",
            "How does naloxone work and when should I use it?",
            "Where can I find needle exchange programs?",
        ]
        
        for query in queries:
            print(f"\n{'─' * 70}")
            print(f"Query: {query}")
            print(f"{'─' * 70}")
            
            results = retriever.retrieve(query, n_results=3)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['metadata']['source']} (Similarity: {result['similarity']:.3f})")
                print(f"   {result['document'][:150]}...")
        
        return True
    
    except Exception as e:
        print(f"Error querying RAG: {e}")
        return False


def check_api():
    """Provide API endpoint information."""
    print("\n" + "=" * 70)
    print("STEP 3: Using the API")
    print("=" * 70)
    
    print("""
The RAG system is now integrated into the FastAPI backend:

Health Check Endpoint:
  GET /rag/health
  
  Response: {
    "status": "healthy",
    "collection": "therfour_docs",
    "document_count": 245,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }

Retrieval Endpoint:
  POST /rag/retrieve
  
  Request Body: {
    "query": "What are harm reduction strategies?",
    "n_results": 5,
    "source_pdf": null  # Optional filter
  }
  
  Response: {
    "query": "What are harm reduction strategies?",
    "n_results": 3,
    "results": [
      {
        "id": "document_name.pdf-00000-abc123",
        "document": "Full text of retrieved chunk...",
        "similarity": 0.85,
        "source": "document_name.pdf",
        "chunk_index": 0
      },
      ...
    ]
  }

Start the server:
  python -m uvicorn app.main:app --reload

Then test with curl:
  curl -X GET http://localhost:8000/rag/health
  
  curl -X POST http://localhost:8000/rag/retrieve \\
    -H "Content-Type: application/json" \\
    -d '{
      "query": "What are harm reduction strategies?",
      "n_results": 3
    }'
""")


def main():
    """Run the quickstart walkthrough."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  TherFour RAG System - Quick Start Guide".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Step 1: Ingest
    if not ingest_doclib():
        print("\n❌ Ingestion failed. Please check the error messages above.")
        return 1
    
    # Step 2: Query
    if not query_rag():
        print("\n❌ Query failed. Please check the error messages above.")
        return 1
    
    # Step 3: API info
    check_api()
    
    print("\n" + "=" * 70)
    print("✅ Quick start complete!")
    print("=" * 70)
    print("""
Next steps:
1. Start the FastAPI server: python -m uvicorn app.main:app --reload
2. Test the API endpoints at http://localhost:8000/docs
3. Integrate RAG retrieval into voice call handling
4. Monitor RAG performance with benchmark suite

For more information:
  - INGESTION.md - Detailed ingestion guide
  - benchmarks/README.md - Benchmark framework
  - tests/test_rag_retriever.py - Test examples
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
