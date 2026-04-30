"""ChromaDB RAG retrieval interface for TherFour voice agent.

Provides simple interface for querying ingested harm reduction documents
stored in ChromaDB collection.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    """Interface for retrieving harm reduction context from ChromaDB."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "therfour_docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize RAG retriever with ChromaDB connection.

        Args:
            persist_dir: Path to ChromaDB persistence directory
            collection_name: Name of the ChromaDB collection
            embedding_model: SentenceTransformer model name for embeddings
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize client and collection
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(name=collection_name)

        # Initialize embedder
        self.embedder = SentenceTransformer(embedding_model)

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Natural language query
            n_results: Number of results to return
            where: Optional ChromaDB where filter

        Returns:
            List of retrieved documents with scores
        """
        # Encode query
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
        )

        # Format results
        retrieved = []
        for idx in range(len(results["ids"][0])):
            distance = results["distances"][0][idx]
            # ChromaDB returns distances in [0, 2] range for normalized embeddings
            # Convert to similarity in [0, 1] range
            similarity = max(0.0, 1.0 - distance)
            
            retrieved.append(
                {
                    "id": results["ids"][0][idx],
                    "document": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": distance,
                    "similarity": similarity,
                }
            )

        return retrieved

    def retrieve_by_source(
        self,
        query: str,
        source_pdf: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents from a specific source PDF.

        Args:
            query: Natural language query
            source_pdf: PDF filename to filter by
            n_results: Number of results to return

        Returns:
            List of retrieved documents from specified source
        """
        return self.retrieve(
            query=query,
            n_results=n_results,
            where={"source": {"$eq": source_pdf}},
        )

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection.

        Returns:
            Collection metadata and statistics
        """
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "embedding_model": self.embedding_model,
            "persist_dir": self.persist_dir,
        }


# Example usage for testing
if __name__ == "__main__":
    try:
        retriever = RAGRetriever()
        info = retriever.get_collection_info()
        print(f"Collection info: {info}")

        # Example query
        results = retriever.retrieve(
            "What are harm reduction strategies?",
            n_results=3,
        )

        print("\nRetrieved documents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['metadata']['source']}")
            print(f"   Similarity: {result['similarity']:.3f}")
            print(f"   Preview: {result['document'][:150]}...")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure to run ingest_doclib_docling_chroma.py first")
