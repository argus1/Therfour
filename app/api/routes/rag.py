"""RAG retrieval endpoint for harm reduction context."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

try:
    from app.services.rag_retriever import RAGRetriever
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

router = APIRouter()


class RAGQueryRequest(BaseModel):
    """Request model for RAG retrieval."""

    query: str = Field(..., description="Natural language query for harm reduction context")
    n_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve (1-20)",
    )
    source_pdf: Optional[str] = Field(
        default=None, description="Optional: filter results to specific PDF source"
    )


class RAGDocumentResult(BaseModel):
    """Retrieved document with metadata."""

    id: str
    document: str
    similarity: float
    source: str
    chunk_index: int


class RAGQueryResponse(BaseModel):
    """Response model for RAG retrieval."""

    query: str
    n_results: int
    results: List[RAGDocumentResult]


@router.post("/rag/retrieve", response_model=RAGQueryResponse)
async def retrieve_context(request: RAGQueryRequest) -> RAGQueryResponse:
    """Retrieve harm reduction documents relevant to a query.

    This endpoint uses semantic similarity search to find relevant harm reduction
    resources from the ingested document collection.

    Args:
        request: Query request with query text and optional filters

    Returns:
        Query response with retrieved documents and similarity scores

    Raises:
        HTTPException: If RAG system is not available or query fails
    """
    if not RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="RAG retrieval system not available. Run: "
            "python ingest_doclib_docling_chroma.py --pdf-dir ./doclib",
        )

    try:
        retriever = RAGRetriever()

        # Retrieve documents
        if request.source_pdf:
            raw_results = retriever.retrieve_by_source(
                query=request.query,
                source_pdf=request.source_pdf,
                n_results=request.n_results,
            )
        else:
            raw_results = retriever.retrieve(
                query=request.query,
                n_results=request.n_results,
            )

        # Format results
        results = [
            RAGDocumentResult(
                id=r["id"],
                document=r["document"],
                similarity=r["similarity"],
                source=r["metadata"]["source"],
                chunk_index=r["metadata"].get("chunk_index", 0),
            )
            for r in raw_results
        ]

        return RAGQueryResponse(
            query=request.query,
            n_results=len(results),
            results=results,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG retrieval failed: {str(e)}",
        )


@router.get("/rag/health")
async def rag_health():
    """Check if RAG system is available and healthy."""
    if not RAG_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "ChromaDB not initialized. Run: python ingest_doclib_docling_chroma.py",
        }

    try:
        retriever = RAGRetriever()
        info = retriever.get_collection_info()
        return {
            "status": "healthy",
            "collection": info["name"],
            "document_count": info["count"],
            "embedding_model": info["embedding_model"],
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"RAG health check failed: {str(e)}",
        }
