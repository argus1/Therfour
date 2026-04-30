# TherFour RAG (Retrieval-Augmented Generation) System

## Overview

The TherFour RAG system provides intelligent harm reduction context retrieval for the voice agent. It:

1. **Ingests PDFs** from the doclib directory using Docling intelligent chunking
2. **Generates Embeddings** using SentenceTransformer models
3. **Stores in ChromaDB** for efficient semantic similarity search
4. **Retrieves Context** for voice conversations with relevance scoring
5. **Exports Precomputed Index** for iOS and client applications

## System Architecture

```
Doclib PDFs
    ↓
Docling Document Converter
    ↓
Hierarchical/Hybrid Chunking
    ↓
SentenceTransformer Embeddings
    ↓
ChromaDB Vector Store
    ↓
RAG API Endpoints
    ↓
Voice Agent + iOS Apps
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `chromadb>=0.4.0` - Vector database
- `docling>=0.1.0` - PDF understanding and chunking
- `sentence-transformers>=2.2.0` - Embedding generation
- `pypdf>=4.0.0` - PDF fallback extraction

### 2. Ingest PDFs

```bash
python ingest_doclib_docling_chroma.py \
  --pdf-dir ./doclib \
  --persist-dir ./chroma_db \
  --collection therfour_docs
```

This will:
- Discover all PDFs in `./doclib`
- Use Docling to intelligently chunk documents
- Generate embeddings for each chunk
- Store in ChromaDB at `./chroma_db`
- Export precomputed index to `./app/data/rag_precomputed_index.json`

### 3. Start the API Server

```bash
python -m uvicorn app.main:app --reload
```

The RAG endpoints are now available:
- `GET /rag/health` - Check RAG system status
- `POST /rag/retrieve` - Retrieve relevant documents

### 4. Query for Context

```bash
# Health check
curl http://localhost:8000/rag/health

# Retrieve documents
curl -X POST http://localhost:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are harm reduction strategies?",
    "n_results": 5
  }'
```

## Components

### ingest_doclib_docling_chroma.py

Main ingestion script that:
- Discovers PDFs in a directory
- Converts PDFs using Docling DocumentConverter
- Applies hierarchical or hybrid chunking
- Generates embeddings with SentenceTransformer
- Stores in ChromaDB with metadata
- Exports precomputed index as JSON

**Features:**
- Automatic fallback from Docling to PyPDF if conversion fails
- Configurable chunk sizes and overlap
- Unique ID generation based on content hash
- Summary reporting of ingestion results

**Usage:**
```bash
python ingest_doclib_docling_chroma.py \
  --pdf-dir ./doclib \
  --persist-dir ./chroma_db \
  --collection therfour_docs \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-json ./app/data/rag_precomputed_index.json \
  --limit 0 \
  --disable-fallback \
  --fallback-max-chars 1200 \
  --fallback-overlap-chars 200
```

### app/services/rag_retriever.py

RAGRetriever class providing:
- Simple query interface: `retriever.retrieve(query, n_results=5)`
- Source filtering: `retriever.retrieve_by_source(query, source_pdf)`
- Collection metadata: `retriever.get_collection_info()`

**Example:**
```python
from app.services.rag_retriever import RAGRetriever

retriever = RAGRetriever()

# Retrieve documents
results = retriever.retrieve(
    query="What is harm reduction?",
    n_results=5
)

for result in results:
    print(f"Source: {result['metadata']['source']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Text: {result['document'][:100]}...")
```

### app/api/routes/rag.py

FastAPI endpoints for RAG retrieval:
- `POST /rag/retrieve` - Main retrieval endpoint
- `GET /rag/health` - System health check

**Request Format:**
```json
{
  "query": "Natural language query",
  "n_results": 5,
  "source_pdf": "optional_filename.pdf"
}
```

**Response Format:**
```json
{
  "query": "Natural language query",
  "n_results": 3,
  "results": [
    {
      "id": "doc_id",
      "document": "Retrieved text chunk",
      "similarity": 0.85,
      "source": "filename.pdf",
      "chunk_index": 0
    }
  ]
}
```

## Configuration

### Embedding Models

Popular options for harm reduction content:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| all-MiniLM-L6-v2 | 22MB | Fast | Good |
| all-mpnet-base-v2 | 438MB | Slow | Excellent |
| all-distilroberta-v1 | 268MB | Medium | Good |
| multi-qa-MiniLM-L6-cos-v1 | 22MB | Fast | Good for QA |

Default: `sentence-transformers/all-MiniLM-L6-v2`

To use a different model:
```bash
python ingest_doclib_docling_chroma.py \
  --pdf-dir ./doclib \
  --embedding-model sentence-transformers/all-mpnet-base-v2
```

### Chunking Strategy

**Docling Intelligent (Primary):**
- Preserves document structure (headings, lists, tables)
- Semantic-aware boundary detection
- Better for complex documents
- Slower, may fail on corrupted PDFs

**Fallback PyPDF + Window (Secondary):**
- Simple sliding window chunking
- Reliable extraction
- May break semantic boundaries
- Used when Docling fails

**Configuration:**
```bash
python ingest_doclib_docling_chroma.py \
  --pdf-dir ./doclib \
  --disable-fallback  # Strict mode: fail if Docling fails
  --fallback-max-chars 1500  # Larger chunks
  --fallback-overlap-chars 300  # More overlap
```

## Testing

### Unit Tests

Run the RAG retriever tests:
```bash
python -m pytest tests/test_rag_retriever.py -v
```

Tests cover:
- ✅ Retriever initialization
- ✅ Document retrieval
- ✅ Source filtering
- ✅ Result sorting by relevance
- ✅ Collection information
- ✅ Various query types

### Benchmark Tests

Evaluate RAG performance:
```bash
python -m pytest benchmarks/test_chroma_benchmark.py -v
```

Metrics tracked:
- Relevance precision/recall
- Retrieval latency
- Stability under load
- Regression detection

## Performance Tuning

### Query Optimization

1. **Result Count**: Start with `n_results=5`, adjust based on context window
2. **Similarity Threshold**: Filter results by `similarity > 0.7` for high relevance
3. **Source Filtering**: Use `source_pdf` to limit domain-specific documents

### Embedding Model Selection

- **Speed Priority**: Use `all-MiniLM-L6-v2` (22MB, fast)
- **Quality Priority**: Use `all-mpnet-base-v2` (438MB, slower)
- **Balance**: Use default or test both

### ChromaDB Configuration

- **Persistence**: Automatically enabled in `./chroma_db/`
- **Index Size**: Monitor disk usage with larger document collections
- **Memory**: Embeddings cached in memory; adjust based on hardware

## Integration with Voice Agent

### Example: Conversational Context Retrieval

```python
from app.services.rag_retriever import RAGRetriever

async def handle_voice_call(user_query: str):
    # Retrieve context
    retriever = RAGRetriever()
    context_docs = retriever.retrieve(
        query=user_query,
        n_results=3
    )
    
    # Format context for LLM
    context = "\n\n".join([
        f"Source: {doc['metadata']['source']}\n{doc['document']}"
        for doc in context_docs
        if doc['similarity'] > 0.7
    ])
    
    # Include in LLM prompt
    system_prompt = f"""You are a harm reduction specialist. 
Use this context to answer questions:

{context}

Be accurate, compassionate, and evidence-based."""
    
    # Generate response with context
    response = await generate_llm_response(
        user_message=user_query,
        system_prompt=system_prompt
    )
    
    return response
```

### Example: Pre-loaded Index for iOS

```python
import json

# Load precomputed index
with open('app/data/rag_precomputed_index.json') as f:
    index = json.load(f)

# Use embeddings for client-side similarity search
for item in index:
    doc_id = item['id']
    embedding = item['embedding']
    text = item['text']
```

## Monitoring & Debugging

### Check Collection Status

```python
from app.services.rag_retriever import RAGRetriever

retriever = RAGRetriever()
info = retriever.get_collection_info()
print(f"Documents in collection: {info['count']}")
print(f"Collection name: {info['name']}")
```

### Validate Precomputed Index

```bash
python -c "
import json
with open('app/data/rag_precomputed_index.json') as f:
    data = json.load(f)
    print(f'Total items: {len(data)}')
    print(f'Embedding dimension: {len(data[0][\"embedding\"])}')
    print(f'Unique sources: {len(set(d[\"source\"] for d in data))}')
"
```

### Test API Endpoint

```bash
# Check health
curl http://localhost:8000/rag/health | jq

# Test retrieval
curl -X POST http://localhost:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "n_results": 3}' | jq
```

## Troubleshooting

### No PDFs Found

```bash
# Check doclib directory
ls -la ./doclib

# Ensure .pdf extension (lowercase)
find ./doclib -name "*.pdf" | wc -l
```

### Docling Conversion Fails

```bash
# Run with fallback enabled (default)
python ingest_doclib_docling_chroma.py --pdf-dir ./doclib

# Check specific PDF
python -c "
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert('path/to/pdf')
"
```

### Out of Memory

```bash
# Process in batches
python ingest_doclib_docling_chroma.py --limit 5
```

### ChromaDB Connection Error

```bash
# Verify persistence directory
ls -la ./chroma_db

# Reinitialize if corrupted
rm -rf ./chroma_db
python ingest_doclib_docling_chroma.py --pdf-dir ./doclib
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| chromadb | ≥0.4.0 | Vector database |
| docling | ≥0.1.0 | PDF intelligent conversion |
| sentence-transformers | ≥2.2.0 | Embeddings generation |
| pypdf | ≥4.0.0 | PDF fallback extraction |

Install all:
```bash
pip install chromadb docling sentence-transformers pypdf
```

## Files & Directories

```
├── ingest_doclib_docling_chroma.py     # Main ingestion script
├── quickstart_rag.py                   # Interactive quickstart guide
├── INGESTION.md                        # Detailed ingestion documentation
├── doclib/                             # Source PDFs
│   ├── *.pdf                           # Harm reduction documents
├── app/
│   ├── services/
│   │   └── rag_retriever.py            # RAG retrieval interface
│   ├── api/routes/
│   │   └── rag.py                      # RAG API endpoints
│   ├── data/
│   │   └── rag_precomputed_index.json  # Precomputed embeddings
│   └── main.py                         # FastAPI app with RAG routes
├── chroma_db/                          # ChromaDB persistence
├── tests/
│   └── test_rag_retriever.py           # RAG retriever tests
├── benchmarks/
│   └── test_chroma_benchmark.py        # RAG performance benchmarks
└── requirements.txt                    # Python dependencies
```

## Next Steps

1. ✅ **Ingestion**: Run PDF ingestion from doclib
2. ✅ **Testing**: Verify RAG system with tests and queries
3. ✅ **Integration**: Add RAG context to voice call handling
4. ✅ **Monitoring**: Track RAG performance with benchmarks
5. ⏳ **Optimization**: Fine-tune embedding model and chunking strategy
6. ⏳ **Deployment**: Container deployment with ChromaDB persistence

## Support & References

- **Docling**: https://github.com/DS4SD/docling
- **ChromaDB**: https://www.trychroma.com/
- **SentenceTransformers**: https://www.sbert.net/
- **Harm Reduction**: https://en.wikipedia.org/wiki/Harm_reduction
