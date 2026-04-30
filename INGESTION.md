# PDF Ingestion Pipeline: Doclib → ChromaDB

This ingestion pipeline processes PDF documents from the `doclib` directory using intelligent document chunking and embeddings, storing results in ChromaDB for efficient RAG retrieval.

## Overview

The ingestion process:
1. **Discovers** all PDFs in the source directory
2. **Converts** PDFs using Docling's intelligent document converter
3. **Chunks** documents using hierarchical/hybrid chunking
4. **Embeds** chunks using SentenceTransformer (all-MiniLM-L6-v2 by default)
5. **Stores** embeddings in ChromaDB for similarity search
6. **Exports** precomputed index as JSON for iOS/client applications

### Fallback Strategy

If Docling chunking fails on any PDF:
- Falls back to PyPDF extraction + simple sliding-window chunking
- Maintains data integrity with fallback metadata
- Continues processing remaining PDFs

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installations
python -c "import chromadb; import docling; import sentence_transformers; print('✓ All dependencies installed')"
```

## Usage

### Basic Usage

```bash
python ingest_doclib_docling_chroma.py \
  --pdf-dir ./doclib \
  --persist-dir ./chroma_db \
  --collection therfour_docs
```

### Advanced Options

```bash
python ingest_doclib_docling_chroma.py \
  --pdf-dir ./doclib \
  --persist-dir ./chroma_db \
  --collection therfour_docs \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-json ./app/data/rag_precomputed_index.json \
  --limit 5 \
  --fallback-max-chars 1500 \
  --fallback-overlap-chars 300
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pdf-dir` | required | Directory containing PDFs to ingest |
| `--persist-dir` | `./chroma_db` | Where to store ChromaDB data |
| `--collection` | `therfour_docs` | ChromaDB collection name |
| `--embedding-model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model to use |
| `--output-json` | `./app/data/rag_precomputed_index.json` | Export path for precomputed index |
| `--limit` | 0 (no limit) | Max PDFs to process |
| `--disable-fallback` | false | Disable fallback if Docling fails |
| `--fallback-max-chars` | 1200 | Max chars per fallback chunk |
| `--fallback-overlap-chars` | 200 | Overlap between fallback chunks |

## Output

After successful ingestion:

### ChromaDB Collection
- **Location**: `./chroma_db/`
- **Collection**: `therfour_docs` (configurable)
- **Query-ready**: Can be queried immediately for similarity search

### Precomputed Index JSON
- **Location**: `./app/data/rag_precomputed_index.json`
- **Format**: Array of objects with `id`, `source`, `text`, `embedding`
- **Use case**: Pre-load for iOS apps or client-side RAG

### Summary Output
```
============================================================
INGESTION SUMMARY
============================================================
Processed PDFs:        10 / 10
Total chunks:          1,245
Fallback conversions:  0
Collection:            therfour_docs
Chroma persist dir:    ./chroma_db
Precomputed JSON:      ./app/data/rag_precomputed_index.json
============================================================
```

## Performance Considerations

### Embedding Model Selection

- **all-MiniLM-L6-v2** (current): Fast, lightweight (22MB), good for general harm reduction content
- **all-mpnet-base-v2**: Better quality, larger (438MB), slower
- **sentence-transformers/all-distilroberta-v1**: Fast alternative

### Chunking Strategy

| Method | Pros | Cons |
|--------|------|------|
| Docling Intelligent | Preserves structure, semantic awareness | Slower, may fail on some PDFs |
| Fallback Window | Fast, reliable | May break context, less semantic awareness |

## Troubleshooting

### Import Errors

```bash
# Reinstall specific dependencies
pip install --force-reinstall chromadb docling sentence-transformers pypdf
```

### PDF Processing Fails

```bash
# Use fallback for problematic PDFs
python ingest_doclib_docling_chroma.py --pdf-dir ./doclib --disable-fallback
# If this fails on a specific PDF, check for corruption
```

### Out of Memory

```bash
# Process in batches
python ingest_doclib_docling_chroma.py --pdf-dir ./doclib --limit 2
```

## Integration with RAGEngine

The precomputed JSON can be loaded into a client application:

```python
import json

with open('./app/data/rag_precomputed_index.json') as f:
    precomputed_index = json.load(f)

# Use embeddings for similarity search
for item in precomputed_index:
    doc_id = item['id']
    embedding = item['embedding']
    text = item['text']
```

## Monitoring & Validation

### Check ChromaDB Collection

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="therfour_docs")
print(f"Total chunks: {collection.count()}")
```

### Validate Embeddings

```bash
python -c "
import json
with open('./app/data/rag_precomputed_index.json') as f:
    data = json.load(f)
    print(f'Indexed items: {len(data)}')
    print(f'Embedding dim: {len(data[0][\"embedding\"])}')
"
```

## Next Steps

1. ✅ Run ingestion: `python ingest_doclib_docling_chroma.py --pdf-dir ./doclib`
2. ✅ Validate output in `./chroma_db/` and `./app/data/`
3. ✅ Test RAG queries against ChromaDB collection
4. ✅ Integrate with voice agent for context retrieval
