"""Batch ingest Doclib PDFs using Docling intelligent chunking and ChromaDB embeddings.

This ingestion approach:
- batch processes an entire PDF directory
- uses Docling for intelligent document understanding and chunking
- embeds chunks with SentenceTransformer
- stores in ChromaDB for RAG retrieval
- exports iOS-ready precomputed index JSON

Usage example:
  python ingest_doclib_docling_chroma.py \\
    --pdf-dir ./doclib \\
    --persist-dir ./chroma_db \\
    --collection therfour_docs \\
    --output-json ./app/data/rag_precomputed_index.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import chromadb
from chromadb.config import Settings
from docling.document_converter import DocumentConverter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def _load_chunker():
    """Load the best available Docling chunker implementation."""
    try:
        from docling.chunking import HybridChunker

        return HybridChunker()
    except Exception:
        pass

    try:
        from docling.chunking import HierarchicalChunker

        return HierarchicalChunker()
    except Exception as exc:
        raise RuntimeError(
            "No supported Docling chunker found. Ensure docling is installed from pip wheel "
            "with chunking modules available."
        ) from exc


def _chunk_document(pdf_path: str) -> List[str]:
    """Convert PDF to document and apply intelligent chunking."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    doc = getattr(result, "document", None) or getattr(result, "doc", None)
    if doc is None:
        raise RuntimeError(f"Docling conversion did not return a document object: {pdf_path}")

    chunker = _load_chunker()
    if hasattr(chunker, "chunk"):
        raw_chunks: Iterable = chunker.chunk(doc)
    elif hasattr(chunker, "split"):
        raw_chunks = chunker.split(doc)
    else:
        raise RuntimeError("Docling chunker has neither chunk() nor split()")

    chunks: List[str] = []
    for c in raw_chunks:
        text = getattr(c, "text", None)
        if text is None:
            text = str(c)
        text = text.strip()
        if text:
            chunks.append(text)

    if not chunks:
        raise RuntimeError(f"Docling returned zero chunks for this PDF: {pdf_path}")

    return chunks


def _extract_text_with_pypdf(pdf_path: str) -> str:
    """Fallback text extraction using pypdf."""
    reader = PdfReader(pdf_path)
    page_texts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            page_texts.append(text)

    full_text = "\n\n".join(page_texts).strip()
    if not full_text:
        raise RuntimeError(f"Fallback extractor returned empty text: {pdf_path}")
    return full_text


def _simple_text_chunks(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Simple sliding-window chunking for fallback extraction."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be < max_chars")

    chunks: List[str] = []
    start = 0
    n = len(normalized)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            split = normalized.rfind(" ", start, end)
            if split > start + (max_chars // 2):
                end = split

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(0, end - overlap_chars)

    return chunks


def _chunk_document_with_fallback(
    pdf_path: str,
    enable_fallback: bool,
    fallback_max_chars: int,
    fallback_overlap_chars: int,
) -> Tuple[List[str], str, str]:
    """Try Docling chunking, fallback to pypdf + windowing if needed."""
    try:
        return _chunk_document(pdf_path), "docling-intelligent", ""
    except Exception as docling_exc:
        if not enable_fallback:
            raise

        text = _extract_text_with_pypdf(pdf_path)
        fallback_chunks = _simple_text_chunks(
            text=text,
            max_chars=fallback_max_chars,
            overlap_chars=fallback_overlap_chars,
        )
        if not fallback_chunks:
            raise RuntimeError(
                f"Fallback chunker returned zero chunks for {pdf_path}; "
                f"docling_error={docling_exc}"
            ) from docling_exc
        return fallback_chunks, "fallback-pypdf-window", str(docling_exc)


def _build_ids(source_pdf: str, chunks: List[str]) -> List[str]:
    """Build unique IDs for each chunk based on content hash."""
    ids: List[str] = []
    base = os.path.basename(source_pdf)
    for idx, chunk in enumerate(chunks):
        digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]
        ids.append(f"{base}-{idx:05d}-{digest}")
    return ids


def _discover_pdfs(pdf_dir: Path) -> List[Path]:
    """Discover all PDF files in directory."""
    return sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])


def main() -> None:
    """Main ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Batch ingest PDFs using Docling and ChromaDB"
    )
    parser.add_argument("--pdf-dir", required=True, help="Directory containing source PDFs")
    parser.add_argument(
        "--persist-dir", default="./chroma_db", help="Chroma persistence directory"
    )
    parser.add_argument(
        "--collection", default="therfour_docs", help="Chroma collection name"
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--output-json",
        default="./app/data/rag_precomputed_index.json",
        help="Output JSON path for precomputed RAG index",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional max number of PDFs to process"
    )
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Disable fallback extraction/chunking if Docling fails",
    )
    parser.add_argument(
        "--fallback-max-chars",
        type=int,
        default=1200,
        help="Max chars per fallback chunk",
    )
    parser.add_argument(
        "--fallback-overlap-chars",
        type=int,
        default=200,
        help="Overlap chars between fallback chunks",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    pdf_dir = (
        (script_dir / args.pdf_dir).resolve()
        if not os.path.isabs(args.pdf_dir)
        else Path(args.pdf_dir)
    )
    persist_dir = (
        (script_dir / args.persist_dir).resolve()
        if not os.path.isabs(args.persist_dir)
        else Path(args.persist_dir)
    )
    output_json = (
        (script_dir / args.output_json).resolve()
        if not os.path.isabs(args.output_json)
        else Path(args.output_json)
    )

    # Validate input
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = _discover_pdfs(pdf_dir)
    if args.limit > 0:
        pdf_files = pdf_files[: args.limit]

    if not pdf_files:
        raise RuntimeError(f"No PDFs found in: {pdf_dir}")

    print(f"Found {len(pdf_files)} PDFs to process")
    print(f"Using embedding model: {args.embedding_model}")
    print()

    # Initialize embedder and ChromaDB
    print("Loading embedding model...")
    embedder = SentenceTransformer(args.embedding_model)

    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=args.collection)

    # Process PDFs
    all_precomputed = []
    total_chunks = 0
    failed_pdfs = []
    fallback_used = 0

    print("\nProcessing PDFs...")
    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
        try:
            chunks, chunker_name, fallback_reason = _chunk_document_with_fallback(
                pdf_path=str(pdf_path),
                enable_fallback=not args.disable_fallback,
                fallback_max_chars=args.fallback_max_chars,
                fallback_overlap_chars=args.fallback_overlap_chars,
            )
            ids = _build_ids(str(pdf_path), chunks)
            embeddings = embedder.encode(chunks, normalize_embeddings=True).tolist()

            metadatas = [
                {
                    "source": pdf_path.name,
                    "chunk_index": chunk_index,
                    "embedding_model": args.embedding_model,
                    "chunker": chunker_name,
                }
                for chunk_index, _ in enumerate(chunks)
            ]

            # Upsert to ChromaDB
            collection.upsert(
                ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
            )

            # Build precomputed index
            for doc_id, chunk_text, emb in zip(ids, chunks, embeddings):
                all_precomputed.append(
                    {
                        "id": doc_id,
                        "source": pdf_path.name,
                        "text": chunk_text,
                        "embedding": emb,
                    }
                )

            total_chunks += len(chunks)
            print(f"    ✓ Added {len(chunks)} chunks")
            if chunker_name != "docling-intelligent":
                fallback_used += 1
                print(f"    ⚠ Fallback used: {chunker_name}")
                if fallback_reason:
                    print(f"    Error: {fallback_reason[:100]}...")
        except Exception as exc:
            failed_pdfs.append({"pdf": pdf_path.name, "error": str(exc)})
            print(f"    ✗ Failed: {exc}")

    # Save precomputed index
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(all_precomputed, f, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Processed PDFs:        {len(pdf_files) - len(failed_pdfs)} / {len(pdf_files)}")
    print(f"Total chunks:          {total_chunks}")
    print(f"Fallback conversions:  {fallback_used}")
    print(f"Collection:            {args.collection}")
    print(f"Chroma persist dir:    {persist_dir}")
    print(f"Precomputed JSON:      {output_json}")
    if failed_pdfs:
        print("\nFailed PDFs:")
        for item in failed_pdfs:
            print(f"  • {item['pdf']}: {item['error']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
