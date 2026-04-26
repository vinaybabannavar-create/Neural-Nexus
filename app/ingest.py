"""
ingest.py — Document ingestion script.

Usage
-----
    python -m app.ingest --source docs_sample/
    python -m app.ingest --source path/to/file.pdf
    python -m app.ingest --source https://example.com/article

Supports: PDF, TXT, Markdown, web URLs
"""
import sys
import argparse
from pathlib import Path
from loguru import logger

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document
from app.utils.contextual_chunker import contextual_chunk
from app.utils.vector_store import get_vectorstore_for_ingestion


def load_source(source: str) -> list[Document]:
    """Load documents from a file path, directory, or URL."""
    # URL
    if source.startswith("http://") or source.startswith("https://"):
        logger.info(f"Loading URL: {source}")
        loader = WebBaseLoader(source)
        return loader.load()

    path = Path(source)

    # Directory
    if path.is_dir():
        logger.info(f"Loading directory: {source}")
        docs = []
        for pdf_file in path.glob("**/*.pdf"):
            docs.extend(PyPDFLoader(str(pdf_file)).load())
        for txt_file in path.glob("**/*.txt"):
            docs.extend(TextLoader(str(txt_file)).load())
        for md_file in path.glob("**/*.md"):
            docs.extend(TextLoader(str(md_file)).load())
        return docs

    # Single file
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        logger.info(f"Loading PDF: {source}")
        return PyPDFLoader(source).load()
    if suffix in (".txt", ".md"):
        logger.info(f"Loading text file: {source}")
        return TextLoader(source).load()

    raise ValueError(f"Unsupported source type: {source}")


def ingest(source: str):
    """Full ingestion pipeline: load → contextual chunk → embed → store."""
    logger.info(f"Starting ingestion for: {source}")

    # 1. Load raw documents
    raw_docs = load_source(source)
    if not raw_docs:
        logger.error("No documents loaded. Check the source path.")
        return

    logger.info(f"Loaded {len(raw_docs)} raw document pages/sections")

    # 2. Apply contextual chunking
    chunks = contextual_chunk(raw_docs)
    logger.info(f"Created {len(chunks)} contextual chunks")

    # 3. Embed and store
    vectorstore = get_vectorstore_for_ingestion()
    vectorstore.add_documents(chunks)
    logger.info(f"Stored {len(chunks)} chunks in vector store ✓")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG pipeline")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a file, directory, or URL to ingest",
    )
    args = parser.parse_args()
    ingest(args.source)


if __name__ == "__main__":
    main()
