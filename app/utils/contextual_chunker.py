"""
contextual_chunker.py — Implements Contextual Chunking.

Every chunk produced here gets a document-level summary prepended
before embedding. This is the technique that improves retrieval
accuracy by up to 80% vs naive chunking.

Algorithm
---------
1. Load the raw document.
2. Generate a 2-3 sentence summary of the entire document using an LLM.
3. Split the document into smaller chunks (RecursiveCharacterTextSplitter).
4. Prepend the summary to each chunk's page_content before storing.
5. Add metadata: source, chunk_index, has_context=True.
"""
from typing import List
from loguru import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from app.config import settings

# ── Splitter config ──────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

from app.utils.llm_factory import get_generator_llm

# ── Summariser LLM ───────────────────────────────────────────
summariser = get_generator_llm()

SUMMARY_PROMPT = (
    "Summarise the following document in 2-3 sentences. "
    "Focus on the main topic, key entities, and purpose. "
    "Be factual and concise.\n\nDOCUMENT:\n{text}\n\nSUMMARY:"
)


def _summarise_document(full_text: str) -> str:
    """Generate a short summary of the full document text."""
    # Truncate to avoid token limits (use first ~6000 chars for summary)
    truncated = full_text[:6000]
    prompt = SUMMARY_PROMPT.format(text=truncated)
    response = summariser.invoke(prompt)
    return response.content.strip()


def contextual_chunk(documents: List[Document]) -> List[Document]:
    """
    Takes a list of raw Documents (e.g. from PyPDFLoader),
    applies contextual chunking, and returns enriched chunks
    ready for embedding and storage.

    Parameters
    ----------
    documents : list of Document objects (one per page or file)

    Returns
    -------
    list of Document objects with context-prepended page_content
    """
    all_chunks: List[Document] = []

    # Group pages by source file so we summarise per document, not per page
    sources: dict[str, List[Document]] = {}
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        sources.setdefault(src, []).append(doc)

    for source, pages in sources.items():
        logger.info(f"Processing document: {source} ({len(pages)} pages)")

        # Combine all pages into one text for summarisation
        full_text = "\n\n".join(p.page_content for p in pages)

        # Generate the document-level summary
        try:
            doc_summary = _summarise_document(full_text)
            logger.debug(f"Summary for {source}: {doc_summary[:100]}...")
        except Exception as e:
            logger.warning(f"Could not summarise {source}: {e}. Skipping context prepend.")
            doc_summary = ""

        # Split into chunks
        raw_chunks = splitter.split_documents(pages)

        # Prepend summary to each chunk
        for idx, chunk in enumerate(raw_chunks):
            if doc_summary:
                chunk.page_content = (
                    f"[DOCUMENT CONTEXT: {doc_summary}]\n\n"
                    f"{chunk.page_content}"
                )
            chunk.metadata.update({
                "source": source,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
                "has_context": bool(doc_summary),
            })
            all_chunks.append(chunk)

        logger.info(f"  → {len(raw_chunks)} contextual chunks created for {source}")

    return all_chunks
