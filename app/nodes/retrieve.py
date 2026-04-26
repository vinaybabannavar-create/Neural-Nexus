"""
nodes/retrieve.py — Node 1: Retrieve documents from the vector store.

Input  state keys : question
Output state keys : documents, sources
"""
import time
from loguru import logger
from app.graph.state import GraphState
from app.utils.vector_store import get_retriever


def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve the top-K most relevant document chunks from the vector store
    for the given question.
    """
    start_time = time.time()
    question = state["question"]
    logger.info(f"[RETRIEVE] Searching for: {question!r}")

    retriever = get_retriever()
    documents = retriever.invoke(question)

    sources = list({
        doc.metadata.get("source", "unknown") for doc in documents
    })

    logger.info(f"[RETRIEVE] Found {len(documents)} chunks from {len(sources)} source(s)")

    return {
        **state,
        "documents": documents,
        "sources": sources,
        "web_search_used": False,
        "retry_count": state.get("retry_count", 0),
        "relevance_score": 0.0,
        "generation": None,
        "node_execution_times": {"retrieve": time.time() - start_time}
    }

