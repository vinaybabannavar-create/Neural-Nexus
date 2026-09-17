"""
nodes/retrieve.py — Node 1: Retrieve documents from the vector store.

Input  state keys : question
Output state keys : documents, sources
"""
import time
from loguru import logger
from langchain_core.documents import Document
from app.graph.state import GraphState
from app.utils.vector_store import get_retriever

GREETINGS = {
    "hi", "hello", "hey", "hola", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "who are you", "what can you do", "help"
}


def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve the top-K most relevant document chunks from the vector store
    for the given question.
    """
    start_time = time.time()
    question = state["question"]
    logger.info(f"[RETRIEVE] Searching for: {question!r}")

    norm_q = question.strip().lower().strip("!?,.:; ")
    if norm_q in GREETINGS:
        logger.info(f"[RETRIEVE] Conversational greeting detected: {question!r} -> injecting system overview context")
        system_doc = Document(
            page_content=(
                "Neural Nexus is an advanced Self-Reflective Corrective RAG (C-RAG) system with a real-time security perimeter, "
                "multi-layer trust scoring, per-node latency telemetry, and LiveKit voice integration. "
                "Users can upload documents (PDF, TXT, Markdown) or ingest Web URLs via the knowledge base sidebar, and ask questions. "
                "The system provides fully grounded, cited answers with zero hallucinations, or triggers autonomous web search if needed."
            ),
            metadata={"source": "system:neural_nexus_assistant"}
        )
        return {
            **state,
            "documents": [system_doc],
            "sources": ["system:neural_nexus_assistant"],
            "web_search_used": False,
            "retry_count": state.get("retry_count", 0),
            "relevance_score": 1.0,
            "generation": None,
            "node_execution_times": {"retrieve": time.time() - start_time}
        }

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

