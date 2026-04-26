"""
pipeline.py — Builds and compiles the LangGraph corrective RAG graph.

Graph topology
--------------

  [START]
     │
     ▼
  transform_query    ← Node 0: de-contextualize if follow-up
     │
     ▼
  retrieve           ← Node 1: vector DB retrieval (K=10)
     │
     ▼
  rerank             ← Node 1.5: filter to top 5
     │
     ▼
  grade_documents    ← Node 2: DeepSeek-R1 relevance grader
     │
     ├─ relevant ──────────────────┐
     │                             │
     └─ not relevant               │
          │                        │
          ▼                        │
       web_search ← Node 3         │
          │                        │
          ▼                        ▼
       generate  ←──────────── generate   ← Node 4
          │
          ▼
  grade_hallucinations   ← Node 5
          │
          ├─ grounded ──► [END]
          │
          └─ hallucinated (retry_count < MAX_RETRIES)
                  │
                  └──► generate  (loop back)
"""
from langgraph.graph import StateGraph, START, END
from loguru import logger

from app.graph.state import GraphState
from app.nodes.transform_query import transform_query
from app.nodes.retrieve import retrieve
from app.nodes.rerank import rerank
from app.nodes.grade_documents import grade_documents
from app.nodes.web_search import web_search
from app.nodes.generate import generate
from app.nodes.grade_hallucinations import grade_hallucinations
from app.config import settings


# ── Conditional edge functions ────────────────────────────────

def decide_after_grading(state: GraphState) -> str:
    """
    After grading, decide whether to generate directly (relevant docs)
    or fall back to web search (irrelevant / empty).
    """
    relevance_score = state.get("relevance_score", 0.0)
    documents = state.get("documents", [])

    # If no docs survived grading OR score below threshold → web search
    if not documents or relevance_score < settings.RELEVANCE_THRESHOLD:
        logger.info(
            f"[ROUTER] Score {relevance_score:.2f} < threshold {settings.RELEVANCE_THRESHOLD} "
            "→ triggering web search"
        )
        return "web_search"

    logger.info(f"[ROUTER] Score {relevance_score:.2f} is sufficient → generating answer")
    return "generate"


def decide_after_hallucination_check(state: GraphState) -> str:
    """
    After hallucination check, decide whether to return the answer
    or retry generation.
    """
    check = state.get("hallucination_check", "grounded")
    retry_count = state.get("retry_count", 0)

    if check == "grounded":
        logger.info("[ROUTER] Answer is grounded → returning to user")
        return "end"

    if retry_count >= settings.MAX_RETRIES:
        logger.warning(
            f"[ROUTER] Hallucination detected but max retries ({settings.MAX_RETRIES}) "
            "reached → returning best available answer"
        )
        return "end"

    logger.info(
        f"[ROUTER] Hallucination detected (retry {retry_count}/{settings.MAX_RETRIES}) "
        "→ regenerating"
    )
    return "regenerate"


# ── Build the graph ───────────────────────────────────────────

def build_graph():
    """Construct and compile the corrective RAG LangGraph."""
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("transform_query", transform_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)
    graph.add_node("grade_hallucinations", grade_hallucinations)

    # Entry point
    graph.add_edge(START, "transform_query")

    # transform_query → retrieve
    graph.add_edge("transform_query", "retrieve")

    # retrieve → rerank
    graph.add_edge("retrieve", "rerank")

    # rerank → grade_documents
    graph.add_edge("rerank", "grade_documents")

    # grade_documents → web_search OR generate (conditional)
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )

    # web_search → generate (always, after getting web docs)
    graph.add_edge("web_search", "generate")

    # generate → grade_hallucinations (always)
    graph.add_edge("generate", "grade_hallucinations")

    # grade_hallucinations → END or loop back to generate
    graph.add_conditional_edges(
        "grade_hallucinations",
        decide_after_hallucination_check,
        {
            "end": END,
            "regenerate": "generate",
        },
    )

    compiled = graph.compile()
    logger.info("[GRAPH] Corrective RAG pipeline compiled successfully")
    return compiled



# Singleton — import this in your app
rag_graph = build_graph()
