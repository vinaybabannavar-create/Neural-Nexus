"""
nodes/rerank.py — Node 1.5: Re-rank retrieved documents.

Uses FlashRank (a fast, local cross-encoder) to re-order the top-K 
documents and keep only the most relevant ones.
"""
import time
from loguru import logger
from langchain_core.documents import Document
from app.graph.state import GraphState
from app.config import settings

# Lazy import/init of FlashRank
_ranker = None

def get_ranker():
    global _ranker
    if _ranker is None:
        try:
            from flashrank import Ranker
            logger.info("[RERANK] Initializing FlashRank (ms-marco-TinyBERT-L-2-v2)...")
            _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")
        except ImportError:
            logger.warning("[RERANK] flashrank not installed. Falling back to identity.")
            return None
    return _ranker


def rerank(state: GraphState) -> GraphState:
    """
    Re-rank the retrieved documents and filter to the top-N.
    """
    start_time = time.time()
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {
            **state,
            "node_execution_times": {"rerank": time.time() - start_time}
        }

    ranker = get_ranker()
    if not ranker:
        return {
            **state,
            "node_execution_times": {"rerank": time.time() - start_time}
        }

    logger.info(f"[RERANK] Re-ranking {len(documents)} docs for: {question!r}")

    # Prepare for FlashRank
    passages = []
    for i, doc in enumerate(documents):
        passages.append({
            "id": i,
            "text": doc.page_content,
            "meta": doc.metadata
        })

    from flashrank import RerankRequest
    req = RerankRequest(query=question, passages=passages)
    
    results = ranker.rerank(req)
    
    # Select top 5 (or fewer if less than 5)
    top_n = results[:5]
    
    reranked_docs = []
    for res in top_n:
        reranked_docs.append(Document(
            page_content=res["text"],
            metadata=res["meta"]
        ))

    logger.info(f"[RERANK] Kept top {len(reranked_docs)} most relevant documents")

    return {
        **state,
        "documents": reranked_docs,
        "node_execution_times": {"rerank": time.time() - start_time}
    }
