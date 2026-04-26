"""
nodes/web_search.py — Node 3 (conditional): Autonomous web search fallback.

Triggered when the grader decides retrieved documents are not relevant enough.
Uses Tavily Search API — purpose-built for AI agents.

Input  state keys : question
Output state keys : documents (replaced with web results), web_search_used, sources
"""
import time
from loguru import logger
from langchain_core.documents import Document

from app.graph.state import GraphState
from app.config import settings


def web_search(state: GraphState) -> GraphState:
    """
    Perform a web search using Tavily and convert results to Documents.
    Replaces (or supplements) the vector store results.
    """
    start_time = time.time()
    question = state["question"]
    logger.info(f"[WEB SEARCH] Context insufficient. Searching web for: {question!r}")

    if not settings.TAVILY_API_KEY:
        logger.error("[WEB SEARCH] TAVILY_API_KEY is not set. Skipping web search.")
        return {
            **state, 
            "web_search_used": True,
            "node_execution_times": {"web_search": time.time() - start_time}
        }

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        response = client.search(
            query=question,
            max_results=4,
            search_depth="advanced",   # deep search for better quality
            include_answer=True,        # Tavily's own extracted answer (bonus)
        )

        web_docs = []
        sources = []

        # Convert Tavily results into LangChain Document objects
        for result in response.get("results", []):
            content = result.get("content", "")
            url = result.get("url", "")
            title = result.get("title", "")

            if not content:
                continue

            doc = Document(
                page_content=f"[WEB SOURCE: {title}]\n\n{content}",
                metadata={
                    "source": url,
                    "title": title,
                    "type": "web_search",
                },
            )
            web_docs.append(doc)
            sources.append(url)

        logger.info(f"[WEB SEARCH] Retrieved {len(web_docs)} web results")

        return {
            **state,
            "documents": web_docs,
            "sources": sources,
            "web_search_used": True,
            "node_execution_times": {"web_search": time.time() - start_time}
        }

    except Exception as e:
        logger.error(f"[WEB SEARCH] Failed: {e}")
        return {
            **state, 
            "web_search_used": True,
            "node_execution_times": {"web_search": time.time() - start_time}
        }

