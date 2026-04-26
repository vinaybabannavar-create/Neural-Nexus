"""
state.py — Defines the shared LangGraph state TypedDict
           that flows through every node in the pipeline.
"""
from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the mutable state object passed between all graph nodes.

    Fields
    ------
    question        : The original user query.
    documents       : List of retrieved (or web-searched) Document objects.
    generation      : The LLM-generated answer string.
    web_search_used : Whether the web search fallback was triggered.
    retry_count     : Number of generation retries attempted.
    relevance_score : Average relevance score from the grader (0.0-1.0).
    sources         : List of source labels shown to the user.
    """
    question: str
    documents: List[Document]
    generation: Optional[str]
    web_search_used: bool
    retry_count: int
    relevance_score: float
    sources: List[str]
    hallucination_check: str
