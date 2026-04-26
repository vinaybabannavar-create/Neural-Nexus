"""
nodes/grade_documents.py — Node 2: Grade retrieved documents for relevance.

Uses DeepSeek-R1 (reasoning model) to evaluate each document chunk and
decide whether it is relevant to the question.

Input  state keys : question, documents
Output state keys : documents (filtered), relevance_score
"""
import time
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.utils.llm_factory import get_grader_llm
from app.config import settings


# ── Structured output schema ─────────────────────────────────
class RelevanceGrade(BaseModel):
    """Binary relevance score for a document chunk."""
    score: str = Field(
        description="'yes' if the document is relevant to the question, 'no' otherwise."
    )
    reasoning: str = Field(
        description="One sentence explaining the relevance decision."
    )


# ── Prompt ───────────────────────────────────────────────────
GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert document relevance grader. "
            "Your job is to decide if a retrieved document chunk contains "
            "information that is useful for answering the user's question.\n\n"
            "Rules:\n"
            "- Score 'yes' if the chunk contains facts, context, or reasoning "
            "  that directly helps answer the question.\n"
            "- Score 'no' if the chunk is off-topic, tangential, or empty.\n"
            "- Be strict: a vaguely related chunk should score 'no'.\n"
            "- Do NOT consider whether you know the answer yourself."
        ),
    ),
    (
        "human",
        "QUESTION: {question}\n\nDOCUMENT CHUNK:\n{document}\n\nGrade this chunk.",
    ),
])


def grade_documents(state: GraphState) -> GraphState:
    """
    Grade each retrieved document for relevance.
    Keeps only relevant documents.
    Calculates an average relevance_score (1.0 = all relevant, 0.0 = none).
    """
    start_time = time.time()
    question = state["question"]
    documents = state["documents"]

    logger.info(f"[GRADE] Grading {len(documents)} chunks for relevance…")

    grader_llm = get_grader_llm()
    structured_grader = grader_llm.with_structured_output(RelevanceGrade)
    chain = GRADE_PROMPT | structured_grader

    relevant_docs = []
    scores = []

    for i, doc in enumerate(documents):
        try:
            result: RelevanceGrade = chain.invoke({
                "question": question,
                "document": doc.page_content[:2000],  # cap to avoid token blow-up
            })
            is_relevant = result.score.strip().lower() == "yes"
            scores.append(1.0 if is_relevant else 0.0)
            logger.debug(
                f"  Chunk {i+1}: {'RELEVANT' if is_relevant else 'IRRELEVANT'} "
                f"— {result.reasoning}"
            )
            if is_relevant:
                relevant_docs.append(doc)
        except Exception as e:
            logger.warning(f"  Grading chunk {i+1} failed: {e}. Keeping it.")
            relevant_docs.append(doc)
            scores.append(0.5)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    logger.info(
        f"[GRADE] Kept {len(relevant_docs)}/{len(documents)} chunks. "
        f"Avg relevance score: {avg_score:.2f}"
    )

    return {
        **state,
        "documents": relevant_docs,
        "relevance_score": avg_score,
        "node_execution_times": {"grade_documents": time.time() - start_time}
    }

