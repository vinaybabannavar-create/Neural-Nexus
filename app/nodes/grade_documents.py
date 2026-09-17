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
            "- Do NOT consider whether you know the answer yourself.\n"
            "- Respond with JSON containing 'score' ('yes' or 'no') and 'reasoning'."
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
    grader_llm = get_grader_llm()

    class BatchRelevanceGrade(BaseModel):
        grades: list[RelevanceGrade] = Field(description="Grades for each document chunk in order.")

    chunks_to_grade = documents[:3] if len(documents) > 3 else documents
    if not chunks_to_grade:
        return {
            **state,
            "documents": [],
            "relevance_score": 0.0,
            "node_execution_times": {"grade_documents": time.time() - start_time}
        }

    if any(doc.metadata.get("source") == "system:neural_nexus_assistant" for doc in chunks_to_grade):
        logger.info("[GRADE] System overview document detected for conversational query → marked relevant (score: 1.0)")
        return {
            **state,
            "documents": chunks_to_grade,
            "relevance_score": 1.0,
            "node_execution_times": {"grade_documents": time.time() - start_time}
        }

    BATCH_GRADE_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert document relevance grader. "
                "Evaluate whether each document chunk contains information useful to answer the user's question.\n"
                "Rules:\n"
                "- Score 'yes' if chunk contains facts/context directly helping answer the question.\n"
                "- Score 'no' if off-topic, tangential, or empty.\n"
                "- Return a JSON list of grades, one per chunk in order."
            ),
        ),
        (
            "human",
            "QUESTION: {question}\n\nCHUNKS TO GRADE:\n{chunks_text}\n\nGrade each chunk in order.",
        ),
    ])

    chunks_text = "\n\n".join(
        f"--- CHUNK {i+1} ---\n{doc.page_content[:1000]}"
        for i, doc in enumerate(chunks_to_grade)
    )

    relevant_docs = []
    scores = []

    try:
        try:
            structured_grader = grader_llm.with_structured_output(BatchRelevanceGrade, method="json_mode")
        except Exception:
            structured_grader = grader_llm.with_structured_output(BatchRelevanceGrade)
        batch_chain = BATCH_GRADE_PROMPT | structured_grader
        result: BatchRelevanceGrade = batch_chain.invoke({
            "question": question,
            "chunks_text": chunks_text
        })
        for i, grade in enumerate(result.grades[:len(chunks_to_grade)]):
            is_relevant = grade.score.strip().lower() == "yes"
            scores.append(1.0 if is_relevant else 0.0)
            logger.debug(f"  Chunk {i+1}: {'RELEVANT' if is_relevant else 'IRRELEVANT'} — {grade.reasoning}")
            if is_relevant:
                relevant_docs.append(chunks_to_grade[i])
    except Exception as e:
        logger.warning(f"Batch grading failed: {e}. Falling back to individual grading.")
        try:
            structured_grader = grader_llm.with_structured_output(RelevanceGrade, method="json_mode")
        except Exception:
            structured_grader = grader_llm.with_structured_output(RelevanceGrade)
        chain = GRADE_PROMPT | structured_grader
        for i, doc in enumerate(chunks_to_grade):
            try:
                res: RelevanceGrade = chain.invoke({
                    "question": question,
                    "document": doc.page_content[:1000],
                })
                is_rel = res.score.strip().lower() == "yes"
                scores.append(1.0 if is_rel else 0.0)
                if is_rel:
                    relevant_docs.append(doc)
            except Exception as ex:
                logger.warning(f"  Chunk {i+1} grading error: {ex}")
                relevant_docs.append(doc)
                scores.append(0.5)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    logger.info(
        f"[GRADE] Kept {len(relevant_docs)}/{len(chunks_to_grade)} chunks. "
        f"Avg relevance score: {avg_score:.2f}"
    )

    return {
        **state,
        "documents": relevant_docs,
        "relevance_score": avg_score,
        "node_execution_times": {"grade_documents": time.time() - start_time}
    }

