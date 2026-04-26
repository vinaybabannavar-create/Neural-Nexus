"""
nodes/grade_hallucinations.py — Node 5: Hallucination checker.

Verifies that the generated answer is factually grounded in the
context documents. If not, signals the graph to loop back to generate.

Input  state keys : generation, documents
Output state keys : (state unchanged — routing decision made in graph edges)
"""
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.utils.llm_factory import get_grader_llm


# ── Structured output ─────────────────────────────────────────
class HallucinationGrade(BaseModel):
    """Grounding check result."""
    grounded: str = Field(
        description="'yes' if the answer is fully grounded in the documents, 'no' if it contains hallucinations."
    )
    reasoning: str = Field(
        description="One sentence explaining the grounding decision."
    )


# ── Prompt ────────────────────────────────────────────────────
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a hallucination detection expert. "
            "Your task is to verify that every factual claim in the ANSWER "
            "is directly supported by the CONTEXT DOCUMENTS.\n\n"
            "Rules:\n"
            "- Score 'yes' only if all key facts in the answer appear in the context.\n"
            "- Score 'no' if the answer introduces ANY fact, number, name, or claim "
            "  not present in the context.\n"
            "- Ignore writing style, formatting, and fluency — only check factual grounding."
        ),
    ),
    (
        "human",
        (
            "CONTEXT DOCUMENTS:\n{documents}\n\n"
            "---\n"
            "ANSWER TO CHECK:\n{generation}\n\n"
            "Is the answer fully grounded in the context documents?"
        ),
    ),
])


def grade_hallucinations(state: GraphState) -> GraphState:
    """
    Check whether the generated answer is grounded in the context.
    Adds 'hallucination_check' to state for routing.
    """
    generation = state.get("generation", "")
    documents = state.get("documents", [])

    logger.info("[HALLUCINATION CHECK] Verifying answer is grounded in context…")

    if not generation:
        logger.warning("[HALLUCINATION CHECK] No generation to check.")
        state["hallucination_check"] = "grounded"
        return state

    context = "\n\n".join(
        f"[Doc {i+1}]: {doc.page_content[:800]}"
        for i, doc in enumerate(documents)
    )

    grader_llm = get_grader_llm()
    structured_grader = grader_llm.with_structured_output(HallucinationGrade)
    chain = HALLUCINATION_PROMPT | structured_grader

    try:
        result: HallucinationGrade = chain.invoke({
            "documents": context,
            "generation": generation,
        })
        is_grounded = result.grounded.strip().lower() == "yes"
        logger.info(
            f"[HALLUCINATION CHECK] {'GROUNDED ✓' if is_grounded else 'HALLUCINATION DETECTED ✗'} "
            f"— {result.reasoning}"
        )
        state["hallucination_check"] = "grounded" if is_grounded else "hallucinated"
    except Exception as e:
        logger.warning(f"[HALLUCINATION CHECK] Check failed: {e}. Assuming grounded.")
        state["hallucination_check"] = "grounded"

    return state
