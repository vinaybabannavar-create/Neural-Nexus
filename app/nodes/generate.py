"""
nodes/generate.py — Node 4: Generate the final answer.

Takes the verified (graded) documents as context and uses the generator
LLM to produce a grounded, cited answer.

Input  state keys : question, documents, retry_count
Output state keys : generation, retry_count
"""
import time
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

from app.graph.state import GraphState
from app.utils.llm_factory import get_generator_llm


# ── Prompt ───────────────────────────────────────────────────
GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful, precise AI assistant. "
            "Answer the user's question using ONLY the provided context documents. "
            "Rules:\n"
            "- If the context fully answers the question, give a clear, well-structured answer.\n"
            "- If the context partially answers the question, answer what you can and note the gaps.\n"
            "- Do NOT invent facts, statistics, or quotes not present in the context.\n"
            "- Cite your sources naturally (e.g. 'According to [source]...').\n"
            "- Be concise but complete."
        ),
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    (
        "human",
        (
            "CONTEXT DOCUMENTS:\n"
            "{context}\n\n"
            "---\n"
            "QUESTION: {question}\n\n"
            "Answer based strictly on the context above:"
        ),
    ),
])


def _format_context(documents) -> str:
    """Combine document chunks into a single context string."""
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Doc {i} | Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate(state: GraphState) -> GraphState:
    """
    Generate an answer from the verified context documents.
    Increments retry_count on each call (used by the hallucination check loop).
    """
    start_time = time.time()
    question = state["question"]
    documents = state["documents"]
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)

    logger.info(
        f"[GENERATE] Generating answer from {len(documents)} docs "
        f"(attempt {retry_count + 1})"
    )

    if not documents:
        logger.warning("[GENERATE] No documents available — returning fallback message.")
        msg = "I could not find relevant information to answer your question. Please try rephrasing or provide more context documents."
        return {
            **state,
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "retry_count": retry_count + 1,
            "node_execution_times": {"generate": time.time() - start_time}
        }

    context = _format_context(documents)
    llm = get_generator_llm()
    chain = GENERATE_PROMPT | llm | StrOutputParser()

    # Limit history to the last 6 messages (3 turns) to save tokens and avoid rate limits
    history = messages[-7:-1] if len(messages) > 1 else []


    try:
        generation = chain.invoke({
            "question": question, 
            "context": context,
            "chat_history": history
        })
        logger.info(f"[GENERATE] Answer generated ({len(generation)} chars)")
    except Exception as e:
        logger.error(f"[GENERATE] LLM call failed: {e}")
        generation = f"Generation failed due to an error: {e}"

    return {
        **state,
        "generation": generation,
        "messages": [AIMessage(content=generation)],
        "retry_count": retry_count + 1,
        "node_execution_times": {"generate": time.time() - start_time}
    }

