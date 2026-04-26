"""
nodes/transform_query.py — Node 0: De-contextualize the user's question.

If this is a follow-up question, the LLM rewrites it to be a standalone 
query that contains all necessary context for retrieval.
"""
import time
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from app.graph.state import GraphState
from app.utils.llm_factory import get_generator_llm


def transform_query(state: GraphState) -> GraphState:
    """
    Rewrites the question based on chat history to ensure it's a standalone query.
    """
    start_time = time.time()
    question = state["question"]
    messages = state.get("messages", [])

    # If this is the first message, no need to transform
    if len(messages) <= 1:
        logger.info("[TRANSFORM] First message, skipping query transformation")
        return {
            **state,
            "node_execution_times": {"transform_query": time.time() - start_time}
        }

    logger.info(f"[TRANSFORM] De-contextualizing question: {question!r}")
    
    llm = get_generator_llm()
    
    system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is. "
        "Return ONLY the rewritten question."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm
    
    # Exclude the current human message from the history passed to the transformer
    history = messages[:-1]
    
    try:
        response = chain.invoke({"chat_history": history, "input": question})
        rewritten_question = response.content.strip()
        logger.info(f"[TRANSFORM] Rewritten question: {rewritten_question!r}")
    except Exception as e:
        logger.error(f"[TRANSFORM] Error rewriting question: {e}")
        rewritten_question = question

    return {
        **state,
        "question": rewritten_question,
        "node_execution_times": {"transform_query": time.time() - start_time}
    }
