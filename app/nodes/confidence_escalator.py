"""
nodes/confidence_escalator.py — Node 6: Escalates low-confidence / hallucinated responses.

Inserted after grade_hallucinations when retry count is exhausted and claims cannot be grounded.
Routes to:
  - human_reviewed_approved: If a manual context override is provided, loops back into generate.
  - pending_human_verification: Flags the answer as requiring human review rather than returning a hallucination.
"""
import time
from typing import Optional
from loguru import logger
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from app.graph.state import GraphState


def confidence_escalator(state: GraphState) -> GraphState:
    """
    Handle low-confidence or hallucinated responses when retries are exhausted.
    """
    start_time = time.perf_counter()
    question = state.get("question", "")
    current_generation = state.get("generation", "")
    documents = state.get("documents", [])
    manual_override: Optional[str] = state.get("manual_context_override")

    logger.warning(
        f"[CONFIDENCE ESCALATOR] Evaluating ungrounded response for: '{question[:60]}...' "
        f"(Retries exhausted: {state.get('retry_count', 0)})"
    )

    # 1. Check if human reviewer provided an approved context override
    if manual_override and manual_override.strip():
        logger.info("[CONFIDENCE ESCALATOR] Manual context override provided → Routing to human_reviewed_approved")
        override_doc = Document(
            page_content=manual_override.strip(),
            metadata={"source": "human_reviewer_override", "verified": True}
        )
        # Prepend the human-reviewed document
        updated_docs = [override_doc] + list(documents)
        
        duration = time.perf_counter() - start_time
        exec_times = dict(state.get("node_execution_times") or {})
        exec_times["confidence_escalator"] = round(duration, 4)

        return {
            **state,
            "documents": updated_docs,
            "escalation_status": "human_reviewed_approved",
            "hallucination_check": "grounded",  # Override check
            "node_execution_times": exec_times,
        }

    # 2. Otherwise, mark as pending human verification and return distinct status
    logger.warning("[CONFIDENCE ESCALATOR] No override provided → Flagging response as pending_human_verification")
    
    status_notice = (
        "⚠️ [STATUS: PENDING_HUMAN_VERIFICATION]\n\n"
        "This response could not be verified with high confidence against source documents and has been flagged for human review. "
        "To prevent misinformation, automated generation has been suspended for this query.\n\n"
        f"**Unverified Draft:**\n{current_generation}"
    )

    duration = time.perf_counter() - start_time
    exec_times = dict(state.get("node_execution_times") or {})
    exec_times["confidence_escalator"] = round(duration, 4)

    return {
        **state,
        "generation": status_notice,
        "messages": [AIMessage(content=status_notice)],
        "escalation_status": "pending_human_verification",
        "node_execution_times": exec_times,
    }
