"""
trust_score_engine.py — Computes composite trust and reliability scores for RAG responses.

Calculates a 0-100 composite trust index based on:
1. Relevance Score (document context match)
2. Groundedness / Hallucination Verification
3. Retrieval & Pipeline Latency Efficiency
4. Escalation & Retry Penalties
"""
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from loguru import logger


class TrustScoreBreakdown(BaseModel):
    """Detailed breakdown of composite trust score components."""
    composite_score: float = Field(description="Overall trust score from 0 to 100")
    relevance_points: float = Field(description="Points from document relevance (max 40)")
    groundedness_points: float = Field(description="Points from hallucination check (max 40)")
    latency_points: float = Field(description="Points from retrieval speed/efficiency (max 20)")
    penalties: float = Field(description="Deductions for retries or escalations")
    rating: str = Field(description="Qualitative confidence rating (High / Medium / Low / Escalated)")
    factors: Dict[str, Any] = Field(default_factory=dict, description="Raw input metrics used in calculation")


class TrustScoreRecord(BaseModel):
    """Stored trust score record associated with a request ID."""
    request_id: str
    timestamp: str
    question: str
    composite_score: float
    rating: str
    breakdown: TrustScoreBreakdown
    escalation_status: Optional[str] = None


class TrustScoreEngine:
    """Computes, caches, and provides API access to pipeline trust scores."""

    def __init__(self):
        # In-memory store keyed by request_id
        self._records: Dict[str, TrustScoreRecord] = {}

    def compute_score(self, state: Dict[str, Any]) -> TrustScoreBreakdown:
        """
        Compute composite trust score (0 - 100) from LangGraph state.
        """
        relevance = float(state.get("relevance_score", 0.0))
        hallucination_check = state.get("hallucination_check", "grounded")
        retry_count = int(state.get("retry_count", 0))
        escalation_status = state.get("escalation_status")
        exec_times = state.get("node_execution_times") or {}
        retrieval_latency = float(exec_times.get("retrieve", 0.5))
        total_latency = sum(exec_times.values()) if exec_times else 1.0

        # 1. Relevance Component (Max 40 points)
        # Directly proportional to the document relevance score (0.0 to 1.0)
        relevance_points = round(min(40.0, max(0.0, relevance * 40.0)), 2)

        # 2. Groundedness Component (Max 40 points)
        if escalation_status == "pending_human_verification":
            groundedness_points = 5.0
        elif hallucination_check == "grounded":
            groundedness_points = 40.0
        elif hallucination_check == "hallucinated":
            groundedness_points = 15.0
        else:
            groundedness_points = 25.0

        # 3. Latency Efficiency Component (Max 20 points)
        # Fast retrieval (< 0.4s) gets 20 pts, scaling down to min 5 pts if > 2.5s
        if retrieval_latency <= 0.4:
            latency_points = 20.0
        elif retrieval_latency <= 1.0:
            latency_points = 16.0
        elif retrieval_latency <= 2.0:
            latency_points = 12.0
        elif retrieval_latency <= 4.0:
            latency_points = 8.0
        else:
            latency_points = 5.0

        # 4. Penalties
        # Deduct 5 points per generation retry
        penalties = 0.0
        if retry_count > 1:
            penalties += (retry_count - 1) * 5.0
        
        if escalation_status == "pending_human_verification":
            penalties += 15.0

        # Calculate composite score clamped to [0, 100]
        raw_composite = (relevance_points + groundedness_points + latency_points) - penalties
        composite_score = round(max(0.0, min(100.0, raw_composite)), 1)

        # Qualitative Rating
        if escalation_status == "pending_human_verification":
            rating = "Flagged for Verification"
        elif composite_score >= 80.0:
            rating = "High Trust"
        elif composite_score >= 55.0:
            rating = "Medium Trust"
        else:
            rating = "Low Trust"

        breakdown = TrustScoreBreakdown(
            composite_score=composite_score,
            relevance_points=relevance_points,
            groundedness_points=groundedness_points,
            latency_points=latency_points,
            penalties=round(penalties, 2),
            rating=rating,
            factors={
                "relevance_score": relevance,
                "hallucination_check": hallucination_check,
                "retry_count": retry_count,
                "retrieval_latency_sec": retrieval_latency,
                "total_latency_sec": round(total_latency, 3),
                "escalation_status": escalation_status,
                "web_search_used": state.get("web_search_used", False),
            },
        )

        logger.info(
            f"[TRUST ENGINE] Trust Score: {composite_score}/100 ({rating}) | "
            f"Rel: {relevance_points}/40, Ground: {groundedness_points}/40, Lat: {latency_points}/20, Pen: -{penalties}"
        )

        return breakdown

    def record(self, request_id: str, state: Dict[str, Any]) -> TrustScoreRecord:
        """Compute and store a trust score record for a given request."""
        breakdown = self.compute_score(state)
        record = TrustScoreRecord(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            question=state.get("question", ""),
            composite_score=breakdown.composite_score,
            rating=breakdown.rating,
            breakdown=breakdown,
            escalation_status=state.get("escalation_status"),
        )
        self._records[request_id] = record
        return record

    def get(self, request_id: str) -> Optional[TrustScoreRecord]:
        """Retrieve stored trust score record by request_id."""
        return self._records.get(request_id)


# Global singleton engine
trust_engine = TrustScoreEngine()
