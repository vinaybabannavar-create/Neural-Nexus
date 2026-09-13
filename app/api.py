"""
api.py — FastAPI REST server for the corrective RAG pipeline.

Endpoints
---------
POST /query          — Ask a question, get a grounded answer
POST /ingest         — Ingest a document or URL
GET  /health         — Health check
GET  /graph/diagram  — Returns a Mermaid diagram of the graph
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger
import tempfile
import shutil
from pathlib import Path

from typing import Optional, List, Dict, Any
import uuid
from app.graph.pipeline import rag_graph
from app.ingest import ingest
from app.trust.trust_score_engine import trust_engine, TrustScoreRecord
from app.security.quarantine_store import quarantine_store

app = FastAPI(
    title="Neural Nexus Enterprise Corrective RAG API",
    description="Self-reflective RAG pipeline with autonomous web search, security perimeter, and confidence escalation",
    version="2.0.0",
)


from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ── Request / Response models ─────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    request_id: Optional[str] = None
    manual_context_override: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Can you explain that in more detail?",
                "history": [
                    {"role": "user", "content": "What is contextual chunking?"},
                    {"role": "assistant", "content": "Contextual chunking is a method..."}
                ],
                "manual_context_override": None
            }
        }


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    sources: list[str]
    web_search_used: bool
    relevance_score: float
    retry_count: int
    escalation_status: Optional[str]
    trust_score: float
    trust_rating: str
    history: list[ChatMessage]
    latency_metrics: dict[str, float]


class IngestURLRequest(BaseModel):
    url: str


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "service": "Neural Nexus C-RAG Enterprise API",
        "security_perimeter": "active",
        "quarantine_store": "online"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run the full corrective RAG pipeline for a question.
    Returns the grounded answer with metadata, escalation status, and trust score.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    req_id = request.request_id or f"req_{uuid.uuid4().hex[:12]}"
    logger.info(f"[API] /query (ID: {req_id}) → {request.question!r}")

    # Convert request history to LangChain messages
    langchain_messages = []
    for msg in request.history:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
    
    # Add current question
    langchain_messages.append(HumanMessage(content=request.question))

    initial_state = {
        "question": request.question,
        "messages": langchain_messages,
        "documents": [],
        "generation": None,
        "web_search_used": False,
        "retry_count": 0,
        "relevance_score": 0.0,
        "sources": [],
        "hallucination_check": "grounded",
        "node_execution_times": {},
        "request_id": req_id,
        "escalation_status": None,
        "manual_context_override": request.manual_context_override,
    }

    try:
        result = rag_graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"[API] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Compute and cache Trust Score
    trust_record = trust_engine.record(req_id, result)

    # Convert back to API format
    out_history = []
    for msg in result.get("messages", []):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        out_history.append(ChatMessage(role=role, content=msg.content))

    return QueryResponse(
        request_id=req_id,
        question=result["question"],
        answer=result.get("generation", "No answer generated."),
        sources=result.get("sources", []),
        web_search_used=result.get("web_search_used", False),
        relevance_score=round(result.get("relevance_score", 0.0), 3),
        retry_count=result.get("retry_count", 0),
        escalation_status=result.get("escalation_status"),
        trust_score=trust_record.composite_score,
        trust_rating=trust_record.rating,
        history=out_history,
        latency_metrics=result.get("node_execution_times", {})
    )


@app.get("/v1/trust-score/{request_id}", response_model=TrustScoreRecord)
def get_trust_score(request_id: str):
    """
    Retrieve computed composite trust score and component breakdown for a given request.
    """
    record = trust_engine.get(request_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No trust score found for request_id '{request_id}'")
    return record


@app.get("/security/quarantine")
def get_quarantined_documents(limit: int = 50, offset: int = 0):
    """
    Retrieve audit trail of documents quarantined during ingestion security validation.
    """
    return {
        "stats": quarantine_store.get_quarantine_stats(),
        "records": quarantine_store.get_quarantined_records(limit=limit, offset=offset)
    }


@app.post("/ingest/url")
async def ingest_url(request: IngestURLRequest):
    """Ingest a web URL into the vector store with security screening."""
    try:
        ingest(request.url)
        return {"status": "success", "source": request.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a PDF or text file with security screening."""
    allowed_types = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_types}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        ingest(tmp_path)
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/graph/diagram")
def graph_diagram():
    """Return a Mermaid.js diagram string of the RAG graph topology."""
    diagram = """
graph TD
    START([Start]) --> transform[Transform Query]
    transform --> retrieve[Retrieve from Vector DB / Moss]
    retrieve --> rerank[Re-rank Documents]
    rerank --> grade[Grade Documents<br/>LLM Grader]
    grade -->|relevant| generate[Generate Answer]
    grade -->|not relevant| web_search[Web Search<br/>Tavily]
    web_search --> generate
    generate --> hallucination[Check Hallucinations]
    hallucination -->|grounded| END([Final Answer])
    hallucination -->|hallucinated & retries left| generate
    hallucination -->|hallucinated & retries exhausted| escalator[Confidence Escalator]
    escalator -->|human reviewed approved| generate
    escalator -->|pending human verification| END
"""
    return {"mermaid": diagram.strip()}

