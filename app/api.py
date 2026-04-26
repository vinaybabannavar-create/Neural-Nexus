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

from app.graph.pipeline import rag_graph
from app.ingest import ingest

app = FastAPI(
    title="Corrective RAG API",
    description="Self-reflective RAG pipeline with autonomous web search fallback",
    version="1.0.0",
)


# ── Request / Response models ─────────────────────────────────

class QueryRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {"question": "What is contextual chunking and why does it improve RAG?"}
        }


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    web_search_used: bool
    relevance_score: float
    retry_count: int


class IngestURLRequest(BaseModel):
    url: str


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Corrective RAG API"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run the full corrective RAG pipeline for a question.
    Returns the grounded answer with metadata.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"[API] /query → {request.question!r}")

    initial_state = {
        "question": request.question,
        "documents": [],
        "generation": None,
        "web_search_used": False,
        "retry_count": 0,
        "relevance_score": 0.0,
        "sources": [],
    }

    try:
        result = rag_graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"[API] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=result["question"],
        answer=result.get("generation", "No answer generated."),
        sources=result.get("sources", []),
        web_search_used=result.get("web_search_used", False),
        relevance_score=round(result.get("relevance_score", 0.0), 3),
        retry_count=result.get("retry_count", 0),
    )


@app.post("/ingest/url")
async def ingest_url(request: IngestURLRequest):
    """Ingest a web URL into the vector store."""
    try:
        ingest(request.url)
        return {"status": "success", "source": request.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a PDF or text file."""
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
    START([Start]) --> retrieve[Retrieve from Vector DB]
    retrieve --> grade[Grade Documents<br/>DeepSeek-R1]
    grade -->|relevant| generate[Generate Answer]
    grade -->|not relevant| web_search[Web Search<br/>Tavily]
    web_search --> generate
    generate --> hallucination[Check Hallucinations<br/>DeepSeek-R1]
    hallucination -->|grounded| END([Final Answer])
    hallucination -->|hallucinated| generate
"""
    return {"mermaid": diagram.strip()}
