# 🧠 Neural Nexus AI Pipeline

> **Advanced Self-Reflective Retrieval-Augmented Generation** — autonomously evaluates context quality and leverages high-precision reasoning to synthesize accurate, grounded answers.

![Architecture](Neural%20Nexus.png)

---

## How It Works

Neural Nexus runs an 8-node agentic pipeline built with LangGraph:

0. **Transform Query** — De-contextualizes follow-up questions using conversation history into standalone queries.
1. **Retrieve** — Fetches top-K relevant chunks from vector storage (ChromaDB / Pinecone / Moss adapter).
2. **Re-rank** — FlashRank local cross-encoder (`ms-marco-TinyBERT-L-2-v2`) re-ranks and filters top chunks.
3. **Grade Documents** — Reasoning model evaluates chunk relevance; if score < `RELEVANCE_THRESHOLD` (0.5), triggers web search.
4. **Web Search** — Tavily fetches live internet results when local context is insufficient.
5. **Generate** — Synthesizes a grounded, cited answer from verified context.
6. **Hallucination Check** — Verifies every factual claim against context; loops back to regenerate if ungrounded.
7. **Confidence Escalator** — Circuit-breaker routing: if hallucinated claims persist after max retries, flags for human review (`pending_human_verification`) or accepts verified overrides (`human_reviewed_approved`).

Each chunk gets a **document-level summary prepended before embedding** (contextual chunking), while incoming documents pass through an **ingest security validator** (`ingest_validator.py`) with SQLite audit trails (`quarantine.db`).

---

## Setup

### 1. Clone and install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

**Required:**
- `OPENAI_API_KEY` — embeddings and generation
- `TAVILY_API_KEY` — web search fallback ([get free key](https://tavily.com))

**Optional but recommended:**
- `DEEPSEEK_API_KEY` — higher-quality reasoning in grader nodes (falls back to `gpt-4o-mini` if not set)

### 3. Ingest documents

```bash
# Ingest the sample document
python main.py ingest --source docs_sample/

# Ingest your own PDF
python main.py ingest --source /path/to/your/file.pdf

# Ingest a web page
python main.py ingest --source https://example.com/article
```

### 4. Ask questions

**CLI:**
```bash
python main.py query "What is contextual chunking?"
python main.py query "How does corrective RAG differ from standard RAG?"
```

**FastAPI server:**
```bash
python main.py serve
# Interactive API docs → http://localhost:8000/docs
```

**Streamlit UI:**
```bash
python main.py ui
# Opens browser → http://localhost:8501
```

---

## Project Structure

```
Neural-Nexus/
├── app/
│   ├── config.py                    # Settings from .env
│   ├── ingest.py                    # Document ingestion pipeline + security screening
│   ├── api.py                       # FastAPI REST server (/query, /v1/trust-score, /security/quarantine)
│   ├── ui.py                        # Glassmorphism Streamlit UI with telemetry breakdown
│   ├── graph/
│   │   ├── state.py                 # LangGraph state TypedDict
│   │   └── pipeline.py              # Graph construction + confidence escalation routing logic
│   ├── nodes/
│   │   ├── transform_query.py       # Node 0: Query de-contextualization
│   │   ├── retrieve.py              # Node 1: Vector DB retrieval
│   │   ├── rerank.py                # Node 1.5: FlashRank cross-encoder re-ranking
│   │   ├── grade_documents.py       # Node 2: DeepSeek-R1 / Qwen relevance grader
│   │   ├── web_search.py            # Node 3: Tavily web search fallback
│   │   ├── generate.py              # Node 4: LLM answer generation
│   │   ├── grade_hallucinations.py  # Node 5: Hallucination & grounding checker
│   │   └── confidence_escalator.py  # Node 6: Circuit-breaker & human review escalation
│   ├── security/
│   │   ├── ingest_validator.py      # Ingestion prompt-injection & jailbreak screening
│   │   └── quarantine_store.py      # SQLite audit trail (quarantine.db) for rejected docs
│   ├── trust/
│   │   ├── trust_score_engine.py    # 0–100 composite trust index & REST cache
│   │   └── latency_tracer.py        # Per-stage latency profiler & breakdown formatter
│   └── utils/
│       ├── vector_store.py          # ChromaDB / Pinecone abstraction
│       ├── contextual_chunker.py    # Contextual chunking implementation
│       ├── moss_adapter.py          # MossContextStore interface (Moss-ready adapter)
│       └── llm_factory.py           # LLM provider factory (Groq, DeepSeek, OpenAI)
├── tests/
│   ├── test_pipeline.py             # Unit tests (17 passed)
│   └── eval_adversarial.py          # 16-case adversarial & hallucination evaluation harness
├── docs_sample/
│   └── sample.txt                   # Sample document to test with
├── main.py                          # CLI entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔒 Security Perimeter & Quarantine Store

- **Ingest Validator (`app/security/ingest_validator.py`)**: Screens raw documents, PDFs, and scraped URLs before embedding. Intercepts prompt-injection attempts, jailbreak triggers, system prompt overrides, delimiter injection, and markdown image data exfiltration links.
- **Quarantine Store (`app/security/quarantine_store.py`)**: Logs all rejected content in `quarantine.db` with timestamps, detected threat reasons, risk scores, and content snippets. Accessible via `GET /security/quarantine`.

---

## 🛡️ Confidence Escalation & Trust Index

- **Confidence Escalator (`app/nodes/confidence_escalator.py`)**: If the hallucination checker flags ungrounded claims and generation retries are exhausted, the query is routed to `confidence_escalator`. It stops hallucinated output and marks the response as `pending_human_verification` or injects a verified manual context override (`human_reviewed_approved`).
- **Composite Trust Score (`app/trust/trust_score_engine.py`)**: Calculates an explainable 0–100 score combining relevance (max 40 pts), factual grounding (max 40 pts), and latency efficiency (max 20 pts), minus penalties. Served via `GET /v1/trust-score/{request_id}` and rendered live in the Streamlit UI.
- **Moss Context Adapter (`app/utils/moss_adapter.py`)**: Standardized `MossContextStore` wrapper with telemetry. Note: *Moss-ready interface, pending SDK access* (currently wraps ChromaDB/Pinecone).

---

## 🧪 Verification & Evaluation

### 1. Unit Test Suite (17 Tests)
```bash
pytest tests/test_pipeline.py -v
```
Output:
```text
============================= 17 passed in 13.68s =============================
```

### 2. Adversarial & Grounding Harness
```bash
python tests/eval_adversarial.py
```
Evaluates 16 labeled test cases across Clean Factual, Out-of-Domain (Tavily search), Adversarial / False Premise, and Hallucination Bait queries:
- **Hallucination Interception Rate**: Autonomous detection and retry/escalation on ungrounded claims.
- **Escalation Triggering**: Routing ungrounded responses to `pending_human_verification`.
- **Per-Stage Latency Profiling**: Measured across transform, retrieve, rerank, grade, web search, generate, and grade hallucinations.

---

## API Keys

| Service  | Purpose                    | Link                        |
|----------|----------------------------|-----------------------------|
| Groq     | Fast LLM generation/grading| console.groq.com            |
| OpenAI   | Embeddings + generation    | platform.openai.com         |
| DeepSeek | Reasoning / grader nodes   | platform.deepseek.com       |
| Tavily   | Web search fallback        | tavily.com                  |
| Pinecone | Cloud vector DB (optional) | pinecone.io                 |
