# 🧠 Neural Nexus AI Pipeline

> **Advanced Self-Reflective Retrieval-Augmented Generation** — autonomously evaluates context quality and leverages high-precision reasoning to synthesize accurate, grounded answers.

![Architecture](Neural%20Nexus.png)

---

## How It Works

Neural Nexus runs a 5-node agentic pipeline built with LangGraph:

1. **Retrieve** — Fetches relevant chunks from the vector store (ChromaDB / Pinecone)
2. **Grade Documents** — DeepSeek-R1 scores each chunk for relevance; if score < `RELEVANCE_THRESHOLD` (default `0.5`), triggers web search fallback
3. **Web Search** — Tavily fetches live results when local context is insufficient
4. **Generate** — LLM synthesizes a grounded answer from verified context
5. **Hallucination Check** — Verifies every claim is supported by context; regenerates if not (up to `MAX_RETRIES`)

Each chunk also gets a **document-level summary prepended before embedding** (contextual chunking), which significantly improves retrieval precision.

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
│   ├── ingest.py                    # Document ingestion pipeline
│   ├── api.py                       # FastAPI REST server
│   ├── ui.py                        # Streamlit chat UI
│   ├── graph/
│   │   ├── state.py                 # LangGraph state TypedDict
│   │   └── pipeline.py              # Graph construction + routing logic
│   ├── nodes/
│   │   ├── retrieve.py              # Node 1: Vector DB retrieval
│   │   ├── grade_documents.py       # Node 2: DeepSeek-R1 relevance grader
│   │   ├── web_search.py            # Node 3: Tavily web search fallback
│   │   ├── generate.py              # Node 4: LLM answer generation
│   │   └── grade_hallucinations.py  # Node 5: Hallucination checker
│   └── utils/
│       ├── vector_store.py          # ChromaDB / Pinecone abstraction
│       ├── contextual_chunker.py    # Contextual chunking implementation
│       └── llm_factory.py           # LLM provider factory
├── tests/
│   └── test_pipeline.py             # Unit tests (no API keys needed)
├── docs_sample/
│   └── sample.txt                   # Sample document to test with
├── main.py                          # CLI entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## API Keys

| Service  | Purpose                    | Link                        |
|----------|----------------------------|-----------------------------|
| OpenAI   | Embeddings + generation    | platform.openai.com         |
| DeepSeek | Reasoning / grader nodes   | platform.deepseek.com       |
| Tavily   | Web search fallback        | tavily.com                  |
| Pinecone | Cloud vector DB (optional) | pinecone.io                 |
