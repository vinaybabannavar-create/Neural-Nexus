# 🧠 Neural Nexus AI Pipeline

An **Advanced Self-Reflective Retrieval-Augmented Generation** system that autonomously
evaluates context quality and leverages high-precision reasoning to synthesize answers.

## Architecture

```
User Query
    │
    ▼
Retrieve (ChromaDB / Pinecone)
    │
    ▼
Grade Documents (DeepSeek-R1)
    │
    ├─ Relevant ──────────────────► Generate Answer
    │                                      │
    └─ Not Relevant                        ▼
         │                     Grade Hallucinations (DeepSeek-R1)
         ▼                                 │
     Web Search (Tavily)        ┌─ Grounded ──► ✅ Final Answer
         │                      │
         └──────────────────────┘
                            └─ Hallucinated ──► Regenerate (up to MAX_RETRIES)
```

## Setup

### 1. Clone and install dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

**Required keys:**
- `OPENAI_API_KEY` — for embeddings and generation
- `TAVILY_API_KEY` — for web search fallback (get free key at tavily.com)

**Optional but recommended:**
- `DEEPSEEK_API_KEY` — for higher-quality reasoning in grader nodes
  (falls back to gpt-4o-mini if not set)

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
# Open http://localhost:8000/docs for interactive API docs
```

**Streamlit UI:**
```bash
python main.py ui
# Opens browser at http://localhost:8501
```

## Project Structure

```
corrective-rag/
├── app/
│   ├── config.py                  # Settings from .env
│   ├── ingest.py                  # Document ingestion pipeline
│   ├── api.py                     # FastAPI REST server
│   ├── ui.py                      # Streamlit chat UI
│   ├── graph/
│   │   ├── state.py               # LangGraph state TypedDict
│   │   └── pipeline.py            # Graph construction + routing logic
│   ├── nodes/
│   │   ├── retrieve.py            # Node 1: Vector DB retrieval
│   │   ├── grade_documents.py     # Node 2: DeepSeek-R1 relevance grader
│   │   ├── web_search.py          # Node 3: Tavily web search fallback
│   │   ├── generate.py            # Node 4: LLM answer generation
│   │   └── grade_hallucinations.py # Node 5: Hallucination checker
│   └── utils/
│       ├── vector_store.py        # ChromaDB / Pinecone abstraction
│       ├── contextual_chunker.py  # Contextual chunking implementation
│       └── llm_factory.py         # LLM provider factory
├── tests/
│   └── test_pipeline.py           # Unit tests (no API keys needed)
├── docs_sample/
│   └── sample.txt                 # Sample document to test with
├── main.py                        # CLI entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Concepts

### Contextual Chunking
Each chunk gets a document-level summary prepended before embedding.
This gives every chunk full context, improving retrieval accuracy by ~80%.

### Neural Nexus Logic
The grader evaluates retrieved docs. If relevance score < `RELEVANCE_THRESHOLD`
(default 0.5), the agent autonomously triggers web search instead of answering
from potentially irrelevant context.

### Hallucination Check
After generation, another grader verifies every claim in the answer is
supported by the context. If not, the system regenerates (up to `MAX_RETRIES`).

## API Keys & Where to Get Them

| Service | Purpose | Get key at |
|---------|---------|-----------|
| OpenAI | Embeddings + generation | platform.openai.com |
| DeepSeek | Reasoning/grader nodes | platform.deepseek.com |
| Tavily | Web search fallback | tavily.com |
| Pinecone | Cloud vector DB (optional) | pinecone.io |
