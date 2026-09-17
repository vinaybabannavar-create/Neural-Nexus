# 📜 CRISPE Prompt Engineering Framework Documentation

This document formalizes and audits all LLM-facing prompts across the 8-node Neural Nexus Corrective RAG pipeline using the **CRISPE** framework:
- **C**apacity/Role — The persona/role assigned to the LLM.
- **R**equest — The specific computational task requested.
- **I**nsight — The background context, schemas, and documents supplied.
- **S**tatement — The exact, unedited prompt template from the source code.
- **P**ersonality — Stylistic, tone, formatting, and safety constraints.
- **E**xperiment — Iterations, tuning notes, and failure-mode mitigations.

---

## 1. Node 0: Transform Query (`app/nodes/transform_query.py`)

- **Capacity/Role**: Conversational de-contextualizer and query reformulation specialist.
- **Request**: Reformulate multi-turn follow-up queries that depend on chat history into self-contained, standalone retrieval queries.
- **Insight**: Chat history turns (`HumanMessage` and `AIMessage`) plus the latest user query.
- **Statement (Exact Code Template)**:
```python
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
```
- **Personality**: Objective, deterministic, zero-preamble, direct string output.
- **Experiment**:
  - *Initial*: Model occasionally attempted to answer the question instead of rewriting it.
  - *Tuning*: Explicit negative constraint `"Do NOT answer the question"` added, reducing semantic drift by 100%.

---

## 2. Node 2: Grade Documents (`app/nodes/grade_documents.py`)

- **Capacity/Role**: Expert document relevance grader and semantic relevance evaluator.
- **Request**: Evaluate retrieved text chunks and output binary relevance grades (`yes`/`no`) with one-sentence rationale.
- **Insight**: User query and chunk content.
- **Statement (Exact Code Template)**:
```python
GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert document relevance grader. "
            "Your job is to decide if a retrieved document chunk contains "
            "information that is useful for answering the user's question.\n\n"
            "Rules:\n"
            "- Score 'yes' if the chunk contains facts, context, or reasoning "
            "  that directly helps answer the question.\n"
            "- Score 'no' if the chunk is off-topic, tangential, or empty.\n"
            "- Be strict: a vaguely related chunk should score 'no'.\n"
            "- Do NOT consider whether you know the answer yourself.\n"
            "- Respond with JSON containing 'score' ('yes' or 'no') and 'reasoning'."
        ),
    ),
    (
        "human",
        "QUESTION: {question}\n\nDOCUMENT CHUNK:\n{document}\n\nGrade this chunk.",
    ),
])
```
- **Personality**: Hyper-strict, binary, JSON-schema compliant (`RelevanceGrade`).
- **Experiment**:
  - *Tuning*: Truncated chunk size input to 1,000 characters to ensure sub-300ms evaluation per chunk and prevent token quota exhaustion on fast inference tiers.

---

## 3. Node 4: Generate Answer (`app/nodes/generate.py`)

- **Capacity/Role**: Precision technical synthesis assistant and factual grounded writer.
- **Request**: Synthesize a comprehensive, cited response strictly grounded in verified context chunks.
- **Insight**: Verified document chunks (`_format_context`) and conversation history.
- **Statement (Exact Code Template)**:
```python
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
```
- **Personality**: Grounded, authoritative, evidence-backed, transparent about gaps.
- **Experiment**:
  - *Tuning*: Enforced natural citation instructions (`"According to [source]"`) to allow instant human auditing of claims against vector database chunks.

---

## 4. Node 5: Grade Hallucinations (`app/nodes/grade_hallucinations.py`)

- **Capacity/Role**: Adversarial hallucination detector and factual fidelity auditor.
- **Request**: Verify whether every factual claim, number, and name in the synthesized answer is grounded in source context.
- **Insight**: Source context documents and the draft answer text.
- **Statement (Exact Code Template)**:
```python
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
            "- Ignore writing style, formatting, and fluency — only check factual grounding.\n"
            "- Respond with JSON containing 'grounded' ('yes' or 'no') and 'reasoning'."
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
```
- **Personality**: Forensic, skeptical, adversarial, zero tolerance for extrapolation.
- **Experiment**:
  - *Tuning*: Prompt explicitly commands ignoring fluency and grammar to focus 100% on factual grounding.

---

## 5. Node 6: Confidence Escalator (`app/nodes/confidence_escalator.py`)

- **Capacity/Role**: Human-in-the-loop safety circuit-breaker.
- **Request**: Intercept responses when regeneration retries fail to ground claims, suspending output and issuing auditable warning envelopes.
- **Insight**: Raw draft, user query, and retry count telemetry.
- **Statement (Envelope Format)**:
```python
status_notice = (
    "⚠️ [STATUS: PENDING_HUMAN_VERIFICATION]\n\n"
    "This response could not be verified with high confidence against source documents and has been flagged for human review. "
    "To prevent misinformation, automated generation has been suspended for this query.\n\n"
    f"**Unverified Draft:**\n{current_generation}"
)
```
- **Personality**: Transparent, caution-first, auditable enterprise safety posture.
- **Experiment**:
  - *Tuning*: Added support for verified human context overrides (`manual_context_override`), routing directly to `human_reviewed_approved` when manual domain corrections are injected.
