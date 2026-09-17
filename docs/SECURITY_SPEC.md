# 🛡️ Neural Nexus Security & Integrity Specification

This document defines the threat model, perimeter defenses, audit stores, and current implementation boundaries for the Neural Nexus enterprise RAG system.

---

## 1. Threat Model

Neural Nexus addresses four primary threat vectors in agentic retrieval pipelines:

1. **Ingestion-Time Prompt Injection**: Malicious instructions embedded in uploaded PDFs, TXT files, or scraped web URLs designed to override downstream LLM behavior upon retrieval (Indirect Prompt Injection).
2. **Jailbreak & Delimiter Injection**: Synthetic tokens (e.g. `<|im_start|>`, `[SYSTEM_PROMPT]`, `=== ADMIN OVERRIDE ===`) injected into source text to hijack generation formatting.
3. **Markdown Exfiltration Payloads**: Embedded image links with URL query parameters intended to leak conversation context or tokens to external adversary servers.
4. **Hallucination & Misinformation Propagation**: Generation models synthesizing plausible but fabricated claims when context is sparse or poisoned.

---

## 2. Defensive Architecture & Mitigations

```
┌─────────────────┐       ┌────────────────────────┐       ┌─────────────────┐
│ Raw Document    │ ────► │ Ingest Validator       │ ────► │ Vector DB       │
│ (PDF / TXT / URL│       │ Regex/Heuristic Screen │       │ Contextual Chunk│
└─────────────────┘       └───────────┬────────────┘       └─────────────────┘
                                      │ (If Malicious)
                                      ▼
                          ┌────────────────────────┐
                          │ Quarantine Store       │
                          │ SQLite (quarantine.db) │
                          └────────────────────────┘
```

### 2.1 Ingest Validator (`app/security/ingest_validator.py`)
- **Inspection Rules**:
  - Direct instruction overrides (`ignore/override previous instructions`).
  - System prompt replacement attempts.
  - Persona hijacking (`act as DAN / unrestricted`).
  - Delimiter and chat template token injection (`<|im_start|>`, `<|system|>`).
  - Markdown image exfiltration URLs.
- **Enforcement**: Documents with `risk_score >= 0.7` are immediately blocked from embedding and quarantined.

### 2.2 Quarantine Store (`app/security/quarantine_store.py`)
- **Audit Persistence**: Stored in SQLite database (`quarantine.db`) in table `quarantined_documents`.
- **Fields Logged**: `id`, `timestamp` (UTC ISO), `source`, `reason`, `risk_score`, `snippet`, `metadata_json`.
- **API Endpoint**: Exposed via `GET /security/quarantine` for real-time security dashboard visibility.

### 2.3 Confidence Escalator Circuit-Breaker (`app/nodes/confidence_escalator.py`)
- Mid-flight halt mechanism triggered when hallucination checking detects ungrounded claims after max retries (`MAX_RETRIES=1` or `2`).
- Returns `pending_human_verification` envelope, preventing hallucinated output from reaching the user.

---

## 3. Implementation Status: Coverage & Gaps

To maintain strict compliance and transparency, the table below outlines fully implemented defenses vs. planned roadmap items:

| Security Layer | Implementation Status | Technical Details |
|---|---|---|
| **Ingestion Screening** | **Fully Implemented** | `app/security/ingest_validator.py` regex/heuristic rules with risk scoring. |
| **Quarantine Audit Trail** | **Fully Implemented** | SQLite `quarantine.db` persistence & `GET /security/quarantine` REST endpoint. |
| **Confidence Escalation** | **Fully Implemented** | LangGraph Node 6 halts ungrounded claims after retry exhaustion. |
| **Composite Trust Scoring** | **Fully Implemented** | `app/trust/trust_score_engine.py` (0–100 index based on relevance, grounding, and latency). |
| **LiveKit Audio WebRTC Security** | **Fully Implemented** | JWT token authorization via `POST /voice/livekit/token`. |
| **Moss Guardrails Mid-Flight Halt** | **Interface Ready (Compatibility Mode)** | Standardized `MossContextStore` interface in `app/utils/moss_adapter.py`. Real-time Moss cloud guardrails await native SDK access. |
| **Cryptographic Provenance Signatures** | **Planned Roadmap** | SHA-256 document hashing is planned for v2.2 to verify raw chunk provenance at retrieval time. |
