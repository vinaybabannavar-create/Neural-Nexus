"""
ingest_validator.py — Ingestion security screening for prompt injection and malicious content.

Inspects incoming document content for prompt injection, jailbreak attempts,
system prompt overrides, and delimiter manipulation before storing in vector DB.
"""
import re
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from loguru import logger
from app.security.quarantine_store import quarantine_store


class ValidationResult(BaseModel):
    """Result of security validation on a document."""
    is_valid: bool = Field(description="True if safe to ingest, False if quarantined")
    status: str = Field(description="'accepted' or 'quarantined'")
    reason: Optional[str] = Field(default=None, description="Explanation if quarantined")
    risk_score: float = Field(default=0.0, description="Risk score from 0.0 (safe) to 1.0 (dangerous)")
    detected_patterns: List[str] = Field(default_factory=list, description="List of matched vulnerability signatures")


# Heuristic Regex Signatures for Ingestion Threat Detection
INJECTION_SIGNATURES: List[Tuple[str, str, float]] = [
    # (regex_pattern, description, risk_weight)
    (
        r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|prompts|directives|rules|context)\b",
        "Direct instruction override attempt",
        1.0,
    ),
    (
        r"(?i)\b(?:system\s+prompt\s+override|new\s+system\s+prompt|replace\s+system\s+message)\b",
        "System prompt replacement attempt",
        1.0,
    ),
    (
        r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:dan|an\s+unrestricted|jailbroken|god\s+mode|developer\s+mode|aim|omega)\b",
        "Jailbreak persona hijacking attempt",
        0.95,
    ),
    (
        r"(?i)\b(?:bypass|disable|ignore)\s+(?:all\s+)?(?:safety|content\s+filter|guardrails|moderation|security\s+rules)\b",
        "Safety guardrail bypass attempt",
        0.9,
    ),
    (
        r"(?i)(?:<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|system\|>|\[system_instruction\]|\[SYSTEM_PROMPT\])",
        "Chat template / delimiter token injection",
        0.95,
    ),
    (
        r"(?i)(?:===\s*(?:SYSTEM|ADMIN|INSTRUCTION|OVERRIDE)\s*===|###\s*(?:SYSTEM\s+PROMPT|ADMIN_OVERRIDE)\s*###)",
        "Synthetic system delimiter injection",
        0.85,
    ),
    (
        r"(?i)!\[.*?\]\((?:https?:)?//[^\s)]+\?[^)]*(?:exfil|token|key|secret|cookie|leak)=.*?\)",
        "Markdown image data exfiltration payload",
        0.95,
    ),
    (
        r"(?i)<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
        "Embedded executable script tag in document",
        0.8,
    ),
    (
        r"(?i)\bprint\s+(?:the\s+)?(?:system\s+prompt|hidden\s+instructions|developer\s+mode\s+prompt)\b",
        "System prompt leakage request",
        0.75,
    ),
]


class IngestValidator:
    """Screens incoming documents for prompt injection and malicious signatures."""

    def __init__(self, risk_threshold: float = 0.7):
        self.risk_threshold = risk_threshold

    def validate(self, text: str, source: str = "unknown", auto_quarantine: bool = True) -> ValidationResult:
        """
        Validate document content. If dangerous patterns exceed risk_threshold,
        quarantine the document and return an invalid result.
        """
        if not text or not text.strip():
            return ValidationResult(
                is_valid=True,
                status="accepted",
                reason=None,
                risk_score=0.0,
                detected_patterns=[],
            )

        detected = []
        max_risk = 0.0

        for pattern, desc, weight in INJECTION_SIGNATURES:
            if re.search(pattern, text):
                detected.append(desc)
                if weight > max_risk:
                    max_risk = weight

        if max_risk >= self.risk_threshold or detected:
            reason = f"Prompt injection detected: {', '.join(detected)}"
            logger.warning(f"[SECURITY] Ingest validation FAILED for '{source}': {reason} (Score: {max_risk:.2f})")
            
            if auto_quarantine:
                quarantine_store.record_quarantine(
                    source=source,
                    reason=reason,
                    snippet=text[:500],
                    risk_score=max_risk,
                    metadata={"detected_patterns": detected}
                )

            return ValidationResult(
                is_valid=False,
                status="quarantined",
                reason=reason,
                risk_score=max_risk,
                detected_patterns=detected,
            )

        logger.info(f"[SECURITY] Ingest validation PASSED for '{source}'")
        return ValidationResult(
            is_valid=True,
            status="accepted",
            reason=None,
            risk_score=0.0,
            detected_patterns=[],
        )


# Global validator singleton
ingest_validator = IngestValidator()


def validate_document(text: str, source: str = "unknown", auto_quarantine: bool = True) -> ValidationResult:
    """Convenience function to validate a document text."""
    return ingest_validator.validate(text, source=source, auto_quarantine=auto_quarantine)
