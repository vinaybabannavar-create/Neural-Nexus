"""
quarantine_store.py — SQLite-backed store for quarantined/rejected documents.

Provides an auditable trail of rejected documents with timestamps, reasons,
and content snippets so security events can be audited.
"""
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

DB_PATH = Path("./quarantine.db")


def init_quarantine_db(db_path: Path = DB_PATH):
    """Ensure the quarantine database and table exist."""
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quarantined_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    snippet TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)
    finally:
        conn.close()


# Initialize upon module import
init_quarantine_db()


class QuarantineStore:
    """Store and retrieve quarantined documents."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        init_quarantine_db(self.db_path)

    def record_quarantine(
        self,
        source: str,
        reason: str,
        snippet: str,
        risk_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record a rejected document in the quarantine database.
        Returns the inserted record ID.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata_str = json.dumps(metadata or {})
        # Truncate snippet to 1000 characters for safety & storage sanity
        safe_snippet = snippet[:1000]

        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO quarantined_documents 
                    (timestamp, source, reason, risk_score, snippet, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, source, reason, risk_score, safe_snippet, metadata_str),
                )
                record_id = cursor.lastrowid
                logger.warning(
                    f"[QUARANTINE] Logged rejected document from '{source}' "
                    f"(ID: {record_id}, Risk: {risk_score:.2f}, Reason: {reason})"
                )
                return record_id
        finally:
            conn.close()

    def get_quarantined_records(
        self, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve recent quarantined records."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, source, reason, risk_score, snippet, metadata_json
                FROM quarantined_documents
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                try:
                    item["metadata"] = json.loads(item.get("metadata_json") or "{}")
                except Exception:
                    item["metadata"] = {}
                results.append(item)
            return results
        finally:
            conn.close()

    def get_quarantine_stats(self) -> Dict[str, Any]:
        """Return summary statistics of quarantined items."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(risk_score) FROM quarantined_documents")
            count, avg_risk = cursor.fetchone()
            return {
                "total_quarantined": count or 0,
                "avg_risk_score": round(avg_risk, 2) if avg_risk is not None else 0.0,
            }
        finally:
            conn.close()

    def clear(self):
        """Clear all records (used for test isolation)."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute("DELETE FROM quarantined_documents")
        finally:
            conn.close()


# Singleton instance
quarantine_store = QuarantineStore()
