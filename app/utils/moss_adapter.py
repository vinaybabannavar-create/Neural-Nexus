"""
moss_adapter.py — Moss Context Store integration adapter.

Provides a standardized MossContextStore interface with retrieve() and get_latency_ms()
methods. Currently wraps the local ChromaDB/Pinecone vector retrieval pipeline,
structured so the native Moss SDK / API can be swapped in seamlessly upon SDK access.

Status: Moss-ready interface, pending SDK access.
"""
import time
from typing import List, Optional
from loguru import logger
from langchain_core.documents import Document
from app.utils.vector_store import get_retriever
from app.config import settings


class MossContextStore:
    """
    Standardized Context Store adapter adhering to the Moss Context Engine interface.
    
    Provides high-performance context retrieval and telemetry tracking.
    Wraps existing vector stores (ChromaDB / Pinecone) while maintaining forward
    compatibility with the native Moss SDK endpoint.
    """

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        self.api_key = api_key
        self.endpoint = endpoint
        self._last_latency_ms: float = 0.0
        self._is_native_active: bool = bool(api_key and endpoint)
        
        if self._is_native_active:
            logger.info(f"[MOSS ADAPTER] Initialized native Moss Context Store at {endpoint}")
        else:
            logger.info("[MOSS ADAPTER] Moss SDK access pending — active in compatibility mode via Chroma/Pinecone")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve context documents for a query and record retrieval latency in milliseconds.
        """
        start_time = time.perf_counter()
        k = top_k or settings.TOP_K_RETRIEVAL

        try:
            if self._is_native_active:
                # Placeholder for native Moss SDK remote call when SDK is available:
                # response = self.client.query(query=query, k=k)
                # documents = [Document(page_content=d.text, metadata=d.meta) for d in response]
                raise NotImplementedError("Native Moss SDK is pending upstream package distribution.")
            else:
                retriever = get_retriever()
                documents = retriever.invoke(query)
                if top_k and len(documents) > top_k:
                    documents = documents[:top_k]
                return documents
        finally:
            self._last_latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.debug(f"[MOSS ADAPTER] Retrieved {len(documents) if 'documents' in locals() else 0} chunks in {self._last_latency_ms:.2f}ms")

    def get_latency_ms(self) -> float:
        """Return the latency of the most recent retrieval operation in milliseconds."""
        return round(self._last_latency_ms, 2)

    def is_native_moss_active(self) -> bool:
        """Return whether native remote Moss endpoint is currently connected."""
        return self._is_native_active


# Singleton default context store instance
moss_context_store = MossContextStore()
