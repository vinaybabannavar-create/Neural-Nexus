"""
config.py — Centralised settings loaded from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
    GRADER_MODEL: str = os.getenv("GRADER_MODEL", "deepseek-reasoner")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Vector store
    VECTOR_STORE: str = os.getenv("VECTOR_STORE", "chroma")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENV: str = os.getenv("PINECONE_ENV", "us-east-1-aws")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "corrective-rag")

    # Web search
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Retrieval
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "4"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))

    # Collection name for ChromaDB
    CHROMA_COLLECTION: str = "corrective_rag_docs"


settings = Settings()
