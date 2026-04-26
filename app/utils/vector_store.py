"""
vector_store.py — Abstraction layer for ChromaDB and Pinecone.
                  Call get_retriever() to get a LangChain retriever
                  regardless of which backend is configured.
"""
from loguru import logger
from langchain_openai import OpenAIEmbeddings
from app.config import settings


def get_embeddings():
    """Return the configured embedding model."""
    if settings.EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"Using local HuggingFace embeddings: {settings.EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )


def get_retriever():
    """
    Return a LangChain retriever backed by ChromaDB or Pinecone
    depending on VECTOR_STORE env variable.
    """
    embeddings = get_embeddings()

    if settings.VECTOR_STORE == "pinecone":
        return _get_pinecone_retriever(embeddings)
    return _get_chroma_retriever(embeddings)


def _get_chroma_retriever(embeddings):
    from langchain_community.vectorstores import Chroma

    logger.info(f"Using ChromaDB at: {settings.CHROMA_PERSIST_DIR}")
    vectorstore = Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K_RETRIEVAL},
    )


def _get_pinecone_retriever(embeddings):
    from pinecone import Pinecone
    from langchain_community.vectorstores import PineconeVectorStore

    logger.info(f"Using Pinecone index: {settings.PINECONE_INDEX_NAME}")
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    return vectorstore.as_retriever(
        search_kwargs={"k": settings.TOP_K_RETRIEVAL}
    )


def get_vectorstore_for_ingestion():
    """
    Return the raw vectorstore object (not retriever) for document ingestion.
    """
    embeddings = get_embeddings()

    if settings.VECTOR_STORE == "pinecone":
        from pinecone import Pinecone
        from langchain_community.vectorstores import PineconeVectorStore
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        return PineconeVectorStore(index=index, embedding=embeddings)

    from langchain_community.vectorstores import Chroma
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
