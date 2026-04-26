"""
llm_factory.py — Returns the correct LLM instances based on config.

Two LLMs are used in this pipeline:
  1. Generator LLM  — answers the question (gpt-4o-mini or groq)
  2. Grader LLM     — evaluates relevance + hallucinations (DeepSeek-R1)
"""
from langchain_openai import ChatOpenAI
from app.config import settings


def get_generator_llm():
    """
    Returns the LLM used to generate the final answer.
    Defaults to OpenAI gpt-4o-mini. Set LLM_PROVIDER=groq to switch.
    """
    if settings.LLM_PROVIDER == "groq":
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=settings.GENERATION_MODEL if "llama" in settings.GENERATION_MODEL else "llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=settings.GROQ_API_KEY,
            )
        except ImportError:
            raise ImportError("Install langchain-groq: pip install langchain-groq")

    return ChatOpenAI(
        model=settings.GENERATION_MODEL,
        temperature=0,
        openai_api_key=settings.OPENAI_API_KEY,
    )


def get_grader_llm():
    """
    Returns the DeepSeek-R1 reasoning model used for grading.
    Falls back to gpt-4o-mini if DeepSeek key is missing.
    """
    if settings.DEEPSEEK_API_KEY:
        return ChatOpenAI(
            model=settings.GRADER_MODEL,
            temperature=0,
            openai_api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )

    # Fallback to OpenAI for grading if no DeepSeek key
    import warnings
    warnings.warn(
        "DEEPSEEK_API_KEY not set — falling back to OpenAI for grading. "
        "Set DEEPSEEK_API_KEY for full reasoning capability.",
        stacklevel=2,
    )
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=settings.OPENAI_API_KEY,
    )
