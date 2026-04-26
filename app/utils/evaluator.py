"""
evaluator.py — Automated evaluation of RAG responses.

Uses an LLM to score the final generation based on:
1. Groundedness (Faithfulness)
2. Completeness (Answer Relevance)
"""
from typing import Dict
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.utils.llm_factory import get_grader_llm

class EvalScore(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the score")

def evaluate_response(question: str, context: str, answer: str) -> Dict[str, EvalScore]:
    """
    Evaluate a response using a grader LLM.
    """
    grader = get_grader_llm().with_structured_output(EvalScore)
    
    # 1. Groundedness check
    grounded_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator checking if an answer is supported by the context. Score 1.0 for perfect grounding, 0.0 for hallucination."),
        ("human", "CONTEXT: {context}\n\nANSWER: {answer}\n\nScore the grounding:")
    ])
    
    # 2. Completeness check
    complete_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator checking if an answer fully addresses the user's question. Score 1.0 for perfect answer, 0.0 for irrelevant/missing answer."),
        ("human", "QUESTION: {question}\n\nANSWER: {answer}\n\nScore the completeness:")
    ])
    
    results = {}
    try:
        results["groundedness"] = (grounded_prompt | grader).invoke({"context": context, "answer": answer})
        results["completeness"] = (complete_prompt | grader).invoke({"question": question, "answer": answer})
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        
    return results
