"""
latency_tracer.py — Latency instrumentation and profiling for Neural Nexus graph nodes.

Wraps pipeline stage executions, captures precise execution durations,
and formats metrics for Streamlit UI breakdown and API consumers.
"""
import time
from functools import wraps
from typing import Dict, Any, List, Optional
from loguru import logger


class LatencyTracer:
    """Helper for measuring, recording, and summarizing pipeline stage latencies."""

    @staticmethod
    def trace_node(stage_name: str):
        """Decorator to measure and record execution time of a LangGraph node."""
        def decorator(func):
            @wraps(func)
            def wrapper(state: dict, *args, **kwargs) -> dict:
                start = time.perf_counter()
                logger.debug(f"[LATENCY TRACER] Starting stage '{stage_name}'")
                try:
                    result = func(state, *args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    logger.info(f"[LATENCY TRACER] Stage '{stage_name}' completed in {duration * 1000:.1f}ms ({duration:.3f}s)")
                    if isinstance(result, dict):
                        exec_times = dict(result.get("node_execution_times") or {})
                        exec_times[stage_name] = round(duration, 4)
                        result["node_execution_times"] = exec_times
                return result
            return wrapper
        return decorator

    @staticmethod
    def get_breakdown_table(node_execution_times: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Convert node execution times into a structured breakdown table
        with milliseconds and percentages.
        """
        if not node_execution_times:
            return []

        total_time = sum(node_execution_times.values())
        rows = []
        
        # Display stages in standard execution order if present
        order = [
            "transform_query",
            "retrieve",
            "rerank",
            "grade_documents",
            "web_search",
            "generate",
            "grade_hallucinations",
            "confidence_escalator",
        ]
        
        # Sort by predefined order first, then any extra keys
        sorted_keys = sorted(
            node_execution_times.keys(),
            key=lambda k: order.index(k) if k in order else 999
        )

        for key in sorted_keys:
            sec = node_execution_times[key]
            ms = sec * 1000
            pct = (sec / total_time * 100) if total_time > 0 else 0.0
            rows.append({
                "Stage": key.replace("_", " ").title(),
                "Stage ID": key,
                "Duration (s)": round(sec, 3),
                "Duration (ms)": round(ms, 1),
                "Percentage (%)": round(pct, 1),
            })
            
        return rows


tracer = LatencyTracer()
