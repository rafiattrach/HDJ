"""Health Data Justice RAG module."""

from .rag import HDJRag
from .evaluate import Evaluator, QueryResult, RetrievedChunk, OverlapDetail
from .audit import log_event, load_events
from .export import generate_report
from .translate import detect_language, translate_query, ensure_packages

__all__ = [
    "HDJRag",
    "Evaluator",
    "QueryResult",
    "RetrievedChunk",
    "OverlapDetail",
    "log_event",
    "load_events",
    "generate_report",
    "detect_language",
    "translate_query",
    "ensure_packages",
]
