"""Shared fixtures for HDJ tests."""

import pytest

from hdj.evaluate import Evaluator, RetrievedChunk, QueryResult, OverlapDetail


@pytest.fixture
def simple_gold():
    """A small gold standard for unit tests."""
    return [
        "Data-driven discrimination affects the poorest communities.",
        "Biometric databases exclude people with worn fingerprints.",
        "Algorithmic sorting categorises migrants based on remote surveillance.",
    ]


@pytest.fixture
def evaluator(simple_gold):
    return Evaluator(simple_gold, overlap_threshold=0.3, semantic_threshold=0.75)


@pytest.fixture
def chunk_factory():
    """Factory for creating RetrievedChunk instances."""
    def _make(content, score=0.5, rank=1, uri=None, pages=None):
        return RetrievedChunk(
            content=content,
            score=score,
            rank=rank,
            document_uri=uri,
            page_numbers=pages or [],
        )
    return _make
