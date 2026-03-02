"""Tests for the report generation module."""

import pytest

from hdj.export import generate_report, _pretty_doc_name
from hdj.evaluate import QueryResult, RetrievedChunk, OverlapDetail


# ---------------------------------------------------------------------------
# _pretty_doc_name — URI → human-readable filename
# ---------------------------------------------------------------------------

class TestPrettyDocName:
    def test_file_uri(self):
        assert _pretty_doc_name("file:///home/user/docs/taylor.pdf") == "taylor.pdf"

    def test_plain_path(self):
        assert _pretty_doc_name("/data/pdfs/report.pdf") == "report.pdf"

    def test_none_returns_unknown(self):
        assert _pretty_doc_name(None) == "Unknown PDF"

    def test_empty_string_returns_unknown(self):
        assert _pretty_doc_name("") == "Unknown PDF"

    def test_url_encoded_spaces(self):
        name = _pretty_doc_name("file:///docs/my%20report.pdf")
        assert name == "my report.pdf"

    def test_windows_style_path(self):
        """Plain path with backslashes or just a filename."""
        name = _pretty_doc_name("document.pdf")
        assert name == "document.pdf"


# ---------------------------------------------------------------------------
# generate_report — Markdown report structure
# ---------------------------------------------------------------------------

class TestGenerateReport:
    @pytest.fixture
    def sample_data(self):
        chunk = RetrievedChunk(
            content="Retrieved text about discrimination",
            score=0.85,
            rank=1,
            document_uri="file:///taylor.pdf",
            page_numbers=[3, 4],
        )
        miss_detail = OverlapDetail(
            gold_text="A long gold passage that was missed",
            gold_text_preview="A long gold passage that was missed",
            matched_chunk=chunk,
            overlap_ratio=0.15,
            overlapping_words=["passage"],
            match_type="none",
            semantic_similarity=0.42,
        )
        result = QueryResult(
            name="test_query",
            query="data justice discrimination",
            recall=0.5,
            precision=0.25,
            found=1,
            total_gold=2,
            retrieved=4,
            found_texts=["Found passage preview..."],
            missed_texts=["Missed passage preview..."],
            retrieved_chunks=[chunk],
            match_details=[],
            miss_details=[miss_detail],
        )
        queries = {"test_query": "data justice discrimination"}
        gold_standard = [
            {"source_file": "taylor.pdf", "text": "Found passage about data justice"},
            {"source_file": "taylor.pdf", "text": "A long gold passage that was missed"},
        ]
        config = {
            "embedding_model": "TestModel",
            "chunk_size": 256,
            "overlap_threshold": 0.3,
            "indexed_pdfs": ["taylor.pdf", "other.pdf"],
        }
        return result, queries, gold_standard, config

    def test_report_contains_header(self, sample_data):
        result, queries, gold, config = sample_data
        report = generate_report([result], queries, gold, config)
        assert "# Search Validation Report" in report

    def test_report_contains_config(self, sample_data):
        result, queries, gold, config = sample_data
        report = generate_report([result], queries, gold, config)
        assert "TestModel" in report
        assert "~256 words" in report

    def test_report_contains_results_table(self, sample_data):
        result, queries, gold, config = sample_data
        report = generate_report([result], queries, gold, config)
        assert "| Question | Coverage | Accuracy | Found |" in report
        assert "test_query" in report
        assert "1/2" in report

    def test_report_contains_gold_standard(self, sample_data):
        result, queries, gold, config = sample_data
        report = generate_report([result], queries, gold, config)
        assert "## Reference Passages (2 passages)" in report
        assert "taylor.pdf" in report

    def test_report_contains_missed_diagnosis(self, sample_data):
        result, queries, gold, config = sample_data
        report = generate_report([result], queries, gold, config)
        assert "Missed passages" in report
        assert "15% word overlap" in report
        assert "Meaning similarity: 42%" in report

    def test_report_contains_indexed_pdfs(self, sample_data):
        result, queries, gold, config = sample_data
        report = generate_report([result], queries, gold, config)
        assert "taylor.pdf" in report
        assert "other.pdf" in report

    def test_report_decision_provenance(self, sample_data):
        result, queries, gold, config = sample_data
        report = generate_report([result], queries, gold, config)
        assert "## How Results Were Determined" in report
        assert "30%" in report

    def test_report_with_audit_trail(self, sample_data):
        result, queries, gold, config = sample_data
        audit = [
            {"timestamp": "2026-03-02T10:00:00", "action": "index_built", "details": {"pdfs": 3}},
        ]
        report = generate_report([result], queries, gold, config, audit_events=audit)
        assert "## Session Activity" in report
        assert "index_built" in report

    def test_report_without_audit_trail(self, sample_data):
        result, queries, gold, config = sample_data
        report = generate_report([result], queries, gold, config)
        assert "Session Activity" not in report

    def test_report_empty_results(self):
        """Report with no evaluation results should still render."""
        report = generate_report(
            results=[],
            queries={},
            gold_standard=[],
            config={"indexed_pdfs": []},
        )
        assert "# Search Validation Report" in report
        assert "## Results Summary" in report

    def test_report_multiple_queries_sorted(self):
        """Results should be sorted by recall (highest first)."""
        low = QueryResult(
            name="low_recall", query="q1", recall=0.2, precision=0.1,
            found=1, total_gold=5, retrieved=10,
            found_texts=[], missed_texts=[],
        )
        high = QueryResult(
            name="high_recall", query="q2", recall=0.8, precision=0.5,
            found=4, total_gold=5, retrieved=8,
            found_texts=[], missed_texts=[],
        )
        report = generate_report(
            [low, high],
            {"low_recall": "q1", "high_recall": "q2"},
            [],
            {"indexed_pdfs": []},
        )
        # "high_recall" should appear before "low_recall" in the table
        high_pos = report.index("high_recall")
        low_pos = report.index("low_recall")
        assert high_pos < low_pos

    def test_report_best_query_identified(self, sample_data):
        result, queries, gold, config = sample_data
        report = generate_report([result], queries, gold, config)
        assert "Best question: test_query" in report

    def test_gold_standard_grouped_by_file(self):
        """Gold standard entries are grouped by source_file."""
        gold = [
            {"source_file": "a.pdf", "text": "passage from a"},
            {"source_file": "b.pdf", "text": "passage from b"},
            {"source_file": "a.pdf", "text": "another from a"},
        ]
        result = QueryResult(
            name="q", query="q", recall=0.0, precision=0.0,
            found=0, total_gold=3, retrieved=0,
            found_texts=[], missed_texts=[],
        )
        report = generate_report([result], {"q": "q"}, gold, {"indexed_pdfs": []})
        assert "### a.pdf (2 passages)" in report
        assert "### b.pdf (1 passages)" in report
