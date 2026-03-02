"""Tests for evaluation logic — the core of the HDJ system."""

import json
import numpy as np
import pytest

from hdj.evaluate import (
    Evaluator,
    RetrievedChunk,
    OverlapDetail,
    QueryResult,
    save_results,
    STOPWORDS,
)


# ---------------------------------------------------------------------------
# _content_words / _tokenize — stopword filtering
# ---------------------------------------------------------------------------

class TestContentWords:
    def test_filters_stopwords(self):
        text = "The data is being used by the government for the people"
        words = Evaluator._content_words(text)
        assert "the" not in words
        assert "is" not in words
        assert "by" not in words
        assert "data" in words
        assert "government" in words
        assert "people" in words

    def test_pure_stopwords_returns_empty(self):
        text = "the and or but if is are was were"
        assert Evaluator._content_words(text) == set()

    def test_no_stopwords_all_retained(self):
        text = "discrimination algorithmic surveillance biometric"
        words = Evaluator._content_words(text)
        assert words == {"discrimination", "algorithmic", "surveillance", "biometric"}

    def test_case_insensitive(self):
        words = Evaluator._content_words("Data JUSTICE Health")
        assert words == {"data", "justice", "health"}

    def test_punctuation_stripped(self):
        words = Evaluator._content_words("data-driven, discrimination.")
        # \w+ splits on hyphens/punctuation, producing individual words
        assert "data" in words
        assert "driven" in words
        assert "discrimination" in words

    def test_tokenize_preserves_all_words(self):
        """_tokenize should NOT filter stopwords (used for substring match reporting)."""
        words = Evaluator._tokenize("the data is here")
        assert "the" in words
        assert "data" in words
        assert "is" in words


# ---------------------------------------------------------------------------
# _text_overlap — the matching algorithm
# ---------------------------------------------------------------------------

class TestTextOverlap:
    def setup_method(self):
        self.ev = Evaluator([], overlap_threshold=0.3)

    def test_exact_substring_gold_in_retrieved(self):
        """Gold text fully contained in a retrieved chunk → substring match."""
        gold = "biometric databases"
        retrieved = "India's biometric databases exclude the poor."
        is_match, ratio, words, mtype = self.ev._text_overlap(retrieved, gold)
        assert is_match is True
        assert ratio == 1.0
        assert mtype == "substring"

    def test_exact_substring_retrieved_in_gold(self):
        """Retrieved chunk is a substring of gold text → substring match."""
        gold = "The biometric database known as Aadhaar has over a billion records."
        retrieved = "biometric database known as aadhaar"
        is_match, ratio, words, mtype = self.ev._text_overlap(retrieved, gold)
        assert is_match is True
        assert ratio == 1.0
        assert mtype == "substring"

    def test_case_insensitive_substring(self):
        gold = "Data Justice"
        retrieved = "We discuss DATA JUSTICE in this paper."
        is_match, _, _, mtype = self.ev._text_overlap(retrieved, gold)
        assert is_match is True
        assert mtype == "substring"

    def test_word_overlap_above_threshold(self):
        """Enough content words in common → word_overlap match."""
        gold = "discrimination affects poorest communities data"
        retrieved = "data discrimination impacts the poorest communities worldwide"
        is_match, ratio, words, mtype = self.ev._text_overlap(retrieved, gold)
        assert is_match is True
        assert mtype == "word_overlap"
        assert ratio >= 0.3
        # Check that the common content words are reported
        assert "discrimination" in words
        assert "poorest" in words
        assert "communities" in words

    def test_word_overlap_below_threshold(self):
        """Too few content words in common → no match."""
        gold = "biometric databases exclude people with worn fingerprints due to hard labour"
        retrieved = "telemedicine provides healthcare access in rural regions"
        is_match, ratio, words, mtype = self.ev._text_overlap(retrieved, gold)
        assert is_match is False
        assert mtype == "none"
        assert ratio < 0.3

    def test_threshold_boundary_exact(self):
        """Exactly at the threshold should match."""
        # Build texts where exactly 30% of gold content words appear in retrieved
        # Gold has 10 content words, retrieved shares exactly 3
        gold = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        retrieved = "alpha bravo charlie kilo lima mike november oscar papa quebec"
        ev = Evaluator([], overlap_threshold=0.3)
        is_match, ratio, _, mtype = ev._text_overlap(retrieved, gold)
        assert is_match is True
        assert mtype == "word_overlap"
        assert abs(ratio - 0.3) < 1e-9

    def test_threshold_boundary_just_below(self):
        """Just below the threshold should NOT match."""
        # Gold has 10 content words, retrieved shares 2 → 20% < 30%
        gold = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        retrieved = "alpha bravo kilo lima mike november oscar papa quebec romeo"
        ev = Evaluator([], overlap_threshold=0.3)
        is_match, ratio, _, mtype = ev._text_overlap(retrieved, gold)
        assert is_match is False
        assert mtype == "none"

    def test_empty_gold_no_content_words(self):
        """Gold text with only stopwords → no match (no content words)."""
        gold = "the and or but if"
        retrieved = "the and or but if"
        is_match, ratio, _, mtype = self.ev._text_overlap(retrieved, gold)
        # Substring match happens first (case-insensitive), so this IS a match
        assert is_match is True
        assert mtype == "substring"

    def test_empty_gold_after_filtering(self):
        """Gold text that has no content words and isn't a substring."""
        gold = "the and or but if"
        retrieved = "completely different text about algorithms"
        is_match, ratio, words, mtype = self.ev._text_overlap(retrieved, gold)
        assert is_match is False
        assert ratio == 0.0
        assert mtype == "none"

    def test_empty_retrieved_is_substring(self):
        """Empty retrieved text is technically a substring of any gold text.

        Python: '' in 'anything' → True. The code treats this as a substring
        match, which is fine in practice since search never returns empty chunks.
        """
        gold = "biometric databases"
        retrieved = ""
        is_match, _, _, mtype = self.ev._text_overlap(retrieved, gold)
        assert is_match is True
        assert mtype == "substring"

    def test_stopwords_dont_inflate_overlap(self):
        """Two texts sharing only stopwords should NOT match."""
        gold = "the government was implementing the system for the people"
        retrieved = "the company was building the platform for the users"
        is_match, ratio, words, mtype = self.ev._text_overlap(retrieved, gold)
        # Content words: {government, implementing, system, people}
        # vs {company, building, platform, users} → 0% overlap
        assert is_match is False
        assert mtype == "none"


# ---------------------------------------------------------------------------
# evaluate() — recall and precision metrics
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_perfect_recall(self, evaluator, simple_gold):
        """All gold standard passages found → 100% recall."""
        retrieved = [
            "Data-driven discrimination affects the poorest communities in many ways.",
            "Biometric databases exclude people with worn fingerprints from services.",
            "Algorithmic sorting categorises migrants based on remote surveillance data.",
        ]
        result = evaluator.evaluate("test", "query", retrieved)
        assert result.recall == 1.0
        assert result.found == len(simple_gold)
        assert result.missed_texts == []

    def test_zero_recall(self, evaluator, simple_gold):
        """No gold passages found → 0% recall."""
        retrieved = [
            "Telemedicine in rural areas improves healthcare access.",
            "Electronic health records streamline administrative processes.",
        ]
        result = evaluator.evaluate("test", "query", retrieved)
        assert result.recall == 0.0
        assert result.found == 0
        assert len(result.missed_texts) == len(simple_gold)

    def test_partial_recall(self, evaluator, simple_gold):
        """Some gold passages found → proportional recall."""
        retrieved = [
            "Data-driven discrimination affects the poorest communities deeply.",
            "Telemedicine provides rural healthcare access.",
        ]
        result = evaluator.evaluate("test", "query", retrieved)
        assert result.found == 1
        assert result.recall == pytest.approx(1 / 3)

    def test_precision_calculation(self, evaluator):
        """Precision = found / retrieved."""
        retrieved = [
            "Data-driven discrimination affects the poorest communities deeply.",
            "Unrelated chunk about weather forecasting today.",
            "Another unrelated chunk about cooking recipes.",
            "Yet another unrelated piece about astronomy.",
        ]
        result = evaluator.evaluate("test", "query", retrieved)
        # 1 gold matched, 4 retrieved → precision = 1/4
        assert result.precision == pytest.approx(0.25)

    def test_empty_retrieved(self, evaluator, simple_gold):
        """No results retrieved → 0 recall, 0 precision."""
        result = evaluator.evaluate("test", "query", [])
        assert result.recall == 0.0
        assert result.precision == 0.0
        assert result.retrieved == 0

    def test_empty_gold_standard(self):
        """Empty gold standard → 0 recall (avoid division by zero)."""
        ev = Evaluator([], overlap_threshold=0.3)
        result = ev.evaluate("test", "query", ["some text"])
        assert result.recall == 0.0
        assert result.total_gold == 0

    def test_with_chunks_objects(self, evaluator, chunk_factory):
        """evaluate() works correctly when passed RetrievedChunk objects."""
        chunks = [
            chunk_factory(
                "Data-driven discrimination affects the poorest communities greatly.",
                score=0.95,
                rank=1,
                uri="file:///taylor.pdf",
                pages=[1, 2],
            ),
        ]
        result = evaluator.evaluate(
            "test", "query", [c.content for c in chunks], retrieved_chunks=chunks
        )
        assert result.found >= 1
        assert result.retrieved_chunks is not None
        assert result.retrieved_chunks[0].score == 0.95
        assert result.retrieved_chunks[0].page_numbers == [1, 2]

    def test_match_details_populated(self, evaluator):
        """Match and miss details should be populated with OverlapDetail objects."""
        retrieved = [
            "Data-driven discrimination affects the poorest communities.",
        ]
        result = evaluator.evaluate("test", "query", retrieved)
        assert len(result.match_details) == 1
        assert len(result.miss_details) == 2
        assert result.match_details[0].match_type in ("substring", "word_overlap")
        assert all(d.match_type == "none" for d in result.miss_details)

    def test_semantic_matrix_integrated(self, evaluator):
        """Semantic similarity from matrix is attached to OverlapDetail."""
        retrieved = [
            "Data-driven discrimination affects the poorest communities.",
            "Biometric databases exclude fingerprint-worn people.",
        ]
        # 3 gold × 2 chunks matrix
        sem_matrix = np.array([
            [0.85, 0.20],
            [0.15, 0.90],
            [0.40, 0.35],
        ])
        result = evaluator.evaluate("test", "query", retrieved, semantic_matrix=sem_matrix)
        # First gold matched first chunk → semantic_similarity should be 0.85
        match_for_first_gold = [
            d for d in result.match_details if "discrimination" in d.gold_text.lower()
        ]
        assert len(match_for_first_gold) == 1
        assert match_for_first_gold[0].semantic_similarity == pytest.approx(0.85)

    def test_long_query_truncated(self, evaluator):
        """Queries longer than 200 chars get truncated in the result."""
        long_query = "x" * 300
        result = evaluator.evaluate("test", long_query, [])
        assert len(result.query) == 203  # 200 + "..."
        assert result.query.endswith("...")

    def test_custom_overlap_threshold(self, simple_gold):
        """Different thresholds change matching behavior."""
        # With a very high threshold, fewer matches
        strict = Evaluator(simple_gold, overlap_threshold=0.9)
        # With a very low threshold, more matches
        loose = Evaluator(simple_gold, overlap_threshold=0.1)

        retrieved = [
            "discrimination communities data poorest",
        ]
        strict_result = strict.evaluate("test", "q", retrieved)
        loose_result = loose.evaluate("test", "q", retrieved)
        assert loose_result.found >= strict_result.found


# ---------------------------------------------------------------------------
# _cosine_matrix — semantic similarity computation
# ---------------------------------------------------------------------------

class TestCosineMatrix:
    def test_identical_vectors(self):
        gold = [[1.0, 0.0, 0.0]]
        chunks = [[1.0, 0.0, 0.0]]
        mat = Evaluator._cosine_matrix(gold, chunks)
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        gold = [[1.0, 0.0]]
        chunks = [[0.0, 1.0]]
        mat = Evaluator._cosine_matrix(gold, chunks)
        assert mat[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        gold = [[1.0, 0.0]]
        chunks = [[-1.0, 0.0]]
        mat = Evaluator._cosine_matrix(gold, chunks)
        assert mat[0, 0] == pytest.approx(-1.0)

    def test_matrix_shape(self):
        gold = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        chunks = [[1, 1, 0], [0, 1, 1]]
        mat = Evaluator._cosine_matrix(gold, chunks)
        assert mat.shape == (3, 2)

    def test_zero_vector_handled(self):
        """Zero vectors should not cause division by zero."""
        gold = [[0.0, 0.0]]
        chunks = [[1.0, 0.0]]
        mat = Evaluator._cosine_matrix(gold, chunks)
        # With zero-norm guard (replaced by 1.0), result is 0
        assert np.isfinite(mat[0, 0])

    def test_high_dimensional(self):
        """Sanity check with realistic embedding dimensions."""
        rng = np.random.default_rng(42)
        gold = rng.standard_normal((5, 2560)).tolist()
        chunks = rng.standard_normal((20, 2560)).tolist()
        mat = Evaluator._cosine_matrix(gold, chunks)
        assert mat.shape == (5, 20)
        # All cosine similarities should be in [-1, 1]
        assert np.all(mat >= -1.0 - 1e-6)
        assert np.all(mat <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Serialization — to_dict() roundtrips
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_retrieved_chunk_to_dict(self, chunk_factory):
        chunk = chunk_factory("text", score=0.8, rank=3, uri="file:///a.pdf", pages=[1, 5])
        d = chunk.to_dict()
        assert d["content"] == "text"
        assert d["score"] == 0.8
        assert d["rank"] == 3
        assert d["document_uri"] == "file:///a.pdf"
        assert d["page_numbers"] == [1, 5]

    def test_overlap_detail_to_dict_with_chunk(self, chunk_factory):
        chunk = chunk_factory("matched content", score=0.9, rank=1)
        detail = OverlapDetail(
            gold_text="gold passage",
            gold_text_preview="gold passage",
            matched_chunk=chunk,
            overlap_ratio=0.75,
            overlapping_words=["gold", "passage"],
            match_type="word_overlap",
            semantic_similarity=0.88,
        )
        d = detail.to_dict()
        assert d["overlap_ratio"] == 0.75
        assert d["matched_chunk"]["content"] == "matched content"
        assert d["semantic_similarity"] == 0.88

    def test_overlap_detail_to_dict_without_chunk(self):
        detail = OverlapDetail(
            gold_text="gold", gold_text_preview="gold",
            matched_chunk=None, overlap_ratio=0.0,
        )
        d = detail.to_dict()
        assert d["matched_chunk"] is None

    def test_query_result_to_dict(self, chunk_factory):
        chunk = chunk_factory("text", score=0.5, rank=1)
        match_detail = OverlapDetail(
            gold_text="g", gold_text_preview="g",
            matched_chunk=chunk, overlap_ratio=0.5,
            match_type="word_overlap",
        )
        result = QueryResult(
            name="test_q",
            query="test query",
            recall=0.5,
            precision=0.25,
            found=1,
            total_gold=2,
            retrieved=4,
            found_texts=["found"],
            missed_texts=["missed"],
            retrieved_chunks=[chunk],
            match_details=[match_detail],
            miss_details=[],
        )
        d = result.to_dict()
        assert d["name"] == "test_q"
        assert d["recall"] == 0.5
        assert d["retrieved_chunks"][0]["content"] == "text"
        assert d["match_details"][0]["overlap_ratio"] == 0.5

    def test_query_result_json_serializable(self, evaluator):
        """Full evaluate() result must be JSON-serializable (no numpy types etc.)."""
        result = evaluator.evaluate("test", "query", ["some text"])
        d = result.to_dict()
        serialized = json.dumps(d)
        roundtrip = json.loads(serialized)
        assert roundtrip["name"] == "test"


# ---------------------------------------------------------------------------
# save_results — file I/O
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_creates_output_dir(self, tmp_path, evaluator):
        output_dir = tmp_path / "results"
        result = evaluator.evaluate("test", "query", ["some chunk"])
        path = save_results([result], output_dir)
        assert output_dir.exists()
        assert path.exists()
        assert path.name.startswith("eval_")
        assert path.suffix == ".json"

    def test_output_structure(self, tmp_path, evaluator):
        result = evaluator.evaluate("test", "query", ["chunk"])
        path = save_results([result], tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert "timestamp" in data
        assert "summary" in data
        assert "results" in data
        assert data["summary"]["best_query"] == "test"
        assert data["summary"]["total_queries"] == 1

    def test_best_query_selected(self, tmp_path):
        gold = ["alpha bravo charlie"]
        ev = Evaluator(gold, overlap_threshold=0.3)
        low = ev.evaluate("low", "q", ["unrelated text here"])
        high = ev.evaluate("high", "q", ["alpha bravo charlie is here"])
        path = save_results([low, high], tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["best_query"] == "high"


# ---------------------------------------------------------------------------
# Evaluator.from_json — loading gold standard
# ---------------------------------------------------------------------------

class TestFromJson:
    def test_loads_gold_standard(self, tmp_path):
        gs = [
            {"source_file": "a.pdf", "text": "passage one"},
            {"source_file": "b.pdf", "text": "passage two"},
        ]
        path = tmp_path / "gold.json"
        path.write_text(json.dumps(gs))
        ev = Evaluator.from_json(path)
        assert len(ev.gold_standard) == 2
        assert ev.gold_standard[0] == "passage one"

    def test_custom_threshold(self, tmp_path):
        gs = [{"source_file": "a.pdf", "text": "text"}]
        path = tmp_path / "gold.json"
        path.write_text(json.dumps(gs))
        ev = Evaluator.from_json(path, overlap_threshold=0.8)
        assert ev.overlap_threshold == 0.8
