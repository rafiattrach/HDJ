"""Evaluation logic for RAG retrieval quality."""

import json
import logging
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .rag import HDJRag

logger = logging.getLogger(__name__)

# Common English stopwords — filtered from word-overlap computation so that
# matches are driven by content words rather than function words like
# "the", "and", "of", etc.
STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "let", "may", "me",
    "might", "more", "most", "must", "my", "myself", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shall", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "upon", "us", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
})

# Common German stopwords — filtered alongside English stopwords so that
# cross-lingual comparisons aren't inflated by function words in either
# language.
GERMAN_STOPWORDS: frozenset[str] = frozenset({
    "ab", "aber", "ach", "alle", "allein", "allem", "allen", "aller",
    "allerdings", "alles", "also", "am", "an", "ander", "andere", "anderem",
    "anderen", "anderer", "anderes", "anderm", "andern", "anders", "auch",
    "auf", "aus", "ausser", "außer", "ausserdem", "außerdem",
    "bei", "beide", "beiden", "beider", "beiderlei", "beides", "beim",
    "bereits", "bestimmte", "bestimmten", "bestimmter", "bin", "bis",
    "bisher", "bist",
    "da", "dabei", "dadurch", "dafür", "dagegen", "daher", "dahin",
    "damals", "damit", "danach", "daneben", "dann", "daran", "darauf",
    "daraus", "darf", "darfst", "darin", "darum", "darunter", "darüber",
    "das", "dass", "daß", "davon", "davor", "dazu", "dein", "deine",
    "deinem", "deinen", "deiner", "deines", "dem", "den", "denn", "dennoch",
    "der", "deren", "des", "deshalb", "dessen", "dich", "die", "dies",
    "diese", "dieselbe", "dieselben", "diesem", "diesen", "dieser", "dieses",
    "dir", "doch", "dort", "drei", "drin", "dritte", "dritten", "dritter",
    "du", "dumm", "durch", "dürfen",
    "eben", "ebenso", "ehe", "eher", "ein", "eine", "einem", "einen",
    "einer", "einige", "einigem", "einigen", "einiger", "einiges", "einmal",
    "er", "erst", "es", "etwa", "etwas", "euch", "euer", "eure", "eurem",
    "euren", "eurer", "eures",
    "für",
    "ganz", "gar", "gegen", "gehen", "geht", "genug", "gerade", "gering",
    "gern", "gerne", "gewesen", "gewiß", "gewiss",
    "hab", "habe", "haben", "habt", "hat", "hatte", "hätte", "hin",
    "hinter",
    "ich", "ihm", "ihn", "ihnen", "ihr", "ihre", "ihrem", "ihren", "ihrer",
    "ihres", "immer", "in", "indem", "infolge", "innen", "ins",
    "irgend", "ist",
    "ja", "jede", "jedem", "jeden", "jeder", "jedes", "jedoch", "jemals",
    "jene", "jenem", "jenen", "jener", "jenes", "jetzt",
    "kann", "kannst", "kaum", "kein", "keine", "keinem", "keinen", "keiner",
    "keines", "können", "könnte",
    "lang", "lange", "längst",
    "machen", "macht", "man", "manch", "manche", "manchem", "manchen",
    "mancher", "mancherlei", "manches", "mehr", "mein", "meine", "meinem",
    "meinen", "meiner", "meines", "mich", "mir", "mit", "mochte", "möchte",
    "morgen", "morgens", "muss", "muß", "müssen", "musste", "müsste",
    "nach", "nachdem", "nachher", "nächst", "neben", "nein", "nicht",
    "nichts", "nie", "niemand", "nirgends", "noch", "nun", "nur",
    "ob", "oben", "oder", "ohne",
    "sehr", "seid", "sein", "seine", "seinem", "seinen", "seiner", "seines",
    "seit", "seitdem", "sich", "sicher", "sie", "sind", "so", "sofort",
    "sogar", "solch", "solche", "solchem", "solchen", "solcher", "soll",
    "sollen", "sollte", "sollten", "solltest", "sondern", "sonst", "soviel",
    "sowie", "über", "überall", "überein",
    "um", "ums", "und", "uns", "unser", "unsere", "unserem", "unseren",
    "unserer", "unseres", "unter",
    "viel", "viele", "vielem", "vielen", "vielleicht", "vom", "von", "vor",
    "vorbei", "vorher",
    "während", "wann", "war", "warum", "was", "weder", "weil", "weit",
    "welch", "welche", "welchem", "welchen", "welcher", "welches", "wem",
    "wen", "wenig", "wenige", "wenigen", "weniger", "wenigstens", "wenn",
    "wer", "werde", "werden", "wessen", "wie", "wieder", "will", "wir",
    "wird", "wirklich", "wo", "wohl", "wollen", "worden", "wurde", "würde",
    "würden",
    "zu", "zum", "zur", "zusammen", "zwar", "zwischen",
})

ALL_STOPWORDS: frozenset[str] = STOPWORDS | GERMAN_STOPWORDS


@dataclass
class RetrievedChunk:
    """A single RAG search result with full metadata."""
    content: str
    score: float  # raw RRF rank-fusion score from hybrid search (NOT a 0–100% relevance)
    rank: int  # 1-based
    document_uri: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    query_similarity: float = 0.0  # cosine similarity between query and this chunk (a true 0–100% relevance)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OverlapDetail:
    """How a gold passage matched (or nearly matched) a chunk."""
    gold_text: str
    gold_text_preview: str
    matched_chunk: RetrievedChunk | None = None
    overlap_ratio: float = 0.0
    overlapping_words: list[str] = field(default_factory=list)
    match_type: str = "none"  # "substring" | "word_overlap" | "semantic" | "none"
    semantic_similarity: float = 0.0  # cosine similarity between embeddings

    def to_dict(self) -> dict:
        d = {
            "gold_text": self.gold_text,
            "gold_text_preview": self.gold_text_preview,
            "matched_chunk": self.matched_chunk.to_dict() if self.matched_chunk else None,
            "overlap_ratio": self.overlap_ratio,
            "overlapping_words": self.overlapping_words,
            "match_type": self.match_type,
            "semantic_similarity": self.semantic_similarity,
        }
        return d


@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    name: str
    query: str
    recall: float
    precision: float
    found: int
    total_gold: int
    retrieved: int
    relevant_retrieved: int = 0  # how many retrieved chunks matched >=1 reference passage
    found_texts: list[str] = field(default_factory=list)
    missed_texts: list[str] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] | None = None
    match_details: list[OverlapDetail] | None = None
    miss_details: list[OverlapDetail] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Use custom serialization for the new fields
        if self.retrieved_chunks is not None:
            d["retrieved_chunks"] = [c.to_dict() for c in self.retrieved_chunks]
        if self.match_details is not None:
            d["match_details"] = [m.to_dict() for m in self.match_details]
        if self.miss_details is not None:
            d["miss_details"] = [m.to_dict() for m in self.miss_details]
        return d


class Evaluator:
    """Evaluates RAG retrieval against gold standard."""

    def __init__(
        self,
        gold_standard: list[str],
        overlap_threshold: float = 0.3,
        semantic_threshold: float = 0.75,
    ):
        self.gold_standard = gold_standard
        self.overlap_threshold = overlap_threshold
        self.semantic_threshold = semantic_threshold
        self._gold_embeddings: list[list[float]] | None = None

    @classmethod
    def from_json(
        cls,
        path: Path,
        overlap_threshold: float = 0.3,
        semantic_threshold: float = 0.75,
    ) -> "Evaluator":
        """Load gold standard from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        texts = [item["text"] for item in data]
        return cls(texts, overlap_threshold, semantic_threshold=semantic_threshold)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Extract lowercase words, stripping punctuation."""
        return set(re.findall(r"\w+", text.lower()))

    @staticmethod
    def _content_words(text: str) -> set[str]:
        """Extract lowercase content words (no stopwords, no punctuation).

        Filters both English and German stopwords so that cross-lingual
        comparisons are driven by meaningful content words.
        """
        return set(re.findall(r"\w+", text.lower())) - ALL_STOPWORDS

    def _text_overlap(
        self, retrieved: str, gold: str
    ) -> tuple[bool, float, list[str], str]:
        """Check if texts overlap significantly.

        Returns (is_match, overlap_ratio, overlapping_words, match_type).

        Stopwords and punctuation are stripped before computing word overlap
        so that matches are driven by meaningful content words.
        """
        retrieved_lower = retrieved.lower()
        gold_lower = gold.lower()

        # Guard: empty strings should not match
        if not retrieved_lower.strip() or not gold_lower.strip():
            return (False, 0.0, [], "none")

        # Substring match — exact containment, no filtering needed
        if gold_lower in retrieved_lower or retrieved_lower in gold_lower:
            all_words = self._tokenize(gold)
            return (True, 1.0, sorted(all_words), "substring")

        # Word overlap with stopword filtering
        gold_words = self._content_words(gold)
        if not gold_words:
            return (False, 0.0, [], "none")

        retrieved_words = self._content_words(retrieved)
        common = gold_words & retrieved_words
        ratio = len(common) / len(gold_words)

        if ratio >= self.overlap_threshold:
            return (True, ratio, sorted(common), "word_overlap")

        return (False, ratio, sorted(common), "none")

    def evaluate(
        self,
        name: str,
        query: str,
        retrieved_texts: list[str],
        retrieved_chunks: list[RetrievedChunk] | None = None,
        semantic_matrix: np.ndarray | None = None,
    ) -> QueryResult:
        """Evaluate retrieved texts against gold standard.

        Parameters
        ----------
        semantic_matrix : ndarray, optional
            Shape ``(n_gold, n_chunks)`` of cosine similarities between gold
            text embeddings and chunk embeddings.  When provided the best
            semantic similarity is recorded on each :class:`OverlapDetail`.
        """
        # Build chunks list — synthetic from texts if not provided
        if retrieved_chunks is not None:
            chunks = retrieved_chunks
        else:
            chunks = [
                RetrievedChunk(content=t, score=0.0, rank=i + 1)
                for i, t in enumerate(retrieved_texts)
            ]

        found_texts = []
        missed_texts = []
        match_details = []
        miss_details = []

        for gold_idx, gold in enumerate(self.gold_standard):
            truncated = gold[:150] + "..." if len(gold) > 150 else gold

            best_ratio = -1.0
            best_words: list[str] = []
            best_type = "none"
            best_chunk: RetrievedChunk | None = None
            best_chunk_idx: int = 0

            for chunk_idx, chunk in enumerate(chunks):
                is_match, ratio, words, mtype = self._text_overlap(chunk.content, gold)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_words = words
                    best_type = mtype
                    best_chunk = chunk
                    best_chunk_idx = chunk_idx

            is_found = best_type != "none"

            # Look up semantic similarity if available
            sem_sim = 0.0
            sem_best_chunk_idx = best_chunk_idx
            if semantic_matrix is not None and len(chunks) > 0:
                sem_best_chunk_idx = int(np.argmax(semantic_matrix[gold_idx]))
                sem_sim = float(semantic_matrix[gold_idx, sem_best_chunk_idx])

            # Semantic fallback: if word-overlap missed but semantic similarity
            # is high enough, upgrade to a "semantic" match.  This is the key
            # enabler for cross-lingual evaluation — a German gold passage and
            # its English equivalent share zero content words but have high
            # cosine similarity via multilingual embeddings.
            if (
                not is_found
                and sem_sim >= self.semantic_threshold
                and semantic_matrix is not None
            ):
                best_type = "semantic"
                best_chunk = chunks[sem_best_chunk_idx]
                best_chunk_idx = sem_best_chunk_idx
                is_found = True

            detail = OverlapDetail(
                gold_text=gold,
                gold_text_preview=truncated,
                matched_chunk=best_chunk,
                overlap_ratio=best_ratio if best_ratio >= 0 else 0.0,
                overlapping_words=best_words,
                match_type=best_type,
                semantic_similarity=sem_sim,
            )

            if is_found:
                found_texts.append(truncated)
                match_details.append(detail)
            else:
                missed_texts.append(truncated)
                miss_details.append(detail)

        found = len(found_texts)
        total = len(self.gold_standard)
        retrieved = len(chunks)

        # Chunk-level precision: of the chunks we retrieved, how many were
        # actually relevant (matched >=1 reference passage)?  This is a per-chunk
        # measure, so it always stays between 0 and 100%.  Do not switch this to
        # found_gold / retrieved_chunks: that compares two different populations
        # (number of reference passages vs number of chunks) and can exceed 100%
        # whenever reference passages outnumber retrieved chunks.
        relevant_chunk_idxs: set[int] = set()
        for chunk_idx, chunk in enumerate(chunks):
            matched = any(
                self._text_overlap(chunk.content, gold)[0]
                for gold in self.gold_standard
            )
            if not matched and semantic_matrix is not None:
                matched = bool(
                    semantic_matrix[:, chunk_idx].max() >= self.semantic_threshold
                )
            if matched:
                relevant_chunk_idxs.add(chunk_idx)
        relevant_retrieved = len(relevant_chunk_idxs)

        return QueryResult(
            name=name,
            query=query[:200] + "..." if len(query) > 200 else query,
            recall=found / total if total else 0,
            precision=relevant_retrieved / retrieved if retrieved else 0,
            found=found,
            total_gold=total,
            retrieved=retrieved,
            relevant_retrieved=relevant_retrieved,
            found_texts=found_texts,
            missed_texts=missed_texts,
            retrieved_chunks=chunks,
            match_details=match_details,
            miss_details=miss_details,
        )

    @staticmethod
    def _cosine_matrix(
        gold_vecs: list[list[float]], chunk_vecs: list[list[float]]
    ) -> np.ndarray:
        """Return (n_gold, n_chunks) cosine-similarity matrix."""
        g = np.asarray(gold_vecs, dtype=np.float32)
        c = np.asarray(chunk_vecs, dtype=np.float32)

        g_norm = np.linalg.norm(g, axis=1, keepdims=True)
        c_norm = np.linalg.norm(c, axis=1, keepdims=True)

        # Avoid division by zero
        g_norm = np.where(g_norm == 0, 1.0, g_norm)
        c_norm = np.where(c_norm == 0, 1.0, c_norm)

        return (g / g_norm) @ (c / c_norm).T

    async def run_query(
        self, rag: HDJRag, name: str, query: str, limit: int = 20
    ) -> QueryResult:
        """Run query through RAG and evaluate results.

        Validation measures *raw* retrieval quality, so the citation filter is
        disabled here (it is a Search-tab display preference, not part of what we
        are scoring). The query↔passage relevance is derived from the chunk
        embeddings we already compute for the semantic-similarity matrix, so no
        extra embedding work is sent to Ollama.
        """
        results = await rag.search(
            query, limit=limit, drop_references=False, with_relevance=False
        )
        chunks = [
            RetrievedChunk(
                content=r["content"],
                score=r.get("score", 0.0),
                rank=i + 1,
                document_uri=r.get("document_uri"),
                page_numbers=r.get("page_numbers", []),
            )
            for i, r in enumerate(results)
        ]
        retrieved_texts = [c.content for c in chunks]

        # Compute the gold×chunk semantic matrix, and reuse the same chunk
        # embeddings to fill each chunk's query↔passage relevance (cosine) —
        # one Ollama round-trip covers both.
        semantic_matrix = None
        if chunks:
            try:
                # Cache gold embeddings across queries
                if self._gold_embeddings is None:
                    self._gold_embeddings = await rag.embed_texts(
                        self.gold_standard
                    )
                embeddings = await rag.embed_texts([query] + retrieved_texts)
                query_embedding, chunk_embeddings = embeddings[0], embeddings[1:]
                semantic_matrix = self._cosine_matrix(
                    self._gold_embeddings, chunk_embeddings
                )
                query_sims = self._cosine_matrix([query_embedding], chunk_embeddings)[0]
                for chunk, sim in zip(chunks, query_sims):
                    chunk.query_similarity = float(sim)
            except Exception:
                logger.warning(
                    "Could not compute semantic similarity (is Ollama running?)",
                    exc_info=True,
                )

        return self.evaluate(
            name,
            query,
            retrieved_texts,
            retrieved_chunks=chunks,
            semantic_matrix=semantic_matrix,
        )


def save_results(results: list[QueryResult], output_dir: Path) -> Path:
    """Save results with timestamp. Returns path to saved file."""
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}.json"

    best = max(results, key=lambda x: x.recall)
    data = {
        "timestamp": timestamp,
        "summary": {
            "best_query": best.name,
            "best_recall": round(best.recall, 3),
            "total_queries": len(results),
        },
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output_path
