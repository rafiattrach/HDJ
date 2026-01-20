"""Evaluation logic for RAG retrieval quality."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .rag import HDJRag


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
    found_texts: list[str]
    missed_texts: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class Evaluator:
    """Evaluates RAG retrieval against gold standard."""

    def __init__(self, gold_standard: list[str], overlap_threshold: float = 0.3):
        self.gold_standard = gold_standard
        self.overlap_threshold = overlap_threshold

    @classmethod
    def from_json(cls, path: Path, overlap_threshold: float = 0.3) -> "Evaluator":
        """Load gold standard from JSON file."""
        with open(path) as f:
            data = json.load(f)
        texts = [item["text"] for item in data]
        return cls(texts, overlap_threshold)

    def _text_overlap(self, retrieved: str, gold: str) -> bool:
        """Check if texts overlap significantly."""
        retrieved_lower = retrieved.lower()
        gold_lower = gold.lower()

        if gold_lower in retrieved_lower or retrieved_lower in gold_lower:
            return True

        gold_words = set(gold_lower.split())
        retrieved_words = set(retrieved_lower.split())

        if not gold_words:
            return False

        overlap = len(gold_words & retrieved_words) / len(gold_words)
        return overlap >= self.overlap_threshold

    def evaluate(self, name: str, query: str, retrieved_texts: list[str]) -> QueryResult:
        """Evaluate retrieved texts against gold standard."""
        found_texts = []
        missed_texts = []

        for gold in self.gold_standard:
            match = any(self._text_overlap(ret, gold) for ret in retrieved_texts)
            truncated = gold[:150] + "..." if len(gold) > 150 else gold
            if match:
                found_texts.append(truncated)
            else:
                missed_texts.append(truncated)

        found = len(found_texts)
        total = len(self.gold_standard)
        retrieved = len(retrieved_texts)

        return QueryResult(
            name=name,
            query=query[:200] + "..." if len(query) > 200 else query,
            recall=found / total if total else 0,
            precision=found / retrieved if retrieved else 0,
            found=found,
            total_gold=total,
            retrieved=retrieved,
            found_texts=found_texts,
            missed_texts=missed_texts,
        )

    async def run_query(
        self, rag: HDJRag, name: str, query: str, limit: int = 20
    ) -> QueryResult:
        """Run query through RAG and evaluate results."""
        results = await rag.search(query, limit=limit)
        retrieved_texts = [r["content"] for r in results]
        return self.evaluate(name, query, retrieved_texts)


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

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path
