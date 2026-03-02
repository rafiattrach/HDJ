#!/usr/bin/env python3
"""CLI for running RAG evaluation."""

import argparse
import asyncio
import json
from pathlib import Path

from src.hdj import HDJRag, Evaluator, QueryResult
from src.hdj.evaluate import save_results

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PDFS_DIR = DATA_DIR / "pdfs"
GOLD_STANDARD_PATH = DATA_DIR / "gold_standard.json"
QUERIES_PATH = ROOT / "queries.json"
RESULTS_DIR = ROOT / "results"
DB_PATH = ROOT / "hdj.lancedb"


def load_queries() -> dict[str, str]:
    """Load queries from queries.json."""
    if not QUERIES_PATH.exists():
        return {}
    with open(QUERIES_PATH) as f:
        return json.load(f)


def _pretty_name(uri: str | None) -> str:
    if not uri:
        return "?"
    from pathlib import PurePosixPath
    return PurePosixPath(uri).name or uri


def print_result(r: QueryResult) -> None:
    """Print a single query result."""
    print(f"\n{'─' * 50}")
    print(f"Query: {r.name}")
    print(f"  Recall:    {r.found}/{r.total_gold} ({r.recall:.1%})")
    print(f"  Precision: {r.found}/{r.retrieved} ({r.precision:.1%})")

    if r.miss_details:
        print(f"\n  Missed ({len(r.miss_details)}):")
        for i, detail in enumerate(r.miss_details[:5]):
            preview = detail.gold_text_preview[:80].replace("\n", " ")
            print(f"    {i+1}. {preview}...")
            if detail.matched_chunk:
                chunk = detail.matched_chunk
                doc = _pretty_name(chunk.document_uri)
                sem = f", {detail.semantic_similarity:.0%} semantic" if detail.semantic_similarity > 0 else ""
                print(
                    f"       Nearest: #{chunk.rank} ({chunk.score:.1%}) from {doc} "
                    f"— {detail.overlap_ratio:.0%} overlap ({len(detail.overlapping_words)} words){sem}"
                )
        if len(r.miss_details) > 5:
            print(f"    ... and {len(r.miss_details) - 5} more")
    elif r.missed_texts:
        print(f"\n  Missed ({len(r.missed_texts)}):")
        for t in r.missed_texts[:3]:
            print(f"    - {t[:80]}...")
        if len(r.missed_texts) > 3:
            print(f"    ... and {len(r.missed_texts) - 3} more")


async def main():
    parser = argparse.ArgumentParser(description="HDJ RAG Evaluation")
    parser.add_argument("--query", "-q", help="Run a single custom query")
    parser.add_argument("--reindex", action="store_true", help="Force re-index all PDFs")
    parser.add_argument("--limit", type=int, default=20, help="Max results per query")
    args = parser.parse_args()

    print("=" * 50)
    print("Health Data Justice - RAG Evaluation")
    print("=" * 50)

    # Load evaluator
    evaluator = Evaluator.from_json(GOLD_STANDARD_PATH)
    print(f"\nGold standard: {len(evaluator.gold_standard)} sections")

    # Build queries
    if args.query:
        queries = {"custom": args.query}
    else:
        queries = load_queries()
        if not queries:
            print("\nNo queries found. Create queries.json or use --query flag.")
            return

    async with HDJRag(DB_PATH) as rag:
        # Index documents
        count = await rag.index_pdfs(PDFS_DIR, force=args.reindex)
        print(f"Indexed documents: {count}")

        # Run queries
        print(f"\nRunning {len(queries)} queries...")
        results = []
        for name, query in queries.items():
            result = await evaluator.run_query(rag, name, query, args.limit)
            results.append(result)
            print_result(result)

        # Summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)

        sorted_results = sorted(results, key=lambda x: x.recall, reverse=True)
        for r in sorted_results:
            print(f"  {r.recall:.0%} recall | {r.precision:.0%} precision | {r.name}")

        best = sorted_results[0]
        print(f"\nBest: '{best.name}' with {best.recall:.1%} recall")

        # Save
        output_path = save_results(results, RESULTS_DIR)
        print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
