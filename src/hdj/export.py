"""Markdown report generator for HDJ RAG evaluation sessions."""

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, unquote


def _pretty_doc_name(document_uri: str | None) -> str:
    if not document_uri:
        return "Unknown PDF"
    try:
        parsed = urlparse(document_uri)
        if parsed.scheme in {"file"} and parsed.path:
            return Path(unquote(parsed.path)).name or document_uri
        if parsed.scheme == "" and document_uri:
            return Path(document_uri).name or document_uri
        return Path(unquote(parsed.path)).name or document_uri
    except Exception:
        return document_uri


def generate_report(
    results: list,
    queries: dict[str, str],
    gold_standard: list[dict],
    config: dict,
    audit_events: list[dict] | None = None,
) -> str:
    """Generate a self-contained Markdown evaluation report.

    Parameters
    ----------
    results : list[QueryResult]
        Evaluation results (must have .name, .query, .recall, .precision,
        .found, .total_gold, .missed_texts attributes).
    queries : dict[str, str]
        Mapping of query name → query text.
    gold_standard : list[dict]
        Gold standard entries, each with ``source_file`` and ``text`` keys.
    config : dict
        Keys: ``embedding_model``, ``chunk_size``, ``search_method``,
        ``results_limit``, ``overlap_threshold``, ``indexed_pdfs``.
    audit_events : list[dict] | None
        Optional audit trail entries to include at the bottom.
    """
    lines: list[str] = []

    embedding_model = config.get("embedding_model", "Qwen3-Embedding-4B (via Ollama)")
    chunk_size = config.get("chunk_size", 512)
    search_method = config.get("search_method", "Hybrid (vector + full-text with RRF reranking)")
    results_limit = config.get("results_limit", 20)
    overlap_threshold = config.get("overlap_threshold", 0.3)
    threshold_pct = int(overlap_threshold * 100)
    semantic_threshold = config.get("semantic_threshold", 0.75)
    semantic_pct = int(semantic_threshold * 100)

    # Header
    lines.append("# Search Validation Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append(f"- Embedding model: {embedding_model}")
    lines.append(f"- Passage size: ~{chunk_size} words")
    lines.append(f"- Search: {search_method}")
    lines.append(f"- Results per question: {results_limit}")
    lines.append(f"- Match strictness: {overlap_threshold} ({threshold_pct}% word match)")
    lines.append(f"- Meaning similarity threshold: {semantic_threshold} ({semantic_pct}% meaning similarity)")
    lines.append("")

    # How Results Were Determined
    lines.append("## How Results Were Determined")
    lines.append("")
    lines.append("How each search result was evaluated:")
    lines.append("")
    lines.append(f"1. **Preparation**: Each PDF is split into short passages (~{chunk_size} words) and indexed for search")
    lines.append("2. **Indexing**: Passages are stored in a local search database")
    lines.append(f"3. **Search**: {search_method}, returning top {results_limit} results per question")
    lines.append(f"4. **Matching**: Each reference passage is compared to every retrieved passage using word-level matching (common words filtered out)")
    lines.append(f"5. **Threshold**: A reference passage counts as \"found\" if ≥{threshold_pct}% of its meaningful words appear in a retrieved passage (or it is an exact text match)")
    lines.append(f"6. **Meaning similarity**: Meaning similarity between reference and retrieved passages is reported alongside word matching")
    lines.append(f"7. **Cross-lingual matching**: When word matching fails but meaning similarity ≥{semantic_pct}%, the passage is counted as found via meaning match (enables German ↔ English validation)")
    lines.append("")
    lines.append("**Interpreting scores:**")
    lines.append("- **Coverage** = proportion of reference passages that were found among retrieved results")
    lines.append("- **Accuracy** = proportion of retrieved passages that matched at least one reference passage")
    lines.append("")

    # Documents indexed
    indexed = config.get("indexed_pdfs", [])
    lines.append("## Documents Indexed")
    if indexed:
        for pdf in indexed:
            lines.append(f"- {pdf}")
    else:
        lines.append("- (none)")
    lines.append("")

    # Reference Passages
    lines.append(f"## Reference Passages ({len(gold_standard)} passages)")
    lines.append("")

    by_file: dict[str, list[dict]] = {}
    for entry in gold_standard:
        src = entry.get("source_file", "unknown")
        by_file.setdefault(src, []).append(entry)

    for filename, entries in by_file.items():
        lines.append(f"### {filename} ({len(entries)} passages)")
        for i, entry in enumerate(entries, 1):
            text = entry.get("text", "")
            preview = text[:120].replace("\n", " ")
            if len(text) > 120:
                preview += "..."
            lines.append(f'{i}. "{preview}"')
        lines.append("")

    # Results summary table
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Question | Coverage | Accuracy | Found |")
    lines.append("|----------|----------|----------|-------|")

    sorted_results = sorted(results, key=lambda r: r.recall, reverse=True)
    for r in sorted_results:
        lines.append(
            f"| {r.name} | {r.recall:.0%} | {r.precision:.0%} | {r.found}/{r.total_gold} |"
        )

    lines.append("")
    if sorted_results:
        best = sorted_results[0]
        lines.append(f"Best question: {best.name} ({best.recall:.0%} coverage)")
    lines.append("")

    # Per-question details
    lines.append("## Question Details")
    lines.append("")

    for r in sorted_results:
        lines.append(f"### {r.name} ({r.recall:.0%} coverage, {r.precision:.0%} accuracy)")
        lines.append("**Question text:**")
        query_text = queries.get(r.name, r.query)
        lines.append(f"> {query_text}")
        lines.append("")
        lines.append(f"**Found {r.found} of {r.total_gold} reference passages.**")
        lines.append("")

        # Retrieved chunks table
        if r.retrieved_chunks:
            lines.append("**Top retrieved passages:**")
            lines.append("")
            lines.append("| Rank | Score | Document | Pages |")
            lines.append("|------|-------|----------|-------|")
            for chunk in r.retrieved_chunks[:10]:
                doc = _pretty_doc_name(chunk.document_uri)
                pages = ", ".join(map(str, chunk.page_numbers)) if chunk.page_numbers else "—"
                lines.append(f"| #{chunk.rank} | {chunk.score:.1%} | {doc} | {pages} |")
            lines.append("")

        # Missed sections with diagnosis
        if r.miss_details:
            lines.append("**Missed passages — why they weren't found:**")
            lines.append("")
            for i, detail in enumerate(r.miss_details, 1):
                preview = detail.gold_text_preview.replace("\n", " ")
                lines.append(f'{i}. "{preview}"')
                if detail.matched_chunk:
                    chunk = detail.matched_chunk
                    doc = _pretty_doc_name(chunk.document_uri)
                    overlap_pct = int(detail.overlap_ratio * 100)
                    gap = threshold_pct - overlap_pct
                    lines.append(
                        f"   - Nearest passage: #{chunk.rank} (score {chunk.score:.1%}) from {doc}"
                    )
                    lines.append(
                        f"   - {overlap_pct}% word overlap ({len(detail.overlapping_words)} content words) "
                        f"is {gap}% below the {threshold_pct}% threshold"
                    )
                    sem = getattr(detail, "semantic_similarity", 0.0)
                    if sem > 0:
                        lines.append(
                            f"   - Meaning similarity: {sem:.0%}"
                        )
                else:
                    lines.append("   - No retrieved passages to compare")
            lines.append("")
        elif r.missed_texts:
            lines.append("Missed:")
            for text in r.missed_texts:
                preview = text[:120].replace("\n", " ")
                if len(text) > 120:
                    preview += "..."
                lines.append(f'- "{preview}"')
            lines.append("")

    # Audit trail (optional)
    if audit_events:
        lines.append("---")
        lines.append("")
        lines.append("## Session Activity")
        lines.append("")
        for event in audit_events:
            ts = event.get("timestamp", "")
            action = event.get("action", "")
            details = event.get("details", {})
            detail_str = ", ".join(f"{k}: {v}" for k, v in details.items()) if details else ""
            if detail_str:
                lines.append(f"- **{ts}** — {action} ({detail_str})")
            else:
                lines.append(f"- **{ts}** — {action}")
        lines.append("")

    return "\n".join(lines)
