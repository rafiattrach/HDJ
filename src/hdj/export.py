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

    # Header
    lines.append("# HDJ RAG Evaluation Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append(f"- Embedding model: {embedding_model}")
    lines.append(f"- Chunk size: {chunk_size} tokens")
    lines.append(f"- Search: {search_method}")
    lines.append(f"- Results per query: {results_limit}")
    lines.append(f"- Gold match strictness: {overlap_threshold} ({threshold_pct}% word overlap)")
    lines.append("")

    # Decision Provenance
    lines.append("## Decision Provenance")
    lines.append("")
    lines.append("How each retrieval decision was made:")
    lines.append("")
    lines.append(f"1. **Embedding**: Text is embedded with `{embedding_model}` into {chunk_size}-token chunks")
    lines.append("2. **Indexing**: Chunks are stored in a LanceDB vector database")
    lines.append(f"3. **Search**: {search_method}, returning top {results_limit} results per query")
    lines.append(f"4. **Matching**: Each gold passage is compared to every retrieved chunk using word-level Jaccard overlap")
    lines.append(f"5. **Threshold**: A gold passage counts as \"found\" if ≥{threshold_pct}% of its words appear in a retrieved chunk (or it is a substring match)")
    lines.append("")
    lines.append("**Interpreting scores:**")
    lines.append("- **Recall** = proportion of gold-standard passages that were found among retrieved chunks")
    lines.append("- **Precision** = proportion of retrieved chunks that matched at least one gold passage")
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

    # Gold standard
    lines.append(f"## Gold Standard ({len(gold_standard)} sections)")
    lines.append("")

    by_file: dict[str, list[dict]] = {}
    for entry in gold_standard:
        src = entry.get("source_file", "unknown")
        by_file.setdefault(src, []).append(entry)

    for filename, entries in by_file.items():
        lines.append(f"### {filename} ({len(entries)} sections)")
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
    lines.append("| Query | Recall | Precision | Found |")
    lines.append("|-------|--------|-----------|-------|")

    sorted_results = sorted(results, key=lambda r: r.recall, reverse=True)
    for r in sorted_results:
        lines.append(
            f"| {r.name} | {r.recall:.0%} | {r.precision:.0%} | {r.found}/{r.total_gold} |"
        )

    lines.append("")
    if sorted_results:
        best = sorted_results[0]
        lines.append(f"Best query: {best.name} ({best.recall:.0%} recall)")
    lines.append("")

    # Per-query details
    lines.append("## Query Details")
    lines.append("")

    for r in sorted_results:
        lines.append(f"### {r.name} ({r.recall:.0%} recall, {r.precision:.0%} precision)")
        lines.append("**Query text:**")
        query_text = queries.get(r.name, r.query)
        lines.append(f"> {query_text}")
        lines.append("")
        lines.append(f"**Found {r.found} of {r.total_gold} gold sections.**")
        lines.append("")

        # Retrieved chunks table
        if r.retrieved_chunks:
            lines.append("**Top retrieved chunks:**")
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
            lines.append("**Missed sections with diagnosis:**")
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
                        f"   - Nearest chunk: #{chunk.rank} (score {chunk.score:.1%}) from {doc}"
                    )
                    lines.append(
                        f"   - {overlap_pct}% overlap ({len(detail.overlapping_words)} words) "
                        f"is {gap}% below the {threshold_pct}% threshold"
                    )
                else:
                    lines.append("   - No retrieved chunks to compare")
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
