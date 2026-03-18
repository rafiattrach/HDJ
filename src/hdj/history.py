"""Evaluation history — load, compare, and track evaluation runs."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """Summary of a single evaluation run."""

    timestamp: str
    filepath: Path
    query_names: list[str]
    recalls: dict[str, float]  # query_name -> recall
    precisions: dict[str, float]  # query_name -> precision
    found_counts: dict[str, int]  # query_name -> found count
    total_gold: int
    best_query: str
    best_recall: float


@dataclass
class RunDiff:
    """Difference between two evaluation runs."""

    recall_changes: dict[str, float]  # query_name -> delta
    precision_changes: dict[str, float]  # query_name -> delta
    gained: dict[str, list[str]]  # query_name -> newly found passage previews
    lost: dict[str, list[str]]  # query_name -> newly missed passage previews
    queries_added: list[str]
    queries_removed: list[str]
    total_gold_changed: int  # new - old (0 means unchanged)


@dataclass
class GoldSectionTracker:
    """Tracks how often a reference passage is found across runs."""

    preview: str
    found_by: dict[str, list[str]] = field(default_factory=dict)  # query -> timestamps
    missed_by: dict[str, list[str]] = field(default_factory=dict)  # query -> timestamps
    ever_found: bool = False
    find_rate: float = 0.0  # found_count / total_appearances


def format_timestamp(ts: str) -> str:
    """Format '20260302_092720' as '2026-03-02 09:27'.

    Returns the input unchanged if it doesn't match the expected format.
    """
    if len(ts) < 13 or ts[8:9] != "_":
        return ts
    try:
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"
    except (IndexError, ValueError):
        return ts


def load_run(filepath: Path) -> RunSummary:
    """Load a single evaluation result file into a RunSummary."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    timestamp = data.get("timestamp", filepath.stem)
    summary = data.get("summary", {})
    results = data.get("results", [])

    query_names = []
    recalls: dict[str, float] = {}
    precisions: dict[str, float] = {}
    found_counts: dict[str, int] = {}
    total_gold = 0

    for r in results:
        name = r.get("name", r.get("query", "unknown"))
        query_names.append(name)
        recalls[name] = r.get("recall", 0.0)
        precisions[name] = r.get("precision", 0.0)
        found_counts[name] = r.get("found", 0)
        total_gold = r.get("total_gold", total_gold)

    return RunSummary(
        timestamp=timestamp,
        filepath=filepath,
        query_names=query_names,
        recalls=recalls,
        precisions=precisions,
        found_counts=found_counts,
        total_gold=total_gold,
        best_query=summary.get("best_query", ""),
        best_recall=summary.get("best_recall", 0.0),
    )


def load_all_runs(results_dir: Path) -> list[RunSummary]:
    """Load all evaluation runs from a directory, sorted chronologically.

    Skips malformed files and non-eval JSON files.
    """
    runs: list[RunSummary] = []
    if not results_dir.is_dir():
        return runs

    for fp in sorted(results_dir.glob("eval_*.json")):
        try:
            runs.append(load_run(fp))
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Skipping malformed file: %s", fp)
    return runs


def _load_result_details(filepath: Path) -> list[dict]:
    """Load the results array from an eval file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def diff_runs(results_dir: Path, ts_a: str, ts_b: str) -> RunDiff:
    """Compare two runs identified by timestamp strings.

    *ts_a* is the baseline, *ts_b* is the comparison.
    """
    runs = load_all_runs(results_dir)
    run_map = {r.timestamp: r for r in runs}

    run_a = run_map[ts_a]
    run_b = run_map[ts_b]

    details_a = _load_result_details(run_a.filepath)
    details_b = _load_result_details(run_b.filepath)

    found_map_a: dict[str, set[str]] = {}
    for r in details_a:
        name = r.get("name", r.get("query", "unknown"))
        found_map_a[name] = set(r.get("found_texts", []))

    found_map_b: dict[str, set[str]] = {}
    for r in details_b:
        name = r.get("name", r.get("query", "unknown"))
        found_map_b[name] = set(r.get("found_texts", []))

    all_queries = set(run_a.query_names) | set(run_b.query_names)
    queries_added = sorted(set(run_b.query_names) - set(run_a.query_names))
    queries_removed = sorted(set(run_a.query_names) - set(run_b.query_names))

    recall_changes: dict[str, float] = {}
    precision_changes: dict[str, float] = {}
    gained: dict[str, list[str]] = {}
    lost: dict[str, list[str]] = {}

    for q in all_queries:
        recall_a = run_a.recalls.get(q, 0.0)
        recall_b = run_b.recalls.get(q, 0.0)
        recall_changes[q] = round(recall_b - recall_a, 6)

        prec_a = run_a.precisions.get(q, 0.0)
        prec_b = run_b.precisions.get(q, 0.0)
        precision_changes[q] = round(prec_b - prec_a, 6)

        found_a = found_map_a.get(q, set())
        found_b = found_map_b.get(q, set())
        gained[q] = sorted(found_b - found_a)
        lost[q] = sorted(found_a - found_b)

    total_gold_changed = run_b.total_gold - run_a.total_gold

    return RunDiff(
        recall_changes=recall_changes,
        precision_changes=precision_changes,
        gained=gained,
        lost=lost,
        queries_added=queries_added,
        queries_removed=queries_removed,
        total_gold_changed=total_gold_changed,
    )


def aggregate_query_performance(
    results_dir: Path, runs: list[RunSummary] | None = None
) -> list[GoldSectionTracker]:
    """Track which reference passages are found/missed across runs.

    Returns trackers sorted by find_rate ascending (hardest first).
    """
    if runs is None:
        runs = load_all_runs(results_dir)

    # passage_preview -> GoldSectionTracker
    trackers: dict[str, GoldSectionTracker] = {}

    for run in runs:
        details = _load_result_details(run.filepath)
        for r in details:
            query_name = r.get("name", r.get("query", "unknown"))
            for preview in r.get("found_texts", []):
                if preview not in trackers:
                    trackers[preview] = GoldSectionTracker(preview=preview)
                t = trackers[preview]
                t.ever_found = True
                t.found_by.setdefault(query_name, []).append(run.timestamp)

            for preview in r.get("missed_texts", []):
                if preview not in trackers:
                    trackers[preview] = GoldSectionTracker(preview=preview)
                t = trackers[preview]
                t.missed_by.setdefault(query_name, []).append(run.timestamp)

    # Compute find_rate for each tracker
    for t in trackers.values():
        total_found = sum(len(ts) for ts in t.found_by.values())
        total_missed = sum(len(ts) for ts in t.missed_by.values())
        total = total_found + total_missed
        t.find_rate = total_found / total if total > 0 else 0.0

    return sorted(trackers.values(), key=lambda t: t.find_rate)
