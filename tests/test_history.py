"""Tests for evaluation history — loading, diffing, and tracking runs."""

import json
import pytest

from hdj.history import (
    RunSummary,
    RunDiff,
    GoldSectionTracker,
    load_run,
    load_all_runs,
    diff_runs,
    aggregate_query_performance,
    format_timestamp,
)


def _make_result(name, recall, precision, found_texts, missed_texts, total_gold=10):
    return {
        "name": name,
        "query": f"query for {name}",
        "recall": recall,
        "precision": precision,
        "found": len(found_texts),
        "total_gold": total_gold,
        "retrieved": 20,
        "found_texts": found_texts,
        "missed_texts": missed_texts,
    }


@pytest.fixture
def write_result(tmp_path):
    """Factory fixture: write an eval JSON file and return its path."""

    def _write(timestamp, results, best_query=None, best_recall=None):
        if best_query is None:
            best_query = max(results, key=lambda r: r["recall"])["name"]
        if best_recall is None:
            best_recall = max(r["recall"] for r in results)

        data = {
            "timestamp": timestamp,
            "summary": {
                "best_query": best_query,
                "best_recall": best_recall,
                "total_queries": len(results),
            },
            "results": results,
        }
        fp = tmp_path / f"eval_{timestamp}.json"
        fp.write_text(json.dumps(data))
        return fp

    return _write


# ---------------------------------------------------------------------------
# TestLoadRun
# ---------------------------------------------------------------------------

class TestLoadRun:
    def test_basic_fields(self, write_result):
        fp = write_result("20260302_090000", [
            _make_result("q1", 0.5, 0.3, ["a", "b"], ["c"]),
        ])
        run = load_run(fp)
        assert run.timestamp == "20260302_090000"
        assert run.filepath == fp
        assert run.query_names == ["q1"]
        assert run.recalls == {"q1": 0.5}
        assert run.precisions == {"q1": 0.3}
        assert run.found_counts == {"q1": 2}
        assert run.best_query == "q1"
        assert run.best_recall == 0.5

    def test_multiple_queries(self, write_result):
        fp = write_result("20260302_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
            _make_result("q2", 0.8, 0.6, ["a", "b", "c"], ["d"]),
        ])
        run = load_run(fp)
        assert len(run.query_names) == 2
        assert run.recalls["q2"] == 0.8

    def test_total_gold_from_results(self, write_result):
        fp = write_result("20260302_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"], total_gold=16),
        ])
        run = load_run(fp)
        assert run.total_gold == 16

    def test_empty_results(self, write_result):
        fp = write_result("20260302_090000", [], best_query="", best_recall=0.0)
        run = load_run(fp)
        assert run.query_names == []
        assert run.recalls == {}
        assert run.total_gold == 0

    def test_old_format_no_chunks(self, tmp_path):
        """Old January format: no retrieved_chunks or match_details."""
        data = {
            "timestamp": "20260120_192047",
            "summary": {"best_query": "q1", "best_recall": 0.8, "total_queries": 1},
            "results": [{
                "name": "q1",
                "query": "some query",
                "recall": 0.8,
                "precision": 0.6,
                "found": 4,
                "total_gold": 5,
                "retrieved": 20,
                "found_texts": ["a", "b", "c", "d"],
                "missed_texts": ["e"],
            }],
        }
        fp = tmp_path / "eval_20260120_192047.json"
        fp.write_text(json.dumps(data))
        run = load_run(fp)
        assert run.best_recall == 0.8
        assert run.recalls["q1"] == 0.8

    def test_malformed_json_raises(self, tmp_path):
        fp = tmp_path / "eval_bad.json"
        fp.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            load_run(fp)

    def test_missing_summary_defaults(self, tmp_path):
        """File with results but no summary section."""
        data = {
            "timestamp": "20260302_100000",
            "results": [
                _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
            ],
        }
        fp = tmp_path / "eval_20260302_100000.json"
        fp.write_text(json.dumps(data))
        run = load_run(fp)
        assert run.best_query == ""
        assert run.best_recall == 0.0


# ---------------------------------------------------------------------------
# TestLoadAllRuns
# ---------------------------------------------------------------------------

class TestLoadAllRuns:
    def test_chronological_order(self, write_result, tmp_path):
        write_result("20260302_090000", [_make_result("q1", 0.5, 0.3, ["a"], ["b"])])
        write_result("20260301_090000", [_make_result("q1", 0.4, 0.2, ["a"], ["b"])])
        write_result("20260303_090000", [_make_result("q1", 0.6, 0.4, ["a"], ["b"])])

        runs = load_all_runs(tmp_path)
        timestamps = [r.timestamp for r in runs]
        assert timestamps == ["20260301_090000", "20260302_090000", "20260303_090000"]

    def test_skips_corrupt_file(self, write_result, tmp_path):
        write_result("20260302_090000", [_make_result("q1", 0.5, 0.3, ["a"], ["b"])])
        (tmp_path / "eval_20260302_100000.json").write_text("broken{")

        runs = load_all_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0].timestamp == "20260302_090000"

    def test_empty_directory(self, tmp_path):
        runs = load_all_runs(tmp_path)
        assert runs == []

    def test_ignores_non_eval_files(self, write_result, tmp_path):
        write_result("20260302_090000", [_make_result("q1", 0.5, 0.3, ["a"], ["b"])])
        (tmp_path / "other.json").write_text('{"not": "eval"}')

        runs = load_all_runs(tmp_path)
        assert len(runs) == 1

    def test_nonexistent_directory(self, tmp_path):
        runs = load_all_runs(tmp_path / "doesnt_exist")
        assert runs == []


# ---------------------------------------------------------------------------
# TestDiffRuns
# ---------------------------------------------------------------------------

class TestDiffRuns:
    def test_recall_deltas(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a", "b"], ["c"]),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.8, 0.5, ["a", "b", "c"], []),
        ])

        diff = diff_runs(tmp_path, "20260301_090000", "20260302_090000")
        assert diff.recall_changes["q1"] == pytest.approx(0.3)

    def test_precision_deltas(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.5, 0.6, ["a"], ["b"]),
        ])

        diff = diff_runs(tmp_path, "20260301_090000", "20260302_090000")
        assert diff.precision_changes["q1"] == pytest.approx(0.3)

    def test_gained_sections(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b", "c"]),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.8, 0.5, ["a", "b"], ["c"]),
        ])

        diff = diff_runs(tmp_path, "20260301_090000", "20260302_090000")
        assert "b" in diff.gained["q1"]

    def test_lost_sections(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a", "b"], ["c"]),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.3, 0.2, ["a"], ["b", "c"]),
        ])

        diff = diff_runs(tmp_path, "20260301_090000", "20260302_090000")
        assert "b" in diff.lost["q1"]

    def test_query_added(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
            _make_result("q2", 0.6, 0.4, ["x"], ["y"]),
        ])

        diff = diff_runs(tmp_path, "20260301_090000", "20260302_090000")
        assert "q2" in diff.queries_added

    def test_query_removed(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
            _make_result("q2", 0.6, 0.4, ["x"], ["y"]),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
        ])

        diff = diff_runs(tmp_path, "20260301_090000", "20260302_090000")
        assert "q2" in diff.queries_removed

    def test_gold_count_change(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"], total_gold=10),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"], total_gold=16),
        ])

        diff = diff_runs(tmp_path, "20260301_090000", "20260302_090000")
        assert diff.total_gold_changed == 6

    def test_no_changes(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
        ])

        diff = diff_runs(tmp_path, "20260301_090000", "20260302_090000")
        assert diff.recall_changes["q1"] == 0.0
        assert diff.gained["q1"] == []
        assert diff.lost["q1"] == []
        assert diff.queries_added == []
        assert diff.queries_removed == []


# ---------------------------------------------------------------------------
# TestAggregateQueryPerformance
# ---------------------------------------------------------------------------

class TestAggregateQueryPerformance:
    def test_never_found(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.0, 0.0, [], ["a", "b"]),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.0, 0.0, [], ["a", "b"]),
        ])

        trackers = aggregate_query_performance(tmp_path)
        for t in trackers:
            assert t.ever_found is False
            assert t.find_rate == 0.0

    def test_always_found(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 1.0, 1.0, ["a", "b"], []),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 1.0, 1.0, ["a", "b"], []),
        ])

        trackers = aggregate_query_performance(tmp_path)
        for t in trackers:
            assert t.ever_found is True
            assert t.find_rate == 1.0

    def test_partial_find_rate(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.5, 0.3, ["b"], ["a"]),
        ])

        trackers = aggregate_query_performance(tmp_path)
        rates = {t.preview: t.find_rate for t in trackers}
        assert rates["a"] == 0.5
        assert rates["b"] == 0.5

    def test_sorted_hardest_first(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["easy", "hard"], ["never"]),
        ])
        write_result("20260302_090000", [
            _make_result("q1", 0.5, 0.3, ["easy"], ["hard", "never"]),
        ])

        trackers = aggregate_query_performance(tmp_path)
        rates = [t.find_rate for t in trackers]
        assert rates == sorted(rates)

    def test_multiple_queries_per_run(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
            _make_result("q2", 1.0, 1.0, ["a", "b"], []),
        ])

        trackers = aggregate_query_performance(tmp_path)
        a_tracker = next(t for t in trackers if t.preview == "a")
        assert "q1" in a_tracker.found_by
        assert "q2" in a_tracker.found_by

    def test_accepts_preloaded_runs(self, write_result, tmp_path):
        write_result("20260301_090000", [
            _make_result("q1", 0.5, 0.3, ["a"], ["b"]),
        ])
        runs = load_all_runs(tmp_path)
        trackers = aggregate_query_performance(tmp_path, runs=runs)
        assert len(trackers) == 2


# ---------------------------------------------------------------------------
# TestFormatTimestamp
# ---------------------------------------------------------------------------

class TestFormatTimestamp:
    def test_standard_format(self):
        assert format_timestamp("20260302_092720") == "2026-03-02 09:27"

    def test_short_passthrough(self):
        assert format_timestamp("2026") == "2026"

    def test_no_underscore_passthrough(self):
        assert format_timestamp("20260302092720") == "20260302092720"
