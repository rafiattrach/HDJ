"""Tests for the audit trail module."""

import json
import pytest

from hdj.audit import log_event, load_events, clear_events


class TestAuditTrail:
    def test_log_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "audit.json"
        log_event(path, "index_built", {"pdfs": ["a.pdf", "b.pdf"]})
        events = load_events(path)
        assert len(events) == 1
        assert events[0]["action"] == "index_built"
        assert events[0]["details"]["pdfs"] == ["a.pdf", "b.pdf"]
        assert "timestamp" in events[0]

    def test_multiple_events_accumulate(self, tmp_path):
        path = tmp_path / "audit.json"
        log_event(path, "first")
        log_event(path, "second")
        log_event(path, "third")
        events = load_events(path)
        assert len(events) == 3
        assert [e["action"] for e in events] == ["first", "second", "third"]

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        events = load_events(path)
        assert events == []

    def test_load_corrupted_json_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{{{invalid json")
        events = load_events(path)
        assert events == []

    def test_clear_events(self, tmp_path):
        path = tmp_path / "audit.json"
        log_event(path, "action_one")
        log_event(path, "action_two")
        assert len(load_events(path)) == 2

        clear_events(path)
        events = load_events(path)
        assert events == []

    def test_clear_nonexistent_file(self, tmp_path):
        """Clearing a file that doesn't exist should not raise."""
        path = tmp_path / "nonexistent.json"
        clear_events(path)  # Should not raise

    def test_log_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "audit.json"
        log_event(path, "test")
        assert path.exists()
        assert len(load_events(path)) == 1

    def test_details_default_to_empty_dict(self, tmp_path):
        path = tmp_path / "audit.json"
        log_event(path, "no_details")
        events = load_events(path)
        assert events[0]["details"] == {}

    def test_timestamp_is_iso_format(self, tmp_path):
        path = tmp_path / "audit.json"
        log_event(path, "check_ts")
        from datetime import datetime
        ts = load_events(path)[0]["timestamp"]
        # Should parse without error
        datetime.fromisoformat(ts)
