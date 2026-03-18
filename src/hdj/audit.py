"""Append-only audit trail for HDJ RAG actions."""

import json
from datetime import datetime
from pathlib import Path


def _migrate_legacy_format(path: Path) -> None:
    """Convert a legacy JSON-array audit file to JSONL in place."""
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content or not content.startswith("["):
        return
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return
    if not isinstance(data, list):
        return
    with open(path, "w", encoding="utf-8") as f:
        for event in data:
            f.write(json.dumps(event) + "\n")


def log_event(path: Path, action: str, details: dict | None = None) -> None:
    """Append one event to the audit log file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Migrate legacy JSON-array format before first append
    if path.exists():
        _migrate_legacy_format(path)
    entry = json.dumps({
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details or {},
    })
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def load_events(path: Path) -> list[dict]:
    """Read the full audit log. Returns an empty list if the file doesn't exist.

    Supports both the current JSONL format (one JSON object per line) and
    the legacy format (a single JSON array).
    """
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    # Legacy format: entire file is a JSON array
    if content.startswith("["):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    # Current format: one JSON object per line (JSONL)
    events = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def clear_events(path: Path) -> None:
    """Reset the audit log."""
    path = Path(path)
    if path.exists():
        path.unlink()
