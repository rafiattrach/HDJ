"""Append-only audit trail for HDJ RAG actions."""

import json
from datetime import datetime
from pathlib import Path


def log_event(path: Path, action: str, details: dict | None = None) -> None:
    """Append one event to the audit log file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    events = load_events(path)
    events.append({
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details or {},
    })

    with open(path, "w") as f:
        json.dump(events, f, indent=2)


def load_events(path: Path) -> list[dict]:
    """Read the full audit log. Returns an empty list if the file doesn't exist."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def clear_events(path: Path) -> None:
    """Reset the audit log."""
    path = Path(path)
    if path.exists():
        with open(path, "w") as f:
            json.dump([], f)
