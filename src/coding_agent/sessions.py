"""Conversation sessions: save / resume / list.

Every run writes its full conversation history to a JSON file, so the work is
auditable later and can be picked back up with `--resume`. The history is plain
JSON (roles, content, tool_calls, tool results), so no extra dependencies are
needed and the files stay human-readable.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

SESSION_VERSION = 1


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def save_session(history: list[dict], task: str, final_answer: str,
                 logs_dir: Path, path: str | Path | None = None) -> Path:
    """Write `history` (plus a little metadata) to a JSON file.

    Defaults to `<logs_dir>/session-<timestamp>.json`; pass `path` to override.
    Returns the path that was written.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    dest = Path(path) if path else logs_dir / f"session-{_timestamp()}.json"
    record = {
        "version": SESSION_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "final_answer": final_answer,
        "history": history,
    }
    dest.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return dest


def load_session(path: str | Path) -> dict:
    """Read a session file back. Raises on a missing or malformed file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_sessions(logs_dir: Path) -> list[Path]:
    """Return saved session files, newest first (filenames are timestamped)."""
    logs_dir = Path(logs_dir)
    if not logs_dir.is_dir():
        return []
    return sorted(logs_dir.glob("session-*.json"), reverse=True)
