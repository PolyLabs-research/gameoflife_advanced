"""JSONL event logging (§6.10): one event per line to `app.jsonl` or `discovery.jsonl` in the log
directory. `print()` does not appear in `src/` outside `cli.py`'s user-facing output — this is the
only other place `src/` writes anything, and it writes structured JSONL.

The log directory is, in order: `$LATTICELIFE_LOG_DIR` if set (the test suite points it at a
temporary directory so test runs never touch the real `out/logs`); else the directory passed to
`configure_logging()`, which commands call with the loaded config's `logging.dir`; else `out/logs`
under the project root — never the current directory, so a command started elsewhere still logs
next to its outputs. A log write that fails (unwritable directory, full disk) drops that event
rather than raising, so logging an error can never replace the error being logged.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from latticelife.config import PROJECT_ROOT

LOG_DIR_ENV = "LATTICELIFE_LOG_DIR"
DEFAULT_LOG_DIR = PROJECT_ROOT / "out" / "logs"

_LOG_FILES = {
    "app": "app.jsonl",
    "discovery": "discovery.jsonl",
}

_configured_dir: Path | None = None


def configure_logging(log_dir: str | Path | None) -> None:
    """Set this process's log directory (None restores the default). A discovery worker
    process calls it again itself, since a spawned child does not inherit the setting."""
    global _configured_dir
    _configured_dir = Path(log_dir) if log_dir is not None else None


def log_dir() -> Path:
    env_dir = os.environ.get(LOG_DIR_ENV)
    if env_dir:
        return Path(env_dir)
    if _configured_dir is not None:
        return _configured_dir
    return DEFAULT_LOG_DIR


def log_path(channel: str) -> Path:
    return log_dir() / _LOG_FILES[channel]


def log_event(channel: str, level: str, event: str, **fields: object) -> None:
    path = log_path(channel)
    record = {"ts": datetime.now(UTC).isoformat(), "level": level, "event": event, **fields}
    line = json.dumps(record, default=str) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(line)
    except OSError:
        return
