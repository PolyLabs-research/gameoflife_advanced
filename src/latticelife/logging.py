"""JSONL event logging (§6.10): one event per line to `out/logs/app.jsonl` or
`out/logs/discovery.jsonl`. `print()` does not appear in `src/` outside `cli.py`'s user-facing
output — this is the only other place `src/` writes anything, and it writes structured JSONL.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

_LOG_PATHS = {
    "app": Path("out/logs/app.jsonl"),
    "discovery": Path("out/logs/discovery.jsonl"),
}


def log_event(channel: str, level: str, event: str, **fields: object) -> None:
    path = _LOG_PATHS[channel]
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(UTC).isoformat(), "level": level, "event": event, **fields}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
