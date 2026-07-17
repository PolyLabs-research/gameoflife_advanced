"""SQLite schema v2 recorder (§6.9). Single-threaded: discovery is a separate process and never
touches this DB — it writes its own JSONL/checkpoints (§6.11). No `check_same_thread=False`, no
lock. Fixed DDL, no runtime ALTER-TABLE migration code: recording starts fresh on schema v2, the
legacy DB's lifespan-derived columns (D-1) and zero-filled metric columns (D-2) are not migrated
(see the salvage manifest).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latticelife.config import Config
    from latticelife.core.lifeform import Lifeform
    from latticelife.session.session import GenerationRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_metadata (
  session_id TEXT PRIMARY KEY,
  started_at TEXT DEFAULT (datetime('now')),
  grid_width INTEGER, grid_height INTEGER,
  shape TEXT, triangle_mode TEXT, wrap INTEGER,
  initial_alive_percentage REAL, n_lifeforms INTEGER,
  seed INTEGER, planned_generations INTEGER
);
CREATE TABLE IF NOT EXISTS lifeform_meta (
  session_id TEXT, lifeform INTEGER,
  birth TEXT, survive TEXT,
  PRIMARY KEY (session_id, lifeform)
);
CREATE TABLE IF NOT EXISTS generation_metrics (
  session_id TEXT, generation INTEGER,
  total_alive INTEGER, changed_cells INTEGER,
  volatility REAL, entropy REAL,
  PRIMARY KEY (session_id, generation)
);
CREATE TABLE IF NOT EXISTS life_records (
  session_id TEXT, generation INTEGER, lifeform INTEGER,
  alive INTEGER, static INTEGER,
  births INTEGER, deaths INTEGER, kills INTEGER,
  dominance REAL, growth_rate REAL, mobility REAL, avg_lifespan REAL,
  cluster_count INTEGER, avg_cluster_size REAL, max_cluster_size INTEGER,
  PRIMARY KEY (session_id, generation, lifeform)
);
"""


def _rule_string(counts: set[int]) -> str:
    return "".join(str(v) for v in sorted(counts))


class Recorder:
    """Context manager guarantees flush-and-close on any exit, including exceptions that
    unwind normally through Python. Bounded loss: up to `batch_size` buffered rows per table
    can be lost only if the process is killed before `__exit__` runs (e.g. SIGKILL) — any
    exception propagating through a `with Recorder(...)` block still flushes first.
    """

    def __init__(self, db_path: str | Path, batch_size: int = 1000) -> None:
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self._connection: sqlite3.Connection | None = None
        self._pending_generation_metrics: list[tuple] = []
        self._pending_life_records: list[tuple] = []

    def __enter__(self) -> Recorder:
        self._connection = sqlite3.connect(self.db_path)
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA user_version = 2")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.flush()
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Recorder must be used as a context manager")
        return self._connection

    def start_session(
        self,
        session_id: str,
        config: Config,
        lifeforms: list[Lifeform],
        planned_generations: int,
        seed: int,
    ) -> None:
        conn = self._conn()
        conn.execute(
            """INSERT INTO session_metadata
               (session_id, grid_width, grid_height, shape, triangle_mode, wrap,
                initial_alive_percentage, n_lifeforms, seed, planned_generations)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                config.grid.width,
                config.grid.height,
                config.grid.shape,
                config.grid.triangle_mode,
                int(config.grid.wrap),
                config.simulation.initial_alive_percentage,
                len(lifeforms),
                seed,
                planned_generations,
            ),
        )
        conn.executemany(
            "INSERT INTO lifeform_meta (session_id, lifeform, birth, survive) VALUES (?, ?, ?, ?)",
            [
                (session_id, lf.id, _rule_string(lf.rule.birth), _rule_string(lf.rule.survive))
                for lf in lifeforms
            ],
        )
        conn.commit()

    def record_generation(self, session_id: str, record: GenerationRecord) -> None:
        self._pending_generation_metrics.append(
            (
                session_id,
                record.generation,
                record.grid_metrics.total_alive,
                record.grid_metrics.changed_cells,
                record.grid_metrics.volatility,
                record.grid_metrics.entropy,
            )
        )
        for lifeform_id, metrics in record.lifeform_metrics.items():
            cluster = (
                record.cluster_metrics.get(lifeform_id)
                if record.cluster_metrics is not None
                else None
            )
            self._pending_life_records.append(
                (
                    session_id,
                    record.generation,
                    lifeform_id,
                    metrics.alive,
                    metrics.static,
                    metrics.births,
                    metrics.deaths,
                    metrics.kills,
                    metrics.dominance,
                    metrics.growth_rate,
                    metrics.mobility,
                    metrics.avg_lifespan,
                    cluster.cluster_count if cluster is not None else None,
                    cluster.avg_cluster_size if cluster is not None else None,
                    cluster.max_cluster_size if cluster is not None else None,
                )
            )

        if (
            len(self._pending_life_records) >= self.batch_size
            or len(self._pending_generation_metrics) >= self.batch_size
        ):
            self.flush()

    def flush(self) -> None:
        if not self._pending_generation_metrics and not self._pending_life_records:
            return
        conn = self._conn()
        if self._pending_generation_metrics:
            conn.executemany(
                """INSERT INTO generation_metrics
                   (session_id, generation, total_alive, changed_cells, volatility, entropy)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                self._pending_generation_metrics,
            )
            self._pending_generation_metrics.clear()
        if self._pending_life_records:
            conn.executemany(
                """INSERT INTO life_records
                   (session_id, generation, lifeform, alive, static, births, deaths, kills,
                    dominance, growth_rate, mobility, avg_lifespan,
                    cluster_count, avg_cluster_size, max_cluster_size)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                self._pending_life_records,
            )
            self._pending_life_records.clear()
        conn.commit()
