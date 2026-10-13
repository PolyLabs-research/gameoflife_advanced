"""SQLite schema v2 recorder (§6.9). Single-threaded: discovery is a separate process and never
touches this DB — it writes its own JSONL/checkpoints (§6.11). No `check_same_thread=False`, no
lock. Fixed DDL, no runtime ALTER-TABLE migration code: recording starts fresh on schema v2, the
legacy DB's lifespan-derived columns (D-1) and zero-filled metric columns (D-2) are not migrated
(see the salvage manifest).

Opening a file never modifies one that is not already ours: a new or empty file (`user_version`
0 and no schema object of any kind) gets the v2 schema, `user_version = 2` and WAL; an existing
schema v2 file (`user_version` 2, at least one of the four v2 tables and no other table; views,
indexes and triggers of the user's own are fine) is checked table by table; any other file (the
legacy `species.db`, a later schema, another application's database that also uses
`user_version` 2, one holding only views/indexes/triggers, not SQLite at all) raises
`RecorderSchemaError` before a single PRAGMA or DDL statement is written to it. A path SQLite
cannot open, read or write (a directory; a file without read or write permission; a file, new
or not, in a folder that cannot be written, where SQLite keeps its journal), or a file another
process holds locked, raises `RecorderOpenError` — while the Recorder opens, not at the first
row recorded: SQLite opens a file it may only read without a word, so the schema is written in
one transaction that also sets `user_version` again, and that write is where a read-only file
fails.

`lifeform_meta.birth`/`survive` hold `core.rules.format_counts` text: the spec's compact digits
(`'3'`, `'23'`) while every count is <= 9, comma-separated once one reaches 10 (`'2,10,12'`,
`'12,'`), which `core.rules.parse_counts` reads back unambiguously (12-neighbour triangles).
"""

from __future__ import annotations

import contextlib
import functools
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from latticelife.core.rules import format_counts
from latticelife.logging import log_event

if TYPE_CHECKING:
    from latticelife.config import Config
    from latticelife.core.lifeform import Lifeform
    from latticelife.session.session import GenerationRecord

SCHEMA_VERSION = 2

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

_INSERT_GENERATION_METRICS = """INSERT INTO generation_metrics
   (session_id, generation, total_alive, changed_cells, volatility, entropy)
   VALUES (?, ?, ?, ?, ?, ?)"""
_INSERT_LIFE_RECORDS = """INSERT INTO life_records
   (session_id, generation, lifeform, alive, static, births, deaths, kills,
    dominance, growth_rate, mobility, avg_lifespan,
    cluster_count, avg_cluster_size, max_cluster_size)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

# One buffered generation: its generation_metrics row and its life_records rows.
_PendingGeneration = tuple[tuple, list[tuple]]


class RecorderSchemaError(ValueError):
    """The DB file is not a LatticeLife schema v2 database (or not SQLite). Raised before
    anything has been written to the file."""


class RecorderOpenError(ValueError):
    """SQLite cannot open, read or write the DB path (a directory, a file without read or write
    permission, a file in a folder that cannot be written), or another process holds it
    locked."""


# Primary SQLite result codes (the low byte of `sqlite_errorcode`) that describe the path, its
# permissions or another process rather than what the file holds.
_IN_USE = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})
_NOT_WRITABLE = frozenset({sqlite3.SQLITE_READONLY, sqlite3.SQLITE_FULL})
_NOT_OPENABLE = frozenset(
    {sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_PERM, sqlite3.SQLITE_AUTH, sqlite3.SQLITE_IOERR}
)


@functools.cache
def _schema_columns() -> dict[str, tuple[str, ...]]:
    """Column names of each v2 table, read from `_SCHEMA` itself so the check below can never
    drift from the DDL."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(_SCHEMA)
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = [row[0] for row in conn.execute(query)]
        return {
            table: tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))
            for table in tables
        }
    finally:
        conn.close()


class Recorder:
    """Context manager guaranteeing flush-and-close on any exit. On a normal exit the final
    flush must succeed (its error propagates). When the `with` body raised, a failing final
    flush is logged (`recorder_flush_failed`) and the body's exception is the one that
    propagates. The connection is closed either way.

    Buffering is per generation: `record_generation` adds a generation's generation_metrics row
    and all its life_records rows as one buffer entry, and `flush` writes the buffer in one
    transaction — so a generation is either fully in the DB or not at all. A failed flush rolls
    back and keeps the buffer; rows leave it only once committed. Bounded loss: at most
    `batch_size` buffered rows per table, and only when the process dies without unwinding
    (SIGKILL, power loss) — any exception, including KeyboardInterrupt, still flushes.
    """

    def __init__(self, db_path: str | Path, batch_size: int = 1000) -> None:
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self._connection: sqlite3.Connection | None = None
        self._pending: list[_PendingGeneration] = []
        self._pending_life_rows = 0

    def __enter__(self) -> Recorder:
        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as exc:
            # os.path.isdir, not Path.is_dir: it never raises (a folder we may not search).
            reason = "it is a directory" if os.path.isdir(self.db_path) else str(exc)
            raise RecorderOpenError(
                f"{self.db_path} cannot be opened as a SQLite database ({reason})"
            ) from None
        try:
            self._prepare_schema(conn)
        except BaseException:
            conn.close()
            raise
        self._connection = conn
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: object, tb: object) -> None:
        conn = self._connection
        if conn is None:
            return
        try:
            self.flush()
        except Exception as flush_error:
            if exc_type is None:
                raise
            log_event(
                "app",
                level="error",
                event="recorder_flush_failed",
                db_path=str(self.db_path),
                error=repr(flush_error),
                lost_generations=len(self._pending),
                while_handling=repr(exc),
            )
        finally:
            self._connection = None
            conn.close()

    def _access_error(self, exc: sqlite3.Error) -> RecorderOpenError | None:
        """The RecorderOpenError for an SQLite error about the path rather than its contents
        (another process holds it; it, or its folder, cannot be opened or written), or None."""
        code = (exc.sqlite_errorcode or 0) & 0xFF
        if code in _IN_USE:
            return RecorderOpenError(
                f"{self.db_path} is in use by another process ({exc}); try again once it has "
                "finished, or record into another file (autorun --db)"
            )
        if exc.sqlite_errorname == "SQLITE_READONLY_DIRECTORY":
            return RecorderOpenError(
                f"{self.db_path} cannot be written ({exc}): its folder cannot be written, and "
                "SQLite keeps its journal there"
            )
        if code in _NOT_WRITABLE:
            return RecorderOpenError(f"{self.db_path} cannot be written ({exc})")
        if code in _NOT_OPENABLE:
            return RecorderOpenError(
                f"{self.db_path} cannot be opened as a SQLite database ({exc})"
            )
        return None

    def _prepare_schema(self, conn: sqlite3.Connection) -> None:
        """Create the v2 schema in a new/empty file, or verify an existing v2 file. Everything
        that writes (WAL, DDL, user_version) happens only after the checks pass."""
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            # Every schema object, not just tables: a file holding only views, indexes or
            # triggers is somebody else's database, not an empty one.
            objects = conn.execute("SELECT type, name FROM sqlite_master").fetchall()
        except sqlite3.DatabaseError as exc:
            access_error = self._access_error(exc)
            if access_error is not None:
                raise access_error from None
            raise RecorderSchemaError(
                f"{self.db_path} is not a SQLite database ({exc}); refusing to use it"
            ) from None
        if version != 0 or objects:
            self._check_v2_schema(conn, version, objects)

        try:
            conn.execute("PRAGMA journal_mode = WAL")
            # Adds any v2 table the file is missing. Setting user_version, even to the value it
            # has, is a write: a file SQLite could open only for reading fails here.
            conn.executescript(
                f"BEGIN IMMEDIATE; {_SCHEMA} PRAGMA user_version = {SCHEMA_VERSION}; COMMIT;"
            )
        except sqlite3.OperationalError as exc:
            if conn.in_transaction:
                conn.rollback()
            access_error = self._access_error(exc)
            if access_error is None:
                raise
            raise access_error from None

    def _check_v2_schema(
        self, conn: sqlite3.Connection, version: int, objects: list[tuple[str, str]]
    ) -> None:
        """Raise RecorderSchemaError unless this non-empty file is a LatticeLife schema v2
        database."""
        tables = {
            name for kind, name in objects if kind == "table" and not name.startswith("sqlite_")
        }
        # user_version 2 alone does not make a file ours: other applications version their own
        # schemas with it too. Ours holds at least one v2 table and no table of anyone else's.
        v2_tables = set(_schema_columns())
        foreign = sorted(tables - v2_tables)
        if version != SCHEMA_VERSION or foreign or not tables & v2_tables:
            others = sorted(f"{kind} {name}" for kind, name in objects if kind != "table")
            not_v2 = (  # which ones, when v2 tables and others are mixed
                f"; not schema v{SCHEMA_VERSION} tables: {', '.join(foreign)}"
                if version == SCHEMA_VERSION and foreign and tables & v2_tables
                else ""
            )
            raise RecorderSchemaError(
                f"{self.db_path} is not a LatticeLife schema v{SCHEMA_VERSION} database "
                f"(user_version={version}, tables: {', '.join(sorted(tables)) or 'none'}{not_v2}"
                f"{'; other objects: ' + ', '.join(others) if others else ''}); "
                "refusing to modify it. Record into a new file instead (autorun --db)."
            )
        # The user's own views and indexes are fine, but not under a v2 table's name: CREATE
        # TABLE IF NOT EXISTS would then fail on it, or skip the table and write into the view.
        clashing = sorted(
            f"{kind} {name}" for kind, name in objects if kind != "table" and name in v2_tables
        )
        if clashing:
            raise RecorderSchemaError(
                f"{self.db_path}: {', '.join(clashing)} takes the name of a schema "
                f"v{SCHEMA_VERSION} table; refusing to modify it. Record into a new file instead "
                "(autorun --db)."
            )
        for table, expected in _schema_columns().items():
            if table not in tables:
                continue
            actual = tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))
            if actual != expected:
                raise RecorderSchemaError(
                    f"{self.db_path}: table {table} has columns {', '.join(actual)}, not the "
                    f"schema v{SCHEMA_VERSION} columns {', '.join(expected)}; refusing to modify it"
                )

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
        if planned_generations < 1:
            raise ValueError(f"planned_generations must be at least 1 (got {planned_generations})")
        conn = self._conn()
        # One transaction: a session is recorded with all of its lifeform rows or not at all.
        with conn:
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
                "INSERT INTO lifeform_meta (session_id, lifeform, birth, survive) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        session_id,
                        lf.id,
                        format_counts(lf.rule.birth),
                        format_counts(lf.rule.survive),
                    )
                    for lf in lifeforms
                ],
            )

    def record_generation(self, session_id: str, record: GenerationRecord) -> None:
        generation_row = (
            session_id,
            record.generation,
            record.grid_metrics.total_alive,
            record.grid_metrics.changed_cells,
            record.grid_metrics.volatility,
            record.grid_metrics.entropy,
        )
        life_rows = []
        for lifeform_id, metrics in record.lifeform_metrics.items():
            cluster = (
                record.cluster_metrics.get(lifeform_id)
                if record.cluster_metrics is not None
                else None
            )
            life_rows.append(
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

        # A single append, so an interrupt can never leave half a generation in the buffer.
        self._pending.append((generation_row, life_rows))
        self._pending_life_rows += len(life_rows)

        if len(self._pending) >= self.batch_size or self._pending_life_rows >= self.batch_size:
            self.flush()

    @property
    def pending_generations(self) -> int:
        """Generations buffered but not yet committed."""
        return len(self._pending)

    def flush(self) -> None:
        """Write every buffered generation in one transaction. On any failure (or an interrupt)
        before the commit has happened, the transaction is rolled back and the buffer is kept
        intact for a later flush."""
        if not self._pending:
            return
        conn = self._conn()
        batch, self._pending, self._pending_life_rows = self._pending, [], 0
        committing = False
        try:
            conn.executemany(_INSERT_GENERATION_METRICS, [row for row, _ in batch])
            conn.executemany(_INSERT_LIFE_RECORDS, [row for _, rows in batch for row in rows])
            committing = True
            conn.commit()
        except BaseException:
            # `in_transaction` tells a failed commit (still open) from an interrupt that arrived
            # just after a successful one (closed, rows durable: do not buffer them again).
            if not committing or conn.in_transaction:
                with contextlib.suppress(sqlite3.Error):
                    conn.rollback()
                self._pending = batch + self._pending
                self._pending_life_rows = sum(len(rows) for _, rows in self._pending)
            raise
