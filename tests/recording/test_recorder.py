import functools
import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from latticelife.config import load_config
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule, parse_counts
from latticelife.recording.recorder import Recorder, RecorderOpenError, RecorderSchemaError
from latticelife.session.metrics import ClusterMetrics, GridMetrics, LifeformMetrics
from latticelife.session.session import GenerationRecord, SimulationSession

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"
CONWAY = Rule(birth={3}, survive={2, 3})


def _config():
    return load_config(DEFAULT_CONFIG_PATH)


def _record(generation, lifeform_ids, with_clusters=True):
    grid_metrics = GridMetrics(total_alive=10, changed_cells=2, volatility=20.0, entropy=0.5)
    lifeform_metrics = {
        lid: LifeformMetrics(
            alive=5,
            static=1,
            births=1,
            deaths=0,
            kills=0,
            dominance=0.5,
            growth_rate=0.1,
            mobility=0.2,
            avg_lifespan=3.0,
        )
        for lid in lifeform_ids
    }
    cluster_metrics = (
        {
            lid: ClusterMetrics(cluster_count=1, avg_cluster_size=5.0, max_cluster_size=5)
            for lid in lifeform_ids
        }
        if with_clusters
        else None
    )
    return GenerationRecord(
        generation, None, grid_metrics, lifeform_metrics, cluster_metrics, alert=None
    )


def test_schema_created_with_user_version_2(tmp_path):
    db_path = tmp_path / "test.db"
    with Recorder(db_path):
        pass

    conn = sqlite3.connect(db_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"session_metadata", "lifeform_meta", "generation_metrics", "life_records"} <= tables
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_start_session_writes_metadata_and_lifeform_rules(tmp_path):
    db_path = tmp_path / "test.db"
    config = _config()
    lifeforms = [Lifeform(1, CONWAY), Lifeform(2, Rule(birth={2}, survive={4, 5}))]

    with Recorder(db_path) as recorder:
        recorder.start_session("sess-1", config, lifeforms, planned_generations=50, seed=123)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT grid_width, grid_height, shape, wrap, n_lifeforms, seed, planned_generations "
        "FROM session_metadata WHERE session_id = ?",
        ("sess-1",),
    ).fetchone()
    assert row == (
        config.grid.width,
        config.grid.height,
        config.grid.shape,
        int(config.grid.wrap),
        2,
        123,
        50,
    )

    rules = dict(
        conn.execute(
            "SELECT lifeform, birth || '/' || survive FROM lifeform_meta WHERE session_id = ?",
            ("sess-1",),
        ).fetchall()
    )
    assert rules[1] == "3/23"
    assert rules[2] == "2/45"


def test_rules_with_counts_of_ten_or_more_are_stored_unambiguously(tmp_path):
    """F3: triangle edge+vertex rules {12}/{0,12} and {1,2}/{0,1,2} both used to be stored as
    '12'/'012'."""
    db_path = tmp_path / "test.db"
    lifeforms = [
        Lifeform(1, Rule(birth={12}, survive={0, 12})),
        Lifeform(2, Rule(birth={1, 2}, survive={0, 1, 2})),
        Lifeform(3, Rule(birth={5, 7, 10}, survive={2, 3})),
    ]

    with Recorder(db_path) as recorder:
        recorder.start_session("sess-t", _config(), lifeforms, planned_generations=5, seed=1)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT lifeform, birth, survive FROM lifeform_meta WHERE session_id = ? ORDER BY 1",
        ("sess-t",),
    ).fetchall()
    assert rows == [(1, "12,", "0,12"), (2, "12", "012"), (3, "5,7,10", "23")]
    for (lifeform_id, birth, survive), lf in zip(rows, lifeforms, strict=True):
        assert lifeform_id == lf.id
        assert Rule(parse_counts(birth), parse_counts(survive)) == lf.rule


def test_record_generation_writes_rows_with_null_clusters_on_non_stride(tmp_path):
    """Cluster columns are NULL (not 0) when no cluster measurement was taken this generation —
    absence of measurement must stay distinguishable from a measured zero."""
    db_path = tmp_path / "test.db"
    config = _config()
    lifeforms = [Lifeform(1, CONWAY)]

    with Recorder(db_path, batch_size=1000) as recorder:
        recorder.start_session("sess-1", config, lifeforms, planned_generations=2, seed=1)
        recorder.record_generation("sess-1", _record(1, [1], with_clusters=False))
        recorder.record_generation("sess-1", _record(2, [1], with_clusters=True))

    conn = sqlite3.connect(db_path)
    gm_rows = conn.execute(
        "SELECT generation, total_alive, entropy FROM generation_metrics ORDER BY generation"
    ).fetchall()
    assert gm_rows == [(1, 10, 0.5), (2, 10, 0.5)]

    cluster_gen1 = conn.execute(
        "SELECT cluster_count FROM life_records WHERE generation = 1"
    ).fetchone()[0]
    cluster_gen2 = conn.execute(
        "SELECT cluster_count FROM life_records WHERE generation = 2"
    ).fetchone()[0]
    assert cluster_gen1 is None
    assert cluster_gen2 == 1


def test_batch_size_triggers_automatic_flush(tmp_path):
    db_path = tmp_path / "test.db"
    config = _config()
    lifeforms = [Lifeform(1, CONWAY)]

    with Recorder(db_path, batch_size=3) as recorder:
        recorder.start_session("sess-1", config, lifeforms, planned_generations=5, seed=1)
        for gen in range(1, 4):
            recorder.record_generation("sess-1", _record(gen, [1]))
        assert recorder.pending_generations == 0  # flushed automatically at batch_size

        recorder.record_generation("sess-1", _record(4, [1]))
        assert recorder.pending_generations == 1  # below batch_size again, still pending


def test_batch_size_counts_life_record_rows_too(tmp_path):
    with Recorder(tmp_path / "test.db", batch_size=4) as recorder:
        recorder.start_session("sess-1", _config(), [], planned_generations=5, seed=1)
        recorder.record_generation("sess-1", _record(1, [1, 2, 3]))
        assert recorder.pending_generations == 1
        recorder.record_generation("sess-1", _record(2, [1, 2, 3]))  # 6 life_records rows
        assert recorder.pending_generations == 0


def test_context_manager_flushes_on_exception(tmp_path):
    db_path = tmp_path / "test.db"
    config = _config()
    lifeforms = [Lifeform(1, CONWAY)]

    with pytest.raises(RuntimeError), Recorder(db_path, batch_size=1000) as recorder:
        recorder.start_session("sess-1", config, lifeforms, planned_generations=1, seed=1)
        recorder.record_generation("sess-1", _record(1, [1]))
        raise RuntimeError("boom")

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM life_records").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM generation_metrics").fetchone()[0] == 1
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


# --- F14: never modify a file that is not already a schema v2 DB -----------------------------


def _journal_mode(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()


def _db_state(db_path):
    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = sorted(
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        )
        rows = {table: conn.execute(f"SELECT * FROM {table}").fetchall() for table in tables}
        return version, tables, rows
    finally:
        conn.close()


def test_legacy_db_is_refused_without_being_modified(tmp_path):
    """F14: the legacy species.db layout (user_version 0, its own session_metadata) used to be
    relabelled user_version=2, switched to WAL and given generation_metrics before the first
    INSERT failed."""
    db_path = tmp_path / "species.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE TABLE session_metadata (session_id INTEGER PRIMARY KEY, started TEXT);"
        "CREATE TABLE life_records (id INTEGER PRIMARY KEY, lifeform INTEGER);"
        "INSERT INTO session_metadata (started) VALUES ('2024-01-01');"
    )
    conn.close()
    before = _db_state(db_path)
    before_bytes = db_path.read_bytes()

    with pytest.raises(RecorderSchemaError, match="user_version=0"), Recorder(db_path):
        pass

    assert _db_state(db_path) == before
    assert db_path.read_bytes() == before_bytes
    assert _journal_mode(db_path) == "delete"
    assert not Path(f"{db_path}-wal").exists()


def test_newer_schema_db_is_refused_and_keeps_its_version(tmp_path):
    db_path = tmp_path / "future.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("CREATE TABLE session_metadata (id TEXT); PRAGMA user_version = 3;")
    conn.close()

    with pytest.raises(RecorderSchemaError, match="user_version=3"), Recorder(db_path):
        pass

    assert _db_state(db_path)[0:2] == (3, ["session_metadata"])
    assert _journal_mode(db_path) == "delete"


def test_v2_db_with_wrong_columns_is_refused(tmp_path):
    db_path = tmp_path / "odd.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("CREATE TABLE life_records (session_id TEXT); PRAGMA user_version = 2;")
    conn.close()

    with pytest.raises(RecorderSchemaError, match="life_records"), Recorder(db_path):
        pass
    assert _db_state(db_path)[1] == ["life_records"]


_V2_GENERATION_METRICS = """CREATE TABLE generation_metrics (
  session_id TEXT, generation INTEGER, total_alive INTEGER, changed_cells INTEGER,
  volatility REAL, entropy REAL, PRIMARY KEY (session_id, generation));"""


@pytest.mark.parametrize(
    ("ddl", "message"),
    [
        (
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
            "INSERT INTO users VALUES (1, 'a');",
            r"user_version=2, tables: users\b",
        ),
        (
            f"{_V2_GENERATION_METRICS} CREATE TABLE users (id INTEGER PRIMARY KEY);",
            r"tables: generation_metrics, users; not schema v2 tables: users\b",
        ),
        ("CREATE VIEW v AS SELECT 1 AS x;", r"user_version=2, tables: none; other objects: view v"),
        ("", r"user_version=2, tables: none\)"),
    ],
    ids=["foreign-tables", "v2-and-foreign-tables", "view-only", "no-objects"],
)
def test_user_version_2_alone_does_not_make_a_db_ours(tmp_path, ddl, message):
    """Another application's database versioned with user_version 2 (users table, no v2 table)
    used to pass every check: it was switched to WAL and given the four LatticeLife tables."""
    db_path = tmp_path / "foreign.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(f"{ddl} PRAGMA user_version = 2;")
    conn.close()
    before = _db_state(db_path)
    before_bytes = db_path.read_bytes()

    with pytest.raises(RecorderSchemaError, match=message), Recorder(db_path):
        pass

    assert db_path.read_bytes() == before_bytes
    assert _db_state(db_path) == before
    assert _journal_mode(db_path) == "delete"


@pytest.mark.parametrize(
    "ddl",
    [
        "CREATE INDEX life_records ON generation_metrics(total_alive);",
        "CREATE VIEW life_records AS SELECT * FROM generation_metrics;",
    ],
    ids=["index", "view"],
)
def test_a_view_or_index_named_like_a_v2_table_is_refused_unchanged(tmp_path, ddl):
    """It passed the schema check: the file went to WAL, then CREATE TABLE IF NOT EXISTS failed
    on the index (traceback), or skipped the table and the first flush wrote into the view."""
    db_path = tmp_path / "clash.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(f"{_V2_GENERATION_METRICS} {ddl} PRAGMA user_version = 2;")
    conn.close()
    before = _db_state(db_path)
    before_bytes = db_path.read_bytes()

    with pytest.raises(RecorderSchemaError, match=r"life_records takes the name of a schema v2"):
        with Recorder(db_path):
            pass

    assert db_path.read_bytes() == before_bytes
    assert _db_state(db_path) == before
    assert _journal_mode(db_path) == "delete"
    assert not Path(f"{db_path}-wal").exists()


def test_v2_db_with_the_users_own_index_view_and_statistics_is_still_ours(tmp_path):
    """Only tables decide whose a file is: an index, a view or ANALYZE's sqlite_stat1 added to a
    recording for analysis does not lock the user out of recording into it again."""
    db_path = tmp_path / "test.db"
    with Recorder(db_path) as recorder:
        recorder.start_session("sess-1", _config(), [Lifeform(1, CONWAY)], 1, seed=1)
        recorder.record_generation("sess-1", _record(1, [1]))
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE INDEX by_lifeform ON life_records (lifeform);"
        "CREATE VIEW alive_by_generation AS SELECT generation, total_alive FROM generation_metrics;"
        "DROP TABLE lifeform_meta; ANALYZE;"
    )
    conn.close()

    with Recorder(db_path) as recorder:
        recorder.start_session("sess-2", _config(), [Lifeform(1, CONWAY)], 1, seed=1)
        recorder.record_generation("sess-2", _record(1, [1]))

    version, tables, rows = _db_state(db_path)
    assert version == 2
    assert {"session_metadata", "lifeform_meta", "generation_metrics", "life_records"} <= set(
        tables
    )
    assert len(rows["generation_metrics"]) == 2
    assert rows["lifeform_meta"] == [("sess-2", 1, "3", "23")]


def test_a_path_sqlite_cannot_open_raises_recorder_open_error(tmp_path):
    """A directory used to escape as sqlite3.OperationalError('unable to open database file')."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    with (
        pytest.raises(RecorderOpenError, match="recordings cannot be opened.*is a directory"),
        Recorder(folder),
    ):
        pass
    assert list(folder.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_db_and_unwritable_folder_raise_recorder_open_error(tmp_path):
    unreadable = tmp_path / "locked.db"
    with Recorder(unreadable):
        pass
    unreadable.chmod(0)
    read_only_folder = tmp_path / "read-only"
    read_only_folder.mkdir(mode=0o555)
    try:
        for db_path in (unreadable, read_only_folder / "new.db"):
            with (
                pytest.raises(RecorderOpenError, match="unable to open database file"),
                Recorder(db_path),
            ):
                pass
        assert list(read_only_folder.iterdir()) == []
    finally:
        unreadable.chmod(0o644)
        read_only_folder.chmod(0o755)


def _recorded_db(db_path, journal_mode="wal"):
    """A schema v2 DB holding one session, in `journal_mode`."""
    with Recorder(db_path) as recorder:
        recorder.start_session("sess-1", _config(), [Lifeform(1, CONWAY)], 1, seed=1)
        recorder.record_generation("sess-1", _record(1, [1]))
    conn = sqlite3.connect(db_path)
    conn.execute(f"PRAGMA journal_mode = {journal_mode}")
    conn.close()
    return db_path


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
def test_a_read_only_db_raises_recorder_open_error_on_opening(tmp_path, journal_mode):
    """SQLite opens a file it may only read without complaint. A read-only v2 DB in WAL mode
    passed every check and escaped as sqlite3.OperationalError ('attempt to write a readonly
    database') from start_session's INSERT; one in rollback-journal mode from the WAL switch."""
    db_path = _recorded_db(tmp_path / "read-only.db", journal_mode)
    before = _db_state(db_path)
    db_path.chmod(0o444)
    try:
        with (
            pytest.raises(RecorderOpenError, match="read-only.db cannot be written .*readonly"),
            Recorder(db_path) as recorder,
        ):
            recorder.start_session("sess-2", _config(), [Lifeform(1, CONWAY)], 1, seed=2)
    finally:
        db_path.chmod(0o644)
    assert _db_state(db_path) == before
    assert _journal_mode(db_path) == journal_mode


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
@pytest.mark.parametrize("existing", ["empty", "v2"])
def test_a_db_in_a_read_only_folder_raises_recorder_open_error(tmp_path, existing):
    """SQLite keeps its journal beside the DB. An existing empty file there escaped as
    sqlite3.OperationalError from the WAL switch; a v2 DB in WAL mode, which cannot even be read
    there, was reported as 'not a SQLite database'."""
    folder = tmp_path / "read-only"
    folder.mkdir()
    db_path = folder / "run.db"
    if existing == "empty":
        db_path.touch()
    else:
        _recorded_db(db_path)
    before = db_path.read_bytes()
    folder.chmod(0o555)
    try:
        with (
            pytest.raises(RecorderOpenError, match="run.db cannot be written .*its folder"),
            Recorder(db_path),
        ):
            pass
        assert sorted(path.name for path in folder.iterdir()) == ["run.db"]
    finally:
        folder.chmod(0o755)
    assert db_path.read_bytes() == before


@pytest.mark.parametrize(
    ("journal_mode", "lock"),
    [("delete", "BEGIN EXCLUSIVE"), ("delete", "BEGIN IMMEDIATE"), ("wal", "BEGIN IMMEDIATE")],
    ids=["exclusive", "reserved", "wal-writer"],
)
def test_a_db_another_connection_holds_is_reported_as_in_use(
    tmp_path, monkeypatch, journal_mode, lock
):
    """A held lock outlasting the busy timeout ('database is locked') was reported as 'is not a
    SQLite database' when it stopped the checks reading the file, and escaped as
    sqlite3.OperationalError when it stopped the schema write."""
    db_path = _recorded_db(tmp_path / "busy.db", journal_mode)
    before = _db_state(db_path)
    holder = sqlite3.connect(db_path, isolation_level=None)
    holder.execute(lock)
    holder.execute("UPDATE session_metadata SET seed = 5")  # uncommitted
    monkeypatch.setattr(sqlite3, "connect", functools.partial(sqlite3.connect, timeout=0.05))
    try:
        with (
            pytest.raises(RecorderOpenError, match="busy.db is in use by another process"),
            Recorder(db_path),
        ):
            pass
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert _db_state(db_path) == before


@pytest.mark.parametrize(
    "ddl",
    [
        "CREATE VIEW v AS SELECT 1 AS x;",
        "CREATE VIEW v AS SELECT 1 AS x;"
        "CREATE TRIGGER t INSTEAD OF INSERT ON v BEGIN SELECT 1; END;",
    ],
    ids=["view", "view-and-trigger"],
)
def test_db_without_tables_but_with_other_objects_is_refused_untouched(tmp_path, ddl):
    """A user_version-0 file with no tables used to count as empty: one holding only views (and
    triggers on them) was switched to WAL, given the v2 tables and relabelled user_version=2."""
    db_path = tmp_path / "views.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(ddl)
    conn.close()
    before_bytes = db_path.read_bytes()

    with pytest.raises(RecorderSchemaError, match=r"user_version=0.*view v"), Recorder(db_path):
        pass

    assert db_path.read_bytes() == before_bytes
    assert _db_state(db_path)[0:2] == (0, [])
    assert _journal_mode(db_path) == "delete"
    assert not Path(f"{db_path}-wal").exists()


def test_existing_empty_file_gets_the_schema(tmp_path):
    db_path = tmp_path / "touched.db"
    db_path.touch()
    with Recorder(db_path):
        pass
    version, tables, _ = _db_state(db_path)
    assert version == 2
    assert {"session_metadata", "lifeform_meta", "generation_metrics", "life_records"} <= set(
        tables
    )


def test_non_sqlite_file_is_refused_untouched(tmp_path):
    db_path = tmp_path / "notes.txt"
    db_path.write_text("not a database\n" * 100)

    with pytest.raises(RecorderSchemaError, match="not a SQLite database"), Recorder(db_path):
        pass
    assert db_path.read_text() == "not a database\n" * 100


def test_reopening_a_v2_db_keeps_earlier_sessions(tmp_path):
    db_path = tmp_path / "test.db"
    for session_id in ("sess-1", "sess-2"):
        with Recorder(db_path) as recorder:
            recorder.start_session(session_id, _config(), [Lifeform(1, CONWAY)], 1, seed=1)
            recorder.record_generation(session_id, _record(1, [1]))

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM session_metadata").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM generation_metrics").fetchone()[0] == 2
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert _journal_mode(db_path) == "wal"


def test_start_session_rejects_a_plan_without_generations(tmp_path):
    with Recorder(tmp_path / "test.db") as recorder, pytest.raises(ValueError, match="at least 1"):
        recorder.start_session("sess-1", _config(), [Lifeform(1, CONWAY)], 0, seed=1)


# --- F15: the final flush ---------------------------------------------------------------------


def _hold_write_lock(db_path):
    blocker = sqlite3.connect(db_path, timeout=0)
    blocker.execute("BEGIN IMMEDIATE")
    return blocker


def test_failed_final_flush_keeps_the_body_exception_and_closes(tmp_path):
    """F15: a locked DB during the final flush used to replace the body's exception with
    'database is locked' and leave the connection open."""
    db_path = tmp_path / "test.db"
    recorder = Recorder(db_path)
    with pytest.raises(RuntimeError, match="real simulation error"), recorder:
        recorder.start_session("sess-1", _config(), [Lifeform(1, CONWAY)], 2, seed=1)
        recorder.record_generation("sess-1", _record(1, [1]))
        blocker = _hold_write_lock(db_path)
        recorder._connection.execute("PRAGMA busy_timeout = 50")
        raise RuntimeError("real simulation error")

    blocker.rollback()
    blocker.close()
    assert recorder._connection is None


def test_failed_final_flush_without_body_exception_raises_and_closes(tmp_path):
    db_path = tmp_path / "test.db"
    recorder = Recorder(db_path)
    with pytest.raises(sqlite3.OperationalError, match="locked"), recorder:
        recorder.start_session("sess-1", _config(), [Lifeform(1, CONWAY)], 2, seed=1)
        recorder.record_generation("sess-1", _record(1, [1]))
        blocker = _hold_write_lock(db_path)
        recorder._connection.execute("PRAGMA busy_timeout = 50")

    blocker.rollback()
    blocker.close()
    assert recorder._connection is None


def test_failed_flush_rolls_back_and_keeps_the_buffer_for_a_retry(tmp_path):
    db_path = tmp_path / "test.db"
    with Recorder(db_path) as recorder:
        recorder.start_session("sess-1", _config(), [Lifeform(1, CONWAY)], 3, seed=1)
        recorder.record_generation("sess-1", _record(1, [1]))
        recorder.record_generation("sess-1", _record(2, [1]))
        blocker = _hold_write_lock(db_path)
        recorder._connection.execute("PRAGMA busy_timeout = 50")
        with pytest.raises(sqlite3.OperationalError):
            recorder.flush()
        assert recorder.pending_generations == 2
        blocker.rollback()
        blocker.close()
        recorder.flush()  # the retry writes everything once
        assert recorder.pending_generations == 0

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT generation FROM generation_metrics").fetchall() == [(1,), (2,)]
    assert conn.execute("SELECT COUNT(*) FROM life_records").fetchone()[0] == 2


def test_failure_in_life_records_insert_keeps_generation_metrics_buffered(tmp_path):
    """The old flush cleared the generation_metrics buffer as soon as its insert ran, so a
    life_records failure lost those rows; now nothing leaves the buffer before the commit."""
    db_path = tmp_path / "test.db"
    with Recorder(db_path) as recorder:
        recorder.start_session("sess-1", _config(), [Lifeform(1, CONWAY)], 2, seed=1)
        recorder._connection.execute(
            "INSERT INTO life_records (session_id, generation, lifeform) VALUES ('sess-1', 2, 1)"
        )
        recorder._connection.commit()
        recorder.record_generation("sess-1", _record(1, [1]))
        recorder.record_generation("sess-1", _record(2, [1]))  # its life_records row collides
        with pytest.raises(sqlite3.IntegrityError):
            recorder.flush()
        assert recorder.pending_generations == 2
        recorder._connection.execute("DELETE FROM life_records")
        recorder._connection.commit()
        recorder.flush()

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT generation FROM generation_metrics").fetchall() == [(1,), (2,)]
    assert conn.execute("SELECT generation FROM life_records").fetchall() == [(1,), (2,)]


def test_interrupt_inside_the_body_still_flushes_whole_generations(tmp_path):
    db_path = tmp_path / "test.db"
    with pytest.raises(KeyboardInterrupt), Recorder(db_path, batch_size=1000) as recorder:
        recorder.start_session("sess-1", _config(), [Lifeform(1, CONWAY)], 9, seed=1)
        for gen in (1, 2, 3):
            recorder.record_generation("sess-1", _record(gen, [1, 2]))
        raise KeyboardInterrupt

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM generation_metrics").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM life_records").fetchone()[0] == 6


# --- F59: a real session streamed into the recorder ---------------------------------------------


def test_real_session_streams_every_generation_past_the_history_limit(tmp_path):
    """Recording must never read the session's bounded history: with history_limit=10, all 50
    generations land in the DB, and the final one has cluster metrics."""
    base = _config()
    config = replace(
        base,
        grid=replace(base.grid, width=12, height=12),
        simulation=replace(base.simulation, history_limit=10, n_lifeforms=2),
    )
    session = SimulationSession.from_config(config, seed=3, auto_pause=False)
    db_path = tmp_path / "x.db"

    with Recorder(db_path, batch_size=7) as recorder:
        recorder.start_session("sess-1", config, session.lifeforms, 50, seed=3)
        for i in range(50):
            session.step(force_clusters=(i == 49))
            recorder.record_generation("sess-1", session.history_records[-1])

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT COUNT(*), MIN(generation), MAX(generation) FROM generation_metrics"
    ).fetchone()
    assert rows == (50, 1, 50)
    assert conn.execute("SELECT COUNT(*) FROM life_records").fetchone()[0] == 100
    last = conn.execute(
        "SELECT cluster_count FROM life_records WHERE generation = 50 ORDER BY lifeform"
    ).fetchall()
    assert all(count is not None for (count,) in last)
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
