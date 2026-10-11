import sqlite3
from pathlib import Path

import pytest

from latticelife.config import load_config
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule, parse_counts
from latticelife.recording.recorder import Recorder
from latticelife.session.metrics import ClusterMetrics, GridMetrics, LifeformMetrics
from latticelife.session.session import GenerationRecord

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
        assert recorder._pending_life_records == []  # flushed automatically at batch_size

        recorder.record_generation("sess-1", _record(4, [1]))
        assert len(recorder._pending_life_records) == 1  # below batch_size again, still pending


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
