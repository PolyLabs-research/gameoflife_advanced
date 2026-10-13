"""The console commands end to end (F59: cli.py had no tests at all). Every test writes its own
config into tmp_path with relative output paths, which resolve next to that config (tmp_path has
no pyproject.toml), and points LATTICELIFE_LOG_DIR at its own directory — nothing touches the
project's real out/."""

import contextlib
import functools
import json
import multiprocessing as mp
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
import tomllib
from pathlib import Path

import pytest

import latticelife
from latticelife import cli
from latticelife import logging as ll_logging
from latticelife.config import DEFAULT_CONFIG_PATH
from latticelife.discovery.worker import DiscoveryWorker, acquire_lineage_lock
from latticelife.logging import LOG_DIR_ENV, configure_logging
from latticelife.session.session import SimulationSession

SRC_DIR = Path(latticelife.__file__).resolve().parents[1]


def _python_env() -> dict[str, str]:
    """The environment for a Python subprocess running this checkout's code. It keeps the
    inherited PYTHONPATH after `src`: that carries the suite's out/ guard (tests/conftest.py)."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC_DIR), os.environ["PYTHONPATH"]])}


# Small enough for sub-second sessions and discovery generations, valid for every shape.
FAST = {
    "grid.width": 12,
    "grid.height": 12,
    "simulation.history_limit": 10,
    "simulation.seed": 7,
    "discovery.mu": 2,
    "discovery.n_offspring": 2,
    "discovery.evals_per_candidate": 1,
    "discovery.grid_size": 8,
    "discovery.steps": 20,
    "discovery.freeze_window": 8,
}


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _dump_toml(data, prefix=""):
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    lines = [f"[{prefix}]"] if prefix and scalars else []
    lines += [f"{key} = {_toml_value(value)}" for key, value in scalars.items()]
    text = "\n".join(lines) + "\n" if lines else ""
    for key, table in tables.items():
        text += "\n" + _dump_toml(table, f"{prefix}.{key}" if prefix else key)
    return text


def write_config(directory, **overrides):
    """The default config plus `FAST` plus `overrides` ({"section.key": value}, with `__` for
    the dot in keyword form), written to `directory/config.toml`."""
    with DEFAULT_CONFIG_PATH.open("rb") as f:
        data = tomllib.load(f)
    for dotted, value in {
        **FAST,
        **{k.replace("__", "."): v for k, v in overrides.items()},
    }.items():
        *sections, key = dotted.split(".")
        table = data
        for section in sections:
            table = table[section]
        table[key] = value
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.toml"
    path.write_text(_dump_toml(data))
    return path


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    path = tmp_path / "logs"
    monkeypatch.setenv(LOG_DIR_ENV, str(path))
    return path


def _events(log_dir, channel):
    path = Path(log_dir) / f"{channel}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _wait_for(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _session_rows(db_path):
    """Per session, in the order they were recorded: (planned, count, min gen, max gen)."""
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            """SELECT s.planned_generations, COUNT(g.generation), MIN(g.generation),
                      MAX(g.generation)
               FROM session_metadata s LEFT JOIN generation_metrics g USING (session_id)
               GROUP BY s.session_id ORDER BY s.rowid"""
        ).fetchall()
    finally:
        conn.close()


def _autorun(config, db, *extra, sessions=2, generations=40, shape="square"):
    argv = ["autorun", "--sessions", str(sessions), "--generations", str(generations)]
    argv += ["--shape", shape, "--config", str(config), "--db", str(db), *extra]
    return cli.main(argv)


# --- autorun ------------------------------------------------------------------------------------


def test_autorun_records_every_generation_past_the_history_limit(tmp_path, log_dir, capsys):
    """F4: only the last history_limit generations of each session were saved (e.g. 201-500
    of 500), while the summary claimed all of them."""
    db = tmp_path / "run.db"
    assert _autorun(write_config(tmp_path), db) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary == {"sessions": 2, "generations": 80, "db_path": str(db)}
    assert _session_rows(db) == [(40, 40, 1, 40), (40, 40, 1, 40)]
    conn = sqlite3.connect(db)
    life_rows = conn.execute("SELECT COUNT(*) FROM life_records").fetchone()[0]
    species = conn.execute("SELECT SUM(n_lifeforms) FROM session_metadata").fetchone()[0]
    assert life_rows == 40 * species
    # §6.6: the final generation of a run always has cluster metrics.
    finals = conn.execute("SELECT cluster_count FROM life_records WHERE generation = 40").fetchall()
    assert finals and all(count is not None for (count,) in finals)


def test_autorun_runs_every_planned_generation_through_sudden_death(tmp_path, log_dir, capsys):
    """F5: autorun stopped a session at its first sudden-death auto-pause (often generation 11,
    during the transient), skewing the data toward sessions that never collapse."""
    config = write_config(
        tmp_path,
        alerts__sudden_death_window=2,
        alerts__sudden_death_drop=0.1,
        simulation__history_limit=2,
    )
    db = tmp_path / "run.db"
    assert _autorun(config, db, sessions=6, generations=30, shape="random") == 0

    assert _session_rows(db) == [(30, 30, 1, 30)] * 6
    # The alert really fired: some session lost >= 10% of its cells over 2 generations.
    conn = sqlite3.connect(db)
    alive = {}
    for session_id, total in conn.execute(
        "SELECT session_id, total_alive FROM generation_metrics ORDER BY session_id, generation"
    ):
        alive.setdefault(session_id, []).append(total)
    assert any(
        old > 0 and (old - new) / old >= 0.1
        for series in alive.values()
        for old, new in zip(series, series[2:], strict=False)
    )


def test_autorun_with_a_seed_is_reproducible(tmp_path, log_dir, capsys):
    config = write_config(tmp_path)
    rows = []
    for name in ("a.db", "b.db"):
        assert _autorun(config, tmp_path / name, sessions=3, generations=15, shape="random") == 0
        conn = sqlite3.connect(tmp_path / name)
        order = {
            sid: i
            for i, (sid,) in enumerate(
                conn.execute("SELECT session_id FROM session_metadata ORDER BY rowid")
            )
        }
        metadata = conn.execute(
            "SELECT shape, n_lifeforms, seed, initial_alive_percentage FROM session_metadata "
            "ORDER BY rowid"
        ).fetchall()
        life = sorted(
            (order[row[0]], *row[1:]) for row in conn.execute("SELECT * FROM life_records")
        )
        rows.append((metadata, life))
    assert rows[0] == rows[1]


def _fail_at_call(monkeypatch, n, exc):
    real_step = SimulationSession.step
    calls = {"n": 0}

    def step(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            raise exc
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(SimulationSession, "step", step)


def test_interrupted_autorun_keeps_what_it_recorded(tmp_path, log_dir, capsys, monkeypatch):
    """F4: a Ctrl+C lost every generation of the session in progress (rows were only written
    after a session finished). Now generations 1-14 of session 2 are flushed and the exit
    status says the batch was interrupted."""
    db = tmp_path / "run.db"
    _fail_at_call(monkeypatch, 40 + 15, KeyboardInterrupt())

    assert _autorun(write_config(tmp_path), db) == cli.EXIT_INTERRUPTED

    summary = json.loads(capsys.readouterr().out)
    assert summary == {"sessions": 1, "generations": 54, "db_path": str(db), "interrupted": True}
    assert _session_rows(db) == [(40, 40, 1, 40), (40, 14, 1, 14)]
    event = _events(log_dir, "app")[-1]
    assert event["event"] == "autorun_interrupted"
    assert event["partial_session_generations"] == 14


def test_failing_autorun_keeps_what_it_recorded_and_logs(tmp_path, log_dir, monkeypatch):
    db = tmp_path / "run.db"
    _fail_at_call(monkeypatch, 23, RuntimeError("simulation exploded"))

    with pytest.raises(RuntimeError, match="simulation exploded"):
        _autorun(write_config(tmp_path), db)

    assert _session_rows(db) == [(40, 22, 1, 22)]
    event = _events(log_dir, "app")[-1]
    assert event["event"] == "unhandled_exception"
    assert event["command"] == "autorun"
    assert "simulation exploded" in event["error"]
    assert "Traceback" in event["traceback"]


@pytest.mark.parametrize(
    "argv",
    [
        ["autorun", "--sessions", "0", "--generations", "5"],
        ["autorun", "--sessions", "-3", "--generations", "5"],
        ["autorun", "--sessions", "2", "--generations", "0"],
        ["autorun", "--sessions", "2", "--generations", "-10"],
        ["autorun", "--sessions", "two", "--generations", "5"],
        ["discover", "--minutes", "0"],
        ["discover", "--minutes", "-1"],
        ["discover", "--minutes", "nan"],
        ["discover", "--minutes", "inf"],
    ],
)
def test_nonsense_counts_are_rejected_before_anything_runs(tmp_path, log_dir, capsys, argv):
    """F58: `--generations -10` exited 0 and recorded sessions with planned_generations=-10."""
    db = tmp_path / "run.db"
    extra = ["--db", str(db)] if argv[0] == "autorun" else []
    with pytest.raises(SystemExit) as excinfo:
        cli.main([*argv, "--config", str(write_config(tmp_path)), *extra])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "must be" in err or "expected" in err
    assert not db.exists()
    assert not (tmp_path / "out").exists()


def test_invalid_config_exits_2_with_one_line(tmp_path, log_dir, capsys):
    config = write_config(tmp_path, discovery__steps=100, discovery__freeze_window=100)
    assert cli.main(["discover", "--config", str(config)]) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith("latticelife discover: ") and "discovery.steps" in err
    assert "Traceback" not in err
    assert _events(log_dir, "app")[-1]["event"] == "invalid_input"


def test_a_discovery_only_rule_does_not_stop_ui_or_autorun(tmp_path, log_dir, capsys, monkeypatch):
    """discovery.steps = 100 is too short to detect a 100-step freeze, which only discovery
    scoring cares about; the check refused the config for every command."""
    config = write_config(tmp_path, discovery__steps=100, discovery__freeze_window=100)

    assert _autorun(config, tmp_path / "run.db", sessions=1, generations=5) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["sessions"] == 1

    from latticelife.ui import app as ui_app

    launched = []

    class FakeApp:
        def __init__(self, loaded):
            launched.append(loaded.discovery.steps)

        def run(self):
            pass

    monkeypatch.setattr(ui_app, "App", FakeApp)
    assert cli.main(["ui", "--config", str(config)]) == cli.EXIT_OK
    assert launched == [100]

    assert cli.main(["discover", "--config", str(config)]) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith(f"latticelife discover: {config}: discovery.steps (100)")
    assert "steps >= 166" in err


@pytest.fixture
def checkout_log_dir(tmp_path, monkeypatch):
    """No LATTICELIFE_LOG_DIR, nothing configured yet, and a stand-in for this checkout's
    out/logs (logging.DEFAULT_LOG_DIR), as in a fresh `latticelife` process."""
    monkeypatch.delenv(LOG_DIR_ENV, raising=False)
    path = tmp_path / "checkout" / "out" / "logs"
    monkeypatch.setattr(ll_logging, "DEFAULT_LOG_DIR", path)
    previous = ll_logging._configured_dir
    configure_logging(None)
    yield path
    configure_logging(previous)


@pytest.mark.parametrize("own_log_dir", [None, "my-logs"])
def test_an_invalid_config_is_logged_next_to_it(tmp_path, checkout_log_dir, capsys, own_log_dir):
    """An invalid --config from another directory logged its invalid_input event to this
    checkout's out/logs (it was never loaded, so its logging.dir was never configured)."""
    overrides = {"discovery__mu": 0}
    if own_log_dir is not None:
        overrides["logging__dir"] = own_log_dir
    config = write_config(tmp_path / "scratch", **overrides)
    assert cli.main(["discover", "--config", str(config)]) == cli.EXIT_USAGE
    assert "discovery.mu" in capsys.readouterr().err

    expected = (tmp_path / "scratch").resolve() / (own_log_dir or "out/logs")
    events = _events(expected, "app")
    assert [e["event"] for e in events] == ["invalid_input"]
    assert "discovery.mu" in events[0]["error"]
    assert not checkout_log_dir.exists()


def _autorun_config_error(config, capsys):
    argv = ["autorun", "--sessions", "1", "--generations", "1", "--config", str(config)]
    assert cli.main(argv) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith(f"latticelife autorun: {config}: ") and "Traceback" not in err
    return err


def test_a_config_that_is_not_utf8_exits_2_and_logs_next_to_it(tmp_path, checkout_log_dir, capsys):
    """It escaped as an unhandled UnicodeDecodeError, with a traceback."""
    config = tmp_path / "scratch" / "latin1.toml"
    config.parent.mkdir()
    config.write_bytes(b'note = "caf\xe9"\n')
    assert "not UTF-8" in _autorun_config_error(config, capsys)
    events = _events(config.parent.resolve() / "out" / "logs", "app")
    assert [e["event"] for e in events] == ["invalid_input"]
    assert not checkout_log_dir.exists()


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_a_config_path_with_no_file_logs_to_the_default_dir(
    tmp_path, checkout_log_dir, capsys, monkeypatch, kind
):
    """No file to log next to (a directory escaped as an unhandled IsADirectoryError): the
    event goes to the default log directory, never into the current directory (F16)."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config = Path("no.toml")
    if kind == "directory":
        config = Path("config.d")
        config.mkdir()
    _autorun_config_error(config, capsys)
    assert [e["event"] for e in _events(checkout_log_dir, "app")] == ["invalid_input"]
    assert not (cwd / "out").exists()


def test_missing_config_exits_2(tmp_path, log_dir, capsys):
    argv = ["autorun", "--sessions", "1", "--generations", "1"]
    assert cli.main([*argv, "--config", str(tmp_path / "nope.toml")]) == cli.EXIT_USAGE
    assert "config file not found" in capsys.readouterr().err


def test_random_shape_needs_a_grid_every_shape_can_use(tmp_path, log_dir, capsys):
    """A 13x13 wrapped grid suits a square but cannot tile hexagons or triangles; autorun with
    --shape random used to crash partway through the batch instead of refusing up front."""
    config = write_config(tmp_path, grid__width=13, grid__height=13)
    db = tmp_path / "run.db"
    assert _autorun(config, db, shape="random") == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert "--shape random" in err and "hexagon" in err
    assert not db.exists()
    assert _autorun(config, db, sessions=1, generations=3, shape="square") == 0


def test_autorun_refuses_a_db_that_is_not_schema_v2(tmp_path, log_dir, capsys):
    """F14 through the CLI: one line, exit 2, the file untouched."""
    db = tmp_path / "species.db"
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE session_metadata (id INTEGER PRIMARY KEY);")
    conn.close()
    before = db.read_bytes()

    assert _autorun(write_config(tmp_path), db) == cli.EXIT_USAGE
    assert "not a LatticeLife schema v2 database" in capsys.readouterr().err
    assert db.read_bytes() == before


def test_autorun_refuses_another_apps_db_that_also_uses_user_version_2(tmp_path, log_dir, capsys):
    """It used to exit 0 after switching the file to WAL and adding the four v2 tables."""
    db = tmp_path / "app.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
        "INSERT INTO users VALUES (1, 'a'); PRAGMA user_version = 2;"
    )
    conn.close()
    before = db.read_bytes()

    assert _autorun(write_config(tmp_path), db) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith("latticelife autorun: ") and err.count("\n") == 1
    assert "not a LatticeLife schema v2 database (user_version=2, tables: users" in err
    assert db.read_bytes() == before
    assert not Path(f"{db}-wal").exists()


def _a_directory(tmp_path):
    (tmp_path / "adir").mkdir()
    return tmp_path / "adir"


def _under_a_file(tmp_path):
    (tmp_path / "afile").write_text("x")
    return tmp_path / "afile" / "run.db"


@pytest.mark.parametrize(
    ("make_db", "message"),
    [
        (_a_directory, "adir cannot be opened as a SQLite database (it is a directory)"),
        (_under_a_file, "cannot create the folder for"),
    ],
    ids=["directory", "parent-is-a-file"],
)
def test_autorun_reports_a_db_path_it_cannot_open_in_one_line(
    tmp_path, log_dir, capsys, make_db, message
):
    """A directory as --db escaped as sqlite3.OperationalError ('unable to open database file')
    with a traceback and exit 1, a file in place of the DB's folder as FileExistsError."""
    db = make_db(tmp_path)

    assert _autorun(write_config(tmp_path), db) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith("latticelife autorun: ") and err.count("\n") == 1
    assert message in err and "Traceback" not in err
    events = [event["event"] for event in _events(log_dir, "app")]
    assert "invalid_input" in events and "unhandled_exception" not in events


def _a_read_only_db(db, cleanup, monkeypatch):
    db.chmod(0o444)
    cleanup.callback(db.chmod, 0o644)
    return db


def _an_empty_file_in_a_read_only_folder(db, cleanup, monkeypatch):
    folder = db.parent / "read-only"
    folder.mkdir()
    (folder / "empty.db").touch()
    folder.chmod(0o555)
    cleanup.callback(folder.chmod, 0o755)
    return folder / "empty.db"


def _a_db_another_process_holds(db, cleanup, monkeypatch):
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode = delete")
    conn.close()
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    cleanup.callback(holder.close)
    # The recorder's busy timeout, cut from 5 s.
    monkeypatch.setattr(sqlite3, "connect", functools.partial(sqlite3.connect, timeout=0.05))
    return db


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
@pytest.mark.parametrize(
    ("make_db", "message"),
    [
        (_a_read_only_db, "run.db cannot be written (attempt to write a readonly database)"),
        (_an_empty_file_in_a_read_only_folder, "empty.db cannot be written"),
        (_a_db_another_process_holds, "run.db is in use by another process (database is locked)"),
    ],
    ids=["read-only-file", "read-only-folder", "in-use"],
)
def test_autorun_reports_a_db_it_cannot_write_in_one_line(
    tmp_path, log_dir, capsys, monkeypatch, make_db, message
):
    """SQLite opens a file it may only read without complaint, so these escaped as
    sqlite3.OperationalError tracebacks with exit 1 at the first write; a DB locked by another
    process was reported as 'is not a SQLite database (database is locked)'."""
    config = write_config(tmp_path)
    recorded = tmp_path / "run.db"
    assert _autorun(config, recorded, sessions=1, generations=3) == 0
    capsys.readouterr()

    with contextlib.ExitStack() as cleanup:
        db = make_db(recorded, cleanup, monkeypatch)
        before = db.read_bytes()
        assert _autorun(config, db, sessions=1, generations=3) == cli.EXIT_USAGE
        err = capsys.readouterr().err
        assert db.read_bytes() == before
    assert err.startswith("latticelife autorun: ") and err.count("\n") == 1
    assert message in err and "Traceback" not in err
    events = [event["event"] for event in _events(log_dir, "app")]
    assert "invalid_input" in events and "unhandled_exception" not in events
    assert _session_rows(recorded) == [(3, 3, 1, 3)]


_AUTORUN_SCRIPT = textwrap.dedent(
    """
    import sys
    from latticelife import cli
    sys.exit(cli.main(sys.argv[1:]))
    """
)


def test_sigterm_flushes_autorun(tmp_path, log_dir):
    """`kill` used to drop up to a whole batch of buffered rows; now SIGTERM unwinds like
    Ctrl+C, so everything recorded so far is flushed and the exit status is 130."""
    db = tmp_path / "run.db"
    config = write_config(tmp_path, recording__batch_size=100000)
    script = tmp_path / "run.py"
    script.write_text(_AUTORUN_SCRIPT)
    argv = ["autorun", "--sessions", "1", "--generations", "1000000", "--shape", "square"]
    proc = subprocess.Popen(
        [sys.executable, str(script), *argv, "--config", str(config), "--db", str(db)],
        env=_python_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert _wait_for(lambda: db.exists())
    time.sleep(1.0)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=30)

    assert proc.returncode == cli.EXIT_INTERRUPTED, err
    summary = json.loads(out)
    assert summary["interrupted"] is True
    [(planned, count, first, last)] = _session_rows(db)
    assert count > 0 and (first, last) == (1, count)
    assert abs(count - summary["generations"]) <= 1  # the signal may land between the two


# --- discover -----------------------------------------------------------------------------------


def test_discover_runs_and_stops_cleanly(tmp_path, log_dir, capsys):
    config = write_config(tmp_path)
    assert cli.main(["discover", "--minutes", "0.02", "--config", str(config)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["shape"] == "square"
    assert Path(summary["checkpoint"]).parent == tmp_path / "out" / "discovery"
    assert summary["generation"] >= 1
    events = [e["event"] for e in _events(log_dir, "discovery")]
    assert events[0] == "worker_started" and events[-1] == "worker_stopped"


def test_discover_reports_a_crashed_worker(tmp_path, log_dir, capsys):
    """F18: a worker crash (corrupt state, bad settings) exited 0 and logged nothing."""
    config = write_config(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "discovery").write_text("a file where the checkpoint directory goes")

    assert cli.main(["discover", "--minutes", "1", "--config", str(config)]) == cli.EXIT_FAILED

    assert "the discovery worker failed (exit code 1)" in capsys.readouterr().err
    assert [e["event"] for e in _events(log_dir, "discovery")] == ["worker_crashed"]
    failed = _events(log_dir, "app")[-1]
    assert (failed["event"], failed["exitcode"]) == ("discovery_worker_failed", 1)


def test_discover_refuses_a_lineage_another_run_holds(tmp_path, log_dir, capsys):
    """F66: a second `discover` on the same checkpoint replayed the first run's search."""
    config_path = write_config(tmp_path)
    from latticelife.config import load_config

    lineage = DiscoveryWorker(load_config(config_path), "square").checkpoint_path
    lock = acquire_lineage_lock(lineage)
    try:
        t0 = time.monotonic()
        assert cli.main(["discover", "--config", str(config_path)]) == cli.EXIT_FAILED
        assert time.monotonic() - t0 < 15
    finally:
        lock.close()
    assert "another discover run is already using" in capsys.readouterr().err


@pytest.fixture
def unstoppable_worker(monkeypatch):
    """A worker whose stop() is lost, with a short stop timeout."""
    monkeypatch.setattr(DiscoveryWorker, "stop", lambda self: None)
    monkeypatch.setattr(cli, "_STOP_TIMEOUT_SECONDS", 0.5)


def test_discover_kills_a_worker_that_does_not_stop(tmp_path, log_dir, capsys, unstoppable_worker):
    config = write_config(tmp_path)
    assert cli.main(["discover", "--minutes", "0.01", "--config", str(config)]) == 1
    assert "killed (did not stop within 0.5 s)" in capsys.readouterr().err
    assert _events(log_dir, "app")[-1]["event"] == "discovery_worker_killed"


def test_second_ctrl_c_kills_the_worker_at_once(
    tmp_path, log_dir, capsys, monkeypatch, unstoppable_worker
):
    config = write_config(tmp_path)
    monkeypatch.setattr(cli, "_STOP_TIMEOUT_SECONDS", 30.0)
    # Two Ctrl+Cs from outside (a helper process, so no thread exists here when the worker is
    # forked): the first asks for a stop that the worker ignores, the second kills it.
    sender = subprocess.Popen(
        ["sh", "-c", f"sleep 1; kill -INT {os.getpid()}; sleep 1; kill -INT {os.getpid()}"]
    )
    t0 = time.monotonic()
    try:
        status = cli.main(["discover", "--minutes", "5", "--config", str(config)])
    finally:
        # A run that ended early (a broken worker) must not get the Ctrl+Cs still to come:
        # pytest itself would take them and abort the whole session.
        sender.kill()
        sender.wait(10)
    assert status == cli.EXIT_INTERRUPTED
    assert time.monotonic() - t0 < 10
    assert "killed (stop requested again)" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


_DISCOVER_SCRIPT = textwrap.dedent(
    """
    import multiprocessing as mp
    import sys
    from latticelife import cli

    if __name__ == "__main__":
        mp.set_start_method(sys.argv[1])
        sys.exit(cli.main(sys.argv[2:]))
    """
)


def _start_discover(tmp_path, method, minutes="5"):
    script = tmp_path / "discover.py"
    script.write_text(_DISCOVER_SCRIPT)
    argv = ["discover", "--minutes", minutes, "--config", str(write_config(tmp_path))]
    return subprocess.Popen(
        [sys.executable, str(script), method, *argv],
        env=_python_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,  # its own process group, like a terminal's foreground job
    )


def _worker_pid(log_dir):
    assert _wait_for(
        lambda: any(e["event"] == "generation_complete" for e in _events(log_dir, "discovery"))
    )
    return _events(log_dir, "discovery")[0]["pid"]


def _gone(pid):
    """True once `pid` has exited (a zombie waiting for its new parent to reap it counts)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):  # gone before, or while, reading
        return True
    return state in ("Z", "X")


@pytest.mark.parametrize("method", mp.get_all_start_methods())
def test_ctrl_c_stops_discover_cleanly_under_every_start_method(tmp_path, log_dir, method):
    """F67/F68: Ctrl+C reaches the whole process group. Under spawn/forkserver the worker died
    with a KeyboardInterrupt traceback and no worker_stopped; under fork its inherited handler
    could deadlock on the stop event's lock, hanging both processes."""
    proc = _start_discover(tmp_path, method)
    try:
        _worker_pid(log_dir)
        os.killpg(proc.pid, signal.SIGINT)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)

    assert proc.returncode == 0, out
    assert "Traceback" not in out and "KeyboardInterrupt" not in out
    stopped = _events(log_dir, "discovery")[-1]
    assert (stopped["event"], stopped["reason"]) == ("worker_stopped", "requested")


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_sigterm_to_discover_alone_stops_its_worker(tmp_path, log_dir):
    """F19: SIGTERM to the parent PID only (kill, an IDE's stop button) left the worker
    orphaned and running forever."""
    proc = _start_discover(tmp_path, mp.get_start_method())
    try:
        worker_pid = _worker_pid(log_dir)
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)

    assert proc.returncode == 0, out
    assert _wait_for(lambda: _gone(worker_pid), timeout=10)
    assert _events(log_dir, "discovery")[-1]["event"] == "worker_stopped"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_sigkill_to_discover_still_stops_its_worker(tmp_path, log_dir):
    proc = _start_discover(tmp_path, mp.get_start_method())
    try:
        worker_pid = _worker_pid(log_dir)
        proc.kill()
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)

    assert _wait_for(lambda: _gone(worker_pid), timeout=15)
    stopped = _events(log_dir, "discovery")[-1]
    assert (stopped["event"], stopped["reason"]) == ("worker_stopped", "parent exited")
