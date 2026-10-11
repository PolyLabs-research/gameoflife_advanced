import json
import multiprocessing as mp
import os
import random
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import latticelife
from latticelife.config import load_config
from latticelife.core.rules import Rule
from latticelife.discovery import es
from latticelife.discovery import worker as worker_module
from latticelife.discovery.worker import EXIT_LOCKED, DiscoveryWorker, acquire_lineage_lock
from latticelife.logging import LOG_DIR_ENV
from latticelife.session.session import SimulationSession

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"
SRC_DIR = Path(latticelife.__file__).resolve().parents[1]


def _fast_config(tmp_path, **discovery_overrides):
    """A discovery config trimmed for fast worker-process tests: small population/grid/steps so
    a generation completes in well under a second, and per-test checkpoint, discovered-rules and
    log paths so a test run never appends to the project's real `out/`."""
    base = load_config(DEFAULT_CONFIG_PATH)
    overrides = {
        "mu": 2,
        "n_offspring": 2,
        "evals_per_candidate": 1,
        "grid_size": 8,
        "steps": 15,
        "checkpoint_path": str(tmp_path / "discovery"),
        "discovered_path": str(tmp_path / "discovered_rules.jsonl"),
    }
    overrides.update(discovery_overrides)
    return replace(
        base,
        discovery=replace(base.discovery, **overrides),
        logging=replace(base.logging, dir=str(tmp_path / "logs")),
    )


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """This test's own JSONL log directory (inherited by worker processes)."""
    path = tmp_path / "logs"
    monkeypatch.setenv(LOG_DIR_ENV, str(path))
    return path


def _events(log_dir, channel="discovery"):
    path = Path(log_dir) / f"{channel}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wait_for(predicate, timeout=20.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _checkpoint(worker):
    return json.loads(worker.checkpoint_path.read_text())


def _stop(worker, timeout=10):
    worker.stop()
    worker.join(timeout=timeout)
    assert not worker.is_alive()


class _StopAfterEvaluations:
    """In-process stand-ins for the worker's stop flag and for `es.evaluate_candidate`: every
    evaluation scores `score` (or raises `error`), and the flag reads True once `limit`
    evaluations have run."""

    def __init__(self, limit, score=0.5, error=None):
        self.limit = limit
        self.score = score
        self.error = error
        self.calls = 0

    @property
    def value(self):
        return self.calls >= self.limit

    def evaluate(self, rule, shape, config, rng):
        if self.error is not None:
            raise self.error
        self.calls += 1
        return self.score(rule) if callable(self.score) else self.score


def _run_in_process(config, shape, monkeypatch, stub, worker=None):
    worker = worker or DiscoveryWorker(config, shape=shape)
    monkeypatch.setattr(es, "evaluate_candidate", stub.evaluate)
    worker_module._run_loop(
        config, shape, worker.checkpoint_path, worker.discovered_path, stub, None
    )
    return worker


# --- the worker process -------------------------------------------------------------------------


def test_worker_runs_and_produces_a_checkpoint(tmp_path):
    config = _fast_config(tmp_path)
    worker = DiscoveryWorker(config, shape="square")
    worker.start()
    try:
        _wait_for(worker.checkpoint_path.exists)
    finally:
        _stop(worker, timeout=5)

    assert worker.checkpoint_path.parent == tmp_path / "discovery"
    assert worker.checkpoint_path.name.startswith("checkpoint-square-")
    data = _checkpoint(worker)
    assert data["generation"] >= 1
    assert data["settings"] == worker.settings
    assert len(data["population"]) == config.discovery.mu
    assert worker.exitcode == 0


def test_stop_latency_under_two_seconds(tmp_path):
    config = _fast_config(tmp_path)
    worker = DiscoveryWorker(config, shape="square")
    worker.start()
    time.sleep(1.0)

    t0 = time.monotonic()
    worker.stop()
    worker.join(timeout=5)
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0
    assert not worker.is_alive()
    assert worker.exitcode == 0


def test_stop_flag_takes_no_lock(tmp_path):
    """F68: stop() used to set an mp.Event, which takes a non-reentrant lock that the
    interrupted code could already hold. The flag is now a raw shared byte."""
    worker = DiscoveryWorker(_fast_config(tmp_path), shape="square")
    assert not hasattr(worker._stop_flag, "get_lock")
    worker.stop()
    assert worker._stop_flag.value is True


def test_main_process_sessions_ignore_global_rng_state_while_a_worker_runs(tmp_path):
    """D-3 (F64): the old version compared two sessions that could not differ. Here the global
    numpy and stdlib RNGs are reseeded differently before each trajectory: a session that drew
    from global state, rather than its own seeded Generator, would diverge."""
    config = _fast_config(tmp_path)
    worker = DiscoveryWorker(config, shape="square")
    worker.start()
    try:

        def trajectory(global_seed: int) -> bytes:
            np.random.seed(global_seed)
            random.seed(global_seed)
            session = SimulationSession.from_config(config, seed=42)
            session.run(20)
            return bytes(session.grid.occupancy)

        a, b = trajectory(1), trajectory(2)
    finally:
        _stop(worker, timeout=5)

    assert a == b


def test_worker_writes_rules_and_logs_where_the_config_says(tmp_path, monkeypatch):
    """F16/F17: the discovered-rules file and the worker's JSONL log follow the config, not the
    current directory. The parent's own log dir is pointed elsewhere under tmp_path, so a child
    that ignored the config would log there (never into the project's real out/)."""
    from latticelife import logging as ll_logging

    monkeypatch.delenv(ll_logging.LOG_DIR_ENV, raising=False)
    monkeypatch.setattr(ll_logging, "_configured_dir", tmp_path / "parent-logs")
    monkeypatch.chdir(tmp_path)
    config = _fast_config(tmp_path, save_threshold=1e-9)
    worker = DiscoveryWorker(config, shape="square")
    assert worker.discovered_path == tmp_path / "discovered_rules.jsonl"

    worker.start()
    try:
        _wait_for(lambda: worker.discovered_path.exists() and worker.checkpoint_path.exists())
    finally:
        _stop(worker, timeout=5)

    events = _events(tmp_path / "logs")
    assert events[0]["event"] == "worker_started"
    rows = [json.loads(line) for line in worker.discovered_path.read_text().splitlines()]
    assert rows
    assert set(rows[0]) == {
        "ts",
        "shape",
        "triangle_mode",
        "grid_size",
        "steps",
        "settings",
        "rule",
        "score",
        "evals",
    }
    assert rows[0]["settings"] == es.settings_id(worker.settings)
    assert not (tmp_path / "parent-logs").exists()
    assert not (tmp_path / "out").exists()


def test_real_worker_resumes_its_lineage(tmp_path, log_dir):
    """F63: the old test (gen2 >= gen1 after two 2 s runs) passed most of the time even with
    resume disabled. A crafted checkpoint at generation 1000 must be continued, not restarted."""
    config = _fast_config(tmp_path)
    worker = DiscoveryWorker(config, shape="square")
    state = es.ESState(
        generation=1000,
        population=[
            es.Candidate(Rule({3}, {2, 3}), 5.0, 5),
            es.Candidate(Rule({3, 6}, {2, 3}), 5.0, 5),
        ],
        shape="square",
        entropy=987654321,
        settings=worker.settings,
    )
    worker_module._save_checkpoint(worker.checkpoint_path, state)

    worker.start()
    try:
        assert _wait_for(lambda: _checkpoint(worker)["generation"] > 1000)
    finally:
        _stop(worker)

    data = _checkpoint(worker)
    assert data["entropy_seed"] == 987654321
    assert _events(log_dir)[0]["event"] == "worker_started"
    assert _events(log_dir)[0]["generation"] == 1000


# --- resume, lineages and checkpoint files (in-process) -----------------------------------------


def test_resume_continues_generation_entropy_and_evidence(tmp_path, log_dir, monkeypatch):
    config = _fast_config(tmp_path)
    worker = DiscoveryWorker(config, shape="square")
    crafted = [Rule({3}, {2, 3}), Rule({3, 6}, {2, 3})]
    state = es.ESState(
        generation=1000,
        population=[es.Candidate(rule, 5.0, 5) for rule in crafted],
        shape="square",
        entropy=42,
        settings=worker.settings,
    )
    worker_module._save_checkpoint(worker.checkpoint_path, state)

    pool_size = config.discovery.mu + config.discovery.n_offspring
    _run_in_process(config, "square", monkeypatch, _StopAfterEvaluations(pool_size), worker)

    data = _checkpoint(worker)
    assert data["generation"] == 1001
    assert data["entropy_seed"] == 42
    # The crafted elites (mean 1.0 over 5 evals) beat offspring scoring 0.5, and their sixth
    # evaluation adds to the same running mean (§6.11).
    assert [Rule(set(p["birth"]), set(p["survive"])) for p in data["population"]] == crafted
    assert data["eval_counts"] == [6, 6]
    assert data["scores"] == pytest.approx([5.5 / 6, 5.5 / 6])


def test_changing_shape_starts_a_separate_lineage(tmp_path, log_dir, monkeypatch):
    """F20: a hexagon run used to resume the square population (counts 7 and 8 included),
    merge its square evaluations and write rows such as 'B0245678/S8' labelled hexagon."""
    config = _fast_config(tmp_path, save_threshold=1e-9)
    pool_size = config.discovery.mu + config.discovery.n_offspring
    square = _run_in_process(
        config, "square", monkeypatch, _StopAfterEvaluations(3 * pool_size, score=0.9)
    )
    square_checkpoint = square.checkpoint_path.read_text()
    assert json.loads(square_checkpoint)["generation"] == 3

    hexagon = _run_in_process(
        config, "hexagon", monkeypatch, _StopAfterEvaluations(pool_size, score=0.9)
    )

    assert hexagon.checkpoint_path != square.checkpoint_path
    assert square.checkpoint_path.read_text() == square_checkpoint
    data = _checkpoint(hexagon)
    assert data["generation"] == 1  # a fresh lineage, not square's generation 3 + 1
    assert all(max([*p["birth"], *p["survive"]], default=0) <= 6 for p in data["population"])
    started = [e for e in _events(log_dir) if e["event"] == "worker_started"]
    assert [e["generation"] for e in started] == [0, 0]
    for row in map(json.loads, hexagon.discovered_path.read_text().splitlines()):
        if row["shape"] == "hexagon":
            rule = Rule.from_bs_string(row["rule"])
            assert max(rule.birth | rule.survive, default=0) <= 6


def test_changing_triangle_mode_or_steps_starts_a_separate_lineage(tmp_path):
    base = _fast_config(tmp_path)
    edge = replace(base, grid=replace(base.grid, triangle_mode="edge"))
    paths = {
        DiscoveryWorker(base, "triangle").checkpoint_path,
        DiscoveryWorker(edge, "triangle").checkpoint_path,
        DiscoveryWorker(
            replace(base, discovery=replace(base.discovery, steps=30)), "triangle"
        ).checkpoint_path,
    }
    assert len(paths) == 3
    assert DiscoveryWorker(edge, "triangle").checkpoint_path.name.startswith(
        "checkpoint-triangle-edge-"
    )


@pytest.mark.parametrize(
    "content",
    ['{"generation": 3, "popul', "", '{"generation": 3}{"tail": 1}', '{"version": 1}'],
)
def test_unusable_checkpoint_is_set_aside_and_the_run_starts_fresh(
    tmp_path, log_dir, monkeypatch, content
):
    """F65: a truncated or torn checkpoint made every later run crash at startup (and exit 0)."""
    config = _fast_config(tmp_path)
    worker = DiscoveryWorker(config, shape="square")
    worker.checkpoint_path.parent.mkdir(parents=True)
    worker.checkpoint_path.write_text(content)

    pool_size = config.discovery.mu + config.discovery.n_offspring
    _run_in_process(config, "square", monkeypatch, _StopAfterEvaluations(pool_size), worker)

    assert _checkpoint(worker)["generation"] == 1
    aside = list(worker.checkpoint_path.parent.glob(worker.checkpoint_path.name + ".unusable-*"))
    assert len(aside) == 1
    assert aside[0].read_text() == content
    assert [e["event"] for e in _events(log_dir)][:2] == ["checkpoint_unusable", "worker_started"]


def test_checkpoint_with_other_settings_is_set_aside(tmp_path, log_dir, monkeypatch):
    config = _fast_config(tmp_path)
    worker = DiscoveryWorker(config, shape="square")
    other = es.search_settings(
        replace(config, discovery=replace(config.discovery, steps=99)), "square"
    )
    state = es.ESState(7, [es.Candidate(Rule({3}, {2, 3}), 0.5, 1)], "square", 1, other)
    worker_module._save_checkpoint(worker.checkpoint_path, state)

    pool_size = config.discovery.mu + config.discovery.n_offspring
    _run_in_process(config, "square", monkeypatch, _StopAfterEvaluations(pool_size), worker)

    assert _checkpoint(worker)["generation"] == 1
    assert "settings differ" in _events(log_dir)[0]["error"]


def test_legacy_checkpoint_json_is_left_alone(tmp_path, log_dir, monkeypatch):
    config = _fast_config(tmp_path)
    legacy = tmp_path / "discovery" / "checkpoint.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"generation": 5, "shape": "square"}')

    _run_in_process(config, "square", monkeypatch, _StopAfterEvaluations(4))

    assert legacy.read_text() == '{"generation": 5, "shape": "square"}'
    assert _events(log_dir)[0]["event"] == "legacy_checkpoint_ignored"


def _save_loop(path, entries, seconds):
    rule = Rule({3}, {2, 3})
    settings = es.search_settings(load_config(DEFAULT_CONFIG_PATH), "square")
    state = es.ESState(1, [es.Candidate(rule, 0.5, 1)] * entries, "square", 1, settings)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        worker_module._save_checkpoint(path, state)


def test_concurrent_checkpoint_saves_never_crash_or_tear(tmp_path):
    """F65: both writers used the fixed name checkpoint.tmp, so one writer's rename failed with
    FileNotFoundError and readers saw empty or torn JSON."""
    path = tmp_path / "discovery" / "checkpoint-square-x.json"
    path.parent.mkdir(parents=True)
    context = mp.get_context()
    writers = [context.Process(target=_save_loop, args=(path, n, 1.5)) for n in (40, 2)]
    for writer in writers:
        writer.start()
    bad_reads = []
    reads = 0
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        try:
            es.from_checkpoint_dict(json.loads(path.read_text()))
            reads += 1
        except FileNotFoundError:
            continue
        except ValueError as exc:
            bad_reads.append(exc)
    for writer in writers:
        writer.join(10)

    assert [writer.exitcode for writer in writers] == [0, 0]
    assert bad_reads == []
    assert reads > 0
    assert list(path.parent.glob("*.tmp")) == []


def test_second_worker_on_a_locked_lineage_exits_and_others_still_run(tmp_path, log_dir):
    """F66: a second run on the same checkpoint replayed the first one's generations and both
    overwrote the checkpoint. It now refuses; a different lineage is unaffected."""
    config = _fast_config(tmp_path)
    square = DiscoveryWorker(config, shape="square")
    lock = acquire_lineage_lock(square.checkpoint_path)
    assert lock is not None
    hexagon = DiscoveryWorker(config, shape="hexagon")
    try:
        square.start()
        hexagon.start()
        square.join(timeout=20)
        assert square.exitcode == EXIT_LOCKED
        assert _wait_for(hexagon.checkpoint_path.exists)
        assert hexagon.is_alive()
    finally:
        _stop(hexagon)
        lock.close()

    assert not square.checkpoint_path.exists()
    locked = [e for e in _events(log_dir) if e["event"] == "checkpoint_locked"]
    assert len(locked) == 1
    assert locked[0]["checkpoint"] == str(square.checkpoint_path)
    assert acquire_lineage_lock(square.checkpoint_path) is not None  # released with the process


# --- discovered rules ---------------------------------------------------------------------------


def test_each_rule_is_written_once_per_lineage(tmp_path, log_dir, monkeypatch):
    """F22: every elite above save_threshold was appended again every generation (one rule 22
    times in 15 generations)."""
    config = _fast_config(tmp_path, save_threshold=0.5)
    pool_size = config.discovery.mu + config.discovery.n_offspring
    worker = _run_in_process(
        config, "square", monkeypatch, _StopAfterEvaluations(4 * pool_size, score=0.9)
    )
    rows = [json.loads(line) for line in worker.discovered_path.read_text().splitlines()]
    rules = [row["rule"] for row in rows]
    assert rules and len(rules) == len(set(rules))

    # A resumed run rebuilds the set from the file and does not repeat them either.
    _run_in_process(config, "square", monkeypatch, _StopAfterEvaluations(4 * pool_size, 0.9))
    rules = [json.loads(line)["rule"] for line in worker.discovered_path.read_text().splitlines()]
    assert len(rules) == len(set(rules))
    assert _checkpoint(worker)["generation"] == 8


def test_rules_already_in_the_file_for_this_lineage_are_not_repeated(tmp_path, monkeypatch):
    config = _fast_config(tmp_path, save_threshold=0.5)
    worker = DiscoveryWorker(config, shape="square")
    elite = Rule({3}, {2, 3})
    state = es.ESState(1, [es.Candidate(elite, 0.9, 1)] * 2, "square", 5, worker.settings)
    worker_module._save_checkpoint(worker.checkpoint_path, state)
    lineage = es.settings_id(worker.settings)
    worker.discovered_path.write_text(
        "not json\n"
        + json.dumps({"rule": "B3/S23", "settings": "another-lineage"})
        + "\n"
        + json.dumps({"rule": "B3/S23", "settings": lineage})
        + "\n"
    )

    def score(rule):
        return 0.9 if rule == elite else 0.1

    _run_in_process(config, "square", monkeypatch, _StopAfterEvaluations(3, score), worker)

    lines = worker.discovered_path.read_text().splitlines()
    assert len(lines) == 3  # nothing appended: B3/S23 was already saved for this lineage


# --- failures and stopping ----------------------------------------------------------------------


def test_crash_is_logged_with_its_traceback_and_re_raised(tmp_path, log_dir, monkeypatch):
    """F18: a crash in the worker left no event behind."""
    config = _fast_config(tmp_path)
    stub = _StopAfterEvaluations(100, error=RuntimeError("evaluation exploded"))
    with pytest.raises(RuntimeError, match="evaluation exploded"):
        _run_in_process(config, "square", monkeypatch, stub)

    crashed = [e for e in _events(log_dir) if e["event"] == "worker_crashed"]
    assert len(crashed) == 1
    assert "evaluation exploded" in crashed[0]["error"]
    assert "Traceback" in crashed[0]["traceback"]
    assert acquire_lineage_lock(DiscoveryWorker(config, "square").checkpoint_path) is not None


def test_crashed_worker_process_reports_a_nonzero_exitcode(tmp_path, log_dir):
    config = _fast_config(tmp_path)
    (tmp_path / "discovery").write_text("a file where the checkpoint directory should be")
    worker = DiscoveryWorker(config, shape="square")
    worker.start()
    worker.join(timeout=20)

    assert worker.exitcode == 1
    assert [e["event"] for e in _events(log_dir)] == ["worker_crashed"]


def test_time_limit_stops_the_worker(tmp_path, log_dir, monkeypatch):
    config = _fast_config(tmp_path)
    monkeypatch.setattr(es, "evaluate_candidate", _StopAfterEvaluations(10**6).evaluate)
    worker = DiscoveryWorker(config, shape="square")
    t0 = time.monotonic()
    worker_module._run_loop(
        config, "square", worker.checkpoint_path, worker.discovered_path, worker._stop_flag, 0.3
    )
    assert time.monotonic() - t0 < 5
    stopped = _events(log_dir)[-1]
    assert (stopped["event"], stopped["reason"]) == ("worker_stopped", "time limit")


def test_in_process_run_restores_signal_handlers(tmp_path, log_dir, monkeypatch):
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    _run_in_process(_fast_config(tmp_path), "square", monkeypatch, _StopAfterEvaluations(4))
    after = {sig: signal.getsignal(sig) for sig in before}
    assert after == before
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == set()


def _wait_for_event(log_dir, event, timeout=30.0):
    assert _wait_for(lambda: any(e["event"] == event for e in _events(log_dir)), timeout)
    return next(e for e in _events(log_dir) if e["event"] == event)


@pytest.fixture(params=mp.get_all_start_methods())
def start_method(request):
    """Run the test with each multiprocessing start method as the process default."""
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method(request.param, force=True)
    yield request.param
    mp.set_start_method(previous, force=True)


def test_worker_ignores_sigint_and_stops_cleanly_on_sigterm(tmp_path, log_dir, start_method):
    """F67/F68: Ctrl+C reaches the worker too. Under spawn/forkserver it used to die with a
    KeyboardInterrupt traceback; under fork it ran the parent's handler, which could deadlock.
    The worker now ignores SIGINT (even one sent while it is still starting up) and stops
    cleanly on SIGTERM."""
    worker = DiscoveryWorker(_fast_config(tmp_path), shape="square")
    worker.start()
    try:
        os.kill(worker.pid, signal.SIGINT)  # during start-up: blocked, then ignored
        _wait_for_event(log_dir, "worker_started")
        os.kill(worker.pid, signal.SIGINT)
        time.sleep(0.5)
        assert worker.is_alive()
        os.kill(worker.pid, signal.SIGTERM)
        worker.join(timeout=10)
    finally:
        _stop(worker)

    assert worker.exitcode == 0
    stopped = _events(log_dir)[-1]
    assert (stopped["event"], stopped["reason"]) == ("worker_stopped", "signal SIGTERM")


_ORPHAN_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from dataclasses import replace
    from pathlib import Path
    from latticelife.config import load_config
    from latticelife.discovery.worker import DiscoveryWorker

    if __name__ == "__main__":
        out = Path(sys.argv[2])
        base = load_config(sys.argv[1])
        config = replace(
            base,
            discovery=replace(
                base.discovery, mu=2, n_offspring=2, evals_per_candidate=1, grid_size=8,
                steps=15, checkpoint_path=str(out / "discovery"),
                discovered_path=str(out / "rules.jsonl"),
            ),
        )
        worker = DiscoveryWorker(config, shape="square")
        worker.start()
        time.sleep(600)
    """
)


def _process_gone(pid):
    """True once `pid` has exited (a zombie waiting for its new parent to reap it counts)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):  # gone before, or while, reading
        return True
    return state in ("Z", "X")


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_worker_exits_when_its_parent_is_killed(tmp_path, log_dir):
    """F19: a SIGKILLed (or OOM-killed) parent left the worker running forever, rewriting the
    checkpoint and the rules file."""
    script = tmp_path / "orphan.py"
    script.write_text(_ORPHAN_SCRIPT)
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}
    parent = subprocess.Popen(
        [sys.executable, str(script), str(DEFAULT_CONFIG_PATH), str(tmp_path)], env=env
    )
    try:
        started = _wait_for_event(log_dir, "worker_started")
        assert _wait_for(lambda: (tmp_path / "discovery").exists())
    finally:
        parent.kill()
        parent.wait()

    assert _wait_for(lambda: _process_gone(started["pid"]), timeout=15)
    stopped = _events(log_dir)[-1]
    assert (stopped["event"], stopped["reason"]) == ("worker_stopped", "parent exited")
