import json
import time
from dataclasses import replace
from pathlib import Path

from latticelife.config import load_config
from latticelife.discovery.worker import DiscoveryWorker
from latticelife.session.session import SimulationSession

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


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


def test_worker_runs_and_produces_a_checkpoint(tmp_path):
    config = _fast_config(tmp_path)
    worker = DiscoveryWorker(config, shape="square")
    worker.start()
    try:
        deadline = time.monotonic() + 20
        while not worker.checkpoint_path.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        worker.stop()
        worker.join(timeout=5)

    assert worker.checkpoint_path.exists()
    data = json.loads(worker.checkpoint_path.read_text())
    assert data["generation"] >= 1
    assert len(data["population"]) == config.discovery.mu
    assert not worker.is_alive()


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


def test_resume_from_checkpoint_does_not_regress_generation(tmp_path):
    config = _fast_config(tmp_path)

    worker1 = DiscoveryWorker(config, shape="square")
    worker1.start()
    time.sleep(2.0)
    worker1.stop()
    worker1.join(timeout=5)
    gen1 = json.loads(worker1.checkpoint_path.read_text())["generation"]

    worker2 = DiscoveryWorker(config, shape="square")
    worker2.start()
    time.sleep(2.0)
    worker2.stop()
    worker2.join(timeout=5)
    gen2 = json.loads(worker2.checkpoint_path.read_text())["generation"]

    assert gen2 >= gen1


def test_discovery_worker_does_not_contaminate_main_process_rng(tmp_path):
    """D-3: a discovery worker running in its own process must never affect the main process's
    own seeded SimulationSession trajectories — no shared RNG, no shared anything."""
    config = _fast_config(tmp_path)
    worker = DiscoveryWorker(config, shape="square")
    worker.start()
    try:

        def trajectory() -> bytes:
            session = SimulationSession.from_config(config, seed=42)
            session.run(20)
            return bytes(session.grid.occupancy)

        a, b = trajectory(), trajectory()
    finally:
        worker.stop()
        worker.join(timeout=5)

    assert a == b


def test_worker_writes_rules_and_logs_where_the_config_says(tmp_path, monkeypatch):
    """F16: the discovered-rules file and the worker's JSONL log follow the config, not the
    current directory. The parent's own log dir is pointed elsewhere under tmp_path, so a child
    that ignored the config would log there (never into the project's real out/)."""
    from latticelife import logging as ll_logging

    monkeypatch.delenv(ll_logging.LOG_DIR_ENV, raising=False)
    monkeypatch.setattr(ll_logging, "_configured_dir", tmp_path / "parent-logs")
    monkeypatch.chdir(tmp_path)
    config = _fast_config(tmp_path)
    worker = DiscoveryWorker(config, shape="square")
    assert worker.discovered_path == tmp_path / "discovered_rules.jsonl"

    worker.start()
    try:
        deadline = time.monotonic() + 20
        while not worker.checkpoint_path.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        worker.stop()
        worker.join(timeout=5)

    events = (tmp_path / "logs" / "discovery.jsonl").read_text().splitlines()
    assert json.loads(events[0])["event"] == "worker_started"
    assert not (tmp_path / "parent-logs").exists()
    assert not (tmp_path / "out").exists()
