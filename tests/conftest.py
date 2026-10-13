"""Suite-wide isolation: no test may touch the project's real `out/` directory. It holds recorded
data, and a test run once replaced its discovered-rules file with test rows: a regression of F17
resolved that file against the current directory, and pytest had been started from the checkout.
Four guards:

- JSONL logs go to a per-session temporary directory via `LATTICELIFE_LOG_DIR`, which overrides
  every configured log directory and is inherited by discovery worker processes.
- Every test runs with its own `tmp_path` as the current directory, so a path that wrongly
  resolves against the current directory lands there, never in the checkout.
- An audit hook (`subprocess_site/out_guard.py`) refuses, with PermissionError, anything beyond
  reading that this process, a worker forked from it or a Python subprocess a test starts does
  to `PROJECT_ROOT/out`: opening a file there for writing; creating, renaming, removing or
  touching anything; connecting SQLite to a database there — by its own path, through a
  symlink, or by a name relative to a directory descriptor open on it. Subprocesses get it from
  `subprocess_site/sitecustomize.py`, so a test that sets PYTHONPATH for one must keep the
  inherited entries after its own (`os.pathsep.join([..., os.environ["PYTHONPATH"]])`). A test
  whose code had a change refused fails at teardown, even if that code caught the error. Tests
  of the guard itself point it at a stand-in with `guard_also`, which keeps the real out/
  guarded too.
- At the end of the session `out/` must hold exactly what it held at the start (names, sizes,
  modification times). That also catches any writer the hook cannot see — including a
  latticelife run of your own that writes there while the suite runs.

Tests that write recordings, checkpoints or discovered rules must pass tmp paths themselves —
`tmp_output_config` does that for a config loaded from `config/default.toml`."""

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from latticelife.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from latticelife.logging import LOG_DIR_ENV

SUBPROCESS_SITE = Path(__file__).resolve().parent / "subprocess_site"
PROJECT_OUT = PROJECT_ROOT / "out"


def _load_out_guard():
    spec = importlib.util.spec_from_file_location("out_guard", SUBPROCESS_SITE / "out_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_guard_module = _load_out_guard()
_OUT_GUARD = _guard_module.OutGuard(PROJECT_OUT)


@pytest.fixture(autouse=True, scope="session")
def _isolated_log_dir(tmp_path_factory):
    patcher = pytest.MonkeyPatch()
    patcher.setenv(LOG_DIR_ENV, str(tmp_path_factory.mktemp("logs")))
    yield
    patcher.undo()


@pytest.fixture(autouse=True, scope="session")
def _project_out_untouched():
    # Subprocesses: the guard's sitecustomize first on PYTHONPATH, then the entries pytest was
    # started with — made absolute, since every test runs in its own directory.
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    patcher = pytest.MonkeyPatch()
    patcher.setenv(_guard_module.PROTECTED_DIRS_ENV, os.pathsep.join(_OUT_GUARD.roots))
    patcher.setenv(
        "PYTHONPATH", os.pathsep.join([str(SUBPROCESS_SITE), *map(os.path.abspath, inherited)])
    )
    # This process and its forked workers. An audit hook cannot be removed: `active` switches
    # this one off after the session.
    sys.addaudithook(_OUT_GUARD)
    before = _OUT_GUARD.snapshot(PROJECT_OUT)
    _OUT_GUARD.active = True
    yield
    _OUT_GUARD.active = False
    patcher.undo()
    changed = _OUT_GUARD.changes(before, _OUT_GUARD.snapshot(PROJECT_OUT))
    if changed:
        pytest.fail(
            f"the project's real out/ ({PROJECT_OUT}) changed during the test session: "
            + ", ".join(changed)
            + ". A test or a process it started wrote there (or a latticelife run of your own"
            " did, while the suite ran). Tests must write to tmp paths.",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Each test runs in its own `tmp_path`, so a path resolved against the current directory
    (F17's bug) lands there instead of in the checkout's `out/`."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _out_guard_check():
    yield
    refused, _OUT_GUARD.refused = _OUT_GUARD.refused, []
    if refused:
        pytest.fail(
            f"tried to change the project's real out/ ({PROJECT_OUT}): " + "; ".join(refused),
            pytrace=False,
        )


@pytest.fixture
def out_guard():
    """The suite's `OutGuard`, for the tests of the guard itself."""
    return _OUT_GUARD


@pytest.fixture
def guard_also(monkeypatch):
    """`guard_also(directory)` guards `directory` like the project's real out/ for the rest of
    the test — in this process, the workers it forks and the Python subprocesses it starts — so
    a test can watch the guard refuse a change without aiming anything at the real out/, which
    stays guarded."""

    def guard(directory) -> None:
        roots = (*_OUT_GUARD.roots, os.path.normpath(directory))
        monkeypatch.setattr(_OUT_GUARD, "roots", roots)
        monkeypatch.setenv(_guard_module.PROTECTED_DIRS_ENV, os.pathsep.join(roots))

    return guard


def redirect_outputs(config, out_dir: Path):
    """`config` with every output path (db, checkpoint, discovered rules, logs) under
    `out_dir`."""
    return replace(
        config,
        recording=replace(config.recording, db_path=str(out_dir / "latticelife.db")),
        discovery=replace(
            config.discovery,
            checkpoint_path=str(out_dir / "discovery"),
            discovered_path=str(out_dir / "discovered_rules.jsonl"),
        ),
        logging=replace(config.logging, dir=str(out_dir / "logs")),
    )


@pytest.fixture
def tmp_output_config(tmp_path):
    """The default config with all output paths redirected under `tmp_path / "out"`."""
    return redirect_outputs(load_config(DEFAULT_CONFIG_PATH), tmp_path / "out")
