import json

import pytest

from latticelife import logging as ll_logging
from latticelife.config import PROJECT_ROOT
from latticelife.logging import LOG_DIR_ENV, configure_logging, log_dir, log_event


@pytest.fixture
def no_env_override(monkeypatch):
    """Drop the suite-wide LATTICELIFE_LOG_DIR and restore the module's configured dir."""
    monkeypatch.delenv(LOG_DIR_ENV, raising=False)
    previous = ll_logging._configured_dir
    yield
    configure_logging(previous)


def test_default_log_dir_is_under_the_project_root_not_the_cwd(
    no_env_override, tmp_path, monkeypatch
):
    """F16: the default must not follow the current directory."""
    configure_logging(None)
    monkeypatch.chdir(tmp_path)
    assert log_dir() == PROJECT_ROOT / "out" / "logs"


def test_configured_dir_receives_events(no_env_override, tmp_path):
    configure_logging(tmp_path / "logs")
    log_event("app", level="info", event="hello", n=1)
    line = (tmp_path / "logs" / "app.jsonl").read_text().strip()
    assert json.loads(line)["event"] == "hello"


def test_env_override_wins_over_configured_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path / "env"))
    previous = ll_logging._configured_dir
    try:
        configure_logging(tmp_path / "configured")
        log_event("discovery", level="info", event="x")
    finally:
        configure_logging(previous)
    assert (tmp_path / "env" / "discovery.jsonl").exists()
    assert not (tmp_path / "configured").exists()


def test_unwritable_log_dir_never_raises(monkeypatch, tmp_path):
    """F16: logging an error must not replace it with an OSError from the log write."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setenv(LOG_DIR_ENV, str(blocker / "logs"))
    log_event("app", level="error", event="unhandled_exception", error="boom")
