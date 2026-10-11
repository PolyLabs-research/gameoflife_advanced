"""Suite-wide isolation: no test may write into the project's real `out/` directory (it holds
recorded data). JSONL logs go to a per-session temporary directory via `LATTICELIFE_LOG_DIR`,
which overrides every configured log directory and is inherited by discovery worker processes.
Tests that write recordings, checkpoints or discovered rules must pass tmp paths themselves —
`tmp_output_config` does that for a config loaded from `config/default.toml`."""

from dataclasses import replace
from pathlib import Path

import pytest

from latticelife.config import DEFAULT_CONFIG_PATH, load_config
from latticelife.logging import LOG_DIR_ENV


@pytest.fixture(autouse=True, scope="session")
def _isolated_log_dir(tmp_path_factory):
    patcher = pytest.MonkeyPatch()
    patcher.setenv(LOG_DIR_ENV, str(tmp_path_factory.mktemp("logs")))
    yield
    patcher.undo()


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
