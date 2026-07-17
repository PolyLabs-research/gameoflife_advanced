from pathlib import Path

from latticelife.config import load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.toml"


def test_load_default_config():
    config = load_config(DEFAULT_CONFIG_PATH)

    assert config.grid.shape == "square"
    assert config.grid.wrap is True
    assert config.simulation.seed == 0
    assert config.discovery.mu == 5
    assert config.criticality.weights.alive == 0.3
    assert config.criticality.alive_band.target_high == 0.40
    assert config.ui.fps == 30
