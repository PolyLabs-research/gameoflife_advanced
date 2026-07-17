from pathlib import Path

import numpy as np

from latticelife.config import load_config
from latticelife.experiments.criticality import score_timeseries

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.toml"


def test_score_timeseries_empty_is_zero():
    config = load_config(DEFAULT_CONFIG_PATH).criticality
    score = score_timeseries(np.array([]), np.array([]), None, steps=100, config=config)
    assert score == 0.0


def test_score_timeseries_mid_band_beats_extinct():
    config = load_config(DEFAULT_CONFIG_PATH).criticality
    steps = 100
    mid_rho = np.full(steps, 0.2)
    mid_activity = np.full(steps, 0.02)
    extinct_rho = np.zeros(steps)
    extinct_activity = np.zeros(steps)

    mid_score = score_timeseries(mid_rho, mid_activity, None, steps=steps, config=config)
    extinct_score = score_timeseries(
        extinct_rho, extinct_activity, None, steps=steps, config=config
    )

    assert mid_score > extinct_score
