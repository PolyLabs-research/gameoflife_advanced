"""Counts over time (`experiments.series.count_series`): the bookkeeping identities hold exactly."""

from pathlib import Path

import numpy as np
import pytest

from latticelife.config import load_config
from latticelife.experiments.series import count_series

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.mark.parametrize("shape, degree", [("square", 8), ("hexagon", 6)])
def test_the_identities_hold(config, shape, degree):
    c = {k: np.array(v) for k, v in count_series(config, "B3/S23", shape, 16, 0.4, 3, 60).items()}
    cells = 16 * 16
    assert np.all(c["live"] == c["live"][0] + c["births"] - c["deaths"])
    assert np.all(c["live"] == sum(c[f"live_{k}"] for k in range(degree + 1)))
    assert np.all(cells == c["live"] + sum(c[f"dead_{k}"] for k in range(degree + 1)))
    weighted = sum(k * (c[f"live_{k}"] + c[f"dead_{k}"]) for k in range(degree + 1))
    assert np.all(weighted == degree * c["live"])
    assert c["births"][0] == c["deaths"][0] == 0 and len(c["live"]) == 60


def test_a_rule_without_births_or_deaths_conserves_the_live_cells(config):
    c = count_series(config, "B/S012345678", "square", 12, 0.3, 1, 20)
    assert len(set(c["live"])) == 1 and c["births"][-1] == c["deaths"][-1] == 0


def test_too_few_steps(config):
    with pytest.raises(ValueError, match="at least 2"):
        count_series(config, "B3/S23", "square", 8, 0.3, 1, 1)
