import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latticelife.config import load_config
from latticelife.experiments.criticality import Band, _band_score, score_timeseries

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


# F64: the ramp, freeze-penalty and damage branches had no tests at all.

BAND = Band(low=0.0, high=1.0, target_low=0.2, target_high=0.6)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.1, 0.0),
        (0.0, 0.0),
        (0.05, 0.25),
        (0.1, 0.5),
        (0.2, 1.0),
        (0.4, 1.0),
        (0.6, 1.0),
        (0.7, 0.75),
        (0.8, 0.5),
        (1.0, 0.0),
        (1.1, 0.0),
        (math.nan, 0.0),
    ],
)
def test_band_score_ramps_linearly_into_and_out_of_the_target(value, expected):
    assert _band_score(value, BAND) == pytest.approx(expected)


def _mid_band(steps=100):
    rng = np.random.default_rng(0)
    rho = 0.2 + 0.05 * rng.standard_normal(steps)
    activity = np.abs(np.diff(rho, prepend=rho[0]))
    return rho, activity


def test_freeze_penalty_scales_with_how_early_the_freeze_was_detected():
    config = load_config(DEFAULT_CONFIG_PATH).criticality
    rho, activity = _mid_band()
    free = score_timeseries(rho, activity, None, steps=100, config=config)
    early = score_timeseries(rho, activity, 10, steps=100, config=config)
    late = score_timeseries(rho, activity, 90, steps=100, config=config)

    assert 0 < free < 1
    assert early == pytest.approx(free * (1 - config.freeze_penalty_weight * 0.9))
    assert late == pytest.approx(free * (1 - config.freeze_penalty_weight * 0.1))


def test_damage_inside_its_bands_raises_the_score():
    config = load_config(DEFAULT_CONFIG_PATH).criticality
    rho, activity = _mid_band()
    steps = 100
    h = np.linspace(0.0, 0.3, steps)  # mean in damage_band's target, growth in its target too
    without = score_timeseries(rho, activity, None, steps=steps, config=config)
    with_damage = score_timeseries(rho, activity, None, steps=steps, config=config, h=h)
    assert with_damage == pytest.approx(min(1.0, without + config.weights.damage * 1.0))

    no_weight = replace(config, weights=replace(config.weights, damage=0.0))
    assert score_timeseries(rho, activity, None, steps, no_weight, h=h) == pytest.approx(without)


@pytest.mark.parametrize("where", ["rho", "activity", "h"])
def test_non_finite_input_scores_zero_not_one(where):
    """A NaN used to slip through every band check and the final clamp as a perfect 1.0."""
    config = load_config(DEFAULT_CONFIG_PATH).criticality
    rho, activity = _mid_band()
    h = np.linspace(0.0, 0.3, 100)
    {"rho": rho, "activity": activity, "h": h}[where][50] = np.nan
    assert score_timeseries(rho, activity, None, steps=100, config=config, h=h) == 0.0
