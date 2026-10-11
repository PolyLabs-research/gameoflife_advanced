from dataclasses import replace

import pytest

from latticelife.config import AlertsConfig
from latticelife.session.alerts import AlertKind, evaluate

CONFIG = AlertsConfig(
    sudden_death_window=10,
    sudden_death_drop=0.8,
    stasis_occupancy=0.80,
    stasis_volatility=5.0,
)


def test_legacy_failure_sequence_preserves_sudden_death_alert():
    """Reproduces the legacy `game.py:update_simulation` scenario that triggered D-4: population
    crashed >80% over the sudden-death window, but the grid also happens to read as high
    occupancy / low volatility this same tick (occupancy > 80%, volatility < 5%) — the exact
    combination where the legacy `else: crash_alert_message = None` branch silently clobbered
    the sudden-death message two lines after it was set. Precedence must keep SUDDEN_DEATH."""
    alive_history = [1000] * 9 + [1000]  # 10 generations ago: 1000 alive
    current_alive = 100  # dropped 90% > the 80% threshold
    occupancy = 0.85  # also reads as stasis-occupied this tick
    volatility = 1.0  # and low-volatility this tick

    alert = evaluate(alive_history, current_alive, occupancy, volatility, CONFIG, generation=42)

    assert alert is not None
    assert alert.kind is AlertKind.SUDDEN_DEATH


def test_stasis_fires_when_sudden_death_condition_absent():
    alive_history = [500] * 10
    current_alive = 500
    occupancy = 0.85
    volatility = 1.0

    alert = evaluate(alive_history, current_alive, occupancy, volatility, CONFIG, generation=11)

    assert alert is not None
    assert alert.kind is AlertKind.STASIS


def test_no_alert_when_neither_condition_holds():
    alive_history = [500] * 10
    current_alive = 480
    occupancy = 0.5
    volatility = 20.0

    alert = evaluate(alive_history, current_alive, occupancy, volatility, CONFIG, generation=11)

    assert alert is None


def test_sudden_death_requires_full_window_of_history():
    alive_history = [1000] * 5  # short of the 10-generation window
    current_alive = 50

    alert = evaluate(alive_history, current_alive, 0.0, 100.0, CONFIG, generation=5)

    assert alert is None


def test_sudden_death_ignores_zero_prior_population():
    alive_history = [0] * 10
    current_alive = 0

    alert = evaluate(alive_history, current_alive, 0.0, 100.0, CONFIG, generation=10)

    assert alert is None


# --- the threshold is inclusive (§6.7 "dropped ≥ sudden_death_drop"; F9) -------------------------


def _sudden_death(previous: int, current: int, config: AlertsConfig = CONFIG):
    """evaluate() with a steady `previous` population over the window and no stasis."""
    history = [previous] * config.sudden_death_window
    return evaluate(history, current, 0.1, 50.0, config, generation=20)


@pytest.mark.parametrize(("previous", "current"), [(1000, 200), (100, 20), (10, 2), (5, 1)])
def test_a_drop_of_exactly_the_threshold_fires(previous, current):
    alert = _sudden_death(previous, current)
    assert alert is not None
    assert alert.kind is AlertKind.SUDDEN_DEATH


def test_a_drop_just_under_the_threshold_does_not_fire():
    assert _sudden_death(1000, 201) is None


def test_total_extinction_fires_with_a_threshold_of_one():
    config = replace(CONFIG, sudden_death_drop=1.0)
    assert _sudden_death(1000, 0, config).kind is AlertKind.SUDDEN_DEATH
    assert _sudden_death(1000, 1, config) is None


def test_threshold_comparison_is_free_of_rounding_error():
    """0.07 * 100 is 7.000000000000001 in floating point, so comparing the drop with the
    product would miss an exact 7% drop; comparing the fraction does not."""
    config = replace(CONFIG, sudden_death_drop=0.07)
    assert _sudden_death(100, 93, config).kind is AlertKind.SUDDEN_DEATH
    assert _sudden_death(100, 94, config) is None
