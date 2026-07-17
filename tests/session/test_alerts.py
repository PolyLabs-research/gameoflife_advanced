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
