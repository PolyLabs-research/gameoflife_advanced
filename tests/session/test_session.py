from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latticelife.config import load_config
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.session.alerts import AlertKind
from latticelife.session.session import SimulationSession

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"
CONWAY = Rule(birth={3}, survive={2, 3})


def _config(**overrides):
    base = load_config(DEFAULT_CONFIG_PATH)
    grid_overrides = {k: v for k, v in overrides.items() if hasattr(base.grid, k)}
    sim_overrides = {k: v for k, v in overrides.items() if hasattr(base.simulation, k)}
    return replace(
        base,
        grid=replace(base.grid, **grid_overrides),
        simulation=replace(base.simulation, **sim_overrides),
    )


def test_still_life_lifespan_reaches_200_after_200_generations():
    """D-1 closed by construction: legacy int8 lifespans wrapped 127->-128; int32 here just
    keeps counting."""
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.0, n_lifeforms=1)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])

    block = [(4, 4), (4, 5), (5, 4), (5, 5)]  # still-life under B3/S23
    for r, c in block:
        session.grid.occupancy[r * session.grid.width + c] = 1

    for _ in range(200):
        session.step()

    for r, c in block:
        assert session.grid.lifespans[r * session.grid.width + c] == 200
    assert session.grid.lifespans.dtype == np.int32


def test_growth_rate_nonzero_across_conway_run():
    config = _config(width=30, height=30, wrap=True, initial_alive_percentage=0.0, n_lifeforms=1)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])

    r_pentomino = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]  # chaotic growth, not a still-life
    origin_r, origin_c = 14, 14
    for dr, dc in r_pentomino:
        r, c = origin_r + dr, origin_c + dc
        session.grid.occupancy[r * session.grid.width + c] = 1

    growth_rates = []
    for _ in range(15):
        session.step()
        growth_rates.append(session.history_records[-1].lifeform_metrics[1].growth_rate)

    assert any(rate != 0.0 for rate in growth_rates)


def test_entropy_positive_for_two_species_run():
    config = _config(width=20, height=20, wrap=True, initial_alive_percentage=0.0, n_lifeforms=2)
    session = SimulationSession(config, [Lifeform(1, CONWAY), Lifeform(2, CONWAY)])

    block_a = [(2, 2), (2, 3), (3, 2), (3, 3)]
    block_b = [(10, 10), (10, 11), (11, 10), (11, 11)]
    for r, c in block_a:
        session.grid.occupancy[r * session.grid.width + c] = 1
    for r, c in block_b:
        session.grid.occupancy[r * session.grid.width + c] = 2

    session.step()

    assert session.history_records[-1].grid_metrics.entropy > 0.0


def test_cluster_metrics_only_computed_on_stride_and_final_generation():
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.0, n_lifeforms=1)
    config = replace(config, metrics=replace(config.metrics, cluster_stride=5))
    session = SimulationSession(config, [Lifeform(1, CONWAY)])

    block = [(4, 4), (4, 5), (5, 4), (5, 5)]
    for r, c in block:
        session.grid.occupancy[r * session.grid.width + c] = 1

    session.run(7)

    records = list(session.history_records)
    assert records[0].cluster_metrics is None
    assert records[3].cluster_metrics is None
    assert records[4].cluster_metrics is not None  # generation 5: stride hit
    assert records[6].cluster_metrics is not None  # generation 7: final tick of run()


def test_sudden_death_pauses_session():
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.0, n_lifeforms=1)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])

    for _ in range(10):
        session.history_alive.append(1000)
    session.grid.occupancy[:] = 0  # next step's alive count is 0 -> full drop vs history[-10]

    session.step()

    assert session.paused is True
    assert session.alert is not None
    assert session.alert.kind is AlertKind.SUDDEN_DEATH


def test_reset_clears_history_and_state():
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.3, n_lifeforms=1)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])
    session.run(5)
    assert session.generation == 5
    assert len(session.history_records) == 5

    session.reset()

    assert session.generation == 0
    assert len(session.history_records) == 0
    assert session.paused is False
    assert session.alert is None


# --- auto_pause (F5) ---------------------------------------------------------------------------


def _collapsing_session(**session_kwargs) -> SimulationSession:
    """A session whose next step drops from 1000 alive (10 generations ago) to 0."""
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.0, n_lifeforms=1)
    session = SimulationSession(config, [Lifeform(1, CONWAY)], **session_kwargs)
    for _ in range(10):
        session.history_alive.append(1000)
    session.grid.occupancy[:] = 0
    return session


def test_auto_pause_off_still_reports_the_alert_but_keeps_running():
    session = _collapsing_session(auto_pause=False)

    session.step()

    assert session.alert is not None
    assert session.alert.kind is AlertKind.SUDDEN_DEATH
    assert session.history_records[-1].alert is session.alert
    assert session.paused is False


def test_run_without_auto_pause_records_every_planned_generation():
    session = _collapsing_session(auto_pause=False)

    session.run(25)

    assert session.generation == 25
    assert session.history_records[0].alert.kind is AlertKind.SUDDEN_DEATH
    assert session.history_records[-1].cluster_metrics is not None


def test_from_config_passes_auto_pause_through():
    config = _config(width=10, height=10, wrap=True)
    assert SimulationSession.from_config(config, auto_pause=False).auto_pause is False
    assert SimulationSession.from_config(config).auto_pause is True


def test_generation_that_auto_pauses_has_cluster_metrics():
    """F5: run() stops at an auto-pause, so that generation is the run's last recorded one and
    must carry cluster metrics (§6.6) even off the cluster stride."""
    session = _collapsing_session()
    assert session.config.metrics.cluster_stride > 1

    session.run(50)

    assert session.paused is True
    assert session.generation == 1
    assert session.history_records[-1].cluster_metrics is not None


# --- manual cell edits (F80) -------------------------------------------------------------------

GLIDER = [(1, 2), (2, 3), (3, 1), (3, 2), (3, 3)]


def test_erasing_the_grid_and_drawing_is_not_a_sudden_death():
    """F80 repro: 80 generations, erase every live cell, draw a glider, step. The old code
    raised 'Sudden death: gen 81 drop 238->5' and re-paused for 9 more generations."""
    config = _config()  # default 50x50 square, Conway, wrap
    session = SimulationSession(config, [Lifeform(1, CONWAY)], seed=12345)
    session.run(80)
    assert session.paused is False
    assert session.history_alive[-1] > 100

    for index in np.flatnonzero(session.grid.occupancy):
        session.set_cell(int(index), 0)
    for r, c in GLIDER:
        session.set_cell(r * session.grid.width + c, 1)

    for _ in range(config.alerts.sudden_death_window + 5):
        session.step()
        assert session.alert is None or session.alert.kind is not AlertKind.SUDDEN_DEATH
        assert session.paused is False


def test_growth_rate_after_an_edit_measures_only_the_rule():
    config = _config(n_lifeforms=2)
    session = SimulationSession(config, [Lifeform(1, CONWAY), Lifeform(2, CONWAY)], seed=777)
    session.run(60)

    empty = np.flatnonzero(session.grid.occupancy == 0)[:150]
    for index in empty:
        assert session.set_cell(int(index), 1) is True
    alive_after_edit = int(np.count_nonzero(session.grid.occupancy == 1))

    session.step()

    metrics = session.history_records[-1].lifeform_metrics[1]
    expected = (metrics.alive - alive_after_edit) / alive_after_edit
    assert metrics.growth_rate == pytest.approx(expected)


def test_set_cell_updates_occupancy_and_lifespans():
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.0, n_lifeforms=2)
    session = SimulationSession(config, [Lifeform(1, CONWAY), Lifeform(2, CONWAY)])

    assert session.set_cell(5, 2) is True
    assert session.grid.occupancy[5] == 2
    assert session.grid.lifespans[5] == 1
    assert session.set_cell(5, 2) is False  # already that species: nothing changes

    session.grid.lifespans[5] = 40
    assert session.set_cell(5, 1) is True  # replacing a species starts a fresh lifespan
    assert session.grid.lifespans[5] == 1

    assert session.set_cell(5, 0) is True
    assert session.grid.occupancy[5] == 0
    assert session.grid.lifespans[5] == 0

    with pytest.raises(IndexError):
        session.set_cell(100, 1)
    with pytest.raises(IndexError):
        session.set_cell(-1, 1)
    with pytest.raises(ValueError):
        session.set_cell(3, 7)


def test_sudden_death_detection_resumes_once_a_full_window_follows_an_edit():
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.0, n_lifeforms=1)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])
    block = [(4, 4), (4, 5), (5, 4), (5, 5)]  # still life: 4 alive every generation
    for r, c in block:
        session.set_cell(r * session.grid.width + c, 1)

    window = config.alerts.sudden_death_window
    for _ in range(window):
        session.step()
        assert session.alert is None

    session.grid.occupancy[:] = 0  # a collapse the dynamics did (not a set_cell edit)
    session.step()

    assert session.alert is not None
    assert session.alert.kind is AlertKind.SUDDEN_DEATH
    assert session.paused is True
