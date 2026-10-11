import tracemalloc
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
    assert growth_rates[0] == pytest.approx(0.2)  # F6: 5 -> 6 cells, not 6 measured against 0


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
    assert list(session.history_alive) == [int(np.count_nonzero(session.grid.occupancy))]
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


@pytest.mark.parametrize("generations_before_edit", [0, 15])
def test_extinction_in_the_first_generation_after_an_edit_raises_sudden_death(
    generations_before_edit,
):
    """The edited grid is generation 0 of the restarted window, as the seeded grid is for a new
    session (F7): the window used to hold only post-edit step totals, so an edited population
    that died in its first step left every baseline at 0 and the alert never fired."""
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.0, n_lifeforms=1)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])
    block = [4 * 10 + 4, 4 * 10 + 5, 5 * 10 + 4, 5 * 10 + 5]  # still life: 4 alive
    for index in block:
        session.set_cell(index, 1)
    session.run(generations_before_edit)
    assert session.alert is None

    for index in block:
        session.set_cell(index, 0)
    for index in (1 * 10 + 1, 5 * 10 + 7, 8 * 10 + 2):  # three isolated cells: dead in one step
        session.set_cell(index, 1)
    window = config.alerts.sudden_death_window

    session.run(window + 5)

    alert_generation = generations_before_edit + window  # a full window after the edit
    assert session.paused is True
    assert session.generation == alert_generation
    assert session.alert.kind is AlertKind.SUDDEN_DEATH
    assert session.alert.message == f"Sudden death: gen {alert_generation} drop 3->0"


def test_each_edit_restarts_the_window_from_the_grid_the_next_step_finds():
    """An edit part-way through a restarted window starts it again, with a new generation 0."""
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.0, n_lifeforms=1)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])
    window = config.alerts.sudden_death_window
    for index in (1 * 10 + 1, 5 * 10 + 7, 8 * 10 + 2, 2 * 10 + 6):  # four isolated cells
        session.set_cell(index, 1)
    session.run(window - 2)  # extinct since generation 1, but no full window yet
    assert session.alert is None

    for index in (1 * 10 + 1, 5 * 10 + 7):  # a second edit restarts the window
        session.set_cell(index, 1)
    session.run(window - 1)
    assert session.alert is None  # the first edit's 4->0 is no longer in the window

    session.step()
    assert session.alert.message == f"Sudden death: gen {2 * window - 2} drop 2->0"
    assert session.paused is True


# --- growth baselines (F6) ---------------------------------------------------------------------


def _alive_by_species(session: SimulationSession) -> dict[int, int]:
    occupancy = session.grid.occupancy
    return {lf.id: int(np.count_nonzero(occupancy == lf.id)) for lf in session.lifeforms}


def _assert_growth_measured_from(session: SimulationSession, before: dict[int, int]) -> None:
    session.step()
    for lifeform_id, metrics in session.history_records[-1].lifeform_metrics.items():
        expected = (metrics.alive - before[lifeform_id]) / max(before[lifeform_id], 1)
        assert metrics.growth_rate == pytest.approx(expected)


def test_generation_one_growth_is_measured_against_the_seeded_grid():
    """F6: generation 1 used to be measured against 0 (666.0 recorded instead of -0.467)."""
    lifeforms = [Lifeform(i, CONWAY) for i in (1, 2, 3)]
    session = SimulationSession(_config(n_lifeforms=3), lifeforms, seed=1234)
    _assert_growth_measured_from(session, _alive_by_species(session))
    for metrics in session.history_records[-1].lifeform_metrics.values():
        assert -1.0 <= metrics.growth_rate <= 1.0

    session.run(5)
    session.reset()
    _assert_growth_measured_from(session, _alive_by_species(session))


def test_hand_placed_still_life_has_zero_growth_from_generation_one():
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.0, n_lifeforms=1)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])
    for r, c in [(4, 4), (4, 5), (5, 4), (5, 5)]:
        session.grid.occupancy[r * 10 + c] = 1

    session.run(3)

    rates = [record.lifeform_metrics[1].growth_rate for record in session.history_records]
    assert rates == [0.0, 0.0, 0.0]


# --- generation 0 in the alive history (F7) ------------------------------------------------------


def test_history_alive_starts_with_the_seeded_population():
    session = SimulationSession(_config(), [Lifeform(1, CONWAY)], seed=5)
    assert list(session.history_alive) == [int(np.count_nonzero(session.grid.occupancy))]
    assert session.history_alive[0] > 0

    session.run(4)
    assert len(session.history_alive) == len(session.history_records) + 1


def test_extinction_in_the_first_generation_raises_sudden_death():
    """F7: with no generation-0 entry every baseline after a first-generation extinction was 0,
    so the alert never fired and the session never paused."""
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.5, n_lifeforms=1)
    session = SimulationSession(config, [Lifeform(1, Rule(birth=set(), survive=set()))], seed=1)
    window = config.alerts.sudden_death_window

    session.run(window + 5)

    assert session.paused is True
    assert session.generation == window  # measured against generation 0, a full window back
    assert session.alert.kind is AlertKind.SUDDEN_DEATH
    assert session.alert.message == f"Sudden death: gen {window} drop 50->0"


# --- pausing on a raised alert, not a persisting one (F8) ---------------------------------------


def test_resuming_after_a_sudden_death_pause_keeps_running():
    """F8: the crash stays in the comparison window for window-1 more generations; each of them
    used to pause again, so every resume advanced exactly one generation."""
    session = _collapsing_session()
    session.step()
    assert session.paused is True
    assert session.paused_by_alert is True

    session.paused = False
    for _ in range(session.config.alerts.sudden_death_window + 5):
        session.step()
        assert session.paused is False
    assert session.history_records[1].alert.kind is AlertKind.SUDDEN_DEATH  # still reported
    assert session.alert is None  # and cleared once the crash left the window


def test_a_later_crash_after_the_alert_cleared_pauses_again():
    session = _collapsing_session()
    session.step()
    session.paused = False
    window = session.config.alerts.sudden_death_window
    session.run(window + 1)
    assert session.alert is None
    assert session.paused is False

    for r, c in [(4, 4), (4, 5), (5, 4), (5, 5)]:  # a still life the dynamics "grew"
        session.grid.occupancy[r * 10 + c] = 1
    session.run(window)
    assert session.paused is False
    session.grid.occupancy[:] = 0  # and a second collapse
    session.step()

    assert session.alert.kind is AlertKind.SUDDEN_DEATH
    assert session.paused is True


# --- pause ownership across reset and new sessions (F48) ----------------------------------------


def test_reset_keeps_a_pause_the_user_chose():
    """F48: reset used to unpause, so the generation-0 grid could never be inspected."""
    session = SimulationSession(_config(), [Lifeform(1, CONWAY)], seed=2)
    session.run(5)
    session.paused = True

    session.reset()

    assert session.paused is True
    assert session.generation == 0
    assert session.run(3) == 0  # still paused: nothing advances by itself
    session.step()  # single-stepping from generation 0 still works
    assert session.generation == 1


def test_reset_lifts_a_pause_the_alert_imposed():
    session = _collapsing_session()
    session.step()
    assert session.paused_by_alert is True

    session.reset()

    assert session.paused is False
    assert session.paused_by_alert is False
    assert session.alert is None


def test_setting_paused_makes_the_pause_the_users():
    session = _collapsing_session()
    session.step()
    session.paused = True  # e.g. the user pressed Pause again over the alert

    assert session.paused_by_alert is False
    session.reset()
    assert session.paused is True


def test_a_crash_stepped_into_while_paused_leaves_the_pause_the_users():
    session = _collapsing_session()
    session.paused = True
    session.step()  # the app's single step (S)

    assert session.alert.kind is AlertKind.SUDDEN_DEATH
    assert session.paused is True
    assert session.paused_by_alert is False


def test_a_session_can_start_paused():
    config = _config(width=10, height=10, wrap=True)
    session = SimulationSession(config, [Lifeform(1, CONWAY)], paused=True)
    assert session.paused is True
    assert session.paused_by_alert is False
    assert SimulationSession.from_config(config, paused=True).paused is True
    assert SimulationSession.from_config(config).paused is False


# --- run() keeps nothing per generation (F69) ----------------------------------------------------


def test_run_returns_the_number_of_generations_stepped():
    config = _config(width=10, height=10, wrap=True)
    assert SimulationSession(config, [Lifeform(1, CONWAY)]).run(7) == 7
    assert _collapsing_session().run(50) == 1  # auto-paused at generation 1
    assert SimulationSession(config, [Lifeform(1, CONWAY)], paused=True).run(5) == 0


def test_run_memory_does_not_grow_with_the_number_of_generations():
    """F69: run(n) used to keep all n UpdateResults (~1 KB each) until it returned."""
    config = _config(width=10, height=10, wrap=True, n_lifeforms=3, history_limit=10)
    lifeforms = [Lifeform(i, CONWAY) for i in (1, 2, 3)]
    session = SimulationSession(config, lifeforms, seed=1, auto_pause=False)
    session.run(50)  # fill the bounded histories first

    tracemalloc.start()
    try:
        session.run(3000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 512 * 1024
