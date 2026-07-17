from dataclasses import replace
from pathlib import Path

import numpy as np

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
