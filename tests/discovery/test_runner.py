from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latticelife.config import load_config
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.experiments.runner import run_damage_spreading, run_timeseries
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
        simulation=replace(base.simulation, **{"n_lifeforms": 1, **sim_overrides}),
    )


def test_run_timeseries_rho_matches_occupancy_and_activity_is_its_delta():
    """F64: rho must be the tracked species' share of all cells. Two species on a non-square
    grid, so counting every species or dividing by the width would both be caught."""
    config = _config(width=12, height=10, wrap=True, n_lifeforms=2)
    lifeforms = [Lifeform(1, CONWAY), Lifeform(2, Rule(birth={3, 6}, survive={2, 3}))]
    session = SimulationSession(config, lifeforms, seed=11)
    twin = SimulationSession(config, lifeforms, seed=11)

    result = run_timeseries(session, lifeform_id=1, steps=20)

    expected = [np.count_nonzero(twin.grid.occupancy == 1) / 120]
    for _ in range(1, 20):
        twin.step()
        assert np.count_nonzero(twin.grid.occupancy == 2) > 0  # the other species is present
        expected.append(np.count_nonzero(twin.grid.occupancy == 1) / 120)
    assert result.rho.tolist() == expected
    assert len(result.activity) == 20
    assert result.activity[0] == 0.0  # legacy never measures activity at step 0
    for i in range(1, 20):
        assert result.activity[i] == abs(result.rho[i] - result.rho[i - 1])


def test_run_timeseries_detects_freeze_on_empty_grid():
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.0)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])
    # Empty grid is a fixpoint (Chunk 1) -> activity is 0 forever -> freezes right after burn-in.
    result = run_timeseries(session, lifeform_id=1, steps=200, burn_in_frac=0.4, freeze_window=50)

    assert result.frozen_at is not None
    assert result.frozen_at >= int(200 * 0.4)


def test_run_damage_spreading_h0_reflects_the_perturbation():
    config = _config(width=20, height=20, wrap=True, initial_alive_percentage=0.0)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])

    h = run_damage_spreading(session, steps=10, rng=np.random.default_rng(0), flip_cells=3)

    assert len(h) == 10
    # Exactly `flip_cells` disagreeing cells out of n_cells at t=0.
    assert h[0] == 3 / (20 * 20)


def test_run_damage_spreading_shares_adjacency_array_identity():
    config = _config(width=10, height=10, wrap=True, initial_alive_percentage=0.3)
    session = SimulationSession(config, [Lifeform(1, CONWAY)])
    grid_a_adjacency = session.grid.adjacency

    run_damage_spreading(session, steps=5, rng=np.random.default_rng(1), flip_cells=1)

    # The function must not have replaced session.grid's own adjacency array.
    assert session.grid.adjacency is grid_a_adjacency


def test_run_damage_spreading_is_deterministic_given_same_rng_seed():
    config = _config(width=15, height=15, wrap=True, initial_alive_percentage=0.3)

    def run():
        session = SimulationSession(config, [Lifeform(1, CONWAY)], seed=99)
        return run_damage_spreading(session, steps=15, rng=np.random.default_rng(5), flip_cells=1)

    h_a = run()
    h_b = run()
    assert np.array_equal(h_a, h_b)


def test_unperturbed_multi_species_twins_never_diverge():
    """F23: the twins used independent RNG streams, so with two species the birth tie-breaks
    differed and unperturbed twins drifted to H ~ 0.26 by step 50. Common random numbers keep
    them identical."""
    config = _config(width=32, height=32, wrap=True, n_lifeforms=2, initial_alive_percentage=0.5)
    lifeforms = [Lifeform(1, CONWAY), Lifeform(2, CONWAY)]
    session = SimulationSession(config, lifeforms, seed=7)

    h = run_damage_spreading(session, steps=50, rng=np.random.default_rng(3), flip_cells=0)

    assert np.count_nonzero(session.grid.occupancy == 2) > 0
    assert not h.any()


def test_perturbed_multi_species_twins_differ_only_by_the_perturbation():
    config = _config(width=32, height=32, wrap=True, n_lifeforms=2, initial_alive_percentage=0.5)
    lifeforms = [Lifeform(1, CONWAY), Lifeform(2, CONWAY)]
    session = SimulationSession(config, lifeforms, seed=7)

    h = run_damage_spreading(session, steps=50, rng=np.random.default_rng(3), flip_cells=1)

    assert h[0] == 1 / (32 * 32)
    assert h.max() < 0.25  # one flipped cell's damage, not tie-break noise across the grid


def test_damage_spreading_needs_a_species():
    config = _config(width=10, height=10, wrap=True)
    session = SimulationSession(config, [])
    with pytest.raises(ValueError, match="at least one species"):
        run_damage_spreading(session, steps=5, rng=np.random.default_rng(0))
