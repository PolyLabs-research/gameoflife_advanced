"""Density sweeps and their analysis helpers (`experiments.sweeps`)."""

from pathlib import Path

import numpy as np
import pytest

from latticelife.config import load_config
from latticelife.core.rules import Rule
from latticelife.experiments.harness import read_jsonl, single_species_session
from latticelife.experiments.sweeps import (
    SweepTask,
    critical_densities,
    fit_finite_size,
    run_sweep,
    survival_table,
    sweep_point,
    sweep_session,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"
LIFE = Rule(birth={3}, survive={2, 3})


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


def _session(config, rule, cells, size=10):
    session = single_species_session(config, rule, "square", size, 0.0, seed=1)
    for r, c in cells:
        session.set_cell(r * size + c, 1)
    return session


def test_sweep_session_stops_once_the_empty_grid_is_reached_and_absorbing(config):
    # A lone cell dies in one step; nothing more needs running.
    result = sweep_session(_session(config, LIFE, [(3, 3)]), steps=100)
    assert result["survived"] is False
    assert result["extinction_step"] == 1
    assert result["rho_final"] == 0.0
    assert result["stopped_at"] == 1
    # Stopped before the empty grid could be seen twice: no cycle was recorded.
    assert (result["period"], result["transient"]) == (None, None)


def test_sweep_session_reads_the_final_density_off_a_detected_cycle(config):
    # A blinker: detected at step 2, the fate at step 99 is known without stepping on.
    result = sweep_session(_session(config, LIFE, [(4, 4), (4, 5), (4, 6)]), steps=100)
    assert result["survived"] is True
    assert result["extinction_step"] is None
    assert result["rho_final"] == 0.03
    assert (result["period"], result["transient"], result["stopped_at"]) == (2, 0, 2)


def test_a_b0_rule_is_not_cut_short_by_an_empty_grid(config):
    """With 0 in B the empty grid fills up next step: the fate at the end is decided by the
    cycle (empty <-> full), and depends on the parity of `steps`."""
    strobe = Rule(birth={0}, survive=set())
    odd = sweep_session(_session(config, strobe, []), steps=5)
    even = sweep_session(_session(config, strobe, []), steps=6)
    assert odd["extinction_step"] == 0 and even["extinction_step"] == 0
    assert (odd["period"], odd["transient"]) == (2, 0)
    assert odd["survived"] is False  # generation 4 is empty
    assert even["survived"] is True  # generation 5 is full
    assert even["rho_final"] == 1.0
    with pytest.raises(ValueError, match="steps"):
        sweep_session(_session(config, strobe, []), steps=0)


def test_sweep_point_and_run_sweep_record_and_resume(config, tmp_path):
    record = sweep_point(SweepTask(config, "B3/S23", "hexagon", 10, 0.3, 2, 30))
    assert (record.rule, record.shape, record.triangle_mode, record.grid) == (
        "B3/S23",
        "hexagon",
        None,
        10,
    )
    assert record.to_json()["survived"] in (True, False)

    out = tmp_path / "sweep.jsonl"
    first = list(run_sweep(config, LIFE, "square", [8], [0.1, 0.5], [1, 2], 20, out))
    assert len(first) == 4
    assert list(run_sweep(config, LIFE, "square", [8], [0.1, 0.5], [1, 2], 20, out)) == []
    more = list(run_sweep(config, LIFE, "square", [8, 10], [0.1, 0.5], [1, 2], 20, out))
    assert len(more) == 4 and all(r.grid == 10 for r in more)
    assert len(read_jsonl(out)) == 8


def test_run_sweep_matches_in_parallel(config, tmp_path):
    args = (config, LIFE, "triangle", [8], [0.2, 0.5, 0.8], [1, 2], 25)
    serial = sorted(
        (r.density, r.seed, r.survived, r.rho_final) for r in run_sweep(*args, tmp_path / "s.jsonl")
    )
    parallel = sorted(
        (r.density, r.seed, r.survived, r.rho_final)
        for r in run_sweep(*args, tmp_path / "p.jsonl", workers=2)
    )
    assert serial == parallel


def test_survival_table_groups_by_size_and_density():
    records = [
        {"grid": 8, "density": 0.2, "survived": True},
        {"grid": 8, "density": 0.2, "survived": False},
        {"grid": 8, "density": 0.4, "survived": True},
        {"grid": 16, "density": 0.2, "survived": False},
    ]
    assert survival_table(records) == {8: [(0.2, 0.5, 2), (0.4, 1.0, 1)], 16: [(0.2, 0.0, 1)]}


def test_critical_densities_interpolates_every_crossing():
    points = [(0.1, 0.0), (0.2, 0.25), (0.3, 0.75), (0.4, 1.0), (0.5, 1.0), (0.6, 0.5), (0.7, 0.0)]
    assert critical_densities(points) == pytest.approx([0.25, 0.6])
    assert critical_densities([(0.1, 0.0), (0.2, 0.2)]) == []
    assert critical_densities([(0.1, 0.0), (0.2, 0.5)]) == [0.2]
    assert critical_densities([]) == []


def test_fit_finite_size_recovers_synthetic_parameters():
    sizes = np.array([16, 24, 32, 48, 64, 96], dtype=float)
    thresholds = 0.41 + 0.9 * sizes ** (-0.75)
    fit = fit_finite_size(sizes, thresholds)
    assert fit.rho_inf == pytest.approx(0.41, abs=2e-3)
    assert fit.amplitude == pytest.approx(0.9, abs=5e-2)
    assert fit.exponent == pytest.approx(0.75, abs=3e-2)
    assert fit.residual < 1e-6
    with pytest.raises(ValueError, match="three or more"):
        fit_finite_size([16, 32], [0.5, 0.45])
