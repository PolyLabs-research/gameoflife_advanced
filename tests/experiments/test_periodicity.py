"""Exact cycle detection (`experiments.periodicity`) and its use in `run_timeseries`."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latticelife.config import load_config
from latticelife.core.geometries import build_adjacency
from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.core.translations import TorusTranslations
from latticelife.experiments.periodicity import (
    CycleDetector,
    TranslationCycleDetector,
    pack_state,
    unpack_state,
)
from latticelife.experiments.runner import run_timeseries
from latticelife.session.session import SimulationSession

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"
LIFE = Rule(birth={3}, survive={2, 3})


def _config(width, height, density=0.0, n_lifeforms=1):
    base = load_config(DEFAULT_CONFIG_PATH)
    return replace(
        base,
        grid=replace(base.grid, shape="square", width=width, height=height, wrap=True),
        simulation=replace(
            base.simulation, initial_alive_percentage=density, n_lifeforms=n_lifeforms
        ),
    )


def _life_grid(width, height, cells):
    grid = LatticeGrid([Lifeform(1, LIFE)], "square", width, height, True)
    for r, c in cells:
        grid.occupancy[r * width + c] = 1
    return grid


def test_pack_and_unpack_round_trip_and_drop_species():
    occupancy = np.array([0, 1, 2, 0, 3, 0, 0, 0, 1, 0], dtype=np.int8)
    packed = pack_state(occupancy)
    assert len(packed) == 2
    assert unpack_state(packed, 10).tolist() == (occupancy != 0).tolist()


def test_cycle_detector_finds_a_blinker_after_its_transient():
    """A pre-blinker (a T-shape minus one cell -> ... ) settles into period 2; the detector
    must report the period and the generation the cycle began."""
    grid = _life_grid(10, 10, [(4, 4), (4, 5), (4, 6)])  # a blinker from the start
    detector = CycleDetector()
    rng = np.random.default_rng(0)
    assert detector.observe(grid.occupancy, 0) is None
    grid.step(rng)
    assert detector.observe(grid.occupancy, 1) is None
    grid.step(rng)
    assert detector.observe(grid.occupancy, 2) == (2, 0)
    assert len(detector) == 2


def test_cycle_detector_reports_a_transient():
    # Three cells in an L become a block (period 1) after one step.
    grid = _life_grid(10, 10, [(4, 4), (4, 5), (5, 4)])
    detector = CycleDetector()
    rng = np.random.default_rng(0)
    detector.observe(grid.occupancy, 0)
    grid.step(rng)
    assert detector.observe(grid.occupancy, 1) is None
    grid.step(rng)
    assert detector.observe(grid.occupancy, 2) == (1, 1)


def test_translation_detector_finds_the_glider_period_and_displacement():
    width, height = 16, 12
    grid = _life_grid(width, height, [(1, 2), (2, 3), (3, 1), (3, 2), (3, 3)])
    translations = TorusTranslations("square", width, height)
    detector = TranslationCycleDetector(translations, grid.adjacency)
    rng = np.random.default_rng(0)
    detector.observe(grid.occupancy, 0)
    found = None
    for step in range(1, 10):
        grid.step(rng)
        found = detector.observe(grid.occupancy, step)
        if found is not None:
            break
    assert found == (4, (1, 1))  # period 4, one cell down and one right


@pytest.mark.parametrize(
    ("shape", "mode"),
    [("hexagon", "edge+vertex"), ("triangle", "edge"), ("triangle", "edge+vertex")],
)
def test_translation_detector_matches_a_translated_state_on_every_lattice(shape, mode):
    """Independent of any rule: feed a state, then its translate, and the detector must name
    the translation that carried it there."""
    width, height = 12, 10
    translations = TorusTranslations(shape, width, height, mode)
    adjacency = build_adjacency(shape, width, height, True, mode)
    detector = TranslationCycleDetector(translations, adjacency)
    rng = np.random.default_rng(2)
    state = (rng.random(width * height) < 0.3).astype(np.int8)
    vector = (2, 4)
    moved = np.zeros_like(state)
    moved[translations.permutation(vector)] = state
    other = (rng.random(width * height) < 0.3).astype(np.int8)

    assert detector.observe(state, 0) is None
    assert detector.observe(other, 1) is None  # a different state, not a translate
    assert detector.observe(moved, 2) == (2, vector)


def test_translation_detector_fingerprint_is_translation_invariant():
    width, height = 12, 10
    translations = TorusTranslations("hexagon", width, height)
    detector = TranslationCycleDetector(
        translations, build_adjacency("hexagon", width, height, True)
    )
    rng = np.random.default_rng(4)
    state = rng.random(width * height) < 0.4
    moved = np.zeros_like(state)
    moved[translations.permutation((3, 5))] = state
    assert detector.fingerprint(state) == detector.fingerprint(moved)
    assert detector.fingerprint(state) != detector.fingerprint(rng.random(width * height) < 0.4)


def test_translation_detector_rejects_a_mismatched_adjacency():
    with pytest.raises(ValueError, match="different grids"):
        TranslationCycleDetector(
            TorusTranslations("square", 8, 6), build_adjacency("square", 8, 8, True)
        )


# --- run_timeseries(detect_cycles=True) -------------------------------------------------------


def _blinker_session(steps_density=0.0):
    config = _config(10, 10)
    session = SimulationSession(config, [Lifeform(1, LIFE)], seed=1)
    for r, c in [(4, 4), (4, 5), (4, 6)]:
        session.set_cell(r * 10 + c, 1)
    return session


def test_run_timeseries_reports_the_exact_period_and_extends_the_series():
    session = _blinker_session()
    result = run_timeseries(session, lifeform_id=1, steps=50, detect_cycles=True)

    assert (result.period, result.transient) == (2, 0)
    assert session.generation == 2  # left on the cycle where the repeat was seen
    assert len(result.rho) == 50
    assert result.rho.tolist() == [3 / 100] * 50
    assert result.activity.tolist() == [0.0] * 50


def test_periodic_extension_matches_stepping_on():
    """The extended series must be exactly what stepping would have produced."""
    config = _config(12, 12, density=0.4)
    extended = run_timeseries(
        SimulationSession(config, [Lifeform(1, LIFE)], seed=9), 1, steps=120, detect_cycles=True
    )
    stepped = run_timeseries(
        SimulationSession(config, [Lifeform(1, LIFE)], seed=9), 1, steps=120, detect_cycles=False
    )
    assert extended.period is not None  # a 12x12 Life soup settles well inside 120 steps
    assert extended.rho.tolist() == stepped.rho.tolist()
    assert extended.activity.tolist() == stepped.activity.tolist()
    assert extended.frozen_at == stepped.frozen_at
    assert stepped.period is None and stepped.transient is None


def test_run_timeseries_without_a_repeat_reports_none():
    session = _blinker_session()
    result = run_timeseries(session, lifeform_id=1, steps=2, detect_cycles=True)
    assert result.period is None and result.transient is None
    assert session.generation == 1


def test_cycle_detection_refuses_several_species():
    config = _config(10, 10, density=0.3, n_lifeforms=2)
    session = SimulationSession(config, [Lifeform(1, LIFE), Lifeform(2, LIFE)], seed=1)
    with pytest.raises(ValueError, match="exactly one species"):
        run_timeseries(session, lifeform_id=1, steps=10, detect_cycles=True)
