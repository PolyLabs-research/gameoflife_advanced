"""Torus sizes: square `L` and rectangular `WxH` grid labels, through every experiment, and the
`no_repeat_before` bound a run that never repeated records."""

from pathlib import Path

import pytest

from latticelife.config import load_config
from latticelife.core.rules import Rule
from latticelife.experiments.census import CensusTask, classify_rule
from latticelife.experiments.cost import cost_hint, measure_seconds_per_step, task_cost
from latticelife.experiments.damage import DamageTask
from latticelife.experiments.harness import (
    grid_dimensions,
    grid_sort_key,
    parse_grid,
    single_species_session,
)
from latticelife.experiments.objects import SoupTask, soup_objects
from latticelife.experiments.sweeps import SweepTask, survival_table, sweep_point

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.mark.parametrize(
    ("given", "label", "dims"),
    [
        (32, 32, (32, 32)),
        ("32", 32, (32, 32)),
        ("32x32", 32, (32, 32)),
        ("24x26", "24x26", (24, 26)),
        ("24X26", "24x26", (24, 26)),
        ((24, 26), "24x26", (24, 26)),
        ((5, 5), 5, (5, 5)),
    ],
)
def test_grid_labels(given, label, dims):
    assert parse_grid(given) == label
    assert grid_dimensions(given) == dims


@pytest.mark.parametrize("bad", ["", "x", "8x", "ax8", "0", "8x0", -3, True, 2.5, (1, 2, 3)])
def test_bad_grid_labels_are_refused(bad):
    with pytest.raises(ValueError):
        parse_grid(bad)


def test_numpy_integers_are_sizes():
    import numpy as np

    assert parse_grid(np.int64(16)) == 16 and isinstance(parse_grid(np.int64(16)), int)


def test_grid_sort_key_orders_by_cells():
    assert sorted([16, "8x10", 8, "4x30"], key=grid_sort_key) == [8, "8x10", "4x30", 16]


def test_a_rectangular_session_has_its_dimensions(config):
    session = single_species_session(config, Rule.from_bs_string("B3/S23"), "square", "6x9", 0.3, 1)
    assert session.grid.n_cells == 54
    assert (session.config.grid.width, session.config.grid.height) == (6, 9)


def test_sweep_records_label_rectangles_and_bound_unsettled_runs(config):
    record = sweep_point(SweepTask(config, "B2/S34", "hexagon", "8x10", 0.5, 1, steps=3000))
    assert record.grid == "8x10" and record.period is not None and record.no_repeat_before is None
    capped = sweep_point(SweepTask(config, "B2/S34", "hexagon", "16x16", 0.5, 1, steps=5))
    assert capped.grid == 16
    assert capped.period is None and capped.no_repeat_before == capped.stopped_at == 4


def test_an_absorbed_extinction_is_settled_not_bounded(config):
    record = sweep_point(SweepTask(config, "B/S", "square", 8, 0.5, 1, steps=50))
    assert record.extinction_step == 1 and record.period is None
    assert record.no_repeat_before is None


def test_census_records_bound_unsettled_runs(config):
    unsettled = classify_rule(CensusTask(config, "B2/S34", "hexagon", "8x10", 3, 1, 0.5, 2))
    assert unsettled.outcome == "unsettled" and unsettled.no_repeat_before == 2
    assert unsettled.grid == "8x10"
    settled = classify_rule(CensusTask(config, "B/S", "square", 8, 20, 1, 0.5, 2))
    assert settled.no_repeat_before is None


def test_soups_run_on_rectangles(config):
    record = soup_objects(SoupTask(config, "B3/S23", "square", "12x16", 1, 40, 0.3, 2, 30, 8))
    assert record["grid"] == "12x16"
    assert record["no_repeat_before"] == (
        record["stopped_at"] if record["soup_period"] is None else None
    )


def test_survival_table_mixes_square_and_rectangular_tori():
    rows = [
        {"grid": 16, "density": 0.5, "survived": True},
        {"grid": "8x10", "density": 0.5, "survived": False},
        {"grid": "8x10", "density": 0.5, "survived": True},
    ]
    assert survival_table(rows) == {"8x10": [(0.5, 0.5, 2)], 16: [(0.5, 1.0, 1)]}


def test_cost_hints(config):
    hint = cost_hint("24x26", 1000, seconds_per_step=0.001)
    assert (hint.cells, hint.cell_steps, hint.seconds) == (624, 624_000, pytest.approx(1.0))
    assert cost_hint(16, 10).seconds is None
    damage = task_cost(DamageTask(config, "B3/S23", "square", 8, 0.5, 1, steps=100, warmup=20))
    assert damage.generations == 220  # warm-up, then two twins per generation
    sweep = task_cost(SweepTask(config, "B3/S23", "square", 8, 0.5, 1, 300))
    assert sweep.generations == 300 and sweep.cell_steps == 64 * 300
    seconds = measure_seconds_per_step(config, "B3/S23", "square", 16, steps=5)
    assert 0 < seconds < 1
