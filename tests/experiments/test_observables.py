"""Observables (`experiments.observables`): numbers with intervals, and an honest bound, never a
guess, for runs that did not repeat within their cap."""

import math
import uuid
from pathlib import Path

import numpy as np
import pytest

from latticelife.config import load_config
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.experiments.harness import read_jsonl
from latticelife.experiments.observables import (
    bootstrap_interval,
    damage_spread,
    density_series,
    median_transient,
    no_repeat_before,
    settling_time,
    survival,
    wilson_interval,
)
from latticelife.experiments.sweeps import run_sweep
from latticelife.recording.recorder import Recorder
from latticelife.session.session import SimulationSession

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


def sweep_record(seed, transient=None, period=None, grid=16, steps=100, **extra):
    record = {
        "rule": "B2/S34",
        "shape": "hexagon",
        "triangle_mode": None,
        "grid": grid,
        "density": 0.5,
        "seed": seed,
        "steps": steps,
        "survived": True,
        "extinction_step": None,
        "rho_final": 0.2,
        "period": period,
        "transient": transient,
        "stopped_at": steps - 1 if transient is None else transient + (period or 1),
    }
    record["no_repeat_before"] = record["stopped_at"] if transient is None else None
    record.update(extra)
    return record


# --- settling time of one record ------------------------------------------------------------


def test_a_settled_record_settles_at_its_transient():
    assert settling_time(sweep_record(1, transient=40, period=2)) == 40
    assert no_repeat_before(sweep_record(1, transient=40, period=2)) is None


def test_a_soup_absorbed_into_the_empty_grid_settles_where_it_died():
    # 0 not in B: the sweep stops at extinction and records no period, but the fate is exact.
    record = sweep_record(1, rule="B3/S23", extinction_step=12, stopped_at=12)
    record["no_repeat_before"] = None
    assert settling_time(record) == 12
    assert no_repeat_before(record) is None


def test_extinction_on_a_rule_with_birth_from_nothing_settles_nothing():
    # 0 in B: the empty grid fills again, so dying says nothing about when the run settles.
    record = sweep_record(1, rule="B0/S", extinction_step=1)
    assert settling_time(record) is None
    assert no_repeat_before(record) == 99


def test_older_records_without_the_field_read_their_bound_from_the_run():
    sweep = sweep_record(1)
    del sweep["no_repeat_before"]
    assert no_repeat_before(sweep) == 99  # stopped_at
    census = {"rule": "B2/S34", "steps": 500, "transient": None, "period": None}
    assert no_repeat_before(census) == 499


# --- median_transient -----------------------------------------------------------------------


def test_complete_row_has_a_median_and_a_bootstrap_interval():
    times = [30, 41, 46, 52, 61, 80]
    rows = median_transient(
        [sweep_record(s, transient=t, period=2) for s, t in enumerate(times, 1)]
    )
    (row,) = rows
    assert row.complete and row.unsettled == 0 and row.seeds == 6
    assert row.median == pytest.approx(49.0)
    assert row.low <= row.median <= row.high
    assert row.settling_times == times
    assert (row.width, row.height, row.cells, row.size) == (16, 16, 256, 16.0)


def test_the_bootstrap_is_seeded_so_rows_are_reproducible():
    records = [sweep_record(s, transient=t, period=2) for s, t in enumerate([5, 9, 30, 31, 70], 1)]
    assert median_transient(records) == median_transient(records)
    first = median_transient(records, bootstrap_seed=1)[0]
    assert (first.low, first.high) == (
        median_transient(records, bootstrap_seed=1)[0].low,
        median_transient(records, bootstrap_seed=1)[0].high,
    )


def test_an_unsettled_seed_makes_the_median_a_bound_not_a_measurement():
    records = [sweep_record(s, transient=t, period=2) for s, t in enumerate([10, 20, 30, 40], 1)]
    records.append(sweep_record(5, steps=100))  # never repeated within 100 generations
    (row,) = median_transient(records)
    assert not row.complete
    assert row.median is None and row.low is None and row.high is None
    assert row.unsettled_seeds == [5] and row.max_cap == 99
    # The unknown transient could be anything from 0 up: the median of five lies in [20, 30].
    assert (row.median_at_least, row.median_at_most) == (20.0, 30.0)
    assert row.settled_median == 25.0


def test_half_unsettled_leaves_the_median_unbounded_above():
    records = [sweep_record(1, transient=10, period=1), sweep_record(2), sweep_record(3)]
    (row,) = median_transient(records)
    assert row.median_at_most is None and row.median_at_least == 0.0


def test_a_rerun_with_a_larger_cap_settles_the_seed():
    records = [
        sweep_record(1, transient=10, period=2),
        sweep_record(2, steps=100),  # unsettled at cap 100 ...
        sweep_record(2, transient=150, period=4, steps=1000),  # ... settled at cap 1000
        sweep_record(3, transient=12, period=2),
    ]
    (row,) = median_transient(records)
    assert row.complete and row.settling_times == [10, 150, 12] and row.median == 12.0


def test_the_largest_cap_tried_is_reported():
    (row,) = median_transient([sweep_record(1, steps=100), sweep_record(1, steps=5000)])
    assert row.max_cap == 4999 and row.unsettled == 1


def test_deterministic_runs_that_disagree_are_an_error_not_noise():
    records = [sweep_record(1, transient=10, period=2), sweep_record(1, transient=11, period=2)]
    with pytest.raises(ValueError, match="must agree"):
        median_transient(records)


def test_rows_are_per_torus_and_ordered_by_cells():
    records = [
        sweep_record(1, transient=5, period=1, grid=16),
        sweep_record(1, transient=3, period=1, grid="8x10"),
        sweep_record(1, transient=4, period=1, grid=8),
        sweep_record(1, transient=4, period=1, grid="8x8"),  # the same torus as 8
    ]
    rows = median_transient(records)
    assert [r.grid for r in rows] == [8, "8x10", 16]
    assert rows[1].size == pytest.approx(math.sqrt(80))


def test_census_records_work_too():
    census = [
        {
            "rule": "B2/S34",
            "shape": "hexagon",
            "triangle_mode": None,
            "grid": 32,
            "density": 0.5,
            "seed": s,
            "steps": 500,
            "outcome": "periodic",
            "period": 2,
            "transient": t,
        }
        for s, t in [(1, 7), (2, 9), (3, 8)]
    ]
    (row,) = median_transient(census)
    assert row.complete and row.median == 8.0


def test_sweeps_rerun_with_a_larger_cap_complete_their_rows(config, tmp_path):
    """The loop's pattern: a short cap leaves seeds unsettled; rerunning just those seeds with a
    larger cap into the same file completes the row, with the transients the long runs found."""
    out = tmp_path / "sweep.jsonl"
    rule = Rule.from_bs_string("B2/S34")
    list(run_sweep(config, rule, "hexagon", [8], [0.5], [1, 2, 3, 4], steps=10, out_path=out))
    (short,) = median_transient(read_jsonl(out))
    assert not short.complete and short.unsettled_seeds
    list(run_sweep(config, rule, "hexagon", [8], [0.5], short.unsettled_seeds, 3000, out))
    (full,) = median_transient(read_jsonl(out))
    assert full.complete and full.seeds == 4
    assert all(t >= 0 for t in full.settling_times)


# --- intervals ------------------------------------------------------------------------------


def test_wilson_interval_known_values():
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-4) and high == 1.0
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_bootstrap_interval_edges():
    assert bootstrap_interval([7.0]) == (7.0, 7.0)
    with pytest.raises(ValueError):
        bootstrap_interval([])
    low, high = bootstrap_interval(np.arange(100.0), np.mean)
    assert low < 49.5 < high


# --- survival -------------------------------------------------------------------------------


def test_survival_counts_per_density_with_wilson_intervals():
    records = [
        sweep_record(s, transient=1, period=1, density=d, survived=alive)
        for d, outcomes in [(0.2, [True, False, False, False]), (0.6, [True, True, True, False])]
        for s, alive in enumerate(outcomes, 1)
    ]
    rows = survival(records)
    assert [(r.density, r.survived, r.runs, r.p) for r in rows] == [
        (0.2, 1, 4, 0.25),
        (0.6, 3, 4, 0.75),
    ]
    assert all(r.low < r.p < r.high for r in rows)


def test_survival_counts_a_seed_once_and_refuses_contradictions():
    record = sweep_record(1, transient=1, period=1)
    (row,) = survival([record, dict(record)])
    assert row.runs == 1
    with pytest.raises(ValueError, match="must agree"):
        survival([record, {**record, "survived": False}])


# --- damage ---------------------------------------------------------------------------------


def test_damage_spread_means_the_final_damage_over_seeds():
    base = {
        "rule": "B2/S34",
        "shape": "hexagon",
        "triangle_mode": None,
        "grid": 16,
        "density": 0.5,
        "steps": 50,
        "flips": 1,
        "warmup": 0,
    }
    records = [
        {**base, "seed": 1, "final": 0.2, "healed_at": None},
        {**base, "seed": 2, "final": 0.0, "healed_at": 9},
        {**base, "seed": 3, "final": 0.4, "healed_at": None},
    ]
    (row,) = damage_spread(records)
    assert row.runs == 3 and row.healed == 1
    assert row.mean_final == pytest.approx(0.2)
    assert row.low <= row.mean_final <= row.high


# --- density over time from an autorun database ---------------------------------------------


def _record_session(recorder, config, rules, seed, generations=6, density=0.4):
    from dataclasses import replace

    session_config = replace(
        config,
        grid=replace(config.grid, shape="square", width=10, height=10, wrap=True),
        simulation=replace(
            config.simulation, initial_alive_percentage=density, n_lifeforms=len(rules)
        ),
    )
    lifeforms = [Lifeform(id=i, rule=r) for i, r in enumerate(rules, 1)]
    session = SimulationSession(session_config, lifeforms, seed=seed, auto_pause=False)
    sid = str(uuid.uuid4())
    recorder.start_session(sid, session_config, lifeforms, generations, seed)
    alive = []
    for _ in range(generations):
        session.step()
        recorder.record_generation(sid, session.history_records[-1])
        alive.append(int(np.count_nonzero(session.grid.occupancy)))
    return alive


def test_density_series_averages_single_species_sessions(config, tmp_path):
    life = Rule(birth={3}, survive={2, 3})
    db = tmp_path / "runs.db"
    with Recorder(db) as recorder:
        a = _record_session(recorder, config, [life], seed=1)
        b = _record_session(recorder, config, [life], seed=2)
        _record_session(recorder, config, [life, life], seed=3)  # two species: left out
    rows = density_series(db, rule="B3/S23")
    assert [r.generation for r in rows] == list(range(1, 7))
    assert {r.sessions for r in rows} == {2}
    for row, alive_a, alive_b in zip(rows, a, b, strict=True):
        assert row.mean == pytest.approx((alive_a + alive_b) / 200)
        assert (row.rule, row.shape, row.width, row.height) == ("B3/S23", "square", 10, 10)
    assert density_series(db, rule="B36/S23") == []
    assert len(density_series(db, max_generation=3)) == 3
