"""Event times (`experiments.observables.event_times`): each seed's time to its first repeated
state, or to extinction, exact or only a bound, so a law can be fitted with a censored likelihood
instead of on a median that is itself a guess."""

from pathlib import Path

import pytest

from latticelife.config import load_config
from latticelife.core.rules import Rule
from latticelife.experiments.harness import read_jsonl
from latticelife.experiments.observables import (
    event_times,
    extinction_outcome,
    first_repeat_time,
    kaplan_meier_median,
)
from latticelife.experiments.sweeps import run_sweep

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


def record(seed, transient=None, period=None, extinction=None, steps=100, rule="B2/S34", **extra):
    settled = transient is not None or (
        extinction is not None and "0" not in rule.split("/")[0][1:]
    )
    stopped = steps - 1
    if transient is not None:
        stopped = transient + (period or 1)
    elif settled:
        stopped = extinction
    data = {
        "rule": rule,
        "shape": "hexagon",
        "triangle_mode": None,
        "grid": 16,
        "density": 0.5,
        "seed": seed,
        "steps": steps,
        "survived": extinction is None,
        "extinction_step": extinction,
        "rho_final": 0.0 if extinction is not None else 0.2,
        "period": period,
        "transient": transient,
        "stopped_at": stopped,
        "no_repeat_before": None if settled else stopped,
    }
    data.update(extra)
    return data


def test_the_first_repeat_is_transient_plus_period():
    assert first_repeat_time(record(1, transient=10, period=3)) == 13
    # Absorbed into the empty grid at 7 (no birth from nothing): the empty state recurs at 8.
    assert first_repeat_time(record(2, extinction=7)) == 8
    assert first_repeat_time(record(3)) is None


def test_a_run_dies_never_dies_or_is_still_alive():
    assert extinction_outcome(record(1, extinction=7)) == ("died", 7)
    assert extinction_outcome(record(2, transient=10, period=3)) == ("never", None)
    assert extinction_outcome(record(3, steps=100)) == ("alive", 99)
    census = {k: v for k, v in record(4).items() if k not in ("extinction_step", "stopped_at")}
    with pytest.raises(ValueError, match="sweep records"):
        extinction_outcome(census)


def test_kaplan_meier_median_known_values():
    assert kaplan_meier_median([1, 2, 3, 4], [False] * 4) == 2
    # A bound at 2 leaves the run at risk until then: 0.8, then x 2/3 at 3, then x 1/2 at 4.
    assert kaplan_meier_median([1, 2, 3, 4, 5], [False, True, False, False, False]) == 4
    assert kaplan_meier_median([1, 2, 3], [False] * 3, never=3) == 3
    assert kaplan_meier_median([1, 2, 3], [False] * 3, never=4) is None
    assert kaplan_meier_median([5, 9], [True, True]) is None


def test_rows_keep_exact_times_and_bounds_apart():
    rows = event_times(
        [record(1, transient=10, period=3), record(2), record(3, transient=4, period=1)]
    )
    (row,) = rows
    assert (row.event, row.seeds, row.events, row.censored, row.never) == (
        "first_repeat",
        3,
        2,
        1,
        0,
    )
    assert (
        row.seed_list == [1, 2, 3]
        and row.times == [13, 99, 5]
        and row.is_censored == [False, True, False]
    )
    assert row.km_median == 13


def test_never_dying_seeds_are_counted_apart_not_as_bounds():
    rows = event_times(
        [record(1, extinction=7), record(2, transient=10, period=3), record(3)], "extinction"
    )
    (row,) = rows
    assert (row.events, row.censored, row.never, row.never_seeds) == (1, 1, 1, [2])
    assert row.times == [7, 99] and row.is_censored == [False, True]
    assert row.km_median is None and row.median_above == 99


def test_a_larger_cap_replaces_a_bound_and_runs_must_agree():
    rows = event_times([record(1, steps=50), record(1, transient=60, period=2, steps=500)])
    assert rows[0].times == [62] and rows[0].is_censored == [False]
    rows = event_times([record(1, steps=50), record(1, steps=500)])
    assert rows[0].times == [499] and rows[0].is_censored == [True]
    with pytest.raises(ValueError, match="must agree"):
        event_times([record(1, transient=10, period=3), record(1, transient=11, period=3)])
    with pytest.raises(ValueError, match="event must be"):
        event_times([], "settling")


def test_a_real_bound_is_honest(config, tmp_path):
    """Every bound a short cap reports is exceeded by the exact time a long cap finds."""
    out = tmp_path / "sweep.jsonl"
    rule = Rule.from_bs_string("B2/S34")
    seeds = [1, 2, 3, 4, 5, 6]
    list(run_sweep(config, rule, "hexagon", [8], [0.5], seeds, steps=20, out_path=out))
    (short,) = event_times(read_jsonl(out))
    assert short.censored > 0
    long_out = tmp_path / "long.jsonl"
    list(run_sweep(config, rule, "hexagon", [8], [0.5], seeds, steps=5000, out_path=long_out))
    (full,) = event_times(read_jsonl(long_out))
    assert full.censored == 0
    exact = dict(zip(full.seed_list, full.times, strict=True))
    for seed, bound, censored in zip(short.seed_list, short.times, short.is_censored, strict=True):
        if censored:
            assert exact[seed] > bound
        else:
            assert exact[seed] == bound
    (extinction,) = event_times(read_jsonl(long_out), "extinction")
    assert extinction.events + extinction.censored + extinction.never == len(seeds)
