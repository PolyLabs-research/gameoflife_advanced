"""Observables: what the experiments' records say, as numbers with uncertainty and stated limits.
Each function takes run records (the dicts the JSONL files hold, or the record dataclasses) and
returns one row per setting. A row says how many runs it rests on, gives an interval, and says
plainly when its number is only a bound, because only the simulator knows that: a run with no
repeated state within its cap does not have a long transient, it has a long transient *or* a long
period, and a law about settling time judged on such a run would be judged on a guess.

`median_transient`: per torus size (and density), the median settling time over seeds with a
bootstrap 95% interval. A seed's settling time is its exact transient: the generation its final
cycle begins (for a soup that dies on a rule without birth from nothing, the generation it
died). Records of the same seed at different caps (`steps`) are merged, since a run is
deterministic given its seed: a seed counts as settled if any of its runs settled, and is
otherwise unsettled up to the largest cap tried (`no_repeat_before`). A row with an unsettled
seed is not `complete`: its `median` is None, `median_at_least`/`median_at_most` bound what the
median could be, and `unsettled_seeds` names the seeds to rerun with a larger cap.

`survival`: per size, density and step count, the share of soups alive at generation
`steps - 1`, with a Wilson score interval. Every sweep run's fate is exact, so nothing here is
censored.

`event_times`: per size and density, each seed's time to an event, exact or only a bound, for a
fit that uses both (a censored likelihood). The event is `first_repeat`, the first generation
whose state repeats an earlier one (transient + period: the quantity a cap really bounds, since
an unsettled run has a long transient *or* a long period), or `extinction`, the first generation
with no live cell (sweep records only). For extinction there is a third kind of seed: one settled
on a cycle with live cells in every state can never die, and is counted apart (`never`), not
treated as a bound. Each row carries the Kaplan-Meier median for people, None when the
censoring leaves it unknown.

`density_series`: mean density over time, averaged over single-species sessions with the same
rule, lattice and torus, read from an `autorun` database (schema v2).

`damage_spread`: per size and density, the mean final fraction of damaged cells over seeds, with
a bootstrap interval, and the share of runs that healed. `damage_curves`: every seed's whole
damage curve, as replicates for a law over time whose error bars come from resampling seeds.

`object_sizes`: from `soup` records, the population of every object found and how often each
distinct object appeared, as samples for a distribution fit (power law, lognormal, ...).

The bootstrap is seeded (`bootstrap_seed`), so the same records always give the same rows.
Rows are frozen dataclasses with `to_json()`; `size` is the square root of the cell count, the
side of a square torus with as many cells, so square and rectangular tori share one axis.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from latticelife.core.geometries import max_degree
from latticelife.core.rules import Rule, parse_counts
from latticelife.experiments.harness import GridLabel, grid_dimensions, grid_sort_key, parse_grid

CONFIDENCE = 0.95
Z95 = 1.959963984540054  # standard normal quantile for a 95% two-sided interval
BOOTSTRAP_RESAMPLES = 2000


def _as_dict(record: Any) -> dict:
    return record if isinstance(record, dict) else record.to_json()


def _geometry(grid: GridLabel) -> dict:
    width, height = grid_dimensions(grid)
    cells = width * height
    return {"width": width, "height": height, "cells": cells, "size": math.sqrt(cells)}


def settling_time(record: dict) -> int | None:
    """The exact settling time of one census or sweep record, or None when it did not settle
    within its cap. A soup that died on a rule with 0 not in B was absorbed at its extinction
    step (the sweep stops there, so it records no period)."""
    record = _as_dict(record)
    if record.get("transient") is not None:
        return int(record["transient"])
    extinct = record.get("extinction_step")
    if extinct is not None and 0 not in Rule.from_bs_string(record["rule"]).birth:
        return int(extinct)
    return None


def no_repeat_before(record: dict) -> int | None:
    """For a record that did not settle, the last generation it reached with no state repeated
    (so transient + period exceeds it); None for a settled record. Records written before the
    field existed are read from `stopped_at` (sweep) or `steps - 1` (census)."""
    record = _as_dict(record)
    if settling_time(record) is not None:
        return None
    if record.get("no_repeat_before") is not None:
        return int(record["no_repeat_before"])
    if record.get("stopped_at") is not None:
        return int(record["stopped_at"])
    return int(record["steps"]) - 1


def bootstrap_interval(
    values: Sequence[float],
    statistic=np.median,
    confidence: float = CONFIDENCE,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval of `statistic` over `values`, from a seeded generator.
    One value gives a zero-width interval at that value."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("no values to bootstrap")
    if data.size == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, data.size, size=(resamples, data.size))
    stats = statistic(data[draws], axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(stats, [tail, 100.0 - tail])
    return float(low), float(high)


def wilson_interval(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (sound at 0 and 1, unlike p +- z se)."""
    if trials < 1:
        raise ValueError("no trials")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    # At p = 0 or 1 the interval ends exactly at 0 or 1; say so rather than leave rounding.
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == trials else min(1.0, centre + half)
    return low, high


# --- settling time --------------------------------------------------------------------------


@dataclass(frozen=True)
class TransientRow:
    rule: str
    shape: str
    triangle_mode: str | None
    grid: GridLabel
    width: int
    height: int
    cells: int
    size: float  # sqrt(cells)
    density: float
    seeds: int  # distinct seeds
    settled: int  # seeds with an exact settling time
    unsettled: int  # seeds that never repeated within any cap tried
    complete: bool  # every seed settled: `median` is a measurement, not a bound
    median: float | None  # median settling time over all seeds, when complete
    low: float | None  # bootstrap 95% interval of that median
    high: float | None
    median_at_least: float | None  # what the median over all seeds could be, from the settled
    median_at_most: float | None  # seeds alone (None: unbounded)
    settled_median: float | None  # median over the settled seeds only, for information
    max_cap: int | None  # the largest no_repeat_before among unsettled seeds
    unsettled_seeds: list[int] = field(default_factory=list)
    settling_times: list[int] = field(default_factory=list)  # settled seeds' times, by seed

    def to_json(self) -> dict:
        return asdict(self)


def _bounded_median(known: list[int], unknown: int) -> tuple[float, float | None]:
    """The median of `known` plus `unknown` values that could be anything from 0 up: lowest
    with them all 0, highest with them all unbounded (None when that leaves it unbounded)."""
    low = float(np.median(sorted(known) + [0] * unknown)) if known or unknown else math.nan
    total = len(known) + unknown
    ordered = sorted(known)
    if total % 2:
        idx = total // 2
        high = float(ordered[idx]) if idx < len(ordered) else None
    else:
        idx = total // 2
        high = float((ordered[idx - 1] + ordered[idx]) / 2) if idx < len(ordered) else None
    return low, high


def median_transient(
    records: Iterable[dict],
    bootstrap_seed: int = 0,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[TransientRow]:
    """Per (rule, shape, triangle mode, torus, density): the median settling time over seeds,
    with a bootstrap 95% interval, from census or sweep records (see the module docstring).
    Raises ValueError if two settled records of one seed disagree: the runs are deterministic,
    so that is a bug or a mixed-up file, never noise."""
    groups: dict[tuple, dict[int, dict]] = {}
    for record in records:
        data = _as_dict(record)
        key = (
            data["rule"],
            data["shape"],
            data.get("triangle_mode"),
            parse_grid(data["grid"]),
            float(data["density"]),
        )
        seeds = groups.setdefault(key, {})
        seed = int(data["seed"])
        time = settling_time(data)
        state = seeds.setdefault(seed, {"time": None, "cap": None})
        if time is not None:
            if state["time"] is not None and state["time"] != time:
                raise ValueError(
                    f"seed {seed} of {key} settled at {state['time']} in one record and {time} "
                    "in another; deterministic runs must agree"
                )
            state["time"] = time
        else:
            cap = no_repeat_before(data)
            state["cap"] = cap if state["cap"] is None else max(state["cap"], cap)

    rows = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], str(k[2]), grid_sort_key(k[3]), k[4])):
        rule, shape, triangle_mode, grid, density = key
        seeds = groups[key]
        known = {s: v["time"] for s, v in sorted(seeds.items()) if v["time"] is not None}
        open_seeds = sorted(s for s, v in seeds.items() if v["time"] is None)
        times = list(known.values())
        complete = not open_seeds
        median = low = high = None
        if complete:
            median = float(np.median(times))
            low, high = bootstrap_interval(times, seed=bootstrap_seed, resamples=resamples)
        at_least, at_most = _bounded_median(times, len(open_seeds))
        caps = [seeds[s]["cap"] for s in open_seeds]
        rows.append(
            TransientRow(
                rule=rule,
                shape=shape,
                triangle_mode=triangle_mode,
                grid=grid,
                **_geometry(grid),
                density=density,
                seeds=len(seeds),
                settled=len(times),
                unsettled=len(open_seeds),
                complete=complete,
                median=median,
                low=low,
                high=high,
                median_at_least=at_least,
                median_at_most=at_most,
                settled_median=float(np.median(times)) if times else None,
                max_cap=max(caps) if caps else None,
                unsettled_seeds=open_seeds,
                settling_times=times,
            )
        )
    return rows


# --- event times ----------------------------------------------------------------------------

EVENTS = ("first_repeat", "extinction")


def first_repeat_time(record: dict) -> int | None:
    """The first generation whose state equals an earlier one's (transient + period), or None
    when no state repeated within the cap (it then exceeds `no_repeat_before`). A soup absorbed
    into the empty grid at generation e, on a rule without birth from nothing, repeats at e + 1."""
    record = _as_dict(record)
    if record.get("period") is not None and record.get("transient") is not None:
        return int(record["transient"]) + int(record["period"])
    time = settling_time(record)
    return None if time is None else time + 1


def extinction_outcome(record: dict) -> tuple[str, int | None]:
    """How one sweep run ends: ("died", e) with e the first generation with no live cell;
    ("never", None) for a run settled on a cycle with live cells in every state, which can never
    die; ("alive", g) for a run alive through generation g, the last it stepped, with its fate
    unsettled (it dies after g, or never). Census records keep no extinction step: ValueError."""
    record = _as_dict(record)
    if "extinction_step" not in record:
        raise ValueError(
            "extinction times need sweep records (census records keep no extinction step)"
        )
    if record["extinction_step"] is not None:
        return "died", int(record["extinction_step"])
    if record.get("period") is not None:
        return "never", None
    return "alive", int(record["stopped_at"])


@dataclass(frozen=True)
class EventTimeRow:
    rule: str
    shape: str
    triangle_mode: str | None
    grid: GridLabel
    width: int
    height: int
    cells: int
    size: float
    density: float
    event: str  # "first_repeat" or "extinction"
    seeds: int
    events: int  # seeds whose time is exact
    censored: int  # seeds known only to take longer than a bound
    never: int  # seeds that can never reach the event (extinction only)
    km_median: float | None  # Kaplan-Meier median over all seeds; None when not reached
    median_above: float | None  # when not reached: the median exceeds this
    seed_list: list[int] = field(default_factory=list)  # seeds with a time or a bound, sorted
    times: list[int] = field(default_factory=list)  # each one's exact time, or its bound
    is_censored: list[bool] = field(default_factory=list)  # True: the time exceeds `times`
    never_seeds: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def kaplan_meier_median(
    times: Sequence[float], censored: Sequence[bool], never: int = 0
) -> float | None:
    """The Kaplan-Meier median of event times where `censored` marks lower bounds and `never`
    more runs can never have the event: the first event time at which the share still waiting
    falls to one half. None when it never does within the data."""
    order = sorted(zip(times, censored, strict=True), key=lambda tc: (tc[0], tc[1]))  # events first
    at_risk = len(order) + never
    waiting = 1.0
    i = 0
    while i < len(order):
        t = order[i][0]
        deaths = leaving = 0
        while i < len(order) and order[i][0] == t:
            deaths += not order[i][1]
            leaving += 1
            i += 1
        if deaths:
            waiting *= 1.0 - deaths / at_risk
            if waiting <= 0.5 + 1e-12:
                return float(t)
        at_risk -= leaving
    return None


def event_times(records: Iterable[dict], event: str = "first_repeat") -> list[EventTimeRow]:
    """Per (rule, shape, triangle mode, torus, density): each seed's time to `event`, exact or a
    bound (see the module docstring). Records of one seed at different caps merge: an exact
    time from any of them wins (and they must agree), otherwise the largest bound."""
    if event not in EVENTS:
        raise ValueError(f"event must be one of {', '.join(EVENTS)} (got {event!r})")
    groups: dict[tuple, dict[int, dict]] = {}
    for record in records:
        data = _as_dict(record)
        key = (
            data["rule"],
            data["shape"],
            data.get("triangle_mode"),
            parse_grid(data["grid"]),
            float(data["density"]),
        )
        seed = int(data["seed"])
        state = groups.setdefault(key, {}).setdefault(
            seed, {"kind": None, "time": None, "bound": None}
        )
        if event == "first_repeat":
            time = first_repeat_time(data)
            kind = "died" if time is not None else "alive"
            value = time if time is not None else no_repeat_before(data)
        else:
            kind, value = extinction_outcome(data)
        if kind == "alive":
            state["bound"] = value if state["bound"] is None else max(state["bound"], value)
            continue
        if state["kind"] is not None and (state["kind"], state["time"]) != (kind, value):
            raise ValueError(
                f"seed {seed} of {key} ended as {state['kind']} {state['time']} in one record and "
                f"{kind} {value} in another; deterministic runs must agree"
            )
        state["kind"], state["time"] = kind, value

    rows = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], str(k[2]), grid_sort_key(k[3]), k[4])):
        rule, shape, triangle_mode, grid, density = key
        seeds = groups[key]
        seed_list, times, flags, never_seeds = [], [], [], []
        for seed, state in sorted(seeds.items()):
            if state["kind"] == "never":
                never_seeds.append(seed)
            elif state["kind"] == "died":
                seed_list.append(seed)
                times.append(state["time"])
                flags.append(False)
            else:
                seed_list.append(seed)
                times.append(state["bound"])
                flags.append(True)
        median = kaplan_meier_median(times, flags, len(never_seeds))
        rows.append(
            EventTimeRow(
                rule=rule,
                shape=shape,
                triangle_mode=triangle_mode,
                grid=grid,
                **_geometry(grid),
                density=density,
                event=event,
                seeds=len(seeds),
                events=flags.count(False),
                censored=flags.count(True),
                never=len(never_seeds),
                km_median=median,
                median_above=None if median is not None or not times else float(max(times)),
                seed_list=seed_list,
                times=times,
                is_censored=flags,
                never_seeds=never_seeds,
            )
        )
    return rows


# --- survival -------------------------------------------------------------------------------


@dataclass(frozen=True)
class SurvivalRow:
    rule: str
    shape: str
    triangle_mode: str | None
    grid: GridLabel
    width: int
    height: int
    cells: int
    size: float
    density: float
    steps: int
    runs: int
    survived: int
    p: float  # share of soups alive at generation steps - 1
    low: float  # Wilson 95% interval
    high: float

    def to_json(self) -> dict:
        return asdict(self)


def survival(records: Iterable[dict]) -> list[SurvivalRow]:
    """Per (rule, shape, triangle mode, torus, density, steps): the share of sweep runs alive at
    generation `steps - 1`, with a Wilson 95% interval. A seed recorded twice counts once (and
    must agree)."""
    groups: dict[tuple, dict[int, bool]] = {}
    for record in records:
        data = _as_dict(record)
        key = (
            data["rule"],
            data["shape"],
            data.get("triangle_mode"),
            parse_grid(data["grid"]),
            float(data["density"]),
            int(data["steps"]),
        )
        seeds = groups.setdefault(key, {})
        seed, alive = int(data["seed"]), bool(data["survived"])
        if seeds.get(seed, alive) != alive:
            raise ValueError(f"seed {seed} of {key} both survived and died; runs must agree")
        seeds[seed] = alive
    rows = []
    for key in sorted(
        groups, key=lambda k: (k[0], k[1], str(k[2]), grid_sort_key(k[3]), k[5], k[4])
    ):
        rule, shape, triangle_mode, grid, density, steps = key
        outcomes = groups[key]
        runs, alive = len(outcomes), sum(outcomes.values())
        low, high = wilson_interval(alive, runs)
        rows.append(
            SurvivalRow(
                rule=rule,
                shape=shape,
                triangle_mode=triangle_mode,
                grid=grid,
                **_geometry(grid),
                density=density,
                steps=steps,
                runs=runs,
                survived=alive,
                p=alive / runs,
                low=low,
                high=high,
            )
        )
    return rows


# --- density over time ----------------------------------------------------------------------


@dataclass(frozen=True)
class DensityRow:
    rule: str
    shape: str
    triangle_mode: str | None
    width: int
    height: int
    generation: int
    mean: float  # mean density over the sessions at this generation
    sem: float  # its standard error (0 for a single session)
    sessions: int
    initial_density_min: float  # the sessions' seeded densities span this range
    initial_density_max: float

    def to_json(self) -> dict:
        return asdict(self)


def density_series(
    db_path: str | Path,
    rule: str | None = None,
    shape: str | None = None,
    grid: int | str | None = None,
    max_generation: int | None = None,
) -> list[DensityRow]:
    """Mean density per generation over the single-species sessions of an `autorun` database,
    grouped by rule, shape, triangle mode and torus (optionally only those matching `rule`,
    `shape` and `grid`). Multi-species sessions are left out: their births are random. The
    database is opened read-only."""
    uri = f"file:{Path(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        sessions = conn.execute(
            """SELECT s.session_id, s.shape, s.triangle_mode, s.grid_width, s.grid_height,
                      s.initial_alive_percentage, l.birth, l.survive
               FROM session_metadata s JOIN lifeform_meta l ON l.session_id = s.session_id
               WHERE s.n_lifeforms = 1"""
        ).fetchall()
        want_dims = grid_dimensions(grid) if grid is not None else None
        want_rule = Rule.from_bs_string(rule) if rule is not None else None
        groups: dict[tuple, list[tuple[str, float]]] = {}
        for sid, sh, tm, w, h, rho0, birth, survive in sessions:
            if shape is not None and sh != shape:
                continue
            if want_dims is not None and (w, h) != want_dims:
                continue
            parsed = Rule(birth=parse_counts(birth), survive=parse_counts(survive))
            if want_rule is not None and (parsed.birth, parsed.survive) != (
                want_rule.birth,
                want_rule.survive,
            ):
                continue
            text = parsed.to_bs_string(max_degree(sh, tm or "edge+vertex"))
            mode = tm if sh == "triangle" else None
            groups.setdefault((text, sh, mode, w, h), []).append((sid, float(rho0)))

        rows: list[DensityRow] = []
        for (text, sh, mode, w, h), members in sorted(groups.items(), key=lambda g: str(g[0])):
            cells = w * h
            series: dict[int, list[float]] = {}
            for sid, _ in members:
                query = (
                    "SELECT generation, total_alive FROM generation_metrics WHERE session_id = ?"
                )
                params: list = [sid]
                if max_generation is not None:
                    query += " AND generation <= ?"
                    params.append(max_generation)
                for generation, alive in conn.execute(query, params):
                    series.setdefault(int(generation), []).append(alive / cells)
            seeded = [rho0 for _, rho0 in members]
            for generation in sorted(series):
                values = np.asarray(series[generation])
                sem = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
                rows.append(
                    DensityRow(
                        rule=text,
                        shape=sh,
                        triangle_mode=mode,
                        width=w,
                        height=h,
                        generation=generation,
                        mean=float(values.mean()),
                        sem=sem,
                        sessions=len(values),
                        initial_density_min=min(seeded),
                        initial_density_max=max(seeded),
                    )
                )
        return rows
    finally:
        conn.close()


# --- damage spreading -----------------------------------------------------------------------


@dataclass(frozen=True)
class DamageRow:
    rule: str
    shape: str
    triangle_mode: str | None
    grid: GridLabel
    width: int
    height: int
    cells: int
    size: float
    density: float
    steps: int
    flips: int
    warmup: int
    runs: int
    healed: int  # runs whose twins agreed again
    mean_final: float  # mean fraction of cells differing at the last generation
    low: float  # bootstrap 95% interval of that mean
    high: float

    def to_json(self) -> dict:
        return asdict(self)


def damage_spread(
    records: Iterable[dict], bootstrap_seed: int = 0, resamples: int = BOOTSTRAP_RESAMPLES
) -> list[DamageRow]:
    """Per (rule, shape, triangle mode, torus, density, steps, flips, warmup): the mean final
    damage over seeds with a bootstrap 95% interval, and how many runs healed."""
    groups: dict[tuple, dict[int, dict]] = {}
    for record in records:
        data = _as_dict(record)
        key = (
            data["rule"],
            data["shape"],
            data.get("triangle_mode"),
            parse_grid(data["grid"]),
            float(data["density"]),
            int(data["steps"]),
            int(data["flips"]),
            int(data["warmup"]),
        )
        groups.setdefault(key, {})[int(data["seed"])] = data
    rows = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], str(k[2]), grid_sort_key(k[3]), k[4:])):
        rule, shape, triangle_mode, grid, density, steps, flips, warmup = key
        runs = list(groups[key].values())
        finals = [float(r["final"]) for r in runs]
        low, high = bootstrap_interval(finals, np.mean, seed=bootstrap_seed, resamples=resamples)
        rows.append(
            DamageRow(
                rule=rule,
                shape=shape,
                triangle_mode=triangle_mode,
                grid=grid,
                **_geometry(grid),
                density=density,
                steps=steps,
                flips=flips,
                warmup=warmup,
                runs=len(runs),
                healed=sum(r["healed_at"] is not None for r in runs),
                mean_final=float(np.mean(finals)),
                low=low,
                high=high,
            )
        )
    return rows


@dataclass(frozen=True)
class CurveRow:
    rule: str
    shape: str
    triangle_mode: str | None
    grid: GridLabel
    width: int
    height: int
    cells: int
    size: float
    density: float
    steps: int
    flips: int
    warmup: int
    seeds: list[int] = field(default_factory=list)
    curves: list[list[float]] = field(default_factory=list)  # per seed: damaged share by generation

    def to_json(self) -> dict:
        return asdict(self)


def damage_curves(records: Iterable[dict]) -> list[CurveRow]:
    """Per (rule, shape, triangle mode, torus, density, steps, flips, warmup): every seed's whole
    damage curve, the share of cells in which the twins differ at generations 0..steps-1. These
    are replicates: points along one curve are correlated, so a law fitted to their mean takes
    its error bars from resampling whole curves, not from the scatter about the mean."""
    groups: dict[tuple, dict[int, list[float]]] = {}
    for record in records:
        data = _as_dict(record)
        key = (
            data["rule"],
            data["shape"],
            data.get("triangle_mode"),
            parse_grid(data["grid"]),
            float(data["density"]),
            int(data["steps"]),
            int(data["flips"]),
            int(data["warmup"]),
        )
        curve = [d / int(data["cells"]) for d in data["differing"]]
        seeds = groups.setdefault(key, {})
        seed = int(data["seed"])
        if seed in seeds and seeds[seed] != curve:
            raise ValueError(f"seed {seed} of {key} has two different curves; runs must agree")
        seeds[seed] = curve
    rows = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], str(k[2]), grid_sort_key(k[3]), k[4:])):
        rule, shape, triangle_mode, grid, density, steps, flips, warmup = key
        seeds = groups[key]
        rows.append(
            CurveRow(
                rule=rule,
                shape=shape,
                triangle_mode=triangle_mode,
                grid=grid,
                **_geometry(grid),
                density=density,
                steps=steps,
                flips=flips,
                warmup=warmup,
                seeds=sorted(seeds),
                curves=[seeds[s] for s in sorted(seeds)],
            )
        )
    return rows


# --- objects --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectSizeRow:
    rule: str
    shape: str
    triangle_mode: str | None
    grid: GridLabel
    width: int
    height: int
    cells: int
    size: float
    density: float
    steps: int
    soups: int
    objects: int  # objects identified, over every soup
    unidentified: int  # components that were not objects (died, unsettled, too large)
    populations: list[int] = field(default_factory=list)  # each object's cells, smallest phase
    abundances: list[int] = field(default_factory=list)  # times each distinct object was seen

    def to_json(self) -> dict:
        return asdict(self)


def object_sizes(records: Iterable[dict]) -> list[ObjectSizeRow]:
    """Per (rule, shape, triangle mode, torus, density, steps), from `soup` records: the
    population of every object found, and how often each distinct object appeared (most common
    first), the samples for asking which distribution sizes and abundances follow. A seed recorded
    twice counts once."""
    groups: dict[tuple, dict[int, dict]] = {}
    for record in records:
        data = _as_dict(record)
        key = (
            data["rule"],
            data["shape"],
            data.get("triangle_mode"),
            parse_grid(data["grid"]),
            float(data["density"]),
            int(data["steps"]),
        )
        groups.setdefault(key, {})[int(data["seed"])] = data
    rows = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], str(k[2]), grid_sort_key(k[3]), k[4:])):
        rule, shape, triangle_mode, grid, density, steps = key
        soups = list(groups[key].values())
        found = [o for soup in soups for o in soup["objects"]]
        counts: dict[tuple, int] = {}
        for o in found:
            k = tuple(tuple(cell) for cell in o["key"])
            counts[k] = counts.get(k, 0) + 1
        rows.append(
            ObjectSizeRow(
                rule=rule,
                shape=shape,
                triangle_mode=triangle_mode,
                grid=grid,
                **_geometry(grid),
                density=density,
                steps=steps,
                soups=len(soups),
                objects=len(found),
                unidentified=sum(sum(soup["unidentified"].values()) for soup in soups),
                populations=sorted(int(o["population"]) for o in found),
                abundances=sorted(counts.values(), reverse=True),
            )
        )
    return rows


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "CONFIDENCE",
    "EVENTS",
    "CurveRow",
    "DamageRow",
    "DensityRow",
    "EventTimeRow",
    "ObjectSizeRow",
    "SurvivalRow",
    "TransientRow",
    "bootstrap_interval",
    "damage_curves",
    "damage_spread",
    "density_series",
    "event_times",
    "extinction_outcome",
    "first_repeat_time",
    "kaplan_meier_median",
    "median_transient",
    "no_repeat_before",
    "object_sizes",
    "settling_time",
    "survival",
    "wilson_interval",
]
