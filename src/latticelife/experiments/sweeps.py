"""Density sweeps: for one rule on one lattice, the probability that a random soup at initial
density rho0 is still alive after `steps` generations, over a grid of densities, torus sizes
and seeds — the raw material for a phase diagram, its critical densities, and finite-size
scaling of those thresholds.

Each run is a single species with exact cycle detection, so it stops early as soon as its
outcome is settled: when the grid is empty and the empty grid is absorbing (0 not in B), or when
a state repeats (then every later state, and the population at generation `steps - 1`, is
known: `rho_final` is read off the cycle). A rule with 0 in B is never cut short by emptiness,
since its empty grid fills with births; its `survived` is simply whether cells are alive at the
last planned generation.

Analysis helpers: `survival_table` turns records into P(survive) per (size, density);
`critical_densities` finds where P crosses 1/2 (there can be two: a soup too sparse to start and
one too dense to survive overcrowding both die); `fit_finite_size` fits
rho*(L) = rho_inf + a L^(-b) by a scan over b with a linear least-squares solve for the rest,
dependency-free and robust for the handful of sizes a sweep typically has.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from latticelife.config import Config
from latticelife.core.geometries import max_degree
from latticelife.core.rules import Rule
from latticelife.experiments.harness import (
    append_jsonl,
    map_tasks,
    resume_keys,
    single_species_session,
)
from latticelife.experiments.periodicity import CycleDetector
from latticelife.session.session import SimulationSession

RECORD_KEY: tuple[str, ...] = ("rule", "shape", "triangle_mode", "grid", "density", "seed", "steps")


@dataclass(frozen=True)
class SweepRecord:
    rule: str
    shape: str
    triangle_mode: str | None
    grid: int
    density: float
    seed: int
    steps: int
    survived: bool  # cells alive at generation steps - 1
    extinction_step: int | None  # first generation with no live cell
    rho_final: float  # density at generation steps - 1
    period: int | None
    transient: int | None
    stopped_at: int  # the last generation actually stepped

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SweepTask:
    config: Config
    rule: str
    shape: str
    grid: int
    density: float
    seed: int
    steps: int


def sweep_session(session: SimulationSession, steps: int) -> dict:
    """Run `session` (one species) until its fate at generation `steps - 1` is settled (see the
    module docstring) and report it."""
    if steps < 1:
        raise ValueError(f"steps must be at least 1 (got {steps})")
    rule = session.lifeforms[0].rule
    absorbing_empty = 0 not in rule.birth
    total = session.grid.n_cells
    detector = CycleDetector()
    rho = [np.count_nonzero(session.grid.occupancy) / total]
    detector.observe(session.grid.occupancy, 0)
    extinction_step = 0 if rho[0] == 0 else None
    period = transient = None
    stopped_at = 0
    for step in range(1, steps):
        if absorbing_empty and rho[-1] == 0:
            break
        session.step()
        stopped_at = step
        rho.append(np.count_nonzero(session.grid.occupancy) / total)
        if extinction_step is None and rho[-1] == 0:
            extinction_step = step
        repeat = detector.observe(session.grid.occupancy, step)
        if repeat is not None:
            period, transient = repeat
            break
    last = steps - 1
    if last < len(rho):
        rho_final = rho[last]
    elif period is not None:
        rho_final = rho[transient + (last - transient) % period]
    else:
        rho_final = rho[-1]  # empty and absorbing: stays empty
    return {
        "survived": bool(rho_final > 0),
        "extinction_step": extinction_step,
        "rho_final": float(rho_final),
        "period": period,
        "transient": transient,
        "stopped_at": stopped_at,
    }


def sweep_point(task: SweepTask) -> SweepRecord:
    rule = Rule.from_bs_string(task.rule)
    triangle_mode = task.config.grid.triangle_mode
    n = max_degree(task.shape, triangle_mode)
    session = single_species_session(
        task.config, rule, task.shape, task.grid, task.density, task.seed
    )
    return SweepRecord(
        rule=rule.to_bs_string(n),
        shape=task.shape,
        triangle_mode=triangle_mode if task.shape == "triangle" else None,
        grid=task.grid,
        density=task.density,
        seed=task.seed,
        steps=task.steps,
        **sweep_session(session, task.steps),
    )


def run_sweep(
    config: Config,
    rule: Rule,
    shape: str,
    grids: Sequence[int],
    densities: Sequence[float],
    seeds: Sequence[int],
    steps: int,
    out_path: str,
    workers: int = 1,
) -> Iterator[SweepRecord]:
    """Every (grid, density, seed) not already recorded in `out_path`, appended to it as it
    finishes and yielded."""
    n = max_degree(shape, config.grid.triangle_mode)
    rule_text = rule.to_bs_string(n)
    triangle_mode = config.grid.triangle_mode if shape == "triangle" else None
    done = resume_keys(out_path, RECORD_KEY)
    tasks = [
        SweepTask(config, rule_text, shape, grid, float(density), seed, steps)
        for grid in grids
        for density in densities
        for seed in seeds
        if (rule_text, shape, triangle_mode, grid, float(density), seed, steps) not in done
    ]
    for record in map_tasks(sweep_point, tasks, workers):
        append_jsonl(out_path, record.to_json())
        yield record


def survival_table(
    records: Iterable[SweepRecord | dict],
) -> dict[int, list[tuple[float, float, int]]]:
    """Per torus size, `(density, P(survive), runs)` sorted by density."""
    counts: dict[tuple[int, float], list[int]] = {}
    for record in records:
        data = record if isinstance(record, dict) else record.to_json()
        cell = counts.setdefault((int(data["grid"]), float(data["density"])), [0, 0])
        cell[0] += int(bool(data["survived"]))
        cell[1] += 1
    table: dict[int, list[tuple[float, float, int]]] = {}
    for (grid, density), (survived, runs) in sorted(counts.items()):
        table.setdefault(grid, []).append((density, survived / runs, runs))
    return table


def critical_densities(points: Sequence[tuple[float, float]]) -> list[float]:
    """The densities at which P(survive) crosses 1/2, by linear interpolation between
    neighbouring `(density, probability)` points (sorted by density). A point exactly at 1/2
    is a crossing itself."""
    crossings: list[float] = []
    ordered = sorted(points)
    for (d0, p0), (d1, p1) in zip(ordered, ordered[1:], strict=False):
        a, b = p0 - 0.5, p1 - 0.5
        if a == 0:
            crossings.append(d0)
        elif a * b < 0:
            crossings.append(d0 + (d1 - d0) * a / (a - b))
    if ordered and ordered[-1][1] == 0.5:
        crossings.append(ordered[-1][0])
    return crossings


@dataclass(frozen=True)
class FiniteSizeFit:
    rho_inf: float
    amplitude: float
    exponent: float
    residual: float  # sum of squared residuals


def fit_finite_size(
    sizes: Sequence[float], thresholds: Sequence[float], exponents: np.ndarray | None = None
) -> FiniteSizeFit:
    """Least-squares fit of rho*(L) = rho_inf + a L^(-b) to the thresholds measured at each
    torus size L, scanning b over `exponents` (default: 400 values from 0.02 to 6, log-spaced)
    and solving rho_inf and a exactly for each. Needs at least three sizes."""
    L = np.asarray(sizes, dtype=float)
    y = np.asarray(thresholds, dtype=float)
    if L.size < 3 or L.size != y.size:
        raise ValueError("finite-size scaling needs thresholds at three or more sizes")
    if exponents is None:
        exponents = np.geomspace(0.02, 6.0, 400)
    best: FiniteSizeFit | None = None
    for b in exponents:
        design = np.column_stack([np.ones_like(L), L ** (-b)])
        solution, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = float(((design @ solution - y) ** 2).sum())
        if best is None or residual < best.residual:
            best = FiniteSizeFit(float(solution[0]), float(solution[1]), float(b), residual)
    assert best is not None
    return best
