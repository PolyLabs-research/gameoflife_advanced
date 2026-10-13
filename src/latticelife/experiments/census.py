"""Rule census: the terminal behaviour of a set of rules on one lattice, one record per
(rule, torus size, seed). The hexagon lattice has 2^14 = 16,384 outer-totalistic rules and the
edge-triangle lattice 2^8 = 256, so both rule spaces can be run in full (`enumerate_rules`);
square (2^18) and edge+vertex triangle (2^26) spaces are sampled or listed by hand.

Each run seeds a torus at a density and steps it with exact cycle detection
(`experiments.runner.run_timeseries(detect_cycles=True)`), so the outcome is a fact about the
finite system, not a guess from the density trace:

- `extinct`: the grid is empty at the end;
- `fixed`: a state repeated after one step (a still configuration);
- `periodic`: a state repeated after more than one step, with the exact `period` and the
  generation the cycle began (`transient`);
- `unsettled`: no state repeated within `steps` — chaotic, or a transient longer than the run.

Beside the measured behaviour each record carries two things computable from the rule alone,
for the analysis to set against it: Langton's lambda (the fraction of the 2^(n+1) neighbourhood
configurations, weighted by multiplicity, that map to a live cell) and the stable fixed points
of the mean-field map rho -> (1-rho) P[Bin(n, rho) in B] + rho P[Bin(n, rho) in S], which knows
only the degree and so measures, by its disagreement with the run, how much the spatial
arrangement of cells matters for that rule. The complement rule (`Rule.complement`) is recorded
as `dual`: it behaves as this rule does on the complemented grid, so a census of one rule per dual
pair (`enumerate_rules(up_to_duality=True)`) covers the whole space.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass

from numpy.polynomial import Polynomial

from latticelife.config import Config
from latticelife.core.geometries import max_degree
from latticelife.core.rules import Rule
from latticelife.experiments.harness import (
    append_jsonl,
    map_tasks,
    resume_keys,
    single_species_session,
)
from latticelife.experiments.runner import run_timeseries
from latticelife.session.metrics import compute_cluster_metrics
from latticelife.session.session import SimulationSession

OUTCOMES: tuple[str, ...] = ("extinct", "fixed", "periodic", "unsettled")

# The fields that identify one census task; a record with all of them equal is that task done.
RECORD_KEY: tuple[str, ...] = ("rule", "shape", "triangle_mode", "grid", "steps", "seed", "density")


def rule_space_size(n: int) -> int:
    """How many outer-totalistic rules a lattice of degree `n` has: a birth set and a survival
    set, each any subset of the counts 0..n."""
    return 2 ** (2 * (n + 1))


def rule_masks(rule: Rule, n: int) -> tuple[int, int]:
    """`rule` as two bit masks over the counts 0..n (bit k set when count k is in the set)."""
    birth = sum(1 << k for k in rule.birth if k <= n)
    survive = sum(1 << k for k in rule.survive if k <= n)
    return birth, survive


def rule_from_masks(birth: int, survive: int, n: int) -> Rule:
    return Rule(
        birth={k for k in range(n + 1) if birth >> k & 1},
        survive={k for k in range(n + 1) if survive >> k & 1},
    )


def enumerate_rules(n: int, up_to_duality: bool = False) -> Iterator[Rule]:
    """Every rule on a lattice of degree `n`, in a fixed order (by birth mask, then survival
    mask). With `up_to_duality`, one rule per complement pair — the one with the smaller masks —
    and a self-dual rule once, so the complement of every rule left out is in the output."""
    for birth in range(1 << (n + 1)):
        for survive in range(1 << (n + 1)):
            rule = rule_from_masks(birth, survive, n)
            if up_to_duality and rule_masks(rule.complement(n), n) < (birth, survive):
                continue
            yield rule


def langton_lambda(rule: Rule, n: int) -> float:
    """Langton's lambda for an outer-totalistic rule: of the 2^(n+1) (own state, neighbourhood)
    configurations, the fraction mapping to a live cell, each count k weighted by the C(n, k)
    neighbourhoods that have it."""
    live = sum(math.comb(n, k) for k in rule.birth if k <= n)
    live += sum(math.comb(n, k) for k in rule.survive if k <= n)
    return live / 2 ** (n + 1)


def mean_field_map(rule: Rule, n: int) -> Polynomial:
    """The mean-field density map f(rho) = (1-rho) P[Bin(n, rho) in B] + rho P[Bin(n, rho) in S]
    as a polynomial in rho: what the density would do if every cell's neighbours were
    independent draws at the current density."""
    rho = Polynomial([0.0, 1.0])
    empty = Polynomial([1.0, -1.0])
    birth = sum(
        (math.comb(n, k) * rho**k * empty ** (n - k) for k in rule.birth if k <= n),
        Polynomial([0.0]),
    )
    survive = sum(
        (math.comb(n, k) * rho**k * empty ** (n - k) for k in rule.survive if k <= n),
        Polynomial([0.0]),
    )
    return empty * birth + rho * survive


def mean_field_fixed_points(rule: Rule, n: int) -> tuple[list[float], list[float]]:
    """(all fixed points, the stable ones) of `mean_field_map` in [0, 1], rounded to 6 places.
    Stable means |f'(rho*)| < 1. The identity rule (nothing is born, everything survives) fixes
    every density; it gets ([], [])."""
    f = mean_field_map(rule, n)
    g = f - Polynomial([0.0, 1.0])
    coefficients = g.trim(1e-12).coef
    if coefficients.size <= 1 and abs(coefficients).max(initial=0.0) < 1e-12:
        return [], []
    fixed: list[float] = []
    for root in g.roots():
        if abs(root.imag) > 1e-7:
            continue
        value = float(root.real)
        if -1e-9 <= value <= 1 + 1e-9:
            value = min(max(value, 0.0), 1.0)
            rounded = round(value, 6)
            if rounded not in fixed:
                fixed.append(rounded)
    fixed.sort()
    slope = f.deriv()
    stable = [point for point in fixed if abs(float(slope(point))) < 1 - 1e-9]
    return fixed, stable


@dataclass(frozen=True)
class RuleRecord:
    rule: str
    shape: str
    triangle_mode: str | None
    grid: int
    steps: int
    seed: int
    density: float
    outcome: str
    period: int | None
    transient: int | None
    frozen_at: int | None
    rho_final: float  # mean density over the last `tail` generations
    rho_min: float  # over the same window
    rho_max: float
    activity_final: float  # mean |delta rho| over the same window
    cluster_count: int  # connected components of live cells at the end
    max_cluster_size: int
    langton_lambda: float
    mean_field: list[float]  # stable fixed points of the mean-field map
    dual: str

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CensusTask:
    config: Config
    rule: str
    shape: str
    grid: int
    steps: int
    seed: int
    density: float
    tail: int


def classify_session(session: SimulationSession, steps: int, tail: int = 50) -> dict:
    """Run `session` (one species) for `steps` generations with cycle detection and measure its
    terminal behaviour: the outcome, period and transient, the density and activity over the
    last `tail` generations, and the connected components of live cells at the end (on the
    cycle, when one was found). A window longer than the run is the whole run."""
    if tail < 1:
        raise ValueError(f"tail must be at least 1 (got {tail})")
    series = run_timeseries(session, lifeform_id=1, steps=steps, detect_cycles=True)
    window = series.rho[-tail:]
    rho_final = float(window.mean())
    if rho_final == 0.0:
        outcome = "extinct"
    elif series.period is None:
        outcome = "unsettled"
    elif series.period == 1:
        outcome = "fixed"
    else:
        outcome = "periodic"
    clusters = compute_cluster_metrics(session.grid, 1)
    return {
        "outcome": outcome,
        "period": series.period,
        "transient": series.transient,
        "frozen_at": series.frozen_at,
        "rho_final": rho_final,
        "rho_min": float(window.min()),
        "rho_max": float(window.max()),
        "activity_final": float(series.activity[-tail:].mean()),
        "cluster_count": clusters.cluster_count,
        "max_cluster_size": clusters.max_cluster_size,
    }


def classify_rule(task: CensusTask) -> RuleRecord:
    """One census record: `task.rule` on a `task.grid` torus of `task.shape`, seeded at
    `task.density` from `task.seed`."""
    rule = Rule.from_bs_string(task.rule)
    triangle_mode = task.config.grid.triangle_mode
    n = max_degree(task.shape, triangle_mode)
    session = single_species_session(
        task.config, rule, task.shape, task.grid, task.density, task.seed
    )
    measured = classify_session(session, task.steps, task.tail)
    _, stable = mean_field_fixed_points(rule, n)
    return RuleRecord(
        rule=rule.to_bs_string(n),
        shape=task.shape,
        triangle_mode=triangle_mode if task.shape == "triangle" else None,
        grid=task.grid,
        steps=task.steps,
        seed=task.seed,
        density=task.density,
        langton_lambda=langton_lambda(rule, n),
        mean_field=stable,
        dual=rule.complement(n).to_bs_string(n),
        **measured,
    )


def census_tasks(
    config: Config,
    shape: str,
    rules: Iterable[Rule],
    grids: Sequence[int],
    seeds: Sequence[int],
    steps: int,
    density: float,
    tail: int,
) -> list[CensusTask]:
    n = max_degree(shape, config.grid.triangle_mode)
    return [
        CensusTask(config, rule.to_bs_string(n), shape, grid, steps, seed, density, tail)
        for rule in rules
        for grid in grids
        for seed in seeds
    ]


def run_census(
    config: Config,
    shape: str,
    rules: Iterable[Rule],
    grids: Sequence[int],
    seeds: Sequence[int],
    steps: int,
    density: float,
    out_path: str,
    workers: int = 1,
    tail: int = 50,
) -> Iterator[RuleRecord]:
    """Classify every (rule, grid, seed) not already recorded in `out_path`, appending each
    record to it as it finishes and yielding it. Stopping at any point keeps what finished; a
    later run continues from there."""
    tasks = census_tasks(config, shape, rules, grids, seeds, steps, density, tail)
    done = resume_keys(out_path, RECORD_KEY)
    triangle_mode = config.grid.triangle_mode if shape == "triangle" else None
    pending = [
        task
        for task in tasks
        if (task.rule, shape, triangle_mode, task.grid, steps, task.seed, density) not in done
    ]
    for record in map_tasks(classify_rule, pending, workers):
        append_jsonl(out_path, record.to_json())
        yield record


def summarise(records: Iterable[RuleRecord]) -> dict:
    """Counts per outcome and the distribution of periods, for a run's closing summary."""
    outcomes = dict.fromkeys(OUTCOMES, 0)
    periods: dict[int, int] = {}
    total = 0
    for record in records:
        total += 1
        outcomes[record.outcome] += 1
        if record.period is not None and record.period > 1:
            periods[record.period] = periods.get(record.period, 0) + 1
    return {
        "records": total,
        "outcomes": outcomes,
        "periods": {str(p): periods[p] for p in sorted(periods)},
    }


__all__ = [
    "OUTCOMES",
    "RECORD_KEY",
    "CensusTask",
    "RuleRecord",
    "census_tasks",
    "classify_rule",
    "classify_session",
    "enumerate_rules",
    "langton_lambda",
    "mean_field_fixed_points",
    "mean_field_map",
    "rule_from_masks",
    "rule_masks",
    "rule_space_size",
    "run_census",
    "summarise",
]
