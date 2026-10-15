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

Beside the measured behaviour each record carries things computable from the rule alone, for
the analysis to set against it: Langton's lambda (the fraction of the 2^(n+1) neighbourhood
configurations, weighted by multiplicity, that map to a live cell), the stable fixed points of the
mean-field map rho -> (1-rho) P[Bin(n, rho) in B] + rho P[Bin(n, rho) in S] (`mean_field`), and
the attractor that map actually reaches from the soup's initial density (`mean_field_attractor`,
`mean_field_period`, `mean_field_lyapunov`: `mean_field_orbit`). The attractor is the prediction
to compare a run with. A rule's largest stable fixed point is not: B6/S56 has stable points 0 and
1 but its map sends 0.5 to 0, and B0/S has no stable fixed point at all while its map strobes
between 0 and 1 as the soup does. Mean field knows only the degree, so its
disagreement with the run measures how much the spatial arrangement of cells matters for that
rule. The complement rule (`Rule.complement`) is recorded as `dual`: it behaves as this rule does
on the complemented grid, so a census of one rule per dual pair
(`enumerate_rules(up_to_duality=True)`) covers the whole space.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction

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


def _poly_divmod(numerator: list[Fraction], divisor: list[Fraction]):
    """Exact long division of polynomials (coefficients lowest degree first)."""
    remainder = list(numerator)
    quotient = [Fraction(0)] * max(len(numerator) - len(divisor) + 1, 1)
    while len(remainder) >= len(divisor) and any(remainder):
        shift = len(remainder) - len(divisor)
        factor = remainder[-1] / divisor[-1]
        quotient[shift] = factor
        for i, c in enumerate(divisor):
            remainder[shift + i] -= factor * c
        while remainder and remainder[-1] == 0:
            remainder.pop()
    return quotient, remainder


def _poly_gcd(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    while b and any(b):
        _, r = _poly_divmod(a, b)
        a, b = b, r
    return [c / a[-1] for c in a]


def _square_free(coefficients: list[int]) -> list[float]:
    """The square-free part g / gcd(g, g') of an integer polynomial, as floats: the same roots,
    each simple, so numerical root finding is accurate even where g has a multiple root."""
    g = [Fraction(c) for c in coefficients]
    derivative = [i * c for i, c in enumerate(g)][1:]
    if not any(derivative):
        return [float(c) for c in g]
    common = _poly_gcd(g, derivative)
    part, _ = _poly_divmod(g, common)
    while part and part[-1] == 0:
        part.pop()
    return [float(c) for c in part]


def mean_field_fixed_points(rule: Rule, n: int) -> tuple[list[float], list[float]]:
    """(all fixed points, the stable ones) of `mean_field_map` in [0, 1], rounded to 6 places.
    Stable means |f'(rho*)| < 1; a multiple root of f(rho) - rho has f' = 1 and is not stable.
    The identity rule (nothing is born, everything survives) fixes every density; it gets
    ([], []).

    f(rho) - rho has integer coefficients, and multiple roots are common (B4/S3456 has a triple
    root at 1, B346/S1456 one at 1/2), which numerical root finding splits into spurious points
    such as 0.999993. So roots at 0 and 1 are divided out exactly (0 is one when 0 is not in B,
    1 when n is in S), and the interior roots are found from the exact square-free part, where
    every root is simple."""
    f = mean_field_map(rule, n)
    g = f - Polynomial([0.0, 1.0])
    coefficients = [round(c) for c in g.coef]  # integers: exact in floating point too
    while coefficients and coefficients[-1] == 0:
        coefficients.pop()
    if not coefficients:
        return [], []
    fixed: list[float] = []
    while len(coefficients) > 1 and coefficients[0] == 0:  # a root at 0
        coefficients = coefficients[1:]
        if 0.0 not in fixed:
            fixed.append(0.0)
    while len(coefficients) > 1 and sum(coefficients) == 0:  # a root at 1: divide by (x - 1)
        quotient = [0] * (len(coefficients) - 1)
        carry = 0
        for i in range(len(coefficients) - 1, 0, -1):
            carry += coefficients[i]
            quotient[i - 1] = carry
        coefficients = quotient
        if 1.0 not in fixed:
            fixed.append(1.0)
    interior = _square_free(coefficients) if len(coefficients) > 1 else []
    for root in Polynomial(interior).roots() if len(interior) > 1 else []:
        if abs(root.imag) > 1e-7:
            continue
        value = float(root.real)
        if 1e-9 < value < 1 - 1e-9:
            rounded = round(value, 6)
            if rounded not in fixed:
                fixed.append(rounded)
    fixed.sort()
    slope = f.deriv()
    stable = [point for point in fixed if abs(float(slope(point))) < 1 - 1e-9]
    return fixed, stable


def _horner(coefficients: list[float], x: float) -> float:
    value = 0.0
    for c in coefficients:
        value = value * x + c
    return value


def mean_field_orbit(
    rule: Rule, n: int, rho0: float = 0.5, transient: int = 20_000, max_period: int = 64
) -> dict:
    """Where the mean-field map goes from `rho0`: the attractor's `period` and its `values`
    (sorted, 6 decimals), or period None and no values when no cycle of period up to
    `max_period` attracts the orbit (chaos, or a longer cycle); and `lyapunov`, the orbit's
    Lyapunov exponent. For a cycle that is log|m| / period, where m is the cycle's multiplier
    (the product of f' around it): negative when the cycle attracts, 0 when it is neutral, and
    positive only when the orbit sits exactly on an unstable cycle, as every rule with
    lambda = 1/2 does at 1/2 (f(1/2) = lambda: a soup at density 1/2 is a fixed point of its own
    map, stable or not). Without a cycle it is the average of log|f'| along 5,000 steps.

    After `transient` steps, the first period p at which the orbit nearly returns is refined by
    Newton's method on f^p(y) = y, so a cycle the orbit approaches slowly (multipliers up to
    0.9996 occur) is still found exactly, at its minimal period, and an orbit creeping onto a
    neutral cycle (|m| = 1: 1 for B2/S23456, where the distance to 1 shrinks like t^(-1/4), or
    the 0-1 strobe for some B0 rules) is reported as reaching it."""
    f = mean_field_map(rule, n)
    coefficients = [float(c) for c in f.coef][::-1]
    slope = [float(c) for c in f.deriv().coef][::-1]

    def step(x: float) -> float:
        return min(1.0, max(0.0, _horner(coefficients, x)))

    x = float(rho0)
    for _ in range(transient):
        x = step(x)
    orbit = [x]
    for _ in range(2 * max_period):
        orbit.append(step(orbit[-1]))
    exact, _ = mean_field_fixed_points(rule, n)
    for period in range(1, max_period + 1):
        if abs(orbit[period] - x) > 1e-4:
            continue
        cycle = _refine_cycle(x, period, step, slope, orbit, exact)
        if cycle is not None:
            values, multiplier = cycle
            lyapunov = math.log(max(abs(multiplier), 1e-300)) / len(values)
            return {
                "period": len(values),
                "values": sorted({round(v, 6) for v in values}),
                "lyapunov": round(max(lyapunov, -50.0), 4),
            }
    total = 0.0
    z = x
    for _ in range(5000):
        total += math.log(max(abs(_horner(slope, z)), 1e-12))
        z = step(z)
    return {"period": None, "values": [], "lyapunov": round(total / 5000, 4)}


def _refine_cycle(x, period, step, slope, orbit, exact) -> tuple[list[float], float] | None:
    """The cycle of `period` near `x` that the orbit (`orbit`, starting at `x`) is reaching, as
    (its points from the one nearest `x`, its multiplier), at its minimal period; or None.

    Newton converges quadratically onto a simple root, and the point it returns is then exact
    to rounding. At a multiple root (a neutral cycle, multiplier 1) it is only linear, and its
    residual vanishes long before the point is reached: |f(y) - y| < 1e-14 at 3e-3 to 9e-3 from
    the sixth-order root at 0 of B/S01234. So when the multiplier comes out within 1e-2 of 1,
    the orbit is creeping, and it is snapped onto the exact fixed point it is creeping toward
    (`exact`: the next one in the direction f(x) - x points), or onto the 0-1 strobe; a simple
    root reached that way is then polished by Newton again, so the multiplier is exact too."""

    def newton(y):
        for _ in range(400):  # on f^p(y) - y
            value, derivative = y, 1.0
            for _ in range(period):
                derivative *= _horner(slope, value)
                value = step(value)
            residual = value - y
            if abs(residual) < 1e-14 or abs(derivative - 1.0) < 1e-15:
                break
            y_next = min(1.0, max(0.0, y - residual / (derivative - 1.0)))
            if y_next == y:
                break
            y = y_next
        return y

    def cycle_from(y):
        points = [y]
        for _ in range(period - 1):
            points.append(step(points[-1]))
        return points

    def multiplier_of(points):
        product = 1.0
        for point in points:
            product *= _horner(slope, point)
        return product

    y = newton(x)
    points = cycle_from(y)
    if abs(step(points[-1]) - y) > 1e-10:
        return None
    for divisor in range(1, period):
        if period % divisor == 0 and abs(points[divisor] - y) < 1e-10:
            points = points[:divisor]
            break
    multiplier = multiplier_of(points)
    if abs(multiplier - 1.0) < 1e-2:
        drift = step(x) - x
        if len(points) == 1 and exact:
            if drift > 0:
                ahead = [p for p in exact if p >= x]
            elif drift < 0:
                ahead = [p for p in exact if p <= x]
            else:
                ahead = []
            target = min(ahead or exact, key=lambda p: abs(p - x))
            if abs(_horner(slope, target) - 1.0) > 1e-6:
                target = min(1.0, max(0.0, newton(target)))  # a simple root: polish it
            points = [target]
        elif len(points) == 2 and step(0.0) == 1.0 and step(1.0) == 0.0:
            low, high = sorted(points)
            if low < 0.5 < high:
                points = [0.0, 1.0] if points[0] < 0.5 else [1.0, 0.0]
        multiplier = multiplier_of(points)
    y = points[0]
    if abs(x - y) < 1e-12 or abs(multiplier) < 1 - 1e-9:
        return points, multiplier  # sitting on it exactly, or it attracts
    if abs(abs(multiplier) - 1.0) < 1e-6:
        # Neutral: accept only if the orbit really is closing in on it.
        gaps = [abs(orbit[k * len(points)] - y) for k in range(len(orbit) // len(points))]
        if all(b <= a for a, b in zip(gaps, gaps[1:], strict=False)):
            return points, multiplier
    return None


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
    changed_final: float  # mean fraction of cells that changed per step, over the same window
    cluster_count: int  # connected components of live cells at the end
    max_cluster_size: int
    langton_lambda: float
    mean_field: list[float]  # stable fixed points of the mean-field map
    mean_field_attractor: list[float]  # where the map goes from `density` (see mean_field_orbit)
    mean_field_period: int | None
    mean_field_lyapunov: float
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
    cycle, when one was found). A window longer than the run is the whole run.

    `changed_final`, the fraction of cells changing per step, tells a chaotic rule whose density
    hardly moves from a quiet one; |delta rho| cannot. It is read from the session's own
    history, the generations actually stepped (so, once a cycle is found, the ones leading on
    to it), over at most `tail` of them and never more than the session keeps
    (`simulation.history_limit`)."""
    if tail < 1:
        raise ValueError(f"tail must be at least 1 (got {tail})")
    series = run_timeseries(session, lifeform_id=1, steps=steps, detect_cycles=True)
    window = series.rho[-tail:]
    stepped = list(session.history_records)[-tail:]
    changed_final = (
        float(sum(r.result.changed_cells for r in stepped) / (len(stepped) * session.grid.n_cells))
        if stepped
        else 0.0
    )
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
        "changed_final": changed_final,
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
    orbit = mean_field_orbit(rule, n, task.density)
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
        mean_field_attractor=orbit["values"],
        mean_field_period=orbit["period"],
        mean_field_lyapunov=orbit["lyapunov"],
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
    "mean_field_orbit",
    "rule_from_masks",
    "rule_masks",
    "rule_space_size",
    "run_census",
    "summarise",
]
