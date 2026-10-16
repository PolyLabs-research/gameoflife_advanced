"""Cost hints: how much simulating a task costs, before running it. The work of a run is about
its generations times its cells (`LatticeGrid.step` is vectorised over cells, so its time grows
with the cell count), and a run stops early when its fate is settled, so a task's cap gives an
upper bound. `measure_seconds_per_step` times a few generations of a real session to turn that
into seconds on this machine; measured once per (shape, torus) it prices every task there.

For reference, the settling-time prototype (2026-10-16) measured 283, 391 and 731 µs per step on
16, 32 and 64 hexagon tori: the cost per step is not proportional to the cells on small tori,
where fixed overhead dominates, which is why the hint takes a measurement rather than a formula.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from latticelife.config import Config
from latticelife.core.rules import Rule
from latticelife.experiments.harness import grid_dimensions, single_species_session


@dataclass(frozen=True)
class CostHint:
    cells: int
    generations: int  # grid updates at most (the cap; runs often stop sooner)
    cell_steps: int  # generations x cells
    seconds_per_step: float | None  # measured on this machine, when given
    seconds: float | None  # generations x seconds_per_step: an upper bound

    def to_json(self) -> dict:
        return asdict(self)


def cost_hint(
    grid: int | str | tuple[int, int], generations: int, seconds_per_step: float | None = None
) -> CostHint:
    width, height = grid_dimensions(grid)
    cells = width * height
    return CostHint(
        cells=cells,
        generations=generations,
        cell_steps=cells * generations,
        seconds_per_step=seconds_per_step,
        seconds=generations * seconds_per_step if seconds_per_step is not None else None,
    )


def task_cost(task, seconds_per_step: float | None = None) -> CostHint:
    """The cost hint for a census, sweep, soup or damage task (anything with `grid` and
    `steps`). A damage task steps two grids per generation after its warm-up; a soup task's
    object identification afterwards is not counted."""
    generations = int(task.steps)
    if hasattr(task, "flips"):
        generations = int(getattr(task, "warmup", 0)) + 2 * int(task.steps)
    return cost_hint(task.grid, generations, seconds_per_step)


def measure_seconds_per_step(
    config: Config,
    rule: Rule | str,
    shape: str,
    grid: int | str | tuple[int, int],
    density: float = 0.5,
    steps: int = 50,
    seed: int = 1,
) -> float:
    """Wall-clock seconds per generation of `rule` on this torus, the median over `steps`
    generations of a fresh soup (the median shrugs off a slow first step)."""
    if steps < 1:
        raise ValueError(f"steps must be at least 1 (got {steps})")
    if isinstance(rule, str):
        rule = Rule.from_bs_string(rule)
    session = single_species_session(config, rule, shape, grid, density, seed)
    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        session.step()
        timings.append(time.perf_counter() - start)
    timings.sort()
    return timings[len(timings) // 2]


__all__ = ["CostHint", "cost_hint", "measure_seconds_per_step", "task_cost"]
