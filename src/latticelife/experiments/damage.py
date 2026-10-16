"""Damage spreading as a headless experiment: how far one flipped cell's effect spreads. For one
rule on one lattice, at each (torus size, initial density, seed), a soup is optionally run for
`warmup` generations, then copied, `flips` random cells of the copy are flipped, and both twins
are stepped side by side (`experiments.runner.run_damage_spreading`, the method discovery
scores rules with). The record holds the number of cells in which the twins differ at every
generation: a chaotic rule spreads the damage over a finite fraction of the torus, a frozen one
heals or keeps it local.

Single species only, so the dynamics are deterministic: the twins step with copies of one random
stream (common random numbers) that has nothing to decide, and once they agree they agree for
ever, so `healed_at` is exact. Everything, the flipped cells included, is reproducible from the
seed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from latticelife.config import Config
from latticelife.core.geometries import max_degree
from latticelife.core.rules import Rule
from latticelife.experiments.harness import (
    GridLabel,
    append_jsonl,
    map_tasks,
    parse_grid,
    resume_keys,
    single_species_session,
)
from latticelife.experiments.runner import run_damage_spreading

RECORD_KEY: tuple[str, ...] = (
    "rule",
    "shape",
    "triangle_mode",
    "grid",
    "density",
    "seed",
    "steps",
    "flips",
    "warmup",
)


@dataclass(frozen=True)
class DamageRecord:
    rule: str
    shape: str
    triangle_mode: str | None
    grid: GridLabel
    density: float
    seed: int
    steps: int  # generations recorded after the flip, the flip's own generation included
    flips: int  # cells flipped in the copy
    warmup: int  # generations run before the flip
    cells: int
    differing: list[int]  # cells in which the twins differ at generations 0..steps-1
    final: float  # differing[-1] / cells
    peak: float  # max(differing) / cells
    healed_at: int | None  # first generation after the flip at which the twins agree again

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DamageTask:
    config: Config
    rule: str
    shape: str
    grid: GridLabel
    density: float
    seed: int
    steps: int
    flips: int = 1
    warmup: int = 0


def damage_point(task: DamageTask) -> DamageRecord:
    """One damage-spreading run (see the module docstring)."""
    if task.steps < 1:
        raise ValueError(f"steps must be at least 1 (got {task.steps})")
    if task.flips < 1 or task.warmup < 0:
        raise ValueError(f"need flips >= 1 and warmup >= 0 (got {task.flips}, {task.warmup})")
    rule = Rule.from_bs_string(task.rule)
    triangle_mode = task.config.grid.triangle_mode
    n = max_degree(task.shape, triangle_mode)
    session = single_species_session(
        task.config, rule, task.shape, task.grid, task.density, task.seed
    )
    for _ in range(task.warmup):
        session.step()
    cells = session.grid.n_cells
    h = run_damage_spreading(session, task.steps, session.spawn_rng(), flip_cells=task.flips)
    differing = [int(v) for v in np.rint(np.asarray(h) * cells)]
    healed_at = next((t for t in range(1, len(differing)) if differing[t] == 0), None)
    return DamageRecord(
        rule=rule.to_bs_string(n),
        shape=task.shape,
        triangle_mode=triangle_mode if task.shape == "triangle" else None,
        grid=parse_grid(task.grid),
        density=task.density,
        seed=task.seed,
        steps=task.steps,
        flips=task.flips,
        warmup=task.warmup,
        cells=cells,
        differing=differing,
        final=differing[-1] / cells,
        peak=max(differing) / cells,
        healed_at=healed_at,
    )


def run_damage(
    config: Config,
    rule: Rule,
    shape: str,
    grids: Sequence[int | str],
    densities: Sequence[float],
    seeds: Sequence[int],
    steps: int,
    out_path: str,
    workers: int = 1,
    flips: int = 1,
    warmup: int = 0,
) -> Iterator[DamageRecord]:
    """Every (grid, density, seed) not already recorded in `out_path`, appended to it as it
    finishes and yielded."""
    n = max_degree(shape, config.grid.triangle_mode)
    rule_text = rule.to_bs_string(n)
    triangle_mode = config.grid.triangle_mode if shape == "triangle" else None
    done = resume_keys(out_path, RECORD_KEY)
    tasks = [
        DamageTask(config, rule_text, shape, grid, float(density), seed, steps, flips, warmup)
        for grid in (parse_grid(g) for g in grids)
        for density in densities
        for seed in seeds
        if (rule_text, shape, triangle_mode, grid, float(density), seed, steps, flips, warmup)
        not in done
    ]
    for record in map_tasks(damage_point, tasks, workers):
        append_jsonl(out_path, record.to_json())
        yield record


def summarise(records: Iterable[DamageRecord | dict]) -> dict:
    """Per torus size: runs, the mean final damage fraction, and the share that healed."""
    by_grid: dict[str, list[dict]] = {}
    for record in records:
        data = record if isinstance(record, dict) else record.to_json()
        by_grid.setdefault(str(data["grid"]), []).append(data)
    return {
        grid: {
            "runs": len(rows),
            "mean_final": float(np.mean([r["final"] for r in rows])),
            "healed": sum(r["healed_at"] is not None for r in rows) / len(rows),
        }
        for grid, rows in by_grid.items()
    }


__all__ = [
    "RECORD_KEY",
    "DamageRecord",
    "DamageTask",
    "damage_point",
    "run_damage",
    "summarise",
]
