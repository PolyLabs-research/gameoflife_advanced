"""Local settling times: when each block of a torus enters its final cycle.

A soup's transient is the first generation T from which the whole grid repeats with its period
P. Every region of the grid repeats with period P from then on, but most regions get there
sooner. For a block of b x b cells (in the grid's row and column indices: contiguous patches on
every lattice), its local transient is the first generation t0 with x_B(t) = x_B(t + P) for all
t >= t0. The latest block's is T, so the largest local transient is the global one, by
construction.

`local_point` runs a soup twice. The first pass finds T and P with exact cycle detection. The
second keeps the last P states, compares each new state x(s) with x(s - P), and records, per
block, the last s - P at which they differed; the local transient is one more than that (0 for a
block that never differed). A soup whose state does not repeat within `steps`, or whose period
exceeds `max_period` (the second pass holds P grids), has no local transients.

The grid's height and width must be multiples of every block size. Local transients are what an
extreme-value account of settling times is about: the global transient is their maximum, and
whether they are independent and the same on every torus can be checked from them.
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
    grid_dimensions,
    map_tasks,
    parse_grid,
    resume_keys,
    single_species_session,
)
from latticelife.experiments.periodicity import CycleDetector

RECORD_KEY: tuple[str, ...] = (
    "rule",
    "shape",
    "triangle_mode",
    "grid",
    "density",
    "seed",
    "steps",
    "blocks",
)


@dataclass(frozen=True)
class LocalTask:
    config: Config
    rule: str
    shape: str
    grid: GridLabel
    density: float
    seed: int
    steps: int
    blocks: tuple[int, ...] = (8,)
    max_period: int = 4096


@dataclass(frozen=True)
class LocalRecord:
    rule: str
    shape: str
    triangle_mode: str | None
    grid: GridLabel
    density: float
    seed: int
    steps: int
    blocks: list[int]
    transient: int | None  # the global transient (None: no repeat within steps)
    period: int | None
    stopped_at: int  # the last generation of the first pass
    # Per block size (as a string), every block's local transient, row-major over the blocks;
    # None when the soup did not repeat within `steps` or its period exceeded max_period.
    local: dict[str, list[int]] | None
    note: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _check_blocks(width: int, height: int, blocks: Sequence[int]) -> None:
    for b in blocks:
        if b < 1 or width % b or height % b:
            raise ValueError(f"block size {b} does not divide a {width}x{height} grid")


def local_point(task: LocalTask) -> LocalRecord:
    rule = Rule.from_bs_string(task.rule)
    triangle_mode = task.config.grid.triangle_mode
    width, height = grid_dimensions(task.grid)
    _check_blocks(width, height, task.blocks)
    common = {
        "rule": rule.to_bs_string(max_degree(task.shape, triangle_mode)),
        "shape": task.shape,
        "triangle_mode": triangle_mode if task.shape == "triangle" else None,
        "grid": parse_grid(task.grid),
        "density": task.density,
        "seed": task.seed,
        "steps": task.steps,
        "blocks": list(task.blocks),
    }

    def session():
        return single_species_session(
            task.config, rule, task.shape, task.grid, task.density, task.seed
        )

    first = session()
    detector = CycleDetector()
    detector.observe(first.grid.occupancy, 0)
    found = None
    stopped_at = 0
    for step in range(1, task.steps):
        first.step()
        stopped_at = step
        found = detector.observe(first.grid.occupancy, step)
        if found is not None:
            break
    if found is None:
        return LocalRecord(
            **common,
            transient=None,
            period=None,
            stopped_at=stopped_at,
            local=None,
            note=f"no repeat within {task.steps} generations",
        )
    period, transient = found
    if period > task.max_period:
        return LocalRecord(
            **common,
            transient=transient,
            period=period,
            stopped_at=stopped_at,
            local=None,
            note=f"period {period} exceeds max_period",
        )

    second = session()
    ring = np.zeros((period, height, width), dtype=bool)
    last = {b: np.full((height // b, width // b), -1, dtype=np.int64) for b in task.blocks}
    for s in range(transient + period + 1):
        state = second.grid.occupancy.reshape(height, width).astype(bool)
        if s >= period:
            differs = state != ring[s % period]
            for b, mark in last.items():
                hit = differs.reshape(height // b, b, width // b, b).any(axis=(1, 3))
                mark[hit] = s - period
        ring[s % period] = state
        if s < transient + period:
            second.step()
    local = {str(b): (mark + 1).ravel().tolist() for b, mark in last.items()}
    if any(max(times) != transient for times in local.values()):
        raise RuntimeError("the largest local transient differs from the global one")
    return LocalRecord(
        **common, transient=transient, period=period, stopped_at=stopped_at, local=local
    )


def run_local(
    config: Config,
    rule: Rule,
    shape: str,
    grids: Sequence[int | str],
    densities: Sequence[float],
    seeds: Sequence[int],
    steps: int,
    out_path: str,
    blocks: Sequence[int] = (8,),
    workers: int = 1,
) -> Iterator[LocalRecord]:
    """Run every (grid, density, seed) task not already in `out_path`, appending one JSONL record
    per run as it finishes (resumable, as the other experiments)."""
    n = max_degree(shape, config.grid.triangle_mode)
    rule_text = rule.to_bs_string(n)
    triangle_mode = config.grid.triangle_mode if shape == "triangle" else None
    done = resume_keys(out_path, RECORD_KEY)
    tasks: list[LocalTask] = []
    for grid in grids:
        for density in densities:
            for seed in seeds:
                key = (
                    rule_text,
                    shape,
                    triangle_mode,
                    parse_grid(grid),
                    float(density),
                    seed,
                    steps,
                    list(blocks),
                )
                if _key(key) in done:
                    continue
                tasks.append(
                    LocalTask(
                        config, rule_text, shape, grid, float(density), seed, steps, tuple(blocks)
                    )
                )
    for record in map_tasks(local_point, tasks, workers):
        append_jsonl(out_path, record.to_json())
        yield record


def _key(values: Iterable) -> tuple:
    return tuple(tuple(v) if isinstance(v, list) else v for v in values)


__all__ = ["RECORD_KEY", "LocalRecord", "LocalTask", "local_point", "run_local"]
