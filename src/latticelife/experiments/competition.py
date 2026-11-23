"""Several species on one grid, without the app: a start with chosen shares, a run under a chosen
tie-break, and what each species did.

Each species has its own B/S rule, and is born and survives on the count of its own kind among
a cell's neighbours. The species meet in two ways only: a cell holds one species at a time, and
a dead cell that more than one species could be born into is settled by the birth tie-break
(`core.lattice.TieBreak`). So a species alone runs exactly as it would with the grid to itself.

`species_start` seeds a wrapped grid: every cell is live with probability `density`,
independently, and a live cell is species i with probability `shares[i]` (equal shares when none
are given). `competition_point` runs one start and returns its record; `run_competition` writes
one resumable JSONL record per task, as the other experiments do.

A record holds, at generation 0, at every `sample_every`-th generation and at the end:

    alive        each species' live cells
    unlike       adjacent pairs of live cells of two different species, each pair once
    like         adjacent pairs of live cells of one species, each pair once
    births       each species' births since the sample before
    contested    the cells the tie-break settled since the sample before

and `extinct_at`, the first generation at which each species had no cell (None for a species
alive at the end), with the sha256 of the start and of the last grid. `unlike` against `like`
says how mixed the species are: sorted into domains, unlike pairs lie only along the borders.

A run is reproducible from its task. The start is drawn from one stream named by the shape,
the grid and the seed, and the tie-break's draws (`random`, `weighted`) from another named by
the seed; `standoff` and `strongest` draw nothing, so their run is a function of the start.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from latticelife.core.geometries import SHAPES, max_degree
from latticelife.core.lattice import LatticeGrid, TieBreak
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.experiments.coarse import digest
from latticelife.experiments.harness import (
    GridLabel,
    append_jsonl,
    grid_dimensions,
    map_tasks,
    parse_grid,
    resume_keys,
)

MAX_SPECIES = 10  # as `config.MAX_LIFEFORMS`: species ids are 1..10

RECORD_KEY: tuple[str, ...] = (
    "rules",
    "shape",
    "triangle_mode",
    "grid",
    "density",
    "shares",
    "tie_break",
    "tie_break_radius",
    "tie_break_power",
    "seed",
    "steps",
    "sample_every",
)


def _shares(shares: Sequence[float] | None, species: int) -> list[float]:
    """The shares as fractions that add up to 1: equal when None."""
    if not 1 <= species <= MAX_SPECIES:
        raise ValueError(f"a competition has 1 to {MAX_SPECIES} species (got {species})")
    if shares is None:
        return [1.0 / species] * species
    values = [float(s) for s in shares]
    if len(values) != species:
        raise ValueError(f"{species} species need {species} shares (got {len(values)})")
    if any(not math.isfinite(s) or s < 0 for s in values) or sum(values) <= 0:
        raise ValueError(f"shares must be numbers >= 0, not all 0 (got {list(shares)})")
    total = sum(values)
    return [s / total for s in values]


def species_start(
    shape: str,
    grid: int | str | tuple[int, int],
    density: float,
    seed: int,
    shares: Sequence[float] | None = None,
    species: int | None = None,
) -> np.ndarray:
    """A start for several species: every cell live with probability `density`, independently,
    and a live cell species i + 1 with probability `shares[i]` (`species` equal shares when
    `shares` is None). An int8 array by cell index, 0 for an empty cell. Reproducible from the
    shape, the grid, the seed and the arguments; the number of live cells is not fixed, as it is
    in a session's own seeding."""
    if shape not in SHAPES:
        raise ValueError(f"Unsupported grid shape: {shape}")
    if not 0 <= density <= 1:
        raise ValueError(f"density must be in [0, 1] (got {density!r})")
    count = len(shares) if shares is not None else species
    if count is None:
        raise ValueError("give the shares, or the number of species")
    fractions = _shares(shares, count)
    width, height = grid_dimensions(grid)
    rng = np.random.default_rng([20261123, 1, SHAPES.index(shape), width, height, int(seed)])
    live = rng.random(width * height) < density
    kind = rng.choice(count, size=width * height, p=fractions) + 1
    return np.where(live, kind, 0).astype(np.int8)


def pair_counts(occupancy: np.ndarray, adjacency: np.ndarray) -> tuple[int, int]:
    """(unlike, like): the adjacent pairs of live cells holding two different species, and
    holding the same one, each pair counted once. For an adjacency in which every link is
    mutual and listed once, as a lattice's own is."""
    held = np.append(occupancy, 0)[adjacency]  # a pad, -1, reads the empty slot at the end
    mine = occupancy[:, None]
    both = (mine > 0) & (held > 0)
    unlike = int(np.count_nonzero(both & (held != mine)))
    like = int(np.count_nonzero(both & (held == mine)))
    return unlike // 2, like // 2


@dataclass(frozen=True)
class CompetitionTask:
    rules: tuple[str, ...]  # one B/S string per species; species i + 1 runs rules[i]
    shape: str
    grid: GridLabel
    density: float
    seed: int
    steps: int
    shares: tuple[float, ...] | None = None  # None: equal shares
    tie_break: str = "random"
    tie_break_radius: int = 2
    tie_break_power: float = 1.0
    sample_every: int = 0  # 0: the start and the end only
    triangle_mode: str = "edge+vertex"


@dataclass(frozen=True)
class CompetitionRecord:
    rules: list[str]
    shape: str
    triangle_mode: str | None
    grid: GridLabel
    density: float
    shares: list[float]
    tie_break: str
    tie_break_radius: int
    tie_break_power: float
    seed: int
    steps: int
    sample_every: int
    cells: int
    generations: list[int]  # the sampled generations: 0, every sample_every-th, and `steps`
    alive: list[list[int]]  # per species, its live cells at each sampled generation
    unlike: list[int]  # adjacent live pairs of two species, at each sampled generation
    like: list[int]  # adjacent live pairs of one species
    births: list[list[int]]  # per species, its births since the sample before
    contested: list[int]  # cells the tie-break settled since the sample before
    extinct_at: list[int | None]  # per species, the first generation it had no cell
    start_sha256: str
    final_sha256: str

    def to_json(self) -> dict:
        return asdict(self)


def _identity(task: CompetitionTask) -> dict:
    """The fields that identify a task, as its record holds them."""
    if not task.rules:
        raise ValueError("a competition needs at least one rule")
    degree = max_degree(task.shape, task.triangle_mode)
    if task.steps < 1:
        raise ValueError(f"steps must be at least 1 (got {task.steps})")
    if task.sample_every < 0:
        raise ValueError(f"sample_every must be 0 or more (got {task.sample_every})")
    tie = TieBreak(task.tie_break, task.tie_break_radius, float(task.tie_break_power))
    return {
        "rules": [Rule.from_bs_string(rule).to_bs_string(degree) for rule in task.rules],
        "shape": task.shape,
        "triangle_mode": task.triangle_mode if task.shape == "triangle" else None,
        "grid": parse_grid(task.grid),
        "density": float(task.density),
        "shares": _shares(task.shares, len(task.rules)),
        "tie_break": tie.kind,
        "tie_break_radius": tie.radius,
        "tie_break_power": float(tie.power),
        "seed": int(task.seed),
        "steps": int(task.steps),
        "sample_every": int(task.sample_every),
    }


def competition_point(task: CompetitionTask) -> CompetitionRecord:
    """One run of several species from `species_start`, for `steps` generations."""
    identity = _identity(task)
    species = len(task.rules)
    width, height = grid_dimensions(task.grid)
    lifeforms = [Lifeform(i + 1, Rule.from_bs_string(rule)) for i, rule in enumerate(task.rules)]
    tie = TieBreak(identity["tie_break"], identity["tie_break_radius"], identity["tie_break_power"])
    grid = LatticeGrid(
        lifeforms, task.shape, width, height, True, task.triangle_mode, tie_break=tie
    )
    start = species_start(
        task.shape, task.grid, identity["density"], identity["seed"], identity["shares"]
    )
    grid.occupancy = start.copy()
    grid.lifespans[start > 0] = 1
    rng = np.random.default_rng([20261123, 2, identity["seed"]])  # the tie-break's draws

    def alive_now() -> list[int]:
        return [int(np.count_nonzero(grid.occupancy == lf.id)) for lf in lifeforms]

    generations = [0]
    alive = [[count] for count in alive_now()]
    first_unlike, first_like = pair_counts(grid.occupancy, grid.adjacency)
    unlike, like = [first_unlike], [first_like]
    births: list[list[int]] = [[0] for _ in lifeforms]
    contested = [0]
    extinct_at: list[int | None] = [0 if series[0] == 0 else None for series in alive]
    born = [0] * species
    settled = 0
    stride = identity["sample_every"]
    for generation in range(1, task.steps + 1):
        result = grid.step(rng)
        settled += result.contested
        for i, lf in enumerate(lifeforms):
            born[i] += int(result.births[lf.id])
            if extinct_at[i] is None and result.alive[lf.id] == 0:
                extinct_at[i] = generation
        if generation == task.steps or (stride and generation % stride == 0):
            generations.append(generation)
            for i, count in enumerate(alive_now()):
                alive[i].append(count)
                births[i].append(born[i])
            pairs = pair_counts(grid.occupancy, grid.adjacency)
            unlike.append(pairs[0])
            like.append(pairs[1])
            contested.append(settled)
            born = [0] * species
            settled = 0
    return CompetitionRecord(
        **identity,
        cells=width * height,
        generations=generations,
        alive=alive,
        unlike=unlike,
        like=like,
        births=births,
        contested=contested,
        extinct_at=extinct_at,
        start_sha256=digest(start),
        final_sha256=digest(grid.occupancy),
    )


def _key(task: CompetitionTask) -> tuple:
    identity = _identity(task)
    return tuple(
        tuple(identity[field]) if isinstance(identity[field], list) else identity[field]
        for field in RECORD_KEY
    )


def run_competition(
    tasks: Iterable[CompetitionTask], out_path: str, workers: int = 1
) -> Iterator[CompetitionRecord]:
    """Run every task not already in `out_path`, appending one JSONL record per run as it
    finishes (resumable, as the other experiments)."""
    done = resume_keys(out_path, RECORD_KEY)
    todo, queued = [], set()
    for task in tasks:
        key = _key(task)
        if key not in done and key not in queued:
            queued.add(key)
            todo.append(task)
    for record in map_tasks(competition_point, todo, workers):
        append_jsonl(out_path, record.to_json())
        yield record


__all__ = [
    "MAX_SPECIES",
    "RECORD_KEY",
    "CompetitionRecord",
    "CompetitionTask",
    "competition_point",
    "pair_counts",
    "run_competition",
    "species_start",
]
