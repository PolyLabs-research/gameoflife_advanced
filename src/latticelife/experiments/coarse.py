"""Zooming out: block projections of a wrapped grid, and the closure test.

To follow a rule to larger scales, replace each block of b x b cells by one supercell and ask
whether some rule on the supercells predicts the zoomed-out run. The **closure test** asks it.
From one start it takes two routes,

    fine route      run the fine world for s * clock generations, then project
    coarse route    project the start, then run the coarse world for s generations

and counts the supercells where the two differ, for s = 1, 2, ... A count of zero at every s
says the coarse world is an exact stand-in for the fine one at that scale; a count near half the
supercells says it knows nothing about it.

**Blocks.** A block of b sends a wrapped W x H grid to the wrapped (W / b) x (H / b) grid of the
same shape, so a coarse world can run what a projection reads (`coarse_grid`). On the square the
block of supercell (R, C) is the rows b R .. b R + b - 1 and the columns b C .. b C + b - 1, and
a place in it is an offset (row, column). On the hexagon the blocks are rhombi in axial
coordinates (q, r) = (c, row - floor(c / 2)), where the six neighbours are the steps +-(1, 0),
+-(0, 1), +-(1, -1): supercell (Q, R) holds the cells (b Q + i, b R + j), and a place in it is
the axial offset (i, j). Its neighbours on the coarse grid are then b steps from it on the fine
one, in the same six directions. b must divide W and H, and the coarse grid must be one
LatticeLife can wrap (an even width on the hexagon). The triangle lattice has no blocks here.

**Projections** of a one-species configuration (`project`):

    identity    the configuration itself (block 1)
    decimate    the cell at a fixed place in each block: one coset of the sublattice b Z^2
    parity      the sum mod 2 of the cells of each block: every coset, added

and of a grid of several species: `block_counts` (each species' cells in every block) and
`leading_species` (the species with the most; none where the top is shared or the block empty).

**Worlds.** `World(shape, rule, grid)` is a B/S rule on the wrapped grid with LatticeLife's own
adjacency; `dilation = k` puts every neighbour k lattice steps away
(`core.geometries.build_adjacency`). Both run on the one update rule, `LatticeGrid.step`, with
one species, so a run is a function of its start.

**Known answers.** A rule that is linear over GF(2) (a parity rule: B1357/S1357 and
B1357/S02468 on the square, B135/S135 and B135/S0246 on the hexagon) is its own coarse rule at
every power of two: 2^j generations are one generation with neighbours 2^j apart (Frobenius), and
each coset of the sublattice then runs as a copy of the rule on the smaller grid. So with
block = clock = 2^j and the same rule on the coarse grid the count is zero for `decimate` at
any offset and for `parity`, from every start; and for the same world at `dilation = 2^j`
against 2^j generations, cell for cell (`identity`). Where the scales do not match the two routes
differ by a non-zero linear function of the start, so from `uniform_start` every supercell
differs with probability exactly one half. lattice-lab's `examples/scale-up-linear` registered
and confirmed both.

`closure_point` runs one test and returns its record; `run_closure` writes one resumable JSONL
record per task, as the other experiments do. `closure` is the library form: several tests that
share their fine world and clock, off one fine run from a start of the caller's choosing.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from latticelife.core.geometries import build_adjacency, max_degree, validate_dimensions
from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.experiments.harness import (
    GridLabel,
    append_jsonl,
    grid_dimensions,
    map_tasks,
    parse_grid,
    resume_keys,
)

SHAPES: tuple[str, ...] = ("square", "hexagon")
PROJECTIONS: tuple[str, ...] = ("identity", "decimate", "parity")

RECORD_KEY: tuple[str, ...] = (
    "rule",
    "shape",
    "grid",
    "block",
    "clock",
    "projection",
    "offset",
    "coarse_rule",
    "coarse_dilation",
    "seed",
    "steps",
)


# --- blocks --------------------------------------------------------------------------------------


def _check_shape(shape: str) -> None:
    if shape not in SHAPES:
        raise ValueError(f"no block geometry for shape {shape!r} (only {', '.join(SHAPES)})")


def coarse_grid(shape: str, grid: int | str | tuple[int, int], block: int) -> GridLabel:
    """The label of the grid a block-`block` projection of `grid` lands on. ValueError when the
    block does not divide the grid, or the grid it would land on cannot wrap."""
    _check_shape(shape)
    width, height = grid_dimensions(grid)
    if isinstance(block, bool) or int(block) != block or block < 1:
        raise ValueError(f"a block is a whole number of cells, at least 1 (got {block!r})")
    if width % block or height % block:
        raise ValueError(f"a block of {block} does not divide a {width}x{height} grid")
    coarse = (width // block, height // block)
    try:
        validate_dimensions(shape, *coarse, True)
    except ValueError as exc:
        raise ValueError(
            f"a block of {block} sends a {width}x{height} {shape} grid to "
            f"{coarse[0]}x{coarse[1]}, which cannot wrap: {exc}"
        ) from None
    return parse_grid(coarse)


@functools.lru_cache(maxsize=64)
def _sample_cells(shape: str, grid: GridLabel, block: int, offset: tuple[int, int]) -> np.ndarray:
    width, height = grid_dimensions(grid)
    coarse_width, coarse_height = grid_dimensions(coarse_grid(shape, grid, block))
    row, column = np.divmod(np.arange(coarse_width * coarse_height), coarse_width)
    if shape == "square":
        r, c = block * row + offset[0], block * column + offset[1]
    else:
        q = block * column + offset[0]
        r, c = block * (row - column // 2) + offset[1] + q // 2, q
    cells = (r % height) * width + c % width
    cells.setflags(write=False)
    return cells


def sample_cells(
    shape: str, grid: int | str | tuple[int, int], block: int, offset: Sequence[int] = (0, 0)
) -> np.ndarray:
    """For every supercell, in the coarse grid's own cell order, the index of the fine cell at
    `offset` in its block ((row, column) on the square, axial (q, r) on the hexagon). Read-only:
    the array is shared between calls."""
    offset = tuple(int(a) for a in offset)
    if len(offset) != 2 or not all(0 <= a < block for a in offset):
        raise ValueError(f"offset {offset} lies outside a block of {block}")
    return _sample_cells(shape, parse_grid(grid), int(block), offset)


@functools.lru_cache(maxsize=16)
def _block_cells(shape: str, grid: GridLabel, block: int) -> np.ndarray:
    cells = np.stack(
        [_sample_cells(shape, grid, block, (i, j)) for i in range(block) for j in range(block)]
    )
    cells.setflags(write=False)
    return cells


def block_cells(shape: str, grid: int | str | tuple[int, int], block: int) -> np.ndarray:
    """A (block * block, supercells) array: every fine cell of every supercell's block. Each
    fine cell appears once. Read-only: the array is shared between calls."""
    coarse_grid(shape, grid, block)
    return _block_cells(shape, parse_grid(grid), int(block))


# --- projections ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    kind: str
    block: int = 1
    offset: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.kind not in PROJECTIONS:
            raise ValueError(f"unknown projection {self.kind!r} (one of {', '.join(PROJECTIONS)})")
        if self.block < 1 or (self.kind == "identity" and self.block != 1):
            raise ValueError(f"no {self.kind} projection with block {self.block}")
        if len(self.offset) != 2 or not all(0 <= a < self.block for a in self.offset):
            raise ValueError(f"offset {tuple(self.offset)} lies outside a block of {self.block}")


def _cells_of(grid: int | str | tuple[int, int], cells: np.ndarray) -> np.ndarray:
    width, height = grid_dimensions(grid)
    cells = np.asarray(cells)
    if cells.shape != (width * height,):
        raise ValueError(f"expected {width * height} cells, got an array of shape {cells.shape}")
    return cells


def project(
    shape: str, grid: int | str | tuple[int, int], projection: Projection, cells: np.ndarray
) -> np.ndarray:
    """The coarse configuration `projection` reads off the one-species configuration `cells`
    (0 or 1 by cell index), as 0 or 1 by the coarse grid's cell index."""
    cells = _cells_of(grid, cells)
    if projection.kind == "identity":
        return cells.astype(np.int8)
    if projection.kind == "decimate":
        chosen = sample_cells(shape, grid, projection.block, projection.offset)
        return cells[chosen].astype(np.int8)
    blocks = cells[block_cells(shape, grid, projection.block)].astype(np.int8)
    return np.bitwise_xor.reduce(blocks, axis=0)


def block_counts(
    shape: str,
    grid: int | str | tuple[int, int],
    block: int,
    occupancy: np.ndarray,
    species: Sequence[int],
) -> np.ndarray:
    """A (len(species), supercells) array: how many cells of each species id (as in
    `LatticeGrid.occupancy`; 0 is an empty cell and may be asked for too) every block holds."""
    held = _cells_of(grid, occupancy)[block_cells(shape, grid, block)]
    return np.stack([np.count_nonzero(held == int(s), axis=0) for s in species])


def leading_species(counts: np.ndarray, species: Sequence[int]) -> np.ndarray:
    """Per supercell, the id of the species with the most cells in `counts` (rows as `species`,
    from `block_counts`). 0 where no species has a cell, and where two or more share the top:
    a block is not given to either of two level species."""
    counts = np.asarray(counts)
    ids = np.asarray(list(species))
    if counts.ndim != 2 or counts.shape[0] != ids.size or 0 in ids:
        raise ValueError("expected one row of counts per species id, none of them 0 (empty)")
    top = counts.max(axis=0)
    alone = (counts == top).sum(axis=0) == 1
    return np.where((top > 0) & alone, ids[counts.argmax(axis=0)], 0)


# --- worlds and runs -----------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    """A rule (a B/S string) on the wrapped `grid` of `shape`, with every neighbour `dilation`
    lattice steps away."""

    shape: str
    rule: str
    grid: GridLabel
    dilation: int = 1

    def __post_init__(self) -> None:
        _check_shape(self.shape)
        Rule.from_bs_string(self.rule)
        object.__setattr__(self, "grid", parse_grid(self.grid))
        _adjacency(self.shape, self.grid, self.dilation)  # raises for a grid that cannot wrap

    @property
    def cells(self) -> int:
        width, height = grid_dimensions(self.grid)
        return width * height


@functools.lru_cache(maxsize=8)
def _adjacency(shape: str, grid: GridLabel, dilation: int) -> np.ndarray:
    width, height = grid_dimensions(grid)
    neighbours = build_adjacency(shape, width, height, True, dilation=dilation)
    neighbours.setflags(write=False)
    return neighbours


class Run:
    """One world running from a given one-species configuration, on LatticeLife's update
    rule."""

    def __init__(self, world: World, cells: np.ndarray) -> None:
        cells = _cells_of(world.grid, cells)
        if not np.isin(cells, (0, 1)).all():
            raise ValueError("expected a configuration of 0 and 1")
        width, height = grid_dimensions(world.grid)
        self.world = world
        self.grid = LatticeGrid(
            [Lifeform(id=1, rule=Rule.from_bs_string(world.rule))],
            world.shape,
            width,
            height,
            True,
            adjacency=_adjacency(world.shape, world.grid, world.dilation),
        )
        self.grid.occupancy = cells.astype(np.int8)
        # One species: the update draws tie-break keys and never needs them.
        self._rng = np.random.default_rng(0)
        self.generation = 0

    def advance(self, generations: int) -> np.ndarray:
        for _ in range(generations):
            self.grid.step(self._rng)
        self.generation += generations
        return self.cells

    @property
    def cells(self) -> np.ndarray:
        return self.grid.occupancy


def uniform_start(shape: str, grid: int | str | tuple[int, int], seed: int) -> np.ndarray:
    """A uniformly random one-species configuration: every cell 0 or 1 with probability 1/2,
    independently, so every configuration is equally likely. Reproducible from the shape, the
    grid and the seed; a record names its start by them. (A session's own seeding fixes the
    number of live cells, which is not this distribution.)"""
    _check_shape(shape)
    width, height = grid_dimensions(grid)
    size = [width] if width == height else [width, height]
    entropy = [20261123, SHAPES.index(shape), *size, int(seed)]
    return np.random.default_rng(entropy).integers(0, 2, size=width * height, dtype=np.int8)


def digest(cells: np.ndarray) -> str:
    """The sha256 of a configuration, as one byte per cell in cell order."""
    return hashlib.sha256(np.asarray(cells, dtype=np.int8).tobytes()).hexdigest()


# --- the closure test ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagram:
    """One closure test, as the square of maps that should commute: the fine world's `clock`
    generations against one generation of the coarse world, through `projection`."""

    fine: World
    coarse: World
    projection: Projection
    clock: int

    def __post_init__(self) -> None:
        if self.fine.shape != self.coarse.shape:
            raise ValueError("the fine and coarse worlds must share a shape")
        if self.fine.dilation != 1:
            raise ValueError("the fine world runs on the lattice's own adjacency (dilation 1)")
        expected = coarse_grid(self.fine.shape, self.fine.grid, self.projection.block)
        if self.coarse.grid != expected:
            raise ValueError(
                f"a block of {self.projection.block} sends the fine grid {self.fine.grid} to "
                f"{expected}, not to the coarse world's {self.coarse.grid}"
            )
        if self.clock < 1:
            raise ValueError("the clock is at least one generation")


@dataclass(frozen=True)
class Closure:
    """What one closure test showed: for s = 1, 2, ... coarse generations, the supercells where
    the two routes differ, and the live supercells on the fine route (two dead grids agree, and
    say nothing)."""

    mismatches: tuple[int, ...]
    live: tuple[int, ...]
    supercells: int


def closure(
    diagrams: Sequence[Diagram], start: np.ndarray, steps: int
) -> tuple[list[Closure], str]:
    """Run the tests from `start` for `steps` coarse generations. They must share their fine
    world and clock, which is then run once. Returns each test's Closure, and the sha256 of the
    fine world's last configuration."""
    if not diagrams:
        raise ValueError("no tests to run")
    fine, clock = diagrams[0].fine, diagrams[0].clock
    if any(d.fine != fine or d.clock != clock for d in diagrams):
        raise ValueError("tests run together must share their fine world and clock")
    shape, grid = fine.shape, fine.grid
    coarse_runs = [Run(d.coarse, project(shape, grid, d.projection, start)) for d in diagrams]
    fine_run = Run(fine, start)
    mismatches: list[list[int]] = [[] for _ in diagrams]
    live: list[list[int]] = [[] for _ in diagrams]
    for _ in range(steps):
        zoomed = fine_run.advance(clock)
        for i, (diagram, run) in enumerate(zip(diagrams, coarse_runs, strict=True)):
            seen = project(shape, grid, diagram.projection, zoomed)
            mismatches[i].append(int(np.count_nonzero(seen != run.advance(1))))
            live[i].append(int(np.count_nonzero(seen)))
    results = [
        Closure(tuple(m), tuple(v), d.coarse.cells)
        for m, v, d in zip(mismatches, live, diagrams, strict=True)
    ]
    return results, digest(fine_run.cells)


# --- tasks and records ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosureTask:
    rule: str
    shape: str
    grid: GridLabel
    block: int
    clock: int
    seed: int
    steps: int  # coarse generations; the fine world runs steps * clock
    projection: str = "decimate"
    offset: tuple[int, int] = (0, 0)
    coarse_rule: str | None = None  # None: the fine rule
    coarse_dilation: int = 1


@dataclass(frozen=True)
class ClosureRecord:
    rule: str
    shape: str
    grid: GridLabel
    block: int
    clock: int
    projection: str
    offset: list[int]
    coarse_rule: str
    coarse_grid: GridLabel
    coarse_dilation: int
    seed: int
    steps: int
    supercells: int
    mismatches: list[int]  # per coarse generation, the supercells where the routes differ
    live: list[int]  # per coarse generation, the live supercells on the fine route
    start_sha256: str  # of `uniform_start(shape, grid, seed)`
    fine_sha256: str  # of the fine world after steps * clock generations

    def to_json(self) -> dict:
        return asdict(self)


def _rule_text(rule: str, shape: str) -> str:
    return Rule.from_bs_string(rule).to_bs_string(max_degree(shape))


def task_diagram(task: ClosureTask) -> Diagram:
    """The closure test a task names, with its rules in the persisted notation."""
    rule = _rule_text(task.rule, task.shape)
    coarse_rule = rule if task.coarse_rule is None else _rule_text(task.coarse_rule, task.shape)
    projection = Projection(task.projection, int(task.block), tuple(task.offset))
    fine = World(task.shape, rule, task.grid)
    coarse = World(
        task.shape,
        coarse_rule,
        coarse_grid(task.shape, task.grid, task.block),
        task.coarse_dilation,
    )
    return Diagram(fine, coarse, projection, int(task.clock))


def _key(task: ClosureTask) -> tuple:
    diagram = task_diagram(task)
    return (
        diagram.fine.rule,
        task.shape,
        diagram.fine.grid,
        diagram.projection.block,
        diagram.clock,
        diagram.projection.kind,
        tuple(diagram.projection.offset),
        diagram.coarse.rule,
        diagram.coarse.dilation,
        task.seed,
        task.steps,
    )


def closure_point(task: ClosureTask) -> ClosureRecord:
    """One closure test from `uniform_start(shape, grid, seed)`, followed for `steps` coarse
    generations."""
    diagram = task_diagram(task)
    start = uniform_start(task.shape, task.grid, task.seed)
    (result,), last = closure([diagram], start, task.steps)
    return ClosureRecord(
        rule=diagram.fine.rule,
        shape=task.shape,
        grid=diagram.fine.grid,
        block=diagram.projection.block,
        clock=diagram.clock,
        projection=diagram.projection.kind,
        offset=list(diagram.projection.offset),
        coarse_rule=diagram.coarse.rule,
        coarse_grid=diagram.coarse.grid,
        coarse_dilation=diagram.coarse.dilation,
        seed=task.seed,
        steps=task.steps,
        supercells=result.supercells,
        mismatches=list(result.mismatches),
        live=list(result.live),
        start_sha256=digest(start),
        fine_sha256=last,
    )


def run_closure(
    tasks: Iterable[ClosureTask], out_path: str, workers: int = 1
) -> Iterator[ClosureRecord]:
    """Run every task not already in `out_path`, appending one JSONL record per test as it
    finishes (resumable, as the other experiments)."""
    done = resume_keys(out_path, RECORD_KEY)
    todo, queued = [], set()
    for task in tasks:
        key = _key(task)
        if key not in done and key not in queued:
            queued.add(key)
            todo.append(task)
    for record in map_tasks(closure_point, todo, workers):
        append_jsonl(out_path, record.to_json())
        yield record


__all__ = [
    "PROJECTIONS",
    "RECORD_KEY",
    "SHAPES",
    "Closure",
    "ClosureRecord",
    "ClosureTask",
    "Diagram",
    "Projection",
    "Run",
    "World",
    "block_cells",
    "block_counts",
    "closure",
    "closure_point",
    "coarse_grid",
    "digest",
    "leading_species",
    "project",
    "run_closure",
    "sample_cells",
    "task_diagram",
    "uniform_start",
]
