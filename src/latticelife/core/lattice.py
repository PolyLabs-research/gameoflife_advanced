"""LatticeGrid: state arrays + the one update rule (§6.3). There is exactly one update
implementation in this codebase; geometry enters only via the adjacency array from
`core.geometries`. Legacy birth and survival semantics are preserved, with the legacy per-cell
Python loops replaced by vectorized numpy (§6.3 steps 2 and 4).

Kill attribution (§6.3 step 4) deliberately departs from legacy `square_grid.py`. The spec
credits "each overcrowding death" to a foreign species holding more than half of the dead cell's
neighbours, but never defines overcrowding. Legacy read it as the species' *own* neighbour count
exceeding max(S). Survival depends only on the own count, so that reading needs own > max(S) and
foreign > total/2 at once, i.e. a degree of at least 2*max(S) + 3: no kill could ever be recorded
for B3/S23 on squares, B2/S34 on hexagons or B46/S345 on edge+vertex triangles. And with an
empty S (max(S) taken as -1) every isolation death counted as overcrowding. Here a death is an
overcrowding death when the cell's live neighbours of *all* species exceed max(S) (more
neighbours than the species can survive among), and the majority test is unchanged. Species
with an empty S die whatever their surroundings, so their deaths are never credited.

The birth tie-break. Each species is born on its own count: a dead cell with a count of species
i's neighbours in B_i is eligible for i. With several species a cell can be eligible for more
than one at once, and `TieBreak` says who gets it. It is a setting of the one update rule, as
the adjacency array is, and every kind treats the species alike (none is favoured by its id):

    random      one of the eligible species, each with the same chance, by a draw from the
                caller's Generator. The rule as it always was, draw for draw; the next grid then
                depends on the random stream as well as on the grid.
    standoff    nobody: a contested cell stays empty.
    strongest   the eligible species with the most cells within `radius` lattice steps of the
                cell; if the top is shared, nobody.
    weighted    a draw again, with each eligible species' chance in proportion to that same
                count raised to `power`. Power 0 is `random`, draw for draw; a large power
                approaches `strongest`, except that a shared top is still drawn for.

With `standoff` or `strongest` the Generator is never used: the next grid is a function of the
grid alone, whatever the number of species. Cells within `radius` steps are those reached by
that many moves along the adjacency array, each counted once and the contested cell not at
all; at radius 1 they are the cell's neighbours. Two species with the same B contest a cell
with the same count each, so `strongest` and `weighted` need a radius of 2 or more to tell
them apart. Contests are counted (`UpdateResult.contested`) whatever settles them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from latticelife.core.geometries import build_adjacency
from latticelife.core.lifeform import Lifeform
from latticelife.core.results import UpdateResult

# Lifespans saturate here instead of wrapping to a negative int32 (static counts, avg_lifespan
# and the static shade all compare lifespans against a positive threshold).
LIFESPAN_MAX = int(np.iinfo(np.int32).max)

TIE_BREAKS: tuple[str, ...] = ("random", "standoff", "strongest", "weighted")
# How far `strongest` and `weighted` may look. The cells within r steps are found by r rounds of
# neighbours-of-neighbours for the contested cells alone, degree ** r candidates each.
MAX_TIE_BREAK_RADIUS = 4


@dataclass(frozen=True)
class TieBreak:
    """Who is born in a dead cell that more than one species is eligible for (see the module
    docstring). `radius` and `power` matter to `strongest` and `weighted` only."""

    kind: str = "random"
    radius: int = 2
    power: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in TIE_BREAKS:
            raise ValueError(
                f"tie-break must be one of {', '.join(TIE_BREAKS)} (got {self.kind!r})"
            )
        if type(self.radius) is not int or not 1 <= self.radius <= MAX_TIE_BREAK_RADIUS:
            raise ValueError(
                f"tie-break radius must be a whole number of lattice steps from 1 to "
                f"{MAX_TIE_BREAK_RADIUS} (got {self.radius!r})"
            )
        power = self.power
        if (
            isinstance(power, bool)
            or not isinstance(power, int | float)
            or not (math.isfinite(power) and power >= 0)
        ):
            raise ValueError(f"tie-break power must be a number, 0 or more (got {power!r})")

    @property
    def deterministic(self) -> bool:
        """True when the update never draws: the next grid is a function of the grid."""
        return self.kind in ("standoff", "strongest")


class LatticeGrid:
    def __init__(
        self,
        lifeforms: list[Lifeform],
        shape: str,
        width: int,
        height: int,
        wrap: bool,
        triangle_mode: str = "edge+vertex",
        static_threshold: int = 10,
        adjacency: np.ndarray | None = None,
        tie_break: TieBreak | None = None,
    ) -> None:
        self.lifeforms = lifeforms
        self.tie_break = tie_break if tie_break is not None else TieBreak()
        self.shape = shape
        self.width = width
        self.height = height
        self.wrap = wrap
        self.triangle_mode = triangle_mode
        self.static_threshold = static_threshold

        # A caller building a twin grid (e.g. damage spreading, §6.12) can pass an existing
        # adjacency array to share it read-only rather than rebuilding an identical one.
        if adjacency is not None:
            self.adjacency = adjacency
        else:
            self.adjacency = build_adjacency(shape, width, height, wrap, triangle_mode)

        n_cells = width * height
        self.occupancy = np.zeros(n_cells, dtype=np.int8)  # 0 = empty, 1..L = lifeform id
        self.lifespans = np.zeros(n_cells, dtype=np.int32)  # never zeros_like(occupancy) — D-1

    @property
    def n_cells(self) -> int:
        return self.width * self.height

    def _neighbor_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-species neighbor counts (L, n_cells) and their sum (n_cells,), via the padded
        gather: pad the flat grid with one sentinel slot, remap -1 adjacency indices to it, then
        reduce. Works uniformly for every shape — convolve2d and scipy are gone."""
        ids = np.array([lf.id for lf in self.lifeforms], dtype=np.int64)
        padded = np.append(self.occupancy, np.zeros(1, dtype=self.occupancy.dtype))
        gather_idx = np.where(self.adjacency >= 0, self.adjacency, self.n_cells)
        neighbor_vals = padded[gather_idx]  # (n_cells, degree)
        counts = (neighbor_vals[None, :, :] == ids[:, None, None]).sum(axis=2)  # (L, n_cells)
        return counts, counts.sum(axis=0)

    def step(self, rng: np.random.Generator) -> UpdateResult:
        """Synchronous update. `rng` is owned by the caller (SimulationSession, §6.4) — this
        method never seeds or creates a Generator itself."""
        lifeforms = self.lifeforms
        n = self.n_cells
        size = (max(lf.id for lf in lifeforms) + 1) if lifeforms else 1

        if not lifeforms:
            zeros = np.zeros(size, dtype=np.int64)
            return UpdateResult(zeros, zeros, zeros.copy(), zeros.copy(), zeros.copy(), 0)

        births = np.zeros(size, dtype=np.int64)
        deaths = np.zeros(size, dtype=np.int64)
        kills = np.zeros(size, dtype=np.int64)

        ids = np.array([lf.id for lf in lifeforms], dtype=np.int64)
        counts, total_neighbors = self._neighbor_counts()

        new_occupancy = self.occupancy.copy()
        new_lifespans = self.lifespans.copy()

        # --- Births (§6.3 step 2): each species on its own count, then the tie-break. ---
        dead = self.occupancy == 0
        eligible = np.zeros((len(lifeforms), n), dtype=bool)
        for i, lf in enumerate(lifeforms):
            eligible[i] = dead & np.isin(counts[i], list(lf.rule.birth))

        contested = 0
        if self.tie_break.kind == "random":
            keys = rng.random((len(lifeforms), n))
            keys[~eligible] = -1.0
            any_eligible = eligible.any(axis=0)
            winner = np.argmax(keys, axis=0)
            if len(lifeforms) > 1:
                contested = int(np.count_nonzero(eligible.sum(axis=0) > 1))
        else:
            any_eligible, winner, contested = self._settle(eligible, rng)
        chosen_ids = ids[winner]

        new_occupancy[any_eligible] = chosen_ids[any_eligible]
        new_lifespans[any_eligible] = 1
        np.add.at(births, chosen_ids[any_eligible], 1)

        # --- Survival / death + kill attribution (§6.3 steps 3-4), per species. ---
        survived = np.zeros(n, dtype=bool)
        for i, lf in enumerate(lifeforms):
            alive_mask = self.occupancy == lf.id
            survive_mask = alive_mask & np.isin(counts[i], list(lf.rule.survive))
            death_mask = alive_mask & ~survive_mask

            survived |= survive_mask
            new_occupancy[death_mask] = 0
            new_lifespans[death_mask] = 0
            deaths[lf.id] = int(np.count_nonzero(death_mask))
            _credit_kills(i, lf, death_mask, counts, total_neighbors, ids, kills)

        # Survivors age by one, saturating at LIFESPAN_MAX rather than wrapping negative.
        np.add(new_lifespans, 1, out=new_lifespans, where=survived & (new_lifespans < LIFESPAN_MAX))

        changed_cells = int(np.sum(new_occupancy != self.occupancy))

        self.occupancy = new_occupancy
        self.lifespans = new_lifespans

        alive = np.zeros(size, dtype=np.int64)
        static = np.zeros(size, dtype=np.int64)
        for lf in lifeforms:
            alive_mask = self.occupancy == lf.id
            alive[lf.id] = int(np.sum(alive_mask))
            static[lf.id] = int(np.sum(alive_mask & (self.lifespans > self.static_threshold)))

        return UpdateResult(births, deaths, kills, alive, static, changed_cells, contested)

    def _settle(
        self, eligible: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Births under a tie-break other than `random`: which cells get one, the row of
        `eligible` that wins each, and how many cells were contested."""
        tie = self.tie_break
        contenders = eligible.sum(axis=0)
        any_eligible = contenders > 0
        cells = np.flatnonzero(contenders > 1)
        if tie.kind == "weighted":
            # The draw `random` makes, so that power 0 is that rule draw for draw.
            keys = rng.random(eligible.shape)
            keys[~eligible] = -1.0
            winner = np.argmax(keys, axis=0)
        else:
            winner = np.argmax(eligible, axis=0)  # the one eligible species, where there is one
        if cells.size == 0:
            return any_eligible, winner, 0
        if tie.kind == "standoff":
            any_eligible[cells] = False
            return any_eligible, winner, int(cells.size)

        strength = self._strength(cells)  # (species, contested cells)
        in_contest = eligible[:, cells]
        if tie.kind == "strongest":
            strength = np.where(in_contest, strength, -1)
            top = strength.max(axis=0)
            winner[cells] = strength.argmax(axis=0)
            any_eligible[cells[(strength == top).sum(axis=0) > 1]] = False
            return any_eligible, winner, int(cells.size)

        # weighted: the largest of u ** (1 / w) falls to species i with chance w_i / sum(w),
        # for independent uniform u; compared as log(u) / w. Where every contender's weight is
        # 0 (no cell of theirs within the radius, and a positive power) the chances are equal.
        weight = np.where(in_contest, strength.astype(np.float64) ** tie.power, 0.0)
        weight[:, weight.max(axis=0) == 0] = 1.0
        usable = in_contest & (weight > 0)
        score = np.full(weight.shape, -np.inf)
        with np.errstate(divide="ignore"):  # log(0): a draw of exactly 0 loses to any other
            np.divide(
                np.log(np.where(usable, keys[:, cells], 1.0)), weight, out=score, where=usable
            )
        winner[cells] = score.argmax(axis=0)
        return any_eligible, winner, int(cells.size)

    def _within(self, cells: np.ndarray, radius: int) -> np.ndarray:
        """For each of `cells`, the cells within `radius` steps along the adjacency array: a
        (len(cells), k) array padded with -1, each cell once and the cell itself left out."""
        reach = frontier = self.adjacency[cells]
        for _ in range(radius - 1):
            onward = np.where(frontier[:, :, None] >= 0, self.adjacency[frontier], -1)
            frontier = onward.reshape(len(cells), -1)
            reach = np.concatenate([reach, frontier], axis=1)
        reach = np.sort(np.where(reach == cells[:, None], -1, reach), axis=1)
        reach[:, 1:][reach[:, 1:] == reach[:, :-1]] = -1
        return reach

    def _strength(self, cells: np.ndarray) -> np.ndarray:
        """A (species, len(cells)) array: how many cells of each species lie within the
        tie-break's radius of each of `cells`."""
        near = self._within(cells, self.tie_break.radius)
        held = np.append(self.occupancy, 0)[near]  # a pad, -1, reads the empty slot at the end
        return np.stack([np.count_nonzero(held == lf.id, axis=1) for lf in self.lifeforms])


def _credit_kills(
    i: int,
    lifeform: Lifeform,
    death_mask: np.ndarray,
    counts: np.ndarray,
    total_neighbors: np.ndarray,
    ids: np.ndarray,
    kills: np.ndarray,
) -> None:
    """§6.3 step 4, for the deaths of `lifeform` (row `i` of `counts`): a death is an
    overcrowding death when the cell's live neighbours of all species exceed max(S), and it is
    credited to the foreign species with the most neighbours there iff that count is more than
    half of all live neighbours (so the killer is unique). See the module docstring for why
    overcrowding counts every species and why an empty S is never credited."""
    if not lifeform.rule.survive or len(ids) < 2:
        return
    overcrowded = np.flatnonzero(death_mask & (total_neighbors > max(lifeform.rule.survive)))
    if overcrowded.size == 0:
        return
    foreign = np.delete(counts[:, overcrowded], i, axis=0)  # (L-1, m)
    killer_row = foreign.argmax(axis=0)
    killer_count = foreign.max(axis=0)
    credited = 2 * killer_count > total_neighbors[overcrowded]
    np.add.at(kills, np.delete(ids, i)[killer_row[credited]], 1)


def clone_grid(grid: LatticeGrid) -> LatticeGrid:
    """A twin grid for damage spreading (§6.12): shares `grid`'s adjacency array read-only
    (never rebuilt or deep-copied) with independently cloned occupancy/lifespans state."""
    twin = LatticeGrid(
        grid.lifeforms,
        grid.shape,
        grid.width,
        grid.height,
        grid.wrap,
        grid.triangle_mode,
        grid.static_threshold,
        adjacency=grid.adjacency,
        tie_break=grid.tie_break,
    )
    twin.occupancy = grid.occupancy.copy()
    twin.lifespans = grid.lifespans.copy()
    return twin
