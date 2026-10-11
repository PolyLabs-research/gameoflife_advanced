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
"""

from __future__ import annotations

import numpy as np

from latticelife.core.geometries import build_adjacency
from latticelife.core.lifeform import Lifeform
from latticelife.core.results import UpdateResult

# Lifespans saturate here instead of wrapping to a negative int32 (static counts, avg_lifespan
# and the static shade all compare lifespans against a positive threshold).
LIFESPAN_MAX = int(np.iinfo(np.int32).max)


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
    ) -> None:
        self.lifeforms = lifeforms
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

        # --- Births (§6.3 step 2): vectorized random tie-break among eligible species. ---
        dead = self.occupancy == 0
        eligible = np.zeros((len(lifeforms), n), dtype=bool)
        for i, lf in enumerate(lifeforms):
            eligible[i] = dead & np.isin(counts[i], list(lf.rule.birth))

        keys = rng.random((len(lifeforms), n))
        keys[~eligible] = -1.0
        any_eligible = eligible.any(axis=0)
        winner = np.argmax(keys, axis=0)
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

        return UpdateResult(births, deaths, kills, alive, static, changed_cells)


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
    )
    twin.occupancy = grid.occupancy.copy()
    twin.lifespans = grid.lifespans.copy()
    return twin
