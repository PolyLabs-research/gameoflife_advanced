"""LatticeGrid: state arrays + the one update rule (§6.3). There is exactly one update
implementation in this codebase; geometry enters only via the adjacency array from
`core.geometries`. Legacy semantics (birth/survival/kill-attribution) are preserved, with the
legacy per-cell Python loops replaced by vectorized numpy (§6.3 steps 2 and 4).
"""

from __future__ import annotations

import numpy as np

from latticelife.core.geometries import build_adjacency
from latticelife.core.lifeform import Lifeform
from latticelife.core.results import UpdateResult


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
    ) -> None:
        self.lifeforms = lifeforms
        self.shape = shape
        self.width = width
        self.height = height
        self.wrap = wrap
        self.triangle_mode = triangle_mode
        self.static_threshold = static_threshold

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
        for i, lf in enumerate(lifeforms):
            alive_mask = self.occupancy == lf.id
            survive_mask = alive_mask & np.isin(counts[i], list(lf.rule.survive))
            death_mask = alive_mask & ~survive_mask

            new_lifespans[survive_mask] += 1
            new_occupancy[death_mask] = 0
            new_lifespans[death_mask] = 0
            deaths[lf.id] = int(np.sum(death_mask))

            max_survive = max(lf.rule.survive) if lf.rule.survive else -1
            overcrowd = death_mask & (counts[i] > max_survive)
            if np.any(overcrowd):
                counts_excl_self = counts.copy()
                counts_excl_self[i] = -1
                killer_row = np.argmax(counts_excl_self, axis=0)
                killer_vals = counts_excl_self[killer_row, np.arange(n)]
                majority = killer_vals > (total_neighbors / 2)
                credit = overcrowd & majority
                if np.any(credit):
                    np.add.at(kills, ids[killer_row][credit], 1)

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
