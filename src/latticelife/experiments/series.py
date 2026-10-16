"""Counts over time for one single-species run: the raw material for finding conserved quantities
(integer combinations of counts that never change) and a check on the simulator's bookkeeping.

    count_series(config, rule, shape, grid, density, seed, steps) -> dict[str, list[int]]

Columns, one value per generation 0..steps-1:

    live              live cells
    births, deaths    cumulative births and deaths since generation 0
    live_k, dead_k    live and dead cells with k live neighbours, k = 0..degree

Four identities hold by construction: live(t) = live(0) + births(t) - deaths(t);
live = sum_k live_k; cells = live + sum_k dead_k; and, on a torus where every cell has `degree`
neighbours, sum_k k (live_k + dead_k) = degree * live. A search for invariants should rediscover
them before anything it reports about a rule is believed; anything beyond them is the rule's own.
"""

from __future__ import annotations

import numpy as np

from latticelife.config import Config
from latticelife.core.rules import Rule
from latticelife.experiments.harness import single_species_session


def _neighbour_counts(occupied: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    padded = np.append(occupied, False)
    gather = np.where(adjacency >= 0, adjacency, len(occupied))
    return padded[gather].sum(axis=1)


def count_series(
    config: Config,
    rule: Rule | str,
    shape: str,
    grid: int | str,
    density: float,
    seed: int,
    steps: int,
) -> dict[str, list[int]]:
    """The counts above for `rule` on a `grid` torus of `shape`, seeded at `density` from `seed`."""
    if steps < 2:
        raise ValueError(f"steps must be at least 2 (got {steps})")
    rule = Rule.from_bs_string(rule) if isinstance(rule, str) else rule
    session = single_species_session(config, rule, shape, grid, density, seed)
    adjacency = session.grid.adjacency
    degree = adjacency.shape[1]
    columns: dict[str, list[int]] = {"live": [], "births": [], "deaths": []}
    for k in range(degree + 1):
        columns[f"live_{k}"] = []
        columns[f"dead_{k}"] = []
    births = deaths = 0
    previous = None
    for step in range(steps):
        if step:
            session.step()
        occupied = session.grid.occupancy > 0
        if previous is not None:
            births += int(np.count_nonzero(occupied & ~previous))
            deaths += int(np.count_nonzero(previous & ~occupied))
        previous = occupied
        counts = _neighbour_counts(occupied, adjacency)
        columns["live"].append(int(np.count_nonzero(occupied)))
        columns["births"].append(births)
        columns["deaths"].append(deaths)
        for k in range(degree + 1):
            columns[f"live_{k}"].append(int(np.count_nonzero(occupied & (counts == k))))
            columns[f"dead_{k}"].append(int(np.count_nonzero(~occupied & (counts == k))))
    return columns


__all__ = ["count_series"]
