"""Real metric implementations (closes D-2 — the legacy `lifeform_metrics = {lf.id: {} ...}`
placeholder that left all committed advanced-metric columns zero-filled).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latticelife.core.lattice import LatticeGrid
from latticelife.core.results import UpdateResult


@dataclass(frozen=True)
class GridMetrics:
    total_alive: int
    changed_cells: int
    volatility: float
    entropy: float


@dataclass(frozen=True)
class LifeformMetrics:
    alive: int
    static: int
    births: int
    deaths: int
    kills: int
    dominance: float
    growth_rate: float
    mobility: float
    avg_lifespan: float


@dataclass(frozen=True)
class ClusterMetrics:
    cluster_count: int
    avg_cluster_size: float
    max_cluster_size: int


def _entropy(alive_per_species: np.ndarray, total_alive: int) -> float:
    if total_alive == 0:
        return 0.0
    p = alive_per_species[alive_per_species > 0] / total_alive
    return float(-(p * np.log2(p)).sum())


def compute_grid_metrics(result: UpdateResult) -> GridMetrics:
    total_alive = int(result.alive[1:].sum())
    volatility = result.changed_cells / max(total_alive, 1) * 100
    entropy = _entropy(result.alive[1:], total_alive)
    return GridMetrics(
        total_alive=total_alive,
        changed_cells=result.changed_cells,
        volatility=volatility,
        entropy=entropy,
    )


def compute_lifeform_metrics(
    lifeform_id: int,
    result: UpdateResult,
    grid: LatticeGrid,
    total_alive: int,
    previous_alive: int,
) -> LifeformMetrics:
    alive = int(result.alive[lifeform_id])
    static = int(result.static[lifeform_id])
    births = int(result.births[lifeform_id])
    deaths = int(result.deaths[lifeform_id])
    kills = int(result.kills[lifeform_id])

    dominance = alive / max(total_alive, 1)
    growth_rate = (alive - previous_alive) / max(previous_alive, 1)
    mobility = (births + deaths) / max(alive, 1)

    mask = grid.occupancy == lifeform_id
    avg_lifespan = float(grid.lifespans[mask].mean()) if np.any(mask) else 0.0

    return LifeformMetrics(
        alive=alive,
        static=static,
        births=births,
        deaths=deaths,
        kills=kills,
        dominance=dominance,
        growth_rate=growth_rate,
        mobility=mobility,
        avg_lifespan=avg_lifespan,
    )


def compute_cluster_metrics(grid: LatticeGrid, lifeform_id: int) -> ClusterMetrics:
    """Connected components of `lifeform_id`'s cells via iterative min-label propagation over
    the adjacency array (§6.6): initialize labels = cell index on the species mask, repeatedly
    gather neighbor labels and take the elementwise min, until fixpoint. Geometry- and
    wrap-correct by construction for every shape — no scipy.ndimage.
    """
    mask = grid.occupancy == lifeform_id
    if not np.any(mask):
        return ClusterMetrics(cluster_count=0, avg_cluster_size=0.0, max_cluster_size=0)

    n = grid.n_cells
    infinity = np.iinfo(np.int64).max
    labels = np.where(mask, np.arange(n, dtype=np.int64), infinity)

    while True:
        padded = np.append(labels, infinity)
        gather_idx = np.where(grid.adjacency >= 0, grid.adjacency, n)
        neighbor_labels = padded[gather_idx]  # (n, degree); unmasked/-1 slots read `infinity`
        candidate = np.minimum(labels, neighbor_labels.min(axis=1))
        candidate = np.where(mask, candidate, infinity)
        if np.array_equal(candidate, labels):
            break
        labels = candidate

    _, counts = np.unique(labels[mask], return_counts=True)
    return ClusterMetrics(
        cluster_count=len(counts),
        avg_cluster_size=float(counts.mean()),
        max_cluster_size=int(counts.max()),
    )
