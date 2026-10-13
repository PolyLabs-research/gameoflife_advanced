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


def component_roots(adjacency: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Connected components of the cells `cells` (flat indices) under `adjacency`, with every
    adjacency entry treated as an undirected edge and edges to cells outside `cells` ignored.
    Returns, for each position p in `cells`, the position of its component's representative
    (the smallest position in that component), so equal values mean the same component.

    Union-find on the edge list, vectorised (the §6.6 "union-find alternative"; the
    Shiloach-Vishkin hook-and-shortcut scheme): every tree root linked by an edge to a tree with
    a smaller root is hooked onto that root, then pointer jumping flattens every tree, and edges
    found inside one tree are dropped for good. Plain min-label propagation needed one
    full-grid pass per step along the longest path inside a cluster (seconds on a winding
    500x500 cluster); this needs a handful of passes over the edges.
    """
    k = cells.size
    if k == 0:
        return np.zeros(0, dtype=np.intp)
    # Position of every cell in `cells`, -1 elsewhere; the extra last slot makes an adjacency
    # pad of -1 read -1 too.
    position = np.full(adjacency.shape[0] + 1, -1, dtype=np.intp)
    position[cells] = np.arange(k)
    neighbor_positions = position[adjacency[cells]]  # (k, degree)
    linked = neighbor_positions >= 0
    u = np.nonzero(linked)[0]
    v = neighbor_positions[linked]

    parent = np.arange(k)
    while u.size:
        root_u, root_v = parent[u], parent[v]  # every parent is a root after the shortcut
        between = root_u != root_v
        u, v = u[between], v[between]
        if not u.size:
            break
        root_u, root_v = root_u[between], root_v[between]
        # Hook the larger root onto the smaller. Several edges may hook one root; whichever
        # write lands is fine, since every candidate is a smaller root of the same component.
        parent[np.maximum(root_u, root_v)] = np.minimum(root_u, root_v)
        while True:  # pointer jumping until every cell points straight at its root
            grandparent = parent[parent]
            if np.array_equal(grandparent, parent):
                break
            parent = grandparent
    return parent


def compute_cluster_metrics(grid: LatticeGrid, lifeform_id: int) -> ClusterMetrics:
    """Connected components of `lifeform_id`'s cells over the grid's adjacency array (§6.6),
    via `component_roots`. Geometry- and wrap-correct by construction for every shape (no
    scipy.ndimage).
    """
    cells = np.flatnonzero(grid.occupancy == lifeform_id)
    if cells.size == 0:
        return ClusterMetrics(cluster_count=0, avg_cluster_size=0.0, max_cluster_size=0)

    roots = component_roots(grid.adjacency, cells)
    sizes = np.bincount(roots)
    sizes = sizes[sizes > 0]
    return ClusterMetrics(
        cluster_count=len(sizes),
        avg_cluster_size=float(sizes.mean()),
        max_cluster_size=int(sizes.max()),
    )
