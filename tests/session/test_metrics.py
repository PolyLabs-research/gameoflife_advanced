import time
from collections import deque

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.results import UpdateResult
from latticelife.core.rules import Rule
from latticelife.session.metrics import (
    component_roots,
    compute_cluster_metrics,
    compute_grid_metrics,
    compute_lifeform_metrics,
)

CONWAY = Rule(birth={3}, survive={2, 3})


def _result(alive, births=None, deaths=None, kills=None, static=None, changed_cells=0):
    size = len(alive)

    def arr(values):
        if values is None:
            return np.zeros(size, dtype=np.int64)
        return np.array(values, dtype=np.int64)

    return UpdateResult(
        births=arr(births),
        deaths=arr(deaths),
        kills=arr(kills),
        alive=np.array(alive, dtype=np.int64),
        static=arr(static),
        changed_cells=changed_cells,
    )


def test_entropy_two_equal_species_is_one_bit():
    metrics = compute_grid_metrics(_result(alive=[0, 50, 50], changed_cells=10))
    assert metrics.total_alive == 100
    assert metrics.entropy == pytest.approx(1.0)
    assert metrics.volatility == pytest.approx(10.0)


def test_entropy_single_species_is_zero():
    metrics = compute_grid_metrics(_result(alive=[0, 100], changed_cells=0))
    assert metrics.entropy == 0.0


def test_entropy_empty_grid_is_zero():
    metrics = compute_grid_metrics(_result(alive=[0, 0], changed_cells=0))
    assert metrics.total_alive == 0
    assert metrics.entropy == 0.0
    assert metrics.volatility == 0.0


def test_lifeform_metrics_formulas():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 4, 4, wrap=True)
    grid.occupancy[0] = 1
    grid.occupancy[1] = 1
    grid.lifespans[0] = 4
    grid.lifespans[1] = 6

    result = _result(alive=[0, 2], births=[0, 1], deaths=[0, 1], static=[0, 1])
    metrics = compute_lifeform_metrics(
        lifeform_id=1, result=result, grid=grid, total_alive=2, previous_alive=1
    )

    assert metrics.alive == 2
    assert metrics.static == 1
    assert metrics.dominance == pytest.approx(1.0)
    assert metrics.growth_rate == pytest.approx(1.0)
    assert metrics.mobility == pytest.approx(1.0)
    assert metrics.avg_lifespan == pytest.approx(5.0)


def test_lifeform_metrics_no_alive_cells_uses_safe_denominators():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 4, 4, wrap=True)
    result = _result(alive=[0, 0])
    metrics = compute_lifeform_metrics(
        lifeform_id=1, result=result, grid=grid, total_alive=0, previous_alive=0
    )

    assert metrics.dominance == 0.0
    assert metrics.growth_rate == 0.0
    assert metrics.mobility == 0.0
    assert metrics.avg_lifespan == 0.0


def test_cluster_metrics_two_cluster_hex_fixture():
    """Hand-built two-cluster hex fixture: a 4x4 non-wrapped odd-q hex grid. (0,0)/(0,1) are
    same-row neighbors (row offsets (0,-1)/(0,1) are in both parity offset sets), forming a
    size-2 cluster. (3,3)'s only in-bounds neighbors without wrap are (2,3) and (3,2), both dead,
    so it's an isolated size-1 cluster."""
    grid = LatticeGrid([Lifeform(1, CONWAY)], "hexagon", width=4, height=4, wrap=False)
    for idx in (0, 1):  # (0,0), (0,1)
        grid.occupancy[idx] = 1
        grid.lifespans[idx] = 1
    isolated = 3 * 4 + 3  # (3,3)
    grid.occupancy[isolated] = 1
    grid.lifespans[isolated] = 1

    metrics = compute_cluster_metrics(grid, lifeform_id=1)

    assert metrics.cluster_count == 2
    assert metrics.max_cluster_size == 2
    assert metrics.avg_cluster_size == pytest.approx(1.5)


def test_cluster_metrics_no_alive_cells_is_zero():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 5, 5, wrap=True)
    metrics = compute_cluster_metrics(grid, lifeform_id=1)
    assert metrics.cluster_count == 0
    assert metrics.avg_cluster_size == 0.0
    assert metrics.max_cluster_size == 0


def test_cluster_metrics_single_connected_blob():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 5, 5, wrap=True)
    for r, c in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        grid.occupancy[r * 5 + c] = 1
        grid.lifespans[r * 5 + c] = 1

    metrics = compute_cluster_metrics(grid, lifeform_id=1)

    assert metrics.cluster_count == 1
    assert metrics.max_cluster_size == 4
    assert metrics.avg_cluster_size == pytest.approx(4.0)


# --- clustering vs a breadth-first reference (F13) ---------------------------------------------

SHAPES = [
    ("square", "edge+vertex"),
    ("hexagon", "edge+vertex"),
    ("triangle", "edge"),
    ("triangle", "edge+vertex"),
]


def _bfs_components(adjacency: np.ndarray, mask: np.ndarray) -> list[set[int]]:
    """Plain BFS over the adjacency read as an undirected graph (an edge listed by either end
    counts), restricted to `mask`."""
    links: dict[int, set[int]] = {i: set() for i in np.flatnonzero(mask).tolist()}
    for i in links:
        for j in adjacency[i].tolist():
            if j >= 0 and j in links and j != i:
                links[i].add(j)
                links[j].add(i)
    seen: set[int] = set()
    components = []
    for start in links:
        if start in seen:
            continue
        component, queue = {start}, deque([start])
        seen.add(start)
        while queue:
            for j in links[queue.popleft()]:
                if j not in seen:
                    seen.add(j)
                    component.add(j)
                    queue.append(j)
        components.append(component)
    return components


def _grid_with(shape, triangle_mode, width, height, wrap, occupancy) -> LatticeGrid:
    grid = LatticeGrid([Lifeform(1, CONWAY)], shape, width, height, wrap, triangle_mode)
    grid.occupancy[:] = occupancy
    return grid


@settings(max_examples=150, deadline=None)
@given(
    st.sampled_from(SHAPES),
    st.booleans(),
    st.integers(min_value=3, max_value=12).map(lambda half: 2 * half),
    st.integers(min_value=2, max_value=12).map(lambda half: 2 * half),
    st.floats(min_value=0.05, max_value=0.95),
    st.integers(min_value=0, max_value=2**32 - 1),
)
def test_clusters_match_breadth_first_search(shape_mode, wrap, width, height, density, seed):
    shape, triangle_mode = shape_mode
    occupancy = (np.random.default_rng(seed).random(width * height) < density).astype(np.int8)
    grid = _grid_with(shape, triangle_mode, width, height, wrap, occupancy)
    mask = grid.occupancy == 1
    expected = _bfs_components(grid.adjacency, mask)

    cells = np.flatnonzero(mask)
    roots = component_roots(grid.adjacency, cells)
    found: dict[int, set[int]] = {}
    for cell, root in zip(cells.tolist(), roots.tolist(), strict=True):
        found.setdefault(root, set()).add(cell)
    assert sorted(map(sorted, found.values())) == sorted(map(sorted, expected))

    metrics = compute_cluster_metrics(grid, 1)
    sizes = [len(component) for component in expected]
    assert metrics.cluster_count == len(sizes)
    assert metrics.max_cluster_size == (max(sizes) if sizes else 0)
    assert metrics.avg_cluster_size == (float(np.mean(sizes)) if sizes else 0.0)


def test_component_roots_ignores_cells_outside_the_set_and_handles_none():
    grid = _grid_with("square", "edge+vertex", 6, 6, True, 0)
    assert component_roots(grid.adjacency, np.array([], dtype=np.intp)).size == 0
    # Cells 0 and 2 are both next to cell 1, which is left out: two components.
    roots = component_roots(grid.adjacency, np.array([0, 2]))
    assert roots.tolist() == [0, 1]


def _serpentine(size: int) -> np.ndarray:
    """One path winding through every other row: a single cluster whose longest internal path
    is about size^2 / 2 cells, the worst case for label propagation."""
    occupancy = np.zeros((size, size), dtype=np.int8)
    occupancy[::2, :] = 1
    for r in range(1, size, 2):
        occupancy[r, size - 1 if (r // 2) % 2 == 0 else 0] = 1
    return occupancy.ravel()


@pytest.mark.parametrize(("shape", "triangle_mode"), SHAPES)
def test_winding_cluster_is_one_component_and_fast(shape, triangle_mode):
    """F13 regression: min-label propagation took ~23 s for this 200x200 unwrapped square case
    (one full-grid pass per cell along the path); union-find takes milliseconds."""
    grid = _grid_with(shape, triangle_mode, 200, 200, False, _serpentine(200))

    started = time.perf_counter()
    metrics = compute_cluster_metrics(grid, 1)
    elapsed = time.perf_counter() - started

    if shape != "triangle" or triangle_mode == "edge+vertex":  # rows connect through the turns
        assert metrics.cluster_count == 1
        assert metrics.max_cluster_size == int(_serpentine(200).sum())
    assert elapsed < 1.0
