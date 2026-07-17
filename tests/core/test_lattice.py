import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from latticelife.core.lattice import LatticeGrid, clone_grid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule

CONWAY = Rule(birth={3}, survive={2, 3})

# shape, triangle_mode, wrap-mode degree — mirrors tests/core/test_geometries.py's table.
SHAPES = [
    ("square", "edge+vertex", 8),
    ("hexagon", "edge+vertex", 6),
    ("triangle", "edge", 3),
    ("triangle", "edge+vertex", 12),
]


def test_empty_grid_is_a_fixpoint():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 10, 10, wrap=True)
    rng = np.random.default_rng(0)

    result = grid.step(rng)

    assert result.changed_cells == 0
    assert np.all(grid.occupancy == 0)
    assert np.all(grid.lifespans == 0)
    assert result.alive[1] == 0
    assert result.births[1] == 0
    assert result.deaths[1] == 0


def _run_trajectory(seed: int, steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    parent = np.random.default_rng(seed)
    init_rng, step_rng = parent.spawn(2)

    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 12, 12, wrap=True)
    alive = init_rng.random(grid.n_cells) < 0.4
    grid.occupancy[alive] = 1
    grid.lifespans[alive] = 1

    for _ in range(steps):
        grid.step(step_rng)

    return grid.occupancy.copy(), grid.lifespans.copy()


def test_single_species_determinism_same_seed_identical_trajectory():
    occ_a, life_a = _run_trajectory(seed=42)
    occ_b, life_b = _run_trajectory(seed=42)

    assert np.array_equal(occ_a, occ_b)
    assert np.array_equal(life_a, life_b)


def test_lifespans_increase_under_survival_and_dtype_is_int32():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 10, 10, wrap=True)
    assert grid.lifespans.dtype == np.int32

    block = [(4, 4), (4, 5), (5, 4), (5, 5)]  # still-life under B3/S23
    for r, c in block:
        grid.occupancy[r * grid.width + c] = 1
        grid.lifespans[r * grid.width + c] = 1

    rng = np.random.default_rng(0)
    for gen in range(1, 51):
        result = grid.step(rng)
        assert result.changed_cells == 0
        for r, c in block:
            assert grid.lifespans[r * grid.width + c] == gen + 1

    assert grid.lifespans.dtype == np.int32


def test_glider_translates_by_1_1_with_period_4_on_wrapped_square():
    width = height = 20
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", width, height, wrap=True)

    glider = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    origin_r, origin_c = 5, 5
    for dr, dc in glider:
        r, c = origin_r + dr, origin_c + dc
        grid.occupancy[r * width + c] = 1
        grid.lifespans[r * width + c] = 1

    rng = np.random.default_rng(0)
    for _ in range(4):
        grid.step(rng)

    expected = {
        ((origin_r + 1 + dr) % height) * width + ((origin_c + 1 + dc) % width)
        for dr, dc in glider
    }
    alive_cells = set(np.nonzero(grid.occupancy == 1)[0].tolist())
    assert alive_cells == expected


@pytest.mark.parametrize(("shape", "triangle_mode", "degree"), SHAPES)
def test_rule_values_beyond_max_degree_are_unreachable(shape, triangle_mode, degree):
    """A survive count one past the geometry's max degree can never be satisfied — neighbor
    counts are bounded by adjacency degree, so out-of-range rule values are naturally clamped
    without any explicit clamping code."""
    unreachable_rule = Rule(birth=set(), survive={degree + 1})
    grid = LatticeGrid(
        [Lifeform(1, unreachable_rule)],
        shape,
        width=8,
        height=8,
        wrap=True,
        triangle_mode=triangle_mode,
    )
    grid.occupancy[:] = 1
    grid.lifespans[:] = 1

    rng = np.random.default_rng(0)
    result = grid.step(rng)

    assert result.alive[1] == 0


@given(st.integers(min_value=0, max_value=1000))
def test_determinism_property_across_seeds(seed):
    occ_a, life_a = _run_trajectory(seed=seed, steps=5)
    occ_b, life_b = _run_trajectory(seed=seed, steps=5)
    assert np.array_equal(occ_a, occ_b)
    assert np.array_equal(life_a, life_b)


def test_clone_grid_shares_adjacency_and_copies_state():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 6, 6, wrap=True)
    grid.occupancy[0] = 1
    grid.lifespans[0] = 5

    twin = clone_grid(grid)

    assert twin.adjacency is grid.adjacency  # shared read-only, never rebuilt or deep-copied
    assert twin.occupancy is not grid.occupancy
    assert twin.lifespans is not grid.lifespans
    assert np.array_equal(twin.occupancy, grid.occupancy)
    assert np.array_equal(twin.lifespans, grid.lifespans)

    twin.occupancy[1] = 1  # mutating the twin must not affect the original
    assert grid.occupancy[1] == 0
