import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from latticelife.core import geometries
from latticelife.core.geometries import (
    build_adjacency,
    max_degree,
    min_wrap_size,
    validate_dimensions,
)
from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule

# shape, triangle_mode, expected wrap-mode degree (§6.1 degree table)
SHAPES = [
    ("square", "edge+vertex", 8),
    ("hexagon", "edge+vertex", 6),
    ("triangle", "edge", 3),
    ("triangle", "edge+vertex", 12),
]


@pytest.mark.parametrize(("shape", "triangle_mode", "expected_degree"), SHAPES)
def test_max_degree_matches_table(shape, triangle_mode, expected_degree):
    assert max_degree(shape, triangle_mode) == expected_degree


@pytest.mark.parametrize(("shape", "triangle_mode", "expected_degree"), SHAPES)
def test_wrap_degree_matches_table_for_every_cell(shape, triangle_mode, expected_degree):
    adjacency = build_adjacency(shape, width=10, height=10, wrap=True, triangle_mode=triangle_mode)
    assert adjacency.shape[1] == expected_degree
    assert np.all((adjacency >= 0).sum(axis=1) == expected_degree)


@pytest.mark.parametrize(("shape", "triangle_mode", "expected_degree"), SHAPES)
def test_clamp_never_exceeds_wrap_degree(shape, triangle_mode, expected_degree):
    adjacency = build_adjacency(
        shape, width=10, height=10, wrap=False, triangle_mode=triangle_mode
    )
    assert adjacency.shape[1] == expected_degree
    assert np.all((adjacency >= 0).sum(axis=1) <= expected_degree)
    # Some interior cell should still reach full degree even without wrap.
    assert np.any((adjacency >= 0).sum(axis=1) == expected_degree)


def _assert_symmetric(shape: str, triangle_mode: str, wrap: bool, width: int, height: int) -> None:
    adjacency = build_adjacency(shape, width, height, wrap, triangle_mode)
    n = width * height
    for i in range(n):
        for slot in adjacency[i]:
            if slot < 0:
                continue
            j = int(slot)
            assert i in adjacency[j], f"{shape}/{triangle_mode} wrap={wrap}: {j}->{i} not mutual"


@pytest.mark.parametrize(("shape", "triangle_mode", "_deg"), SHAPES)
@pytest.mark.parametrize("wrap", [True, False])
def test_adjacency_symmetry(shape, triangle_mode, _deg, wrap):
    _assert_symmetric(shape, triangle_mode, wrap, width=8, height=8)


# Documented wrap constraints (module docstring of core.geometries): minimum size per shape, and
# which dimensions must be even. Written out independently of the implementation.
_MIN_WRAP = {
    ("square", "edge+vertex"): (3, 3),
    ("hexagon", "edge+vertex"): (4, 3),
    ("triangle", "edge"): (4, 2),
    ("triangle", "edge+vertex"): (6, 4),
}
_EVEN = {"square": (False, False), "hexagon": (True, False), "triangle": (True, True)}


def _documented_invalid(shape, triangle_mode, width, height, wrap) -> bool:
    if not wrap:
        return False
    even_w, even_h = _EVEN[shape]
    if (even_w and width % 2) or (even_h and height % 2):
        return True
    min_w, min_h = _MIN_WRAP[(shape, triangle_mode)]
    return width < min_w or height < min_h


@given(
    st.sampled_from(SHAPES),
    st.booleans(),
    st.integers(min_value=1, max_value=15),
    st.integers(min_value=1, max_value=15),
)
def test_adjacency_symmetry_property(shape_data, wrap, width, height):
    """F1/F11: any size either builds a symmetric adjacency with no self or repeated neighbours,
    or is rejected — and only the documented odd/too-small wrapped sizes are rejected."""
    shape, triangle_mode, _degree = shape_data
    invalid = _documented_invalid(shape, triangle_mode, width, height, wrap)
    if invalid:
        with pytest.raises(ValueError):
            build_adjacency(shape, width, height, wrap, triangle_mode)
        return
    _assert_symmetric(shape, triangle_mode, wrap, width, height)
    adjacency = build_adjacency(shape, width, height, wrap, triangle_mode)
    for i, row in enumerate(adjacency):
        neighbors = [int(j) for j in row if j >= 0]
        assert i not in neighbors
        assert len(set(neighbors)) == len(neighbors)


def _raw_neighbor_lists(shape, triangle_mode, width, height):
    """The wrapped neighbour lists build_adjacency would produce, without its validation."""
    if shape == "triangle":
        up, down = geometries._triangle_offsets(triangle_mode)

        def offset_fn(r, c):
            return up if (r + c) % 2 == 0 else down
    else:
        offset_fn = {"square": geometries._square_offsets, "hexagon": geometries._hex_offsets}[
            shape
        ]
    return geometries._neighbor_lists(width, height, True, offset_fn)


def _raw_defective(shape, triangle_mode, width, height) -> bool:
    lists = _raw_neighbor_lists(shape, triangle_mode, width, height)
    sets = [set(neighbors) for neighbors in lists]
    for i, neighbors in enumerate(lists):
        if i in neighbors or len(set(neighbors)) != len(neighbors):
            return True  # self-neighbour or a neighbour counted twice (F11)
        for j in neighbors:
            if i not in sets[j]:
                return True  # one-way link across the seam (F1)
            if shape == "triangle" and triangle_mode == "edge":
                ri, ci = divmod(i, width)
                rj, cj = divmod(j, width)
                if (ri + ci) % 2 == (rj + cj) % 2:
                    return True  # same-orientation "edge" neighbours (F1)
    return False


@pytest.mark.parametrize(("shape", "triangle_mode", "_deg"), SHAPES)
def test_validation_rejects_exactly_the_sizes_whose_wrapped_adjacency_is_broken(
    shape, triangle_mode, _deg
):
    for width in range(1, 13):
        for height in range(1, 13):
            try:
                validate_dimensions(shape, width, height, True, triangle_mode)
                rejected = False
            except ValueError:
                rejected = True
            assert rejected == _raw_defective(shape, triangle_mode, width, height), (
                width,
                height,
            )


@pytest.mark.parametrize(
    ("shape", "triangle_mode", "width", "height", "message"),
    [
        ("hexagon", "edge+vertex", 51, 50, "even width"),
        ("hexagon", "edge+vertex", 5, 5, "even width"),
        ("triangle", "edge+vertex", 51, 50, "even width and height"),
        ("triangle", "edge+vertex", 50, 51, "even width and height"),
        ("triangle", "edge", 51, 50, "even width and height"),
        ("square", "edge+vertex", 2, 5, "at least 3 x 3"),
        ("triangle", "edge+vertex", 4, 6, "at least 6 x 4"),
        ("square", "edge+vertex", 0, 5, "at least 1"),
    ],
)
def test_build_adjacency_rejects_wrapped_sizes_that_do_not_tile(
    shape, triangle_mode, width, height, message
):
    with pytest.raises(ValueError, match=message):
        build_adjacency(shape, width, height, True, triangle_mode)


def test_clamped_grids_accept_any_positive_size():
    for shape, triangle_mode, _deg in SHAPES:
        build_adjacency(shape, 51, 51, False, triangle_mode)
        build_adjacency(shape, 1, 1, False, triangle_mode)
    with pytest.raises(ValueError, match="at least 1"):
        build_adjacency("square", 0, 5, False)
    with pytest.raises(ValueError, match="at least 1"):
        build_adjacency("square", 5, -1, False)


def test_min_wrap_size_table():
    for (shape, triangle_mode), size in _MIN_WRAP.items():
        assert min_wrap_size(shape, triangle_mode) == size


@pytest.mark.parametrize("width", [4, 6, 8])
@pytest.mark.parametrize("height", [2, 4, 6])
def test_triangle_edge_neighbours_have_opposite_orientation(width, height):
    adjacency = build_adjacency("triangle", width, height, True, "edge")
    for i, row in enumerate(adjacency):
        ri, ci = divmod(i, width)
        for j in row:
            rj, cj = divmod(int(j), width)
            assert (ri + ci) % 2 != (rj + cj) % 2


def _shift(state: np.ndarray, width: int, height: int, dr: int, dc: int) -> np.ndarray:
    return np.roll(state.reshape(height, width), (dr, dc), axis=(0, 1)).reshape(-1)


@pytest.mark.parametrize(("shape", "triangle_mode", "_deg"), SHAPES)
@pytest.mark.parametrize(("dr", "dc"), [(0, 2), (2, 0)])
def test_wrapped_update_commutes_with_parity_preserving_translation(
    shape, triangle_mode, _deg, dr, dc
):
    """Translation invariance on the torus: every cell, seam or not, follows the same rule."""
    width, height = 12, 10
    lifeform = Lifeform(1, Rule(birth={2, 3}, survive={2, 3, 4}))
    rng = np.random.default_rng(7)
    start = (rng.random(width * height) < 0.4).astype(np.int8)

    def step(state):
        grid = LatticeGrid([lifeform], shape, width, height, True, triangle_mode)
        grid.occupancy = state.copy()
        grid.lifespans = state.astype(np.int32)
        grid.step(np.random.default_rng(0))
        return grid.occupancy

    shifted_then_stepped = step(_shift(start, width, height, dr, dc))
    stepped_then_shifted = _shift(step(start), width, height, dr, dc)
    assert np.array_equal(shifted_then_stepped, stepped_then_shifted)


# --- orientation pins (F62) --------------------------------------------------------------------
# Degree and symmetry checks pass for a mirrored layout too (even-q hexagons, or triangles
# pointing the other way), so these compare the adjacency with the documented drawing: odd-q
# hexagons (odd columns shifted down half a cell) and triangles pointing up where (r + c) is
# even. The layouts are written out here, independently of core.geometries and ui.render.


def _hex_centre(r: int, c: int) -> tuple[float, float]:
    return 1.5 * c, math.sqrt(3) * (r + 0.5 * (c % 2))  # unit circumradius


def _triangle_corners(r: int, c: int) -> set[tuple[float, float]]:
    x, y, h = c / 2, r * math.sqrt(3) / 2, math.sqrt(3) / 2  # unit side
    tip, base = (y - h / 2, y + h / 2) if (r + c) % 2 == 0 else (y + h / 2, y - h / 2)
    return {(round(px, 6), round(py, 6)) for px, py in ((x, tip), (x - 0.5, base), (x + 0.5, base))}


def _neighbour_sets(shape: str, triangle_mode: str, width: int, height: int) -> list[set[int]]:
    adjacency = build_adjacency(shape, width, height, False, triangle_mode)
    return [{int(j) for j in row if j >= 0} for row in adjacency]


def test_hexagon_adjacency_is_odd_q():
    width, height = 7, 6
    cells = [divmod(i, width) for i in range(width * height)]
    neighbour_sets = _neighbour_sets("hexagon", "edge+vertex", width, height)
    for i, neighbours in enumerate(neighbour_sets):
        xi, yi = _hex_centre(*cells[i])
        touching = {
            j
            for j, (r, c) in enumerate(cells)
            if j != i and math.isclose(math.dist((xi, yi), _hex_centre(r, c)), math.sqrt(3))
        }
        assert neighbours == touching, cells[i]

    # Spelled out: (r=0, c=1), an odd column, touches (1, 0) and (1, 2) diagonally below it;
    # (1, 2), an even column, touches (0, 1) and (0, 3) diagonally above it.
    assert {1 * width + 0, 1 * width + 2} <= neighbour_sets[0 * width + 1]
    assert {0 * width + 1, 0 * width + 3} <= neighbour_sets[1 * width + 2]


@pytest.mark.parametrize(("triangle_mode", "shared_corners"), [("edge", 2), ("edge+vertex", 1)])
def test_triangle_adjacency_matches_the_drawn_tiling(triangle_mode, shared_corners):
    width, height = 8, 6
    cells = [divmod(i, width) for i in range(width * height)]
    corners = [_triangle_corners(r, c) for r, c in cells]
    for i, neighbours in enumerate(_neighbour_sets("triangle", triangle_mode, width, height)):
        touching = {
            j
            for j in range(len(cells))
            if j != i and len(corners[i] & corners[j]) >= shared_corners
        }
        assert neighbours == touching, cells[i]
