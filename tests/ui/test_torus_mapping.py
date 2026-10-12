"""Grid -> torus mapping (§6.14) in numpy: cell colours (F37), which cell each surface point
shows, its orientation (F35) and its neighbours, and the mesh. No GL context needed."""

import numpy as np
import pytest

from latticelife.core.geometries import build_adjacency
from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.ui import render
from latticelife.ui.torus.geometry import generate_torus_mesh
from latticelife.ui.torus.texture_mapper import (
    cell_colors,
    lattice_cell,
    lattice_point,
    surface_cells,
)

CONWAY = Rule(birth={3}, survive={2, 3})
# (shape, width, height) that tile when wrapped.
WRAPPED_SIZES = [("square", 7, 5), ("hexagon", 8, 5), ("triangle", 8, 6)]
# Bounded grids may have any size; odd ones have no period that keeps hex column or triangle
# up/down parity, so a cell drawn a grid width or height away from its place loses its shape.
BOUNDED_SIZES = [
    ("square", 7, 5),
    ("hexagon", 7, 5),
    ("hexagon", 9, 6),
    ("hexagon", 8, 5),
    ("triangle", 7, 6),
    ("triangle", 8, 5),
    ("triangle", 7, 5),
]
ALL_SIZES = [(*size, True) for size in WRAPPED_SIZES] + [(*size, False) for size in BOUNDED_SIZES]


def _lifeforms(n: int) -> list[Lifeform]:
    return [Lifeform(id=i, rule=CONWAY) for i in range(1, n + 1)]


def test_cell_colors_use_the_grids_static_threshold_like_the_2d_view():
    """F37: the torus folded alive/static with a hard-coded threshold of 10."""
    lifeforms = _lifeforms(1)
    grid = LatticeGrid(lifeforms, "square", 4, 1, False, static_threshold=50)
    grid.occupancy[:] = 1
    grid.lifespans[:] = [5, 20, 40, 60]
    palette = render.build_palette(lifeforms)

    colors = cell_colors(grid, palette)

    expected = palette[render.occupancy_state(grid, palette)].reshape(1, 4, 3)
    np.testing.assert_array_equal(colors, expected)
    alive, static = palette[1], palette[2]
    np.testing.assert_array_equal(colors[0], [alive, alive, alive, static])


def test_cell_colors_are_laid_out_row_by_row():
    lifeforms = _lifeforms(2)
    grid = LatticeGrid(lifeforms, "hexagon", 4, 3, True)
    grid.occupancy[grid.width * 2 + 1] = 2  # row 2, column 1
    palette = render.build_palette(lifeforms)
    colors = cell_colors(grid, palette)
    assert colors.shape == (3, 4, 3)
    np.testing.assert_array_equal(colors[2, 1], palette[2])
    assert (colors.reshape(-1, 3) == palette[0]).all(axis=1).sum() == grid.n_cells - 1


def _random_points(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-6, 30, n), rng.uniform(-6, 30, n)


def test_hexagon_lookup_matches_the_2d_picking():
    xs, ys = _random_points()
    rows, cols = lattice_cell("hexagon", xs, ys)
    expected = [render._hex_at(x, y, 1.0) for x, y in zip(xs, ys, strict=True)]
    assert list(zip(rows.tolist(), cols.tolist(), strict=True)) == expected


def test_triangle_lookup_matches_the_2d_picking():
    # Lattice x is in half cells and y in rows; a 2D triangle of size 2 has those units as px
    # across and sqrt(3) px down.
    xs, ys = _random_points()
    rows, cols = lattice_cell("triangle", xs, ys)
    expected = [render._triangle_at(x, y * np.sqrt(3), 2.0) for x, y in zip(xs, ys, strict=True)]
    assert list(zip(rows.tolist(), cols.tolist(), strict=True)) == expected


def _surface_samples(shape, width, height, wrap, n=900):
    """Flat cell index (-1 for none) at the centres of an n x n grid of surface points; sample
    [i, j] is at u = (j + 0.5) / n, v = (i + 0.5) / n."""
    centres = (np.arange(n) + 0.5) / n
    u, v = np.meshgrid(centres, centres)
    rows, cols = surface_cells(shape, width, height, u, v, wrap=wrap)
    return np.where(rows >= 0, rows * width + cols, -1)


def _touching(cells: np.ndarray, wrap: bool) -> set[tuple[int, int]]:
    """Pairs of different cells shown in neighbouring samples. Across the seams only when the
    grid wraps: a bounded grid's opposite edges meet there on the closed surface, but nothing
    joins them, which the background notches make plain."""
    pairs = set()
    for axis in (0, 1):
        a, b = cells, np.roll(cells, -1, axis=axis)
        if not wrap:
            a, b = (a[:-1], b[:-1]) if axis == 0 else (a[:, :-1], b[:, :-1])
        shown = (a != b) & (a >= 0) & (b >= 0)
        low, high = np.minimum(a, b)[shown], np.maximum(a, b)[shown]
        pairs |= set(zip(low.tolist(), high.tolist(), strict=True))
    return pairs


def _pairs(adjacency: np.ndarray) -> set[tuple[int, int]]:
    """Every neighbour pair of an adjacency array, smaller cell first."""
    rows = enumerate(adjacency.tolist())
    return {(min(i, j), max(i, j)) for i, row in rows for j in row if j >= 0}


def _square_edge_pairs(width: int, height: int, wrap: bool) -> set[tuple[int, int]]:
    """Square cells that share an edge (the 4 of the 8 Moore neighbours left, right, up, down)."""
    pairs = set()
    for r in range(height):
        for c in range(width):
            for r2, c2 in ((r, c + 1), (r + 1, c)):
                if wrap:
                    r2, c2 = r2 % height, c2 % width
                elif r2 == height or c2 == width:
                    continue
                i, j = r * width + c, r2 * width + c2
                pairs.add((min(i, j), max(i, j)))
    return pairs


@pytest.mark.parametrize("shape, width, height, wrap", ALL_SIZES)
def test_every_cell_appears_on_the_surface(shape, width, height, wrap):
    cells = _surface_samples(shape, width, height, wrap, n=400)
    assert set(cells.ravel().tolist()) - {-1} == set(range(width * height))


@pytest.mark.parametrize("shape, width, height, wrap", ALL_SIZES)
def test_cells_touch_on_the_surface_only_where_they_are_neighbours(shape, width, height, wrap):
    """The 3D view folded lattice cells onto the grid modulo its width and height from the grid's
    middle, so on a bounded hexagon grid of odd width half the columns had their half-cell
    shift inverted (triangles of odd width or height: up and down swapped), and the surface
    showed cells side by side that are not neighbours, and real neighbours apart."""
    cells = _surface_samples(shape, width, height, wrap)
    touching = _touching(cells, wrap)
    assert touching <= _pairs(build_adjacency(shape, width, height, wrap))
    # ... and every pair that shares an edge in the grid shares one on the surface.
    if shape == "square":
        edge_pairs = _square_edge_pairs(width, height, wrap)
    else:
        edge_pairs = _pairs(build_adjacency(shape, width, height, wrap, "edge"))
    assert edge_pairs <= touching


@pytest.mark.parametrize("shape, width, height", [s for s in BOUNDED_SIZES if s[0] != "square"])
def test_a_bounded_grid_shows_every_cell_whole_with_the_background_beyond(shape, width, height):
    """A bounded grid is drawn like its 2D footprint: no edge cell is cut by a seam (all cells
    cover the same area), and the notches between edge hexagons or triangles show no cell."""
    cells = _surface_samples(shape, width, height, False)
    areas = np.bincount(cells[cells >= 0], minlength=width * height)
    assert areas.max() / areas.min() < 1.03
    assert 0.05 < (cells < 0).mean() < 0.2


@pytest.mark.parametrize("shape, width, height", WRAPPED_SIZES)
def test_a_wrapped_grid_meets_itself_at_the_seams(shape, width, height):
    rng = np.random.default_rng(1)
    u, v = rng.uniform(0, 1, 2000), rng.uniform(0, 1, 2000)
    here = surface_cells(shape, width, height, u, v, wrap=True)
    for du, dv in ((1, 0), (0, 1), (-1, -1)):
        there = surface_cells(shape, width, height, u + du, v + dv, wrap=True)
        np.testing.assert_array_equal(here[0], there[0])
        np.testing.assert_array_equal(here[1], there[1])


@pytest.mark.parametrize("shape, width, height, wrap", ALL_SIZES)
def test_surface_orientation_matches_the_2d_view(shape, width, height, wrap):
    """F35: on the outer wall facing the camera (u = 0.5, v rising up the wall from 0.5), rows
    must run downward and columns to the right, as in 2D; the torus showed the grid mirrored."""
    v = np.linspace(0.2, 0.8, 61)
    rows_up_the_wall = lattice_point(shape, width, height, np.full_like(v, 0.5), v, wrap=wrap)[1]
    assert np.all(np.diff(rows_up_the_wall) < 0)  # higher on screen = earlier row
    u = np.linspace(0.2, 0.8, 61)
    cols_rightwards = lattice_point(shape, width, height, u, np.full_like(u, 0.5), wrap=wrap)[0]
    assert np.all(np.diff(cols_rightwards) > 0)


@pytest.mark.parametrize(
    "shape, width, height, wrap, middle",
    [
        ("square", 10, 8, True, (4, 5)),
        ("square", 10, 8, False, (4, 5)),
        ("hexagon", 7, 5, False, (2, 3)),
        ("hexagon", 9, 5, False, (2, 4)),
        ("triangle", 7, 5, False, (2, 3)),
    ],
)
def test_the_middle_of_the_grid_faces_the_camera(shape, width, height, wrap, middle):
    rows, cols = surface_cells(shape, width, height, 0.501, 0.499, wrap=wrap)
    assert (int(rows), int(cols)) == middle


def test_mesh_lies_on_the_torus_with_unit_normals_and_uv_in_range():
    R, r = 1.0, 0.4
    vertices, indices = generate_torus_mesh(R, r, 16, 8)
    assert len(vertices) == 17 * 9
    assert indices.dtype == np.uint32 and len(indices) == 16 * 8 * 6
    assert indices.max() < len(vertices)
    pos = vertices["in_position"].astype(np.float64)
    ring = np.hypot(pos[:, 0], pos[:, 1]) - R
    np.testing.assert_allclose(ring**2 + pos[:, 2] ** 2, r**2, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(vertices["in_normal"], axis=1), 1.0, atol=1e-6)
    uv = vertices["in_uv"]
    assert uv.min() == 0.0 and uv.max() == 1.0


def test_v_rises_up_the_camera_facing_outer_wall():
    """The orientation contract `lattice_point` relies on: at theta = 0 (u = 0.5, the side a
    camera at azimuth 0 sees), v increases with height on the outer wall (cos phi > 0)."""
    vertices, _ = generate_torus_mesh(1.0, 0.4, 16, 32)
    front = (vertices["in_uv"][:, 0] == 0.5) & (vertices["in_normal"][:, 0] > 0.1)
    v = vertices["in_uv"][front, 1]
    z = vertices["in_position"][front, 2]
    wall = (v >= 0.5) & (v < 0.75)  # from the outer equator up to the top of the tube
    assert wall.sum() > 4
    assert np.all(np.diff(z[wall][np.argsort(v[wall])]) > 0)
    middle = (vertices["in_uv"] == 0.5).all(axis=1)  # the middle of the grid: facing +x
    np.testing.assert_allclose(vertices["in_position"][middle], [[1.4, 0.0, 0.0]], atol=1e-6)


def test_the_mesh_seams_are_at_the_back_of_the_ring_and_on_the_inner_equator():
    """Where uv starts over (u = 0 / 1, v = 0 / 1) is where the shader puts the grid's edges, so
    those must be the places a camera at azimuth 0 sees least of: the far side of the ring, and
    the inside of the tube."""
    R, r = 1.0, 0.4
    vertices, _ = generate_torus_mesh(R, r, 16, 8)
    pos = vertices["in_position"].astype(np.float64)
    u, v = vertices["in_uv"][:, 0], vertices["in_uv"][:, 1]
    at_u_seam = (u == 0) | (u == 1)
    np.testing.assert_allclose(pos[at_u_seam, 1], 0.0, atol=1e-6)
    assert np.all(pos[at_u_seam, 0] < 0)  # theta = pi: the back of the ring
    at_v_seam = (v == 0) | (v == 1)
    np.testing.assert_allclose(np.hypot(pos[at_v_seam, 0], pos[at_v_seam, 1]), R - r, atol=1e-6)
    np.testing.assert_allclose(pos[at_v_seam, 2], 0.0, atol=1e-6)  # phi = pi: inner equator
