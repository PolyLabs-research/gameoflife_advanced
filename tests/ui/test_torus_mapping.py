"""Grid -> torus mapping (§6.14) in numpy: cell colours (F37), which cell each surface point
shows and its orientation (F35), and the mesh. No GL context needed."""

import numpy as np
import pytest

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


@pytest.mark.parametrize("shape, width, height", WRAPPED_SIZES)
def test_every_cell_appears_on_the_surface(shape, width, height):
    u, v = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 400))
    rows, cols = surface_cells(shape, width, height, u, v)
    assert set(zip(rows.ravel().tolist(), cols.ravel().tolist(), strict=True)) == {
        (r, c) for r in range(height) for c in range(width)
    }


@pytest.mark.parametrize("shape, width, height", WRAPPED_SIZES)
def test_a_wrapped_grid_meets_itself_at_the_seams(shape, width, height):
    rng = np.random.default_rng(1)
    u, v = rng.uniform(0, 1, 2000), rng.uniform(0, 1, 2000)
    here = surface_cells(shape, width, height, u, v)
    for du, dv in ((1, 0), (0, 1), (-1, -1)):
        there = surface_cells(shape, width, height, u + du, v + dv)
        np.testing.assert_array_equal(here[0], there[0])
        np.testing.assert_array_equal(here[1], there[1])


@pytest.mark.parametrize("shape, width, height", WRAPPED_SIZES)
def test_surface_orientation_matches_the_2d_view(shape, width, height):
    """F35: on the outer wall facing the camera (u = 0, v rising up the wall), rows must run
    downward and columns to the right, as in 2D; the torus showed the grid mirrored."""
    v = np.linspace(-0.3, 0.3, 61)
    rows_up_the_wall = lattice_point(shape, width, height, np.zeros_like(v), v)[1]
    assert np.all(np.diff(rows_up_the_wall) < 0)  # higher on screen = earlier row
    u = np.linspace(-0.3, 0.3, 61)
    cols_rightwards = lattice_point(shape, width, height, u, np.zeros_like(u))[0]
    assert np.all(np.diff(cols_rightwards) > 0)


def test_the_middle_of_the_grid_faces_the_camera():
    rows, cols = surface_cells("square", 10, 8, 0.001, -0.001)
    assert (int(rows), int(cols)) == (4, 5)


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
    """The orientation contract `lattice_point` relies on: at theta = 0 (u = 0, the side a
    camera at azimuth 0 sees), v increases with height on the outer wall (cos phi > 0)."""
    vertices, _ = generate_torus_mesh(1.0, 0.4, 16, 32)
    front = (vertices["in_uv"][:, 0] == 0) & (vertices["in_normal"][:, 0] > 0.1)
    v = vertices["in_uv"][front, 1]
    z = vertices["in_position"][front, 2]
    wall = v < 0.25  # from the outer equator up to the top of the tube
    assert np.all(np.diff(z[wall][np.argsort(v[wall])]) > 0)
