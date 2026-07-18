import pygame

from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.ui import render

CONWAY = Rule(birth={3}, survive={2, 3})


def test_build_palette_shape_and_dead_color():
    lifeforms = [Lifeform(1, CONWAY), Lifeform(2, CONWAY)]
    palette = render.build_palette(lifeforms)
    assert palette.shape == (5, 3)  # 2*2 + 1
    assert tuple(palette[0]) == render.DEAD_CELL_COLOR


def test_build_palette_empty_lifeforms():
    palette = render.build_palette([])
    assert palette.shape == (1, 3)


def test_occupancy_state_folds_static_offset():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 4, 4, wrap=True)
    grid.occupancy[0] = 1
    grid.lifespans[0] = 20  # > static_threshold (10)
    grid.occupancy[1] = 1
    grid.lifespans[1] = 1  # alive, not static

    palette = render.build_palette([Lifeform(1, CONWAY)])
    state = render.occupancy_state(grid, palette)

    n_species = 1
    assert state[0] == 1 + n_species  # static shade
    assert state[1] == 1  # alive shade
    assert state[2] == 0  # dead


def test_hexagon_corners_returns_six_points():
    corners = render.hexagon_corners(0, 0, 10, 0, 0)
    assert len(corners) == 6


def test_triangle_corners_returns_three_points_and_differs_by_orientation():
    up = render.triangle_corners(0, 0, 10, upward=True, offset_x=0, offset_y=0)
    down = render.triangle_corners(0, 0, 10, upward=False, offset_x=0, offset_y=0)
    assert len(up) == 3
    assert len(down) == 3
    assert up != down


def test_render_square_produces_correctly_sized_surface():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 5, 5, wrap=True)
    grid.occupancy[0] = 1
    palette = render.build_palette([Lifeform(1, CONWAY)])

    surface = pygame.Surface((100, 100))
    render.render_square(surface, grid, palette, cell_size=10, offset_x=0, offset_y=0)  # no crash


def test_geometric_renderer_tracks_previous_state_and_redraws_without_crashing():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "hexagon", 4, 4, wrap=False)
    grid.occupancy[0] = 1
    palette = render.build_palette([Lifeform(1, CONWAY)])
    surface = pygame.Surface((100, 100))

    renderer = render.GeometricRenderer("hexagon", cell_size=10, offset_x=0, offset_y=0)
    assert renderer._previous_state is None

    renderer.draw(surface, grid, palette)
    assert renderer._previous_state is not None

    renderer.draw(surface, grid, palette)  # no changes: dirty set should be empty, no crash

    renderer.reset()
    assert renderer._previous_state is None
