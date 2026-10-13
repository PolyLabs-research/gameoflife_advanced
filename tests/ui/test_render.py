from dataclasses import replace

import numpy as np
import pygame
import pytest

from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.ui import render

CONWAY = Rule(birth={3}, survive={2, 3})
BACKGROUND = (18, 18, 24)
SHAPES = ["square", "hexagon", "triangle"]


def _grid(shape, width, height, fill=0.0, seed=0):
    grid = LatticeGrid([Lifeform(1, CONWAY)], shape, width, height, wrap=True)
    alive = np.random.default_rng(seed).random(grid.n_cells) < fill
    grid.occupancy[alive] = 1
    grid.lifespans[alive] = 1
    return grid


def test_build_palette_shape_and_dead_color():
    lifeforms = [Lifeform(1, CONWAY), Lifeform(2, CONWAY)]
    palette = render.build_palette(lifeforms)
    assert palette.shape == (5, 3)  # 2*2 + 1
    assert tuple(palette[0]) == render.DEAD_CELL_COLOR


def test_build_palette_empty_lifeforms():
    palette = render.build_palette([])
    assert palette.shape == (1, 3)


def test_ten_species_get_distinct_colours_and_an_eleventh_is_refused():
    """F49: species 11 used to silently reuse species 1's colours."""
    ten = [Lifeform(i, CONWAY) for i in range(1, 11)]
    palette = render.build_palette(ten)
    shades = {tuple(int(v) for v in row) for row in palette[1:]}
    assert len(shades) == 20  # 10 alive + 10 static, all different

    with pytest.raises(ValueError, match="distinct colours"):
        render.build_palette(ten + [Lifeform(11, CONWAY)])


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


def test_render_square_draws_each_cell_at_its_offset():
    """F64: the old test only checked for no crash; this pins colours and placement."""
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 5, 5, wrap=True)
    grid.occupancy[0] = 1
    palette = render.build_palette([Lifeform(1, CONWAY)])
    surface = pygame.Surface((100, 100))
    surface.fill(BACKGROUND)

    render.render_square(surface, grid, palette, cell_size=10, offset_x=7, offset_y=3)

    assert tuple(surface.get_at((7 + 5, 3 + 5)))[:3] == render.ALIVE_COLORS[0]
    assert tuple(surface.get_at((7 + 15, 3 + 5)))[:3] == render.DEAD_CELL_COLOR
    assert tuple(surface.get_at((7 + 50, 3 + 5)))[:3] == BACKGROUND  # right of the grid


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(("width", "height"), [(6, 4), (20, 16), (50, 50)])
def test_every_cell_centre_picks_its_own_cell(shape, width, height):
    """F26: the click mapping is the inverse of the drawing layout, for every cell."""
    layout = render.fit_grid(shape, width, height, pygame.Rect(316, 232, 868, 552))
    for index in range(width * height):
        row, col = divmod(index, width)
        assert layout.cell_at(*layout.cell_center(row, col)) == index


@pytest.mark.parametrize(
    ("shape", "width", "height"), [("hexagon", 100, 100), ("triangle", 200, 200)]
)
def test_the_smallest_unsampled_cells_each_own_the_pixel_at_their_centre(shape, width, height):
    """Down to MIN_POLYGON_SIZE px a cell is drawn one sample per pixel, and the pixel holding
    its centre is drawn as (and picks) that cell — every cell can be clicked."""
    layout = render.fit_grid(shape, width, height, pygame.Rect(316, 232, 868, 552))
    assert render.samples_per_pixel(layout.cell_size) == 1
    assert layout.cell_size < render.MIN_POLYGON_SIZE + 0.3
    for index in range(width * height):
        assert layout.cell_at(*layout.cell_center(*divmod(index, width))) == index


@pytest.mark.parametrize("shape", SHAPES)
def test_points_outside_the_grid_pick_nothing(shape):
    area = pygame.Rect(316, 232, 868, 552)
    layout = render.fit_grid(shape, 20, 16, area)
    bounds = layout.rect
    for point in [
        (bounds.left - 1, bounds.top - 1),
        (bounds.right + 1, bounds.centery),
        (bounds.centerx, bounds.bottom + 1),
        (area.right - 1, area.bottom - 1) if bounds.right < area.right - 2 else (-5, -5),
    ]:
        assert layout.cell_at(*point) is None


def _colourful_grid(shape, width, height):
    """Ten species laid out so that no two cells sharing an edge or a corner share a colour
    (neighbours are at most 2 columns and 1 row apart, so 3 * row + col differs by 1-5)."""
    lifeforms = [Lifeform(i, CONWAY) for i in range(1, 11)]
    grid = LatticeGrid(lifeforms, shape, width, height, wrap=True)
    rows, cols = np.divmod(np.arange(grid.n_cells), width)
    grid.occupancy[:] = (3 * rows + cols) % 10 + 1
    grid.lifespans[:] = 1
    return grid, render.build_palette(lifeforms)


@pytest.mark.parametrize("shape", ["hexagon", "triangle"])
@pytest.mark.parametrize(("width", "height"), [(10, 10), (50, 50), (100, 100)])
def test_every_drawn_pixel_shows_the_cell_a_click_there_picks(shape, width, height):
    """F26 + F28, and on cell edges too: what is drawn at a pixel is the cell `cell_at` returns
    for it (or the outline). pygame's polygons were drawn up to a pixel up-left of the geometry
    `cell_at` inverts, so 5-33% of a cell's own pixels picked a neighbour — a click near its
    left or top edge toggled the wrong cell. The old test skipped every pixel near an edge."""
    area = pygame.Rect(316, 232, 868, 552)
    grid, palette = _colourful_grid(shape, width, height)
    layout = render.fit_grid(shape, width, height, area)
    surface = pygame.Surface((1200, 800))
    surface.fill(BACKGROUND)
    render.GeometricRenderer(BACKGROUND).draw(surface, grid, palette, layout)
    pixels = pygame.surfarray.array3d(surface)

    def check(x, y):
        """The cell picked at (x, y), after checking the pixel shows it (or the outline)."""
        index = layout.cell_at(x, y)
        color = tuple(int(v) for v in pixels[x, y])
        if index is None:
            assert color == BACKGROUND, (x, y)
            return None
        own = tuple(int(v) for v in palette[grid.occupancy[index]])
        assert color in (own, render.GRID_LINE_COLOR), (x, y, index)
        return index if color == own else None

    # Every pixel over and around a few cells, their edges and corners included ...
    n = width * height
    for index in {0, width - 1, n // 2 + width // 2, n - width, n - 1}:
        corners = layout.corners(*divmod(index, width))
        xs = [x for x, _ in corners]
        ys = [y for _, y in corners]
        shown = 0
        for x in range(int(min(xs)) - 2, int(max(xs)) + 3):
            for y in range(int(min(ys)) - 2, int(max(ys)) + 3):
                shown += check(x, y) == index
        assert shown > 0  # the cell shows its own colour, not only outline
    # ... and pixels spread over the whole area.
    rng = np.random.default_rng(1)
    for _ in range(2000):
        check(int(rng.integers(area.left, area.right)), int(rng.integers(area.top, area.bottom)))


@pytest.mark.parametrize(("shape", "width", "height"), [("hexagon", 6, 4), ("triangle", 10, 6)])
def test_the_raster_is_the_click_mapping_at_every_pixel(shape, width, height):
    """Drawing (`rasterise`) and picking (`cell_at`) are one lookup: they agree at every pixel
    of the area, outside the footprint included."""
    area = pygame.Rect(40, 30, 150, 110)
    layout = render.fit_grid(shape, width, height, area)
    cells = render.rasterise(replace(layout, left=0, top=0))
    assert cells.shape == layout.pixel_size
    for x in range(area.left - 3, area.right + 3):
        for y in range(area.top - 3, area.bottom + 3):
            i, j = x - layout.left, y - layout.top
            inside = 0 <= i < cells.shape[0] and 0 <= j < cells.shape[1]
            expected = int(cells[i, j]) if inside else -1
            assert layout.cell_at(x, y) == (None if expected < 0 else expected), (x, y)
    assert set(np.unique(cells)) == {-1, *range(width * height)}


@pytest.mark.parametrize("shape", ["hexagon", "triangle"])
def test_every_cell_keeps_some_fill_inside_its_outline(shape):
    """At the smallest outlined size, the 1-px outline never covers a whole cell."""
    width = height = 150 if shape == "triangle" else 70
    area = pygame.Rect(316, 232, 868, 552)
    layout = render.fit_grid(shape, width, height, area)
    assert render.MIN_OUTLINE_SIZE <= layout.cell_size < render.MIN_OUTLINE_SIZE + 0.6
    grid, palette = _colourful_grid(shape, width, height)
    surface = pygame.Surface((1200, 800))
    render.GeometricRenderer(BACKGROUND).draw(surface, grid, palette, layout)

    cells = render.rasterise(replace(layout, left=0, top=0))
    pixels = pygame.surfarray.array3d(surface.subsurface(layout.rect))
    outline = np.all(pixels == render.GRID_LINE_COLOR, axis=-1)
    assert 0.2 < outline[cells >= 0].mean() < 0.8
    filled = np.bincount(cells[~outline & (cells >= 0)], minlength=width * height)
    assert filled.min() >= 1


@pytest.mark.parametrize("shape", ["hexagon", "triangle"])
@pytest.mark.parametrize(("width", "height"), [(6, 4), (10, 10), (50, 50), (400, 400)])
@pytest.mark.parametrize("area", [(316, 232, 868, 552), (256, 202, 528, 382)])
def test_polygons_stay_inside_the_area_they_are_fitted_to(shape, width, height, area):
    """F28/F29: the drawing used to start half a cell up-left of the grid area (over the
    panel) and, for tall hex grids, run off the bottom of the window."""
    area = pygame.Rect(area)
    layout = render.fit_grid(shape, width, height, area)
    assert area.contains(layout.rect)
    for index in {0, width - 1, width * height - width, width * height - 1, width * height // 2}:
        for x, y in layout.corners(*divmod(index, width)):
            assert area.left - 1e-6 <= x <= area.right + 1e-6
            assert area.top - 1e-6 <= y <= area.bottom + 1e-6


@pytest.mark.parametrize(("width", "height"), [(500, 500), (600, 50), (50, 50)])
def test_square_pixels_belong_to_the_cell_a_click_there_picks(width, height):
    """Square grids too big for the area (an 800x600 window's) are shrunk to fit, and every
    drawn pixel shows the cell `cell_at` picks there."""
    area = pygame.Rect(256, 202, 528, 382)
    grid = _grid("square", width, height, fill=0.5, seed=width)
    palette = render.build_palette([Lifeform(1, CONWAY)])
    layout = render.fit_grid("square", width, height, area)
    assert area.contains(layout.rect)
    surface = pygame.Surface((800, 600))
    surface.fill(BACKGROUND)

    render.render_square(surface, grid, palette, layout.cell_size, layout.left, layout.top)

    pixels = pygame.surfarray.array3d(surface)
    for x in range(area.left, area.right, 3):
        for y in range(area.top, area.bottom, 7):
            index = layout.cell_at(x, y)
            expected = BACKGROUND if index is None else palette[grid.occupancy[index]]
            assert tuple(pixels[x, y]) == tuple(expected)


def test_geometric_renderer_keeps_unchanged_cells_on_screen_every_frame():
    """F25: after a full first draw only changed cells were repainted, but the app clears the
    screen every frame, so unchanged cells vanished from frame 2 on."""
    grid = _grid("hexagon", 10, 10, fill=0.4)
    palette = render.build_palette([Lifeform(1, CONWAY)])
    layout = render.fit_grid("hexagon", 10, 10, pygame.Rect(0, 0, 400, 400))
    renderer = render.GeometricRenderer(BACKGROUND)
    surface = pygame.Surface((400, 400))

    frames = []
    for _ in range(3):
        surface.fill(BACKGROUND)  # as App._draw does
        renderer.draw(surface, grid, palette, layout)
        frames.append(pygame.surfarray.array3d(surface).copy())

    assert np.array_equal(frames[0], frames[1])
    assert np.array_equal(frames[0], frames[2])
    centre = layout.cell_center(*divmod(int(np.flatnonzero(grid.occupancy == 0)[0]), 10))
    assert tuple(surface.get_at(tuple(int(v) for v in centre)))[:3] == render.DEAD_CELL_COLOR


def test_geometric_renderer_repaints_changed_cells_and_full_on_palette_change():
    grid = _grid("triangle", 10, 10, fill=0.0)
    palette = render.build_palette([Lifeform(1, CONWAY)])
    layout = render.fit_grid("triangle", 10, 10, pygame.Rect(0, 0, 400, 400))
    renderer = render.GeometricRenderer(BACKGROUND)
    surface = pygame.Surface((400, 400))
    centre = tuple(int(v) for v in layout.cell_center(3, 4))

    renderer.draw(surface, grid, palette, layout)
    assert tuple(surface.get_at(centre))[:3] == render.DEAD_CELL_COLOR

    grid.occupancy[3 * 10 + 4] = 1
    grid.lifespans[3 * 10 + 4] = 1
    renderer.draw(surface, grid, palette, layout)
    assert tuple(surface.get_at(centre))[:3] == render.ALIVE_COLORS[0]

    recoloured = palette.copy()
    recoloured[0] = (1, 2, 3)
    renderer.draw(surface, grid, recoloured, layout)
    elsewhere = tuple(int(v) for v in layout.cell_center(0, 0))
    assert tuple(surface.get_at(elsewhere))[:3] == (1, 2, 3)


@pytest.mark.parametrize(
    ("shape", "width", "height"), [("triangle", 500, 500), ("hexagon", 400, 400)]
)
def test_tiny_cells_still_show_their_colour(shape, width, height):
    """F29: at 1 px the outline covered every triangle's fill, so a full grid showed no live
    colour at all; cells under MIN_POLYGON_SIZE are now drawn larger and scaled down."""
    area = pygame.Rect(316, 232, 868, 552)
    grid = _grid(shape, width, height, fill=1.0)
    palette = render.build_palette([Lifeform(1, CONWAY)])
    layout = render.fit_grid(shape, width, height, area)
    assert layout.cell_size < render.MIN_POLYGON_SIZE
    surface = pygame.Surface((1200, 800))
    surface.fill(BACKGROUND)

    render.GeometricRenderer(BACKGROUND).draw(surface, grid, palette, layout)

    pixels = pygame.surfarray.array3d(surface.subsurface(layout.rect)).reshape(-1, 3)
    near_alive = np.all(np.abs(pixels.astype(int) - render.ALIVE_COLORS[0]) <= 3, axis=1)
    assert near_alive.mean() > 0.9
    outside = pygame.surfarray.array3d(surface).copy()
    outside[area.left : area.right, area.top : area.bottom] = BACKGROUND
    assert np.all(outside == BACKGROUND)  # nothing drawn outside the area


@pytest.mark.parametrize(
    ("shape", "width", "height"), [("triangle", 500, 500), ("hexagon", 400, 400)]
)
def test_tiny_cells_are_drawn_where_the_layout_puts_them(shape, width, height):
    """Below MIN_POLYGON_SIZE a pixel blends several cells (and a cell holding no pixel centre
    cannot be clicked), but the samples are averaged down in place: the pixel holding a cell's
    centre always shows some of that cell's colour."""
    area = pygame.Rect(316, 232, 868, 552)
    grid = _grid(shape, width, height, fill=0.0)
    palette = render.build_palette([Lifeform(1, CONWAY)])
    layout = render.fit_grid(shape, width, height, area)
    assert render.samples_per_pixel(layout.cell_size) > 1
    renderer = render.GeometricRenderer(BACKGROUND)
    surface = pygame.Surface((1200, 800))
    n = width * height
    for index in (0, width - 1, n // 2 + width // 3, n - 1, 1234):
        grid.occupancy[:] = 0
        grid.occupancy[index] = 1
        renderer.draw(surface, grid, palette, layout)
        x, y = (int(v) for v in layout.cell_center(*divmod(index, width)))
        red, green, _ = (int(v) for v in surface.get_at((x, y))[:3])
        assert red - green >= 5  # dead cells and background have red <= green
