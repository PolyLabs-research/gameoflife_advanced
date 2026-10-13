"""F64: `test_render_square_produces_correctly_sized_surface` asserts nothing, so a renderer that
ignored offset_x/offset_y (or drew the wrong colours) passed. This checks actual pixels."""

import pygame

from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.ui import render

CONWAY = Rule(birth={3}, survive={2, 3})
BACKGROUND = (1, 2, 3)


def test_render_square_draws_each_cell_at_the_offset_in_its_colour():
    lifeforms = [Lifeform(1, CONWAY), Lifeform(2, CONWAY)]
    grid = LatticeGrid(lifeforms, "square", 5, 4, wrap=True)
    grid.occupancy[0] = 1  # row 0, col 0
    grid.occupancy[7] = 2  # row 1, col 2
    palette = render.build_palette(lifeforms)
    surface = pygame.Surface((100, 100))
    surface.fill(BACKGROUND)

    render.render_square(surface, grid, palette, cell_size=10, offset_x=7, offset_y=3)

    def pixel(x, y):
        return tuple(surface.get_at((x, y)))[:3]

    assert pixel(7 + 5, 3 + 5) == tuple(palette[1])  # cell (0, 0): species 1
    assert pixel(7 + 25, 3 + 15) == tuple(palette[2])  # cell (1, 2): species 2
    assert pixel(7 + 15, 3 + 5) == render.DEAD_CELL_COLOR  # cell (0, 1): empty
    assert pixel(7 + 49, 3 + 39) == render.DEAD_CELL_COLOR  # last pixel of the grid
    assert pixel(0, 0) == BACKGROUND  # left of / above the offset: untouched
    assert pixel(6, 50) == BACKGROUND
    assert pixel(7 + 50, 3 + 5) == BACKGROUND  # right of the 5-column grid
    assert pixel(20, 3 + 40) == BACKGROUND  # below the 4-row grid
