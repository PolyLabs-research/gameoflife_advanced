"""Cell rendering (§6.14). Pure draw math ported from the legacy `square_cell.py`,
`hexagon_cell.py`, and `triangle_cell.py` — those classes are discarded; only their
corner/position math survives here as plain functions.

Square rendering is a single vectorized palette-lookup blit per frame, not 40k individual draw
calls. Hex/triangle do a full geometric draw on the first frame (and on reset/resize), then only
redraw cells whose palette index changed since the previous frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform

DEAD_CELL_COLOR = (32, 34, 42)
GRID_LINE_COLOR = (45, 48, 58)

# Ported from the legacy settings.py LIFEFORM_COLORS_ALIVE / LIFEFORM_COLORS_STATIC.
ALIVE_COLORS = [
    (239, 68, 68),
    (34, 197, 94),
    (59, 130, 246),
    (250, 204, 21),
    (168, 85, 247),
    (236, 72, 153),
    (20, 184, 166),
    (249, 115, 22),
    (132, 204, 22),
    (6, 182, 212),
]
STATIC_COLORS = [
    (252, 165, 165),
    (134, 239, 172),
    (147, 197, 253),
    (253, 224, 71),
    (216, 180, 254),
    (249, 168, 212),
    (94, 234, 212),
    (253, 186, 116),
    (190, 242, 100),
    (103, 232, 249),
]


def fit_cell_size(
    shape: str, width: int, height: int, available_width: int, available_height: int
) -> int:
    """Largest integer cell size that fits a `width` x `height` grid of `shape` cells within
    `available_width` x `available_height` pixels. Square cells tile edge-to-edge (1 unit of
    size per column/row), but hex/triangle cells overlap the way `hexagon_corners`/
    `triangle_corners` place them, so their pixel footprint per column/row is a different
    ratio — using the square ratio for them undersizes or oversizes the whole grid by a
    constant factor no choice of width/height can compensate for."""
    if shape == "square":
        factor_w, factor_h = float(width), float(height)
    elif shape == "hexagon":
        factor_w, factor_h = 1.5 * width + 0.5, math.sqrt(3) * (height + 0.5)
    elif shape == "triangle":
        factor_w, factor_h = (width + 1) / 2, math.sqrt(3) / 2 * height
    else:
        raise ValueError(f"Unsupported grid shape: {shape}")
    return max(1, int(min(available_width / factor_w, available_height / factor_h)))


def build_palette(lifeforms: list[Lifeform]) -> np.ndarray:
    """(2L+1, 3) uint8 palette: index 0 is dead, 1..L are alive shades per species id, and
    L+1..2L are static shades per species id (offset by L). L = max lifeform id, or 0 if there
    are none."""
    n_species = max((lf.id for lf in lifeforms), default=0)
    palette = np.zeros((2 * n_species + 1, 3), dtype=np.uint8)
    palette[0] = DEAD_CELL_COLOR
    for i, lf in enumerate(lifeforms):
        palette[lf.id] = ALIVE_COLORS[i % len(ALIVE_COLORS)]
        palette[n_species + lf.id] = STATIC_COLORS[i % len(STATIC_COLORS)]
    return palette


def occupancy_state(grid: LatticeGrid, palette: np.ndarray) -> np.ndarray:
    """Fold occupancy id + alive/static shade into a single palette index per cell (flat,
    length n_cells)."""
    n_species = (palette.shape[0] - 1) // 2
    static_mask = grid.lifespans > grid.static_threshold
    return grid.occupancy.astype(np.int32) + n_species * static_mask.astype(np.int32)


def render_square(
    surface: pygame.Surface,
    grid: LatticeGrid,
    palette: np.ndarray,
    cell_size: int,
    offset_x: int,
    offset_y: int,
) -> None:
    state = occupancy_state(grid, palette).reshape(grid.height, grid.width)
    rgb = palette[state]  # (height, width, 3)
    cell_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))  # pygame wants (w, h, 3)
    scaled = pygame.transform.scale(cell_surface, (grid.width * cell_size, grid.height * cell_size))
    surface.blit(scaled, (offset_x, offset_y))


def hexagon_corners(
    q: int, r: int, size: float, offset_x: float, offset_y: float
) -> list[tuple[float, float]]:
    """Odd-q offset hex corners, ported from the legacy `HexagonCell.draw_static`."""
    center_x = size * 1.5 * q + offset_x
    center_y = size * math.sqrt(3) * (r + 0.5 * (q & 1)) + offset_y
    return [
        (
            center_x + size * math.cos(math.radians(60 * i)),
            center_y + size * math.sin(math.radians(60 * i)),
        )
        for i in range(6)
    ]


def triangle_corners(
    x: int, y: int, size: float, upward: bool, offset_x: float, offset_y: float
) -> list[tuple[float, float]]:
    """Ported from the legacy `TriangleCell.draw_static`."""
    x0 = x * (size / 2) + offset_x
    y0 = y * (size * math.sqrt(3) / 2) + offset_y
    half_size = size / 2
    height = size * math.sqrt(3) / 2
    if upward:
        return [
            (x0, y0 - height / 2),
            (x0 - half_size, y0 + height / 2),
            (x0 + half_size, y0 + height / 2),
        ]
    return [
        (x0, y0 + height / 2),
        (x0 - half_size, y0 - height / 2),
        (x0 + half_size, y0 - height / 2),
    ]


@dataclass
class GeometricRenderer:
    """Hex/triangle draw state: full redraw on the first frame (or after `reset()`), dirty-cell
    redraw thereafter, comparing against the previous frame's palette-index state."""

    shape: str  # "hexagon" | "triangle"
    cell_size: int
    offset_x: int
    offset_y: int
    _previous_state: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_state = None

    def draw(self, surface: pygame.Surface, grid: LatticeGrid, palette: np.ndarray) -> None:
        state = occupancy_state(grid, palette)
        if self._previous_state is None or self._previous_state.shape != state.shape:
            dirty = np.arange(state.size)
        else:
            dirty = np.flatnonzero(state != self._previous_state)

        for idx in dirty:
            r, c = divmod(int(idx), grid.width)
            color = tuple(int(v) for v in palette[state[idx]])
            corners = self._corners(r, c)
            pygame.draw.polygon(surface, color, corners)
            pygame.draw.polygon(surface, GRID_LINE_COLOR, corners, 1)

        self._previous_state = state

    def _corners(self, r: int, c: int) -> list[tuple[float, float]]:
        if self.shape == "hexagon":
            return hexagon_corners(c, r, self.cell_size, self.offset_x, self.offset_y)
        upward = (r + c) % 2 == 0
        return triangle_corners(c, r, self.cell_size, upward, self.offset_x, self.offset_y)
