"""Cell rendering (§6.14). Pure draw math ported from the legacy `square_cell.py`,
`hexagon_cell.py`, and `triangle_cell.py` — those classes are discarded; only their
corner/position math survives here as plain functions.

Square rendering is a single vectorized palette-lookup blit per frame, not 40k individual draw
calls. Hex/triangle cells are polygons kept on a persistent offscreen surface: a full geometric
draw on the first frame (and on reset, resize, shape or palette change), then only the cells
whose palette index changed since the previous frame are redrawn into it — and the whole surface
is blitted to the screen every frame, so cells that did not change stay visible.

`GridLayout` is the single description of where a grid's cells are on screen. Drawing and click
picking (`GridLayout.cell_at`) both go through it, so a click always lands on the cell drawn
under the pointer, for every shape. The layout places the polygons' bounding box inside the
area it was fitted to: a hexagon or triangle reaches half a cell beyond its centre, so the first
cell's centre sits that far in from the area's corner, never on it.

Very large hexagon/triangle grids (cells smaller than `MIN_POLYGON_SIZE` px) are drawn offscreen
at that size and scaled down to fit, and cells under `MIN_OUTLINE_SIZE` px get no outline (a
1-px outline would cover a tiny cell's fill entirely).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pygame

from latticelife.config import MAX_LIFEFORMS
from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform

DEAD_CELL_COLOR = (32, 34, 42)
GRID_LINE_COLOR = (45, 48, 58)

SQRT3 = math.sqrt(3)
# Hex/triangle cells smaller than this are drawn at this size offscreen and scaled down.
MIN_POLYGON_SIZE = 3.0
# Hex/triangle cells smaller than this are drawn without an outline.
MIN_OUTLINE_SIZE = 4.0
# Smallest cell size a layout is given, so a vanishing grid area cannot divide by zero.
_MIN_FIT = 0.05

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
if not len(ALIVE_COLORS) == len(STATIC_COLORS) >= MAX_LIFEFORMS:
    raise RuntimeError("every allowed species needs its own alive and static colour")


def _footprint_factors(shape: str, width: int, height: int) -> tuple[float, float]:
    """A `width` x `height` grid's drawn width and height, in units of cell size. Square cells
    tile edge-to-edge, but hex/triangle cells overlap the way `hexagon_corners`/
    `triangle_corners` place them, so their footprint per column/row is a different ratio."""
    if shape == "square":
        return float(width), float(height)
    if shape == "hexagon":
        return 1.5 * width + 0.5, SQRT3 * (height + 0.5)
    if shape == "triangle":
        return (width + 1) / 2, SQRT3 / 2 * height
    raise ValueError(f"Unsupported grid shape: {shape}")


def fit_cell_size(
    shape: str, width: int, height: int, available_width: float, available_height: float
) -> float:
    """Largest cell size at which a `width` x `height` grid of `shape` cells fits within
    `available_width` x `available_height` pixels. Square cells of 1 px or more get a whole
    number of pixels (crisp blocks); smaller squares, and hexagons and triangles (whose spacing
    is irrational anyway), get the exact fit — never a size that overflows the area."""
    factor_w, factor_h = _footprint_factors(shape, width, height)
    fit = max(_MIN_FIT, min(available_width / factor_w, available_height / factor_h))
    if shape == "square" and fit >= 1:
        return float(int(fit))
    return fit


def hexagon_corners(
    q: int, r: int, size: float, offset_x: float, offset_y: float
) -> list[tuple[float, float]]:
    """Odd-q offset hex corners, ported from the legacy `HexagonCell.draw_static`. The offset
    is the centre of cell (0, 0)."""
    center_x = size * 1.5 * q + offset_x
    center_y = size * SQRT3 * (r + 0.5 * (q & 1)) + offset_y
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
    """Ported from the legacy `TriangleCell.draw_static`. The offset is the midpoint of cell
    (0, 0)'s vertical extent."""
    x0 = x * (size / 2) + offset_x
    y0 = y * (size * SQRT3 / 2) + offset_y
    half_size = size / 2
    height = size * SQRT3 / 2
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


@dataclass(frozen=True)
class GridLayout:
    """Where a `width` x `height` grid of `shape` cells is drawn: cell size in px and the
    top-left corner of its footprint. Built by `fit_grid`; shared by drawing and picking."""

    shape: str
    width: int
    height: int
    cell_size: float
    left: int
    top: int

    @property
    def pixel_size(self) -> tuple[int, int]:
        """The footprint in whole pixels (a square grid's exact block size)."""
        factor_w, factor_h = _footprint_factors(self.shape, self.width, self.height)
        if self.shape == "square":
            return max(1, int(factor_w * self.cell_size)), max(1, int(factor_h * self.cell_size))
        # The tolerance keeps float noise (382.0000000001) from adding a pixel past the fit.
        return (
            math.ceil(factor_w * self.cell_size - 1e-6),
            math.ceil(factor_h * self.cell_size - 1e-6),
        )

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect((self.left, self.top), self.pixel_size)

    def _origin(self) -> tuple[float, float]:
        """The corner-math offset (`hexagon_corners`/`triangle_corners`) that puts the
        polygons' bounding box at (left, top)."""
        if self.shape == "hexagon":
            return self.left + self.cell_size, self.top + SQRT3 / 2 * self.cell_size
        if self.shape == "triangle":
            return self.left + self.cell_size / 2, self.top + SQRT3 / 4 * self.cell_size
        return float(self.left), float(self.top)

    def corners(self, row: int, col: int) -> list[tuple[float, float]]:
        ox, oy = self._origin()
        size = self.cell_size
        if self.shape == "hexagon":
            return hexagon_corners(col, row, size, ox, oy)
        if self.shape == "triangle":
            return triangle_corners(col, row, size, (row + col) % 2 == 0, ox, oy)
        step_x, step_y = self.pixel_size[0] / self.width, self.pixel_size[1] / self.height
        x0, y0 = ox + col * step_x, oy + row * step_y
        return [(x0, y0), (x0 + step_x, y0), (x0 + step_x, y0 + step_y), (x0, y0 + step_y)]

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        """The centroid of the cell's polygon — a point that is always inside the cell."""
        points = self.corners(row, col)
        return (
            sum(x for x, _ in points) / len(points),
            sum(y for _, y in points) / len(points),
        )

    def cell_at(self, x: float, y: float) -> int | None:
        """The flat index of the cell drawn at pixel (x, y), or None outside the grid."""
        ox, oy = self._origin()
        lx, ly = x - ox, y - oy
        if self.shape == "hexagon":
            row, col = _hex_at(lx, ly, self.cell_size)
        elif self.shape == "triangle":
            row, col = _triangle_at(lx, ly, self.cell_size)
        else:
            pixel_w, pixel_h = self.pixel_size
            col = math.floor(lx * self.width / pixel_w)
            row = math.floor(ly * self.height / pixel_h)
        if 0 <= row < self.height and 0 <= col < self.width:
            return row * self.width + col
        return None


def _hex_at(lx: float, ly: float, size: float) -> tuple[int, int]:
    """(row, col) of the hexagon containing (lx, ly), relative to cell (0, 0)'s centre, on the
    unbounded odd-q lattice. Regular hexagons tile as the Voronoi cells of their centres, so the
    nearest centre among the neighbouring candidates is the containing cell."""
    q0 = round(lx / (1.5 * size))
    best: tuple[float, int, int] | None = None
    for q in (q0 - 1, q0, q0 + 1):
        shift = 0.5 * (q & 1)
        r0 = round(ly / (SQRT3 * size) - shift)
        for r in (r0 - 1, r0, r0 + 1):
            dx = lx - 1.5 * size * q
            dy = ly - SQRT3 * size * (r + shift)
            distance = dx * dx + dy * dy
            if best is None or distance < best[0]:
                best = (distance, r, q)
    assert best is not None
    return best[1], best[2]


def _triangle_at(lx: float, ly: float, size: float) -> tuple[int, int]:
    """(row, col) of the triangle containing (lx, ly), relative to cell (0, 0)'s midpoint, on
    the unbounded lattice: the row from y, then the column whose up/down triangle contains the
    point most deeply (ties on a shared edge go to the left cell)."""
    tri_h = SQRT3 / 2 * size
    row = math.floor((ly + tri_h / 2) / tri_h)
    depth = (ly - (row * tri_h - tri_h / 2)) / tri_h  # 0 at the row's top edge, 1 at its bottom
    c0 = math.floor(lx / (size / 2))
    best: tuple[float, int] | None = None
    for col in (c0 - 1, c0, c0 + 1, c0 + 2):
        upward = (row + col) % 2 == 0
        half_width = (depth if upward else 1 - depth) * size / 2
        margin = half_width - abs(lx - col * size / 2)
        if best is None or margin > best[0]:
            best = (margin, col)
    assert best is not None
    return row, best[1]


def fit_grid(shape: str, width: int, height: int, area: pygame.Rect) -> GridLayout:
    """The layout of a `width` x `height` grid of `shape` cells fitted into `area`, its
    footprint anchored at the area's top-left corner."""
    cell_size = fit_cell_size(shape, width, height, area.width, area.height)
    return GridLayout(shape, width, height, cell_size, area.x, area.y)


def build_palette(lifeforms: list[Lifeform]) -> np.ndarray:
    """(2L+1, 3) uint8 palette: index 0 is dead, 1..L are alive shades per species id, and
    L+1..2L are static shades per species id (offset by L). L = max lifeform id, or 0 if there
    are none. Raises ValueError for more species than there are distinct colours."""
    if len(lifeforms) > len(ALIVE_COLORS):
        raise ValueError(
            f"{len(lifeforms)} species, but only {len(ALIVE_COLORS)} have distinct colours"
        )
    n_species = max((lf.id for lf in lifeforms), default=0)
    palette = np.zeros((2 * n_species + 1, 3), dtype=np.uint8)
    palette[0] = DEAD_CELL_COLOR
    for i, lf in enumerate(lifeforms):
        palette[lf.id] = ALIVE_COLORS[i]
        palette[n_species + lf.id] = STATIC_COLORS[i]
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
    cell_size: float,
    offset_x: int,
    offset_y: int,
) -> None:
    """Blit the square grid with its top-left at (offset_x, offset_y), each cell `cell_size` px
    (below 1 px, the grid is shrunk to `cell_size` times its cell count). The nearest-neighbour
    scale puts cell (j * height // H, i * width // W) at footprint pixel (i, j) — the integer
    mapping `GridLayout.cell_at` inverts, so a click picks exactly the cell drawn under it."""
    state = occupancy_state(grid, palette).reshape(grid.height, grid.width)
    rgb = palette[state]  # (height, width, 3)
    cell_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))  # pygame wants (w, h, 3)
    layout = GridLayout("square", grid.width, grid.height, cell_size, offset_x, offset_y)
    scaled = pygame.transform.scale(cell_surface, layout.pixel_size)
    surface.blit(scaled, (offset_x, offset_y))


def _all_corners(layout: GridLayout) -> np.ndarray:
    """Every cell's polygon at once, (n_cells, k, 2) float — the vectorized twin of
    `GridLayout.corners`."""
    rows, cols = np.divmod(np.arange(layout.width * layout.height), layout.width)
    ox, oy = layout._origin()
    size = layout.cell_size
    if layout.shape == "hexagon":
        cx = size * 1.5 * cols + ox
        cy = size * SQRT3 * (rows + 0.5 * (cols & 1)) + oy
        angles = np.radians(60 * np.arange(6))
        xs = cx[:, None] + size * np.cos(angles)[None, :]
        ys = cy[:, None] + size * np.sin(angles)[None, :]
        return np.stack([xs, ys], axis=-1)
    x0 = cols * (size / 2) + ox
    y0 = rows * (size * SQRT3 / 2) + oy
    half, tri_h = size / 2, size * SQRT3 / 2
    flip = np.where((rows + cols) % 2 == 0, 1.0, -1.0)  # +1 upward, -1 downward
    xs = np.stack([x0, x0 - half, x0 + half], axis=1)
    ys = np.stack([y0 - flip * tri_h / 2, y0 + flip * tri_h / 2, y0 + flip * tri_h / 2], axis=1)
    return np.stack([xs, ys], axis=-1)


class GeometricRenderer:
    """Hex/triangle draw state (§6.14): a persistent offscreen surface holding every cell's
    polygon, fully redrawn the first time and whenever the grid, cell size or palette changes
    (or after `reset()`), otherwise updated only where a cell's palette index changed since the
    previous frame. Each `draw` blits the whole surface — scaled down when cells are below
    `MIN_POLYGON_SIZE` px — so the grid is complete on every frame."""

    def __init__(self, background: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.background = background
        self._surface: pygame.Surface | None = None
        self._scaled: pygame.Surface | None = None
        self._key: tuple | None = None
        self._corners: np.ndarray | None = None
        self._previous_state: np.ndarray | None = None

    def reset(self) -> None:
        """Repaint every cell on the next draw."""
        self._previous_state = None

    def draw(
        self,
        surface: pygame.Surface,
        grid: LatticeGrid,
        palette: np.ndarray,
        layout: GridLayout,
    ) -> None:
        draw_size = max(layout.cell_size, MIN_POLYGON_SIZE)
        offscreen = replace(layout, cell_size=draw_size, left=0, top=0)
        key = (grid.shape, grid.width, grid.height, draw_size, palette.tobytes())
        if key != self._key or self._surface is None:
            width, height = offscreen.pixel_size
            self._surface = pygame.Surface((width + 1, height + 1))
            self._corners = _all_corners(offscreen)
            self._scaled = None
            self._key = key
            self._previous_state = None

        state = occupancy_state(grid, palette)
        if self._previous_state is None or self._previous_state.shape != state.shape:
            self._surface.fill(self.background)
            dirty = np.arange(state.size)
        else:
            dirty = np.flatnonzero(state != self._previous_state)
        self._previous_state = state

        colors = [tuple(int(v) for v in rgb) for rgb in palette]
        outline = draw_size >= MIN_OUTLINE_SIZE
        assert self._corners is not None
        for idx in dirty.tolist():
            corners = self._corners[idx].tolist()
            pygame.draw.polygon(self._surface, colors[state[idx]], corners)
            if outline:
                pygame.draw.polygon(self._surface, GRID_LINE_COLOR, corners, 1)

        image = self._surface
        if draw_size != layout.cell_size:
            target_size = layout.pixel_size
            if self._scaled is None or dirty.size or self._scaled.get_size() != target_size:
                self._scaled = pygame.transform.smoothscale(self._surface, target_size)
            image = self._scaled
        surface.blit(image, (layout.left, layout.top))
