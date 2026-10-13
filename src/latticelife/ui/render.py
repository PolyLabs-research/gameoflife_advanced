"""Cell rendering (§6.14). Pure draw math ported from the legacy `square_cell.py`,
`hexagon_cell.py`, and `triangle_cell.py` — those classes are discarded; only their
corner/position math survives here as plain functions.

Square rendering is a single vectorized palette-lookup blit per frame, not 40k individual draw
calls. Hex/triangle cells are kept on a persistent offscreen image: a full geometric draw on the
first frame (and on reset, resize, shape or palette change), then only the cells whose palette
index changed since the previous frame are redrawn into it — and the whole image is blitted to
the screen every frame, so cells that did not change stay visible.

`GridLayout` is the single description of where a grid's cells are on screen. Drawing and click
picking (`GridLayout.cell_at`) both go through it, so a click always lands on the cell drawn
under the pointer, for every shape. The layout places the polygons' bounding box inside the
area it was fitted to: a hexagon or triangle reaches half a cell beyond its centre, so the first
cell's centre sits that far in from the area's corner, never on it.

Hex/triangle drawing and picking share one rasterised geometry. `rasterise` gives each pixel
the cell under the pixel's centre — the very lookup `cell_at` makes for a click on that pixel —
and a cell is drawn by painting exactly its pixels. (`pygame.draw.polygon` was not used for
this: it truncates float corners and fills both ends of every span, so its polygons sat up to a
pixel up-left of the geometry, overlapped their neighbours, and a click near an edge toggled the
neighbour of the cell drawn under it.) Cell edges get a 1-px outline from the same raster.

Very large hexagon/triangle grids (cells smaller than `MIN_POLYGON_SIZE` px) are sampled more
finely than the screen, a whole number of samples per pixel each way, and averaged down to fit;
cells under `MIN_OUTLINE_SIZE` px get no outline (a 1-px outline would cover a tiny cell's fill
entirely).
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
# Hex/triangle cells smaller than this many px are sampled at least this finely offscreen (a
# whole number of samples per pixel each way) and averaged down. A cell this big holds at least
# one sample point: even a triangle's inscribed circle is then wider than a pixel's diagonal.
MIN_POLYGON_SIZE = 3.0
# Hex/triangle cells smaller than this are drawn without an outline.
MIN_OUTLINE_SIZE = 4.0
# Smallest cell size a layout is given, so a vanishing grid area cannot divide by zero.
_MIN_FIT = 0.05
# Samples looked up per batch while rasterising, which bounds the lookup's scratch memory.
_RASTER_BATCH = 1 << 18

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

    def _footprint_origin(self) -> tuple[float, float]:
        """Cell (0, 0)'s corner-math offset (`hexagon_corners`/`triangle_corners`) from the
        footprint's top-left corner: it puts the polygons' bounding box at that corner."""
        if self.shape == "hexagon":
            return self.cell_size, SQRT3 / 2 * self.cell_size
        if self.shape == "triangle":
            return self.cell_size / 2, SQRT3 / 4 * self.cell_size
        return 0.0, 0.0

    def _origin(self) -> tuple[float, float]:
        """The corner-math offset in window pixels."""
        ox, oy = self._footprint_origin()
        return self.left + ox, self.top + oy

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
        """The flat index of the cell drawn at window pixel (x, y), or None where no cell is
        drawn; a fractional position picks the pixel it lies in. A square-grid pixel shows the
        cell `render_square`'s nearest-neighbour scale puts there, a hexagon/triangle pixel the
        cell under the pixel's centre — the point `rasterise` looks up to draw it."""
        i, j = math.floor(x) - self.left, math.floor(y) - self.top
        if self.shape == "square":
            pixel_w, pixel_h = self.pixel_size
            col = math.floor(i * self.width / pixel_w)
            row = math.floor(j * self.height / pixel_h)
            if 0 <= row < self.height and 0 <= col < self.width:
                return row * self.width + col
            return None
        index = int(self.cells_at(i + 0.5, j + 0.5))
        return index if index >= 0 else None

    def cells_at(self, fx, fy) -> np.ndarray:
        """Hexagon/triangle grids: the flat index of the cell containing each point (fx, fy),
        given in px from the footprint's top-left corner, or -1 where no cell does. Vectorized:
        the points may be scalars or arrays of one shape, and every point is looked up with the
        same float operations, so a pixel is drawn and picked as the same cell."""
        ox, oy = self._footprint_origin()
        lx = np.asarray(fx, dtype=np.float64) - ox
        ly = np.asarray(fy, dtype=np.float64) - oy
        if self.shape == "hexagon":
            rows, cols = _hex_cells(lx, ly, self.cell_size)
        elif self.shape == "triangle":
            rows, cols = _triangle_cells(lx, ly, self.cell_size)
        else:
            raise ValueError(f"cells_at is for hexagon and triangle grids, not {self.shape}")
        inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        return np.where(inside, rows * self.width + cols, -1)


def _hex_cells(lx, ly, size: float) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of the hexagons containing the points (lx, ly), relative to cell (0, 0)'s
    centre, on the unbounded odd-q lattice. Regular hexagons tile as the Voronoi cells of their
    centres, so the nearest centre among the neighbouring candidates is the containing cell (on
    a tie, the first candidate in the order below)."""
    lx = np.asarray(lx, dtype=np.float64)
    ly = np.asarray(ly, dtype=np.float64)
    q0 = np.round(lx / (1.5 * size)).astype(np.int64)
    best = np.full(lx.shape, np.inf)
    rows = np.zeros(lx.shape, dtype=np.int64)
    cols = np.zeros(lx.shape, dtype=np.int64)
    for dq in (-1, 0, 1):
        q = q0 + dq
        shift = 0.5 * (q & 1)
        r0 = np.round(ly / (SQRT3 * size) - shift).astype(np.int64)
        for dr in (-1, 0, 1):
            r = r0 + dr
            dx = lx - 1.5 * size * q
            dy = ly - SQRT3 * size * (r + shift)
            distance = dx * dx + dy * dy
            closer = distance < best
            best = np.where(closer, distance, best)
            rows = np.where(closer, r, rows)
            cols = np.where(closer, q, cols)
    return rows, cols


def _triangle_cells(lx, ly, size: float) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of the triangles containing the points (lx, ly), relative to cell (0, 0)'s
    midpoint, on the unbounded lattice: the row from y, then the column whose up/down triangle
    contains the point most deeply (ties on a shared edge go to the left cell)."""
    lx = np.asarray(lx, dtype=np.float64)
    ly = np.asarray(ly, dtype=np.float64)
    tri_h = SQRT3 / 2 * size
    rows = np.floor((ly + tri_h / 2) / tri_h).astype(np.int64)
    depth = (ly - (rows * tri_h - tri_h / 2)) / tri_h  # 0 at the row's top edge, 1 at its bottom
    c0 = np.floor(lx / (size / 2)).astype(np.int64)
    best = np.full(lx.shape, -np.inf)
    cols = np.zeros(lx.shape, dtype=np.int64)
    for dc in (-1, 0, 1, 2):
        col = c0 + dc
        upward = (rows + col) % 2 == 0
        half_width = np.where(upward, depth, 1 - depth) * size / 2
        margin = half_width - np.abs(lx - col * size / 2)
        deeper = margin > best
        best = np.where(deeper, margin, best)
        cols = np.where(deeper, col, cols)
    return rows, cols


def _hex_at(lx: float, ly: float, size: float) -> tuple[int, int]:
    """`_hex_cells` for one point, as plain ints."""
    rows, cols = _hex_cells(lx, ly, size)
    return int(rows), int(cols)


def _triangle_at(lx: float, ly: float, size: float) -> tuple[int, int]:
    """`_triangle_cells` for one point, as plain ints."""
    rows, cols = _triangle_cells(lx, ly, size)
    return int(rows), int(cols)


def samples_per_pixel(cell_size: float) -> int:
    """How many samples per pixel, each way, a hex/triangle grid of `cell_size` px cells is
    rasterised with: 1, or for cells under `MIN_POLYGON_SIZE` px the fewest that make every
    cell at least that many samples across."""
    if cell_size >= MIN_POLYGON_SIZE:
        return 1
    return math.ceil(MIN_POLYGON_SIZE / cell_size)


def rasterise(layout: GridLayout, samples: int = 1) -> np.ndarray:
    """The flat index of the cell at every sample of a hex/triangle layout's footprint (-1 where
    there is none), as a (samples * px width, samples * px height) int32 array in pygame's
    (x, y) order. Sample (i, j) is the point ((i + 0.5) / samples, (j + 0.5) / samples) px from
    the footprint's top-left corner; at one sample per pixel that is the pixel's centre, the
    very point `GridLayout.cell_at` looks up for a click on that pixel."""
    pixel_w, pixel_h = layout.pixel_size
    width, height = pixel_w * samples, pixel_h * samples
    cells = np.empty((width, height), dtype=np.int32)
    ys = (np.arange(height) + 0.5) / samples
    step = max(1, _RASTER_BATCH // height)
    for x0 in range(0, width, step):
        xs = (np.arange(x0, min(x0 + step, width)) + 0.5) / samples
        fx, fy = np.meshgrid(xs, ys, indexing="ij")
        cells[x0 : x0 + step] = layout.cells_at(fx, fy)
    return cells


def _outline(cells: np.ndarray) -> np.ndarray:
    """The pixels of a 1-px line along every cell edge in a `rasterise` array: each cell pixel
    whose left or upper neighbour is another cell or none (so a shared edge is drawn once, on
    one side of it) or whose right or lower neighbour is none (so the grid's border is closed
    on every side). A cell that would be all outline keeps its pixels."""
    padded = np.pad(cells, 1, constant_values=-1)
    inner = padded[1:-1, 1:-1]
    left, right = padded[:-2, 1:-1], padded[2:, 1:-1]
    up, down = padded[1:-1, :-2], padded[1:-1, 2:]
    outline = (inner >= 0) & ((left != inner) | (up != inner) | (right < 0) | (down < 0))
    n_cells = int(cells.max()) + 1
    if n_cells > 0:
        kept = np.bincount(cells[~outline & (cells >= 0)], minlength=n_cells)
        starved = kept == 0
        if starved.any():
            outline &= ~starved[np.maximum(cells, 0)]
    return outline


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


class GeometricRenderer:
    """Hex/triangle draw state (§6.14). On the first draw for a grid shape, size and cell size,
    the footprint is rasterised (`rasterise`, one sample per pixel unless cells are under
    `MIN_POLYGON_SIZE` px), its outline and background are painted once into a persistent
    offscreen image, and each cell's samples are listed. A draw paints whole cells into that
    image: every cell the first time and whenever the geometry or palette changes (or after
    `reset()`), otherwise only the cells whose palette index changed since the previous frame.
    Each `draw` blits the whole image — averaged down to the screen when it was sampled more
    finely — so the grid is complete on every frame."""

    def __init__(self, background: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.background = background
        self._geometry: tuple | None = None
        self._samples = 1
        # Every fill sample's flat index in `_pixels`, grouped by cell: cell k's samples are
        # _order[_starts[k]:_starts[k + 1]]. Outline and background samples are in no group.
        self._order: np.ndarray | None = None
        self._starts: np.ndarray | None = None
        self._pixels: np.ndarray | None = None  # mapped colours, (width, height) as pygame
        self._surface: pygame.Surface | None = None
        self._scaled: pygame.Surface | None = None
        self._palette_key: bytes | None = None
        self._mapped: np.ndarray | None = None  # the palette as `_surface` pixel values
        self._previous_state: np.ndarray | None = None

    def reset(self) -> None:
        """Repaint every cell on the next draw."""
        self._previous_state = None

    def _build(self, layout: GridLayout) -> None:
        """Rasterise `layout` and paint its background and outline offscreen."""
        samples = samples_per_pixel(layout.cell_size)
        cells = rasterise(replace(layout, left=0, top=0), samples)
        surface = pygame.Surface(cells.shape)
        pixels = np.full(cells.shape, surface.map_rgb(self.background), dtype=np.uint32)
        fill = cells
        if samples == 1 and layout.cell_size >= MIN_OUTLINE_SIZE:
            outline = _outline(cells)
            pixels[outline] = surface.map_rgb(GRID_LINE_COLOR)
            fill = np.where(outline, -1, cells)

        flat = fill.ravel()
        samples_in_cells = np.flatnonzero(flat >= 0)
        cell_of_sample = flat[samples_in_cells]
        order = samples_in_cells[np.argsort(cell_of_sample, kind="stable")]
        self._order = order.astype(np.int32) if flat.size < 2**31 else order
        counts = np.bincount(cell_of_sample, minlength=layout.width * layout.height)
        self._starts = np.concatenate(([0], np.cumsum(counts)))

        self._geometry = (layout.shape, layout.width, layout.height, layout.cell_size)
        self._samples = samples
        self._pixels = pixels
        self._surface = surface
        self._scaled = None
        self._palette_key = None  # the colours are mapped for this surface
        self._previous_state = None

    def _paint(self, cells: np.ndarray, colors: np.ndarray) -> None:
        """Give every sample of each of `cells` its colour (`colors`, one per cell)."""
        assert self._order is not None and self._starts is not None and self._pixels is not None
        starts = self._starts[cells]
        counts = self._starts[cells + 1] - starts
        # Each cell's run of `_order`: its start, then one entry per sample.
        runs = np.repeat(starts - (np.cumsum(counts) - counts), counts)
        runs += np.arange(runs.size)
        self._pixels.reshape(-1)[self._order[runs]] = np.repeat(colors, counts)

    def draw(
        self,
        surface: pygame.Surface,
        grid: LatticeGrid,
        palette: np.ndarray,
        layout: GridLayout,
    ) -> None:
        if (layout.shape, layout.width, layout.height, layout.cell_size) != self._geometry:
            self._build(layout)
        assert self._surface is not None and self._pixels is not None
        palette_key = palette.tobytes()
        if palette_key != self._palette_key:
            self._palette_key = palette_key
            self._mapped = np.array(
                [self._surface.map_rgb(tuple(int(v) for v in rgb)) for rgb in palette],
                dtype=np.uint32,
            )
            self._previous_state = None

        state = occupancy_state(grid, palette)
        if self._previous_state is None or self._previous_state.shape != state.shape:
            dirty = np.arange(state.size)
        else:
            dirty = np.flatnonzero(state != self._previous_state)
        self._previous_state = state

        if dirty.size:
            assert self._mapped is not None
            self._paint(dirty, self._mapped[state[dirty]])
            pygame.surfarray.blit_array(self._surface, self._pixels)
            if self._samples > 1:
                self._scaled = pygame.transform.smoothscale(self._surface, layout.pixel_size)
        image = self._scaled if self._samples > 1 else self._surface
        assert image is not None
        surface.blit(image, (layout.left, layout.top))
