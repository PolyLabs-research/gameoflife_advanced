"""Grid -> torus texture, and the numpy twin of the shader's cell lookup.

`cell_colors` gives the RGB texture the torus samples: one texel per cell, coloured through
`render.occupancy_state`, the same alive/static fold the 2D view uses (with the grid's own
static threshold), so both views always agree on what a cell looks like.

`surface_window` is the stretch of the shape's lattice one trip around the torus covers; the
renderer hands it to the shader as two uniforms. `surface_cells` answers, in numpy, the
question `shaders.CELL_LOOKUP` answers per pixel: which cell does surface point (u, v) show? It
is the reference the GL tests check the shader against, and is documented in `shaders.py`.
"""

from __future__ import annotations

import numpy as np

from latticelife.core.lattice import LatticeGrid
from latticelife.ui import render

SQRT3 = np.sqrt(3.0)


def cell_colors(grid: LatticeGrid, palette: np.ndarray) -> np.ndarray:
    """(height, width, 3) uint8: cell (row, col)'s colour at [row, col]."""
    state = render.occupancy_state(grid, palette)
    return palette[state].reshape(grid.height, grid.width, 3)


def surface_window(
    shape: str, width: int, height: int, wrap: bool
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The lattice rectangle the torus shows, in `lattice_cell`'s units: its corner (x, y) at
    u = 0, v = 1 (the back of the ring, the inner equator), and its size (x across one trip
    around the ring, y down one trip around the tube).

    The corner sits on the grid's left and top edges, so the seams are where the grid's edges
    are and each cell keeps its own place on the lattice (its hex column's half-cell shift, its
    triangle's up/down). Wrapped, the window is exactly one period (`width` x `height` cells
    from the edges' average line), so the surface continues across the seams with no break.
    Bounded, it is the 2D view's footprint (`render._footprint_factors`): every cell whole, and
    the notches between the edge cells of a hexagon or triangle grid show the background."""
    if shape == "square":
        return (0.0, 0.0), (float(width), float(height))
    if shape == "hexagon":  # x: 1.5 per column; y: sqrt 3 per row, odd columns half a row down
        if wrap:
            return (-0.75, -SQRT3 / 4), (1.5 * width, SQRT3 * height)
        return (-1.0, -SQRT3 / 2), (1.5 * width + 0.5, SQRT3 * (height + 0.5))
    if shape == "triangle":  # x: half a cell per column; y: one per row
        if wrap:
            return (-0.5, -0.5), (float(width), float(height))
        return (-1.0, -0.5), (width + 1.0, float(height))
    raise ValueError(f"Unsupported grid shape: {shape}")


def lattice_point(
    shape: str, width: int, height: int, u, v, *, wrap: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Surface coordinates -> a point (x, y) on the shape's unbounded lattice: columns run with
    u (around the ring) and rows against v (down the outer wall)."""
    (x0, y0), (span_x, span_y) = surface_window(shape, width, height, wrap)
    x = x0 + np.asarray(u, dtype=np.float64) * span_x
    y = y0 + (1.0 - np.asarray(v, dtype=np.float64)) * span_y
    return x, y


def lattice_cell(shape: str, x, y) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) on the unbounded lattice of the cell containing lattice point (x, y).

    Square points are in cells from the grid's corner. Hexagon points are in units of the
    hexagon size with cell (0, 0)'s centre at the origin (`render._hex_at` with size 1). Triangle
    x is in half cells and y in rows, both from cell (0, 0)'s midpoint (`render._triangle_at`
    with x scaled by size/2 and y by size*sqrt(3)/2)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if shape == "square":
        return np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
    if shape == "hexagon":
        q0 = np.round(x / 1.5).astype(np.int64)
        best = np.full(x.shape, np.inf)
        rows = np.zeros(x.shape, dtype=np.int64)
        cols = np.zeros(x.shape, dtype=np.int64)
        for dq in (-1, 0, 1):
            q = q0 + dq
            shift = 0.5 * (q & 1)
            r0 = np.round(y / SQRT3 - shift).astype(np.int64)
            for dr in (-1, 0, 1):
                r = r0 + dr
                d2 = (x - 1.5 * q) ** 2 + (y - SQRT3 * (r + shift)) ** 2
                closer = d2 < best
                best = np.where(closer, d2, best)
                rows = np.where(closer, r, rows)
                cols = np.where(closer, q, cols)
        return rows, cols
    if shape == "triangle":
        rows = np.floor(y + 0.5).astype(np.int64)
        depth = y + 0.5 - rows
        c0 = np.floor(x).astype(np.int64)
        up = (rows + c0) % 2 == 0
        margin = np.where(up, depth, 1.0 - depth) - (x - c0)
        return rows, np.where(margin >= 0, c0, c0 + 1)
    raise ValueError(f"Unsupported grid shape: {shape}")


def surface_cells(
    shape: str, width: int, height: int, u, v, *, wrap: bool
) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) of the grid cell the torus shows at surface coordinates (u, v): lattice cells
    beyond a wrapped grid's edges are the cells they wrap to; beyond a bounded grid's edges
    there is no cell, (-1, -1), and the surface shows the background."""
    x, y = lattice_point(shape, width, height, u, v, wrap=wrap)
    rows, cols = lattice_cell(shape, x, y)
    if wrap:
        return rows % height, cols % width
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return np.where(inside, rows, -1), np.where(inside, cols, -1)
