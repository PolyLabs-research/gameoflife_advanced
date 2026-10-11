"""Grid -> torus texture, and the numpy twin of the shader's cell lookup.

`cell_colors` gives the RGB texture the torus samples: one texel per cell, coloured through
`render.occupancy_state`, the same alive/static fold the 2D view uses (with the grid's own
static threshold), so both views always agree on what a cell looks like.

`surface_cells` answers, in numpy, the question `shaders.CELL_LOOKUP` answers per pixel: which
cell does surface point (u, v) show? It is the reference the GL tests check the shader against,
and is documented in `shaders.py`.
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


def lattice_point(shape: str, width: int, height: int, u, v) -> tuple[np.ndarray, np.ndarray]:
    """Surface coordinates -> a point (x, y) on the shape's unbounded lattice."""
    x = (np.asarray(u, dtype=np.float64) + 0.5) * width
    y = (0.5 - np.asarray(v, dtype=np.float64)) * height
    if shape == "hexagon":
        return x * 1.5, y * SQRT3
    if shape in ("square", "triangle"):
        return x, y
    raise ValueError(f"Unsupported grid shape: {shape}")


def lattice_cell(shape: str, x, y) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) on the unbounded lattice of the cell containing lattice point (x, y).

    Hexagon points are in units of the hexagon size with cell (0, 0)'s centre at the origin
    (`render._hex_at` with size 1). Triangle x is in half cells and y in rows, both from cell
    (0, 0)'s midpoint (`render._triangle_at` with x scaled by size/2 and y by size*sqrt(3)/2)."""
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


def surface_cells(shape: str, width: int, height: int, u, v) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) of the grid cell the torus shows at surface coordinates (u, v)."""
    x, y = lattice_point(shape, width, height, u, v)
    rows, cols = lattice_cell(shape, x, y)
    return rows % height, cols % width
