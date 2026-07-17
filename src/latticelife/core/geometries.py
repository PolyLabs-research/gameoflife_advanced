"""Geometry as data: build_adjacency() turns a lattice shape into a uniform adjacency array.

Cell index = r * width + c. The returned array has shape (n_cells, max_degree), dtype int32,
padded with -1 for cells with fewer neighbors than the geometry's max degree (i.e. wrap=False
clamped edges/corners). wrap=True uses modulo indexing; wrap=False omits out-of-range neighbors.
This is the only place lattice geometry is expressed — `core.lattice` consumes the array
uniformly regardless of shape.

Hex (odd-q offset) and triangle (edge / edge+vertex) neighbor discovery are ported from the
legacy `hexagon_grid_numpy.py:_create_neighbor_map` and
`triangle_grid_numpy.py:_build_neighbor_offsets` (see `docs/adjacency.md`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from functools import cache

import numpy as np

Offsets = Iterable[tuple[int, int]]
OffsetFn = Callable[[int, int], Offsets]

_SQUARE_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)

_HEX_ODD_OFFSETS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (1, 1))
_HEX_EVEN_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
)


def _square_offsets(r: int, c: int) -> Offsets:
    return _SQUARE_OFFSETS


def _hex_offsets(r: int, c: int) -> Offsets:
    return _HEX_ODD_OFFSETS if c % 2 == 1 else _HEX_EVEN_OFFSETS


def _triangle_corners(r: int, c: int, upward: bool, size: float = 2.0) -> list[tuple[float, float]]:
    """Corners of the notional triangle at (r, c). Used only for adjacency discovery."""
    x0 = c * (size / 2)
    y0 = r * (size * math.sqrt(3) / 2)
    half_size = size / 2
    height = size * math.sqrt(3) / 2
    if upward:
        corners = [
            (x0, y0 - height / 2),
            (x0 - half_size, y0 + height / 2),
            (x0 + half_size, y0 + height / 2),
        ]
    else:
        corners = [
            (x0, y0 + height / 2),
            (x0 - half_size, y0 - height / 2),
            (x0 + half_size, y0 - height / 2),
        ]
    return [(round(cx, 6), round(cy, 6)) for cx, cy in corners]


@cache
def _triangle_offsets(mode: str) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    """Relative neighbor offsets for upward/downward triangles, discovered geometrically."""
    if mode not in ("edge", "edge+vertex"):
        raise ValueError(f"Unsupported triangle neighborhood mode: {mode}")
    min_shared = 2 if mode == "edge" else 1

    def discover(base_r: int, base_c: int, upward: bool) -> frozenset[tuple[int, int]]:
        origin_corners = set(_triangle_corners(base_r, base_c, upward))
        offsets: set[tuple[int, int]] = set()
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = base_r + dr, base_c + dc
                neighbor_upward = (nr + nc) % 2 == 0
                corners = set(_triangle_corners(nr, nc, neighbor_upward))
                if len(origin_corners & corners) >= min_shared:
                    offsets.add((dr, dc))
        return frozenset(offsets)

    offsets_up = discover(0, 0, True)  # even parity
    offsets_down = discover(0, 1, False)  # odd parity
    if len(offsets_up) != len(offsets_down):
        raise ValueError("Triangle neighbor offsets differ between orientations")
    return offsets_up, offsets_down


def max_degree(shape: str, triangle_mode: str = "edge+vertex") -> int:
    """The wrap-mode neighbor count for a shape (adjacency degree table)."""
    shape = shape.lower()
    if shape == "square":
        return 8
    if shape == "hexagon":
        return 6
    if shape == "triangle":
        offsets_up, _ = _triangle_offsets(triangle_mode)
        return len(offsets_up)
    raise ValueError(f"Unsupported grid shape: {shape}")


def _cell_index(r: int, c: int, width: int) -> int:
    return r * width + c


def _neighbor_lists(width: int, height: int, wrap: bool, offset_fn: OffsetFn) -> list[list[int]]:
    lists: list[list[int]] = []
    for r in range(height):
        for c in range(width):
            neighbors: list[int] = []
            for dr, dc in offset_fn(r, c):
                nr, nc = r + dr, c + dc
                if wrap:
                    nr %= height
                    nc %= width
                    neighbors.append(_cell_index(nr, nc, width))
                elif 0 <= nr < height and 0 <= nc < width:
                    neighbors.append(_cell_index(nr, nc, width))
            lists.append(neighbors)
    return lists


def _pack(neighbor_lists: list[list[int]], degree: int) -> np.ndarray:
    adjacency = -np.ones((len(neighbor_lists), degree), dtype=np.int32)
    for i, neighbors in enumerate(neighbor_lists):
        adjacency[i, : len(neighbors)] = neighbors
    return adjacency


def build_adjacency(
    shape: str, width: int, height: int, wrap: bool, triangle_mode: str = "edge+vertex"
) -> np.ndarray:
    """Build the (n_cells, max_degree) int32 adjacency array, padded with -1."""
    shape = shape.lower()
    degree = max_degree(shape, triangle_mode)

    offset_fn: OffsetFn
    if shape == "square":
        offset_fn = _square_offsets
    elif shape == "hexagon":
        offset_fn = _hex_offsets
    elif shape == "triangle":
        offsets_up, offsets_down = _triangle_offsets(triangle_mode)

        def offset_fn(r: int, c: int) -> Offsets:
            return offsets_up if (r + c) % 2 == 0 else offsets_down
    else:
        raise ValueError(f"Unsupported grid shape: {shape}")

    neighbor_lists = _neighbor_lists(width, height, wrap, offset_fn)
    return _pack(neighbor_lists, degree)
