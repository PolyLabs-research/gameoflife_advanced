"""Geometry as data: build_adjacency() turns a lattice shape into a uniform adjacency array.

Cell index = r * width + c. The returned array has shape (n_cells, max_degree), dtype int32,
padded with -1 for cells with fewer neighbors than the geometry's max degree (i.e. wrap=False
clamped edges/corners). wrap=True uses modulo indexing; wrap=False omits out-of-range neighbors.
This is the only place lattice geometry is expressed — `core.lattice` consumes the array
uniformly regardless of shape.

Hex (odd-q offset) and triangle (edge / edge+vertex) neighbor discovery are ported from the
legacy `hexagon_grid_numpy.py:_create_neighbor_map` and
`triangle_grid_numpy.py:_build_neighbor_offsets` (see `docs/adjacency.md`).

Not every width/height can wrap. Hex offsets depend on column parity and triangle orientation on
(row + column) parity, so a wrapped hexagon grid needs an even width and a wrapped triangle grid
an even width and height — otherwise the seam joins cells of the same parity and the §6.1
symmetry invariant breaks. A wrapped grid narrower than the neighbor stencil folds offsets onto
each other (a cell becomes its own neighbor, or one neighbor is counted twice).
`validate_dimensions()` rejects both; `build_adjacency()` calls it, and config loading and the
settings panel call it to report the problem before anything is built.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from functools import cache

import numpy as np

SHAPES: tuple[str, ...] = ("square", "hexagon", "triangle")
TRIANGLE_MODES: tuple[str, ...] = ("edge", "edge+vertex")

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


def _offset_classes(shape: str, triangle_mode: str) -> tuple[frozenset[tuple[int, int]], ...]:
    """Every distinct offset set a cell of `shape` can use (one per parity class)."""
    if shape == "square":
        return (frozenset(_SQUARE_OFFSETS),)
    if shape == "hexagon":
        return (frozenset(_HEX_EVEN_OFFSETS), frozenset(_HEX_ODD_OFFSETS))
    if shape == "triangle":
        return _triangle_offsets(triangle_mode)
    raise ValueError(f"Unsupported grid shape: {shape}")


@cache
def min_wrap_size(shape: str, triangle_mode: str = "edge+vertex") -> tuple[int, int]:
    """Smallest (width, height) a wrapped `shape` grid can have, derived from the stencil:
    along each axis, the size must exceed the largest gap between two offsets of one cell
    (counting the cell itself as offset (0, 0)), or modulo wrapping folds them onto the same
    cell. Rounded up to even where the shape's parity rule requires it (see module docstring).
    square: 3x3, hexagon: 4x3, triangle edge: 4x2, triangle edge+vertex: 6x4."""
    shape = shape.lower()
    span_r = span_c = 0
    for offsets in _offset_classes(shape, triangle_mode):
        rows = [dr for dr, _ in offsets] + [0]
        cols = [dc for _, dc in offsets] + [0]
        span_r = max(span_r, max(rows) - min(rows))
        span_c = max(span_c, max(cols) - min(cols))
    min_width, min_height = span_c + 1, span_r + 1
    even_width, even_height = _parity_requirements(shape)
    if even_width:
        min_width += min_width % 2
    if even_height:
        min_height += min_height % 2
    return min_width, min_height


def _parity_requirements(shape: str) -> tuple[bool, bool]:
    """(width must be even, height must be even) for a wrapped grid of `shape`."""
    if shape == "hexagon":
        return True, False
    if shape == "triangle":
        return True, True
    return False, False


def validate_dimensions(
    shape: str, width: int, height: int, wrap: bool, triangle_mode: str = "edge+vertex"
) -> None:
    """Raise ValueError, with a message fit to show the user, if `shape` cannot be built at
    `width` x `height` with this `wrap` setting (see module docstring). Odd sizes are rejected,
    never rounded: the caller decides what to do about them."""
    shape = shape.lower()
    max_degree(shape, triangle_mode)  # unknown shape / triangle mode -> ValueError
    if width < 1 or height < 1:
        raise ValueError(f"Grid width and height must be at least 1 (got {width} x {height}).")
    if not wrap:
        return

    even_width, even_height = _parity_requirements(shape)
    if even_width and even_height and (width % 2 or height % 2):
        raise ValueError(
            f"A wrapped triangle grid needs an even width and height (got {width} x {height}): "
            "triangles alternate up/down, so an odd size breaks the pattern across the seam."
        )
    if even_width and width % 2:
        raise ValueError(
            f"A wrapped hexagon grid needs an even width (got {width}): hex columns alternate, "
            "so an odd width breaks the pattern across the seam."
        )

    min_width, min_height = min_wrap_size(shape, triangle_mode)
    if width < min_width or height < min_height:
        name = f"triangle ({triangle_mode})" if shape == "triangle" else shape
        raise ValueError(
            f"A wrapped {name} grid must be at least {min_width} x {min_height} "
            f"(got {width} x {height}): a smaller grid makes a cell its own neighbor or counts "
            "one neighbor twice."
        )


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
    """Build the (n_cells, max_degree) int32 adjacency array, padded with -1. Raises
    ValueError for sizes `validate_dimensions` rejects."""
    shape = shape.lower()
    degree = max_degree(shape, triangle_mode)
    validate_dimensions(shape, width, height, wrap, triangle_mode)

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
