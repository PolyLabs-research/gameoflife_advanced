"""Torus translations as data: the translation group of a wrapped lattice, acting on flat cell
indices, built on the same conventions as `core.geometries` (odd-q offset hexagon columns,
triangles alternating in orientation by (row + column) parity).

A translation is named by where it sends the cell at row 0, column 0: the `(dr, dc)` of that
cell's image, reduced modulo (height, width). A translation is determined by the image of any one
cell, so two names are the same translation exactly when they agree modulo the torus. On the
square lattice the name is the ordinary shift. On the hexagon lattice a one-column move from an
even column keeps the row while the same move from an odd column also goes one row down (odd-q
offset), so the name is the move as column 0 (even) sees it, and the arithmetic goes through
axial coordinates, where every translation is a plain vector sum. On the triangle lattice only
`(dr, dc)` with `dr + dc` even are translations: up and down triangles are not translates of one
another, so the translation group has index 2 in the shifts, and `between()` is None for two
cells of different orientation.

Every translation is an automorphism of the wrapped adjacency array; the test suite checks that
for each shape over a range of sizes, because it is the property every use relies on: a translated
pattern evolves exactly as the original, translated (`experiments.objects`). Wrap is required —
a clamped grid has no translations.

`pattern_key` turns a set of cells into a translation-invariant identity that does not depend on
the torus it was read from (anchor at the origin, the other cells in the window centred on it),
provided the pattern spans less than half the torus each way; `place` puts such a key back onto
any torus. Both keep the lattice's constraints: a hexagon pattern is only ever moved by whole
pairs of columns' worth of parity, i.e. its anchor stays on an even column, and a triangle
pattern's anchor keeps its orientation, so a key names one geometric shape.

Point groups. Outer-totalistic rules only count neighbours, so they commute with every symmetry
of the lattice, not just its translations: a rotated or reflected object is the same object. A
key's cells are plane coordinates (offset coordinates relative to an origin cell of the even
class), so the point group acts on keys directly (`point_group`, `transform_key`), and
`canonical_key` names a pattern up to translation, rotation and reflection: the smallest key
among its images. The square (Moore) lattice has the 8 symmetries of the square; the hexagon
lattice the 12 of the regular hexagon, done exactly in cube coordinates; the triangle lattice the
12 of the triangular tiling, done on cell centroids rotated about a tiling vertex (a rotation by 60
degrees turns up triangles into down ones, which `pattern_key` keeps apart by the anchor's
orientation). The test suite checks that each element maps neighbours to neighbours and commutes
with `LatticeGrid.step` for random rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from functools import cache

import numpy as np

from latticelife.core.geometries import min_wrap_size, validate_dimensions

Vector = tuple[int, int]
PatternKey = tuple[tuple[int, int], ...]


class TorusTranslations:
    def __init__(
        self, shape: str, width: int, height: int, triangle_mode: str = "edge+vertex"
    ) -> None:
        shape = shape.lower()
        validate_dimensions(shape, width, height, True, triangle_mode)
        self.shape = shape
        self.width = width
        self.height = height
        self.triangle_mode = triangle_mode
        self.n_cells = width * height
        cells = np.arange(self.n_cells)
        self._rows = cells // width
        self._cols = cells % width

    @property
    def order(self) -> int:
        """How many distinct translations the torus has: one per cell, except that a triangle
        can only go to a triangle of its own orientation."""
        return self.n_cells // 2 if self.shape == "triangle" else self.n_cells

    def reduce(self, vector: Vector) -> Vector:
        """The name of `vector`'s translation reduced modulo the torus, so equal translations
        compare equal. Raises ValueError for a triangle shift that changes orientation."""
        dr, dc = int(vector[0]), int(vector[1])
        if self.shape == "triangle" and (dr + dc) % 2:
            raise ValueError(
                f"({dr}, {dc}) is not a triangle translation: up and down triangles are not "
                "translates of each other, so dr + dc must be even"
            )
        return dr % self.height, dc % self.width

    def _images(self, rows: np.ndarray, cols: np.ndarray, vector: Vector) -> np.ndarray:
        """Flat indices of the cells at (rows, cols) moved by the translation `vector` names."""
        dr, dc = self.reduce(vector)
        if self.shape == "hexagon":
            # The name is column 0's view; as an axial vector it is (dq, dr_axial).
            dq = dc
            dr_axial = dr - (dq - (dq & 1)) // 2
            r_axial = rows - (cols - (cols & 1)) // 2
            q = cols + dq
            r = r_axial + dr_axial + (q - (q & 1)) // 2
        else:
            r = rows + dr
            q = cols + dc
        return (r % self.height) * self.width + (q % self.width)

    def permutation(self, vector: Vector) -> np.ndarray:
        """The image of every cell: `permutation(v)[i]` is where cell `i` goes."""
        return self._images(self._rows, self._cols, vector)

    def apply(self, cells: Iterable[int], vector: Vector) -> np.ndarray:
        """The images of `cells` (flat indices) under the translation `vector` names."""
        cells = np.asarray(list(cells) if not isinstance(cells, np.ndarray) else cells)
        cells = cells.astype(np.int64, copy=False)
        return self._images(cells // self.width, cells % self.width, vector)

    def between(self, source: int, target: int) -> Vector | None:
        """The name of the translation carrying cell `source` to cell `target`, or None when
        there is none (a triangle of the other orientation)."""
        rs, cs = divmod(int(source), self.width)
        rt, ct = divmod(int(target), self.width)
        if self.shape == "triangle":
            if (rs + cs) % 2 != (rt + ct) % 2:
                return None
            return self.reduce((rt - rs, ct - cs))
        if self.shape == "hexagon":
            dq = ct - cs
            dr_axial = (rt - (ct - (ct & 1)) // 2) - (rs - (cs - (cs & 1)) // 2)
            return self.reduce((dr_axial + (dq - (dq & 1)) // 2, dq))
        return self.reduce((rt - rs, ct - cs))

    def inverse(self, vector: Vector) -> Vector:
        image = int(self.apply(np.zeros(1, dtype=np.int64), vector)[0])
        inverse = self.between(image, 0)
        assert inverse is not None  # a translation's inverse always exists
        return inverse

    def find(self, source: np.ndarray, target: np.ndarray) -> Vector | None:
        """A translation carrying the boolean pattern `source` onto `target` (same cell count,
        every image alive), or None. Tries only the translations that take `source`'s first
        live cell to a live cell of `target`, each checked in time proportional to the pattern
        size, not the grid. Two empty patterns match under the identity."""
        source = np.asarray(source, dtype=bool).ravel()
        target = np.asarray(target, dtype=bool).ravel()
        if source.shape != (self.n_cells,) or target.shape != (self.n_cells,):
            raise ValueError("patterns must be boolean arrays with one entry per cell")
        source_cells = np.flatnonzero(source)
        target_cells = np.flatnonzero(target)
        if source_cells.size != target_cells.size:
            return None
        if source_cells.size == 0:
            return (0, 0)
        anchor = int(source_cells[0])
        for candidate in target_cells:
            vector = self.between(anchor, int(candidate))
            if vector is None:
                continue
            if target[self.apply(source_cells, vector)].all():
                return vector
        return None

    def centred(self, vector: Vector) -> Vector:
        """`vector` reduced into the window (-height/2, height/2] x (-width/2, width/2]: the
        shortest way of writing a translation, e.g. a spaceship's displacement."""
        dr, dc = self.reduce(vector)
        if dr > self.height // 2:
            dr -= self.height
        if dc > self.width // 2:
            dc -= self.width
        return dr, dc

    def _origin(self, cell: int) -> int:
        """The cell a pattern's anchor is moved to for its key: (0, 0), or (0, 1) for a
        triangle of the other orientation."""
        if self.shape == "triangle" and (cell // self.width + cell % self.width) % 2:
            return 1
        return 0

    def pattern_key(self, cells: Iterable[int]) -> PatternKey | None:
        """Translation-invariant identity of the pattern `cells` (flat indices): with one of its
        cells moved to the origin, the sorted `(row, column)` of every cell read in the window
        centred on the origin; the smallest such tuple over every choice of anchor. Equal for
        two patterns exactly when one is a translate of the other, and the same whatever torus
        the pattern sits on, as long as it spans less than half the torus each way — None when
        it does not, since its cells then have no unambiguous relative positions."""
        cells = np.unique(np.asarray(list(cells), dtype=np.int64))
        if cells.size == 0:
            return ()
        half_height, half_width = self.height // 2, self.width // 2
        best: PatternKey | None = None
        for anchor in cells:
            vector = self.between(int(anchor), self._origin(int(anchor)))
            assert vector is not None  # the origin has the anchor's orientation
            moved = self.apply(cells, vector)
            rows = moved // self.width
            cols = moved % self.width
            rows = np.where(rows > half_height, rows - self.height, rows)
            cols = np.where(cols > half_width, cols - self.width, cols)
            if rows.max() - rows.min() >= half_height or cols.max() - cols.min() >= half_width:
                return None
            key = tuple(sorted(zip(rows.tolist(), cols.tolist(), strict=True)))
            if best is None or key < best:
                best = key
        return best

    def place(self, key: PatternKey, row: int, column: int) -> np.ndarray:
        """Flat indices of `key` put on this torus with the key's origin at (`row`, `column`).
        The key's frame has an even column, even-parity cell at its origin, so the origin must
        land on one: an even column on hexagons, `row + column` even on triangles. Every cell
        of the key then keeps its column parity and orientation, and so its geometry. Raises
        ValueError otherwise."""
        if self.shape == "hexagon" and column % 2:
            raise ValueError(
                f"a hexagon pattern can only be placed on an even column (got {column})"
            )
        if self.shape == "triangle" and (row + column) % 2:
            raise ValueError(
                f"a triangle pattern must keep its orientation: its origin must have even "
                f"row + column (got ({row}, {column}))"
            )
        if not key:
            return np.zeros(0, dtype=np.int64)
        rows = np.array([r for r, _ in key], dtype=np.int64) + row
        cols = np.array([c for _, c in key], dtype=np.int64) + column
        return (rows % self.height) * self.width + (cols % self.width)


def key_extent(key: PatternKey) -> tuple[int, int, int, int]:
    """(min_row, min_col, max_row, max_col) of a key; (0, 0, -1, -1) for the empty key."""
    if not key:
        return 0, 0, -1, -1
    rows = [r for r, _ in key]
    cols = [c for _, c in key]
    return min(rows), min(cols), max(rows), max(cols)


# --- point groups -------------------------------------------------------------------------------

Cell = tuple[int, int]
CellMap = Callable[[Cell], Cell]


def _square_group() -> list[CellMap]:
    def rotate(cell: Cell) -> Cell:
        r, c = cell
        return c, -r

    def reflect(cell: Cell) -> Cell:
        r, c = cell
        return r, -c

    return _closure(rotate, reflect, 8)


def _hex_to_cube(cell: Cell) -> tuple[int, int, int]:
    r, c = cell
    x = c
    z = r - (c - (c & 1)) // 2
    return x, -x - z, z


def _cube_to_hex(x: int, y: int, z: int) -> Cell:
    assert x + y + z == 0
    return z + (x - (x & 1)) // 2, x


def _hexagon_group() -> list[CellMap]:
    def rotate(cell: Cell) -> Cell:  # 60 degrees about the centre of cell (0, 0)
        x, y, z = _hex_to_cube(cell)
        return _cube_to_hex(-z, -x, -y)

    def reflect(cell: Cell) -> Cell:
        x, y, z = _hex_to_cube(cell)
        return _cube_to_hex(x, z, y)

    return _closure(rotate, reflect, 12)


_SQRT3 = math.sqrt(3.0)


def _triangle_centroid(cell: Cell) -> tuple[float, float]:
    """Centroid of the triangle at (row, column) in the geometry `core.geometries` builds the
    adjacency from (side 2): up triangles on even row + column, apex up."""
    r, c = cell
    up = (r + c) % 2 == 0
    return float(c), r * _SQRT3 + (_SQRT3 / 6 if up else -_SQRT3 / 6)


def _triangle_cell(x: float, y: float) -> Cell:
    c = round(x)
    if abs(x - c) > 1e-6:
        raise ValueError(f"({x}, {y}) is not a triangle centroid")
    for offset, up in ((_SQRT3 / 6, True), (-_SQRT3 / 6, False)):
        r_float = (y - offset) / _SQRT3
        r = round(r_float)
        if abs(r_float - r) < 1e-6 and ((r + c) % 2 == 0) == up:
            return r, c
    raise ValueError(f"({x}, {y}) is not a triangle centroid")


def _triangle_group() -> list[CellMap]:
    # Rotation by 60 degrees about the apex of the up triangle (0, 0), a vertex of the tiling.
    vx, vy = 0.0, -_SQRT3 / 2
    cos60, sin60 = 0.5, _SQRT3 / 2

    def rotate(cell: Cell) -> Cell:
        x, y = _triangle_centroid(cell)
        dx, dy = x - vx, y - vy
        return _triangle_cell(vx + cos60 * dx - sin60 * dy, vy + sin60 * dx + cos60 * dy)

    def reflect(cell: Cell) -> Cell:  # the vertical line through that vertex
        r, c = cell
        return r, -c

    return _closure(rotate, reflect, 12)


def _closure(rotate: CellMap, reflect: CellMap, order: int) -> list[CellMap]:
    """The `order` maps rotate^k and rotate^k after reflect."""
    turns = order // 2

    def element(k: int, mirrored: bool) -> CellMap:
        def apply(cell: Cell) -> Cell:
            if mirrored:
                cell = reflect(cell)
            for _ in range(k):
                cell = rotate(cell)
            return cell

        return apply

    return [element(k, mirrored) for mirrored in (False, True) for k in range(turns)]


@cache
def point_group(shape: str) -> tuple[CellMap, ...]:
    """Every symmetry of `shape`'s lattice fixing a point, as maps on plane cell coordinates
    (offset row and column; a key's cells are such coordinates). The identity comes first."""
    shape = shape.lower()
    if shape == "square":
        return tuple(_square_group())
    if shape == "hexagon":
        return tuple(_hexagon_group())
    if shape == "triangle":
        return tuple(_triangle_group())
    raise ValueError(f"Unsupported grid shape: {shape}")


def plane_key(cells: Iterable[Cell], shape: str, triangle_mode: str = "edge+vertex") -> PatternKey:
    """`pattern_key` of a pattern given by plane cell coordinates (any integers), read on a torus
    large enough that the answer cannot depend on it."""
    cells = list(cells)
    if not cells:
        return ()
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    # Shift into the positive quadrant by a translation the lattice allows (an even number of
    # columns keeps hexagon column parity; an even row + column shift keeps triangle orientation).
    shift_r, shift_c = -min(rows), -min(cols)
    if shape == "hexagon" and shift_c % 2:
        shift_c += 1
    if shape == "triangle" and (shift_r + shift_c) % 2:
        shift_c += 1
    span_r = max(rows) - min(rows) + 2
    span_c = max(cols) - min(cols) + 2
    min_width, min_height = min_wrap_size(shape, triangle_mode)
    height = max(2 * span_r + 2, min_height)
    width = max(2 * span_c + 2, min_width)
    height += height % 2
    width += width % 2
    torus = TorusTranslations(shape, width, height, triangle_mode)
    indices = [(r + shift_r) * width + (c + shift_c) for r, c in cells]
    key = torus.pattern_key(indices)
    assert key is not None  # the torus is more than twice the pattern's extent
    return key


def transform_key(
    key: PatternKey, element: CellMap, shape: str, triangle_mode: str = "edge+vertex"
) -> PatternKey:
    """The key of `key`'s pattern moved by one point-group element."""
    return plane_key((element(cell) for cell in key), shape, triangle_mode)


def symmetric_images(
    key: PatternKey, shape: str, triangle_mode: str = "edge+vertex"
) -> set[PatternKey]:
    """The distinct keys of `key`'s pattern under the lattice's point group: how many there are
    is the pattern's number of orientations (the group order over its stabiliser's)."""
    return {transform_key(key, g, shape, triangle_mode) for g in point_group(shape)}


def canonical_key(key: PatternKey, shape: str, triangle_mode: str = "edge+vertex") -> PatternKey:
    """The identity of `key`'s pattern up to translation, rotation and reflection: the smallest
    of its images' keys."""
    return min(symmetric_images(key, shape, triangle_mode))
