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
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from latticelife.core.geometries import validate_dimensions

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
