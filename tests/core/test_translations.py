"""The torus translation group (`core.translations`): every translation must be an automorphism
of the wrapped adjacency, since `experiments.objects` relies on a translated pattern evolving as
the original does. Checked for every shape over a range of sizes, not just the square lattice
where translations are plain shifts."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from latticelife.core.geometries import build_adjacency, min_wrap_size
from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.core.translations import TorusTranslations, key_extent

SHAPES = [
    ("square", "edge+vertex"),
    ("hexagon", "edge+vertex"),
    ("triangle", "edge"),
    ("triangle", "edge+vertex"),
]
SIZES = [(6, 4), (6, 6), (8, 6), (10, 8)]  # (width, height), all valid for every shape


def _is_automorphism(adjacency: np.ndarray, perm: np.ndarray) -> bool:
    """perm maps each cell's neighbour set onto the neighbour set of its image."""
    for i in range(adjacency.shape[0]):
        expected = sorted(perm[adjacency[i]].tolist())
        actual = sorted(adjacency[perm[i]].tolist())
        if expected != actual:
            return False
    return True


@pytest.mark.parametrize(("shape", "mode"), SHAPES)
@pytest.mark.parametrize(("width", "height"), SIZES)
def test_every_translation_is_an_adjacency_automorphism(shape, mode, width, height):
    t = TorusTranslations(shape, width, height, mode)
    adjacency = build_adjacency(shape, width, height, True, mode)
    seen = set()
    for dr in range(height):
        for dc in range(width):
            if shape == "triangle" and (dr + dc) % 2:
                continue
            perm = t.permutation((dr, dc))
            assert sorted(perm.tolist()) == list(range(t.n_cells))  # a bijection
            assert _is_automorphism(adjacency, perm), f"{shape} ({dr}, {dc})"
            seen.add(perm.tobytes())
    # Distinct names give distinct translations, and there are exactly `order` of them.
    assert len(seen) == t.order


@pytest.mark.parametrize(("shape", "mode"), SHAPES)
def test_between_is_the_translation_carrying_source_to_target(shape, mode):
    t = TorusTranslations(shape, 8, 6, mode)
    for source in range(t.n_cells):
        for target in range(t.n_cells):
            vector = t.between(source, target)
            same_orientation = (
                shape != "triangle"
                or (source // 8 + source % 8) % 2 == (target // 8 + target % 8) % 2
            )
            if not same_orientation:
                assert vector is None
                continue
            assert vector is not None
            assert int(t.permutation(vector)[source]) == target
            assert t.apply([source], vector).tolist() == [target]


def test_hexagon_one_column_move_goes_down_a_row_from_an_odd_column():
    """Odd-q offset: the name (0, 1) is column 0's move to its right-hand neighbour on the same
    row; an odd column's same translation lands one row further down."""
    t = TorusTranslations("hexagon", 8, 6)
    perm = t.permutation((0, 1))
    assert divmod(int(perm[0]), 8) == (0, 1)  # (0, 0) -> (0, 1)
    assert divmod(int(perm[1]), 8) == (1, 2)  # (0, 1) -> (1, 2)
    assert divmod(int(perm[2 * 8 + 2]), 8) == (2, 3)  # (2, 2) -> (2, 3)
    # Two one-column moves along the lattice land one row down: the plain two-column shift
    # (0, 2) is a different lattice vector.
    assert np.array_equal(perm[perm], t.permutation((1, 2)))
    assert not np.array_equal(perm[perm], t.permutation((0, 2)))


@pytest.mark.parametrize(("shape", "mode"), SHAPES)
def test_names_are_reduced_modulo_the_torus(shape, mode):
    t = TorusTranslations(shape, 8, 6, mode)
    base = (2, 2)
    assert np.array_equal(t.permutation(base), t.permutation((2 + 6, 2)))
    assert np.array_equal(t.permutation(base), t.permutation((2, 2 + 8)))
    assert np.array_equal(t.permutation(base), t.permutation((2 - 6, 2 - 16)))
    assert t.reduce((2 - 6, 2 - 16)) == base


def test_triangle_rejects_an_orientation_changing_shift():
    t = TorusTranslations("triangle", 8, 6, "edge")
    with pytest.raises(ValueError, match="dr \\+ dc must be even"):
        t.permutation((0, 1))


@pytest.mark.parametrize(("shape", "mode"), SHAPES)
def test_inverse_undoes_a_translation(shape, mode):
    t = TorusTranslations(shape, 8, 6, mode)
    for vector in [(0, 0), (1, 1), (3, 3), (5, 7), (2, 4)]:
        perm = t.permutation(vector)
        back = t.permutation(t.inverse(vector))
        assert np.array_equal(back[perm], np.arange(t.n_cells))


@pytest.mark.parametrize(("shape", "mode"), SHAPES)
@given(
    pattern_seed=st.integers(min_value=0, max_value=10_000),
    dr=st.integers(min_value=0, max_value=5),
    dc=st.integers(min_value=0, max_value=7),
)
@settings(max_examples=30, deadline=None)
def test_find_recovers_a_translation_applied_to_a_random_pattern(shape, mode, pattern_seed, dr, dc):
    t = TorusTranslations(shape, 8, 6, mode)
    if shape == "triangle" and (dr + dc) % 2:
        dc += 1
    rng = np.random.default_rng(pattern_seed)
    source = rng.random(t.n_cells) < 0.3
    target = np.zeros_like(source)
    target[t.permutation((dr, dc))] = source

    found = t.find(source, target)

    assert found is not None
    moved = np.zeros_like(source)
    moved[t.permutation(found)] = source
    assert np.array_equal(moved, target)


@pytest.mark.parametrize(("shape", "mode"), SHAPES)
def test_find_is_none_for_patterns_that_are_not_translates(shape, mode):
    t = TorusTranslations(shape, 8, 6, mode)
    source = np.zeros(t.n_cells, dtype=bool)
    source[[0, 2]] = True  # two cells two columns apart
    target = np.zeros(t.n_cells, dtype=bool)
    target[[0, 8 * 2]] = True  # two cells two rows apart: not a translate on these lattices
    assert t.find(source, target) is None
    assert t.find(source, np.zeros(t.n_cells, dtype=bool)) is None  # different cell counts
    assert t.find(np.zeros(t.n_cells, dtype=bool), np.zeros(t.n_cells, dtype=bool)) == (0, 0)


def test_find_also_sees_a_translation_of_the_whole_dense_grid():
    t = TorusTranslations("square", 8, 6)
    rng = np.random.default_rng(1)
    source = rng.random(t.n_cells) < 0.6
    target = np.zeros_like(source)
    target[t.permutation((3, 5))] = source
    assert t.find(source, target) is not None


@pytest.mark.parametrize(("shape", "mode"), SHAPES)
def test_pattern_key_is_the_same_for_every_translate_and_torus(shape, mode):
    """The key of a small pattern does not depend on where it sits or how big the torus is."""
    small = TorusTranslations(shape, 10, 8, mode)
    big = TorusTranslations(shape, 16, 14, mode)
    cells = np.array([r * 10 + c for r, c in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]])
    key = small.pattern_key(cells)
    assert key is not None
    assert (0, 0) in key or (0, 1) in key  # an anchor at the key frame's origin
    for vector in [(1, 1), (4, 2), (6, 6), (7, 9)]:
        if shape == "triangle" and sum(vector) % 2:
            vector = (vector[0], vector[1] + 1)
        assert small.pattern_key(small.apply(cells, vector)) == key
    # The same shape placed on the bigger torus: its key must not change with the torus.
    on_big = big.place(key, 4, 6)
    assert big.pattern_key(on_big) == key
    for vector in [(3, 3), (9, 11), (12, 14)]:
        if shape == "triangle" and sum(vector) % 2:
            vector = (vector[0], vector[1] + 1)
        assert big.pattern_key(big.apply(on_big, vector)) == key


@pytest.mark.parametrize(("shape", "mode"), SHAPES)
def test_pattern_key_distinguishes_patterns_that_are_not_translates(shape, mode):
    t = TorusTranslations(shape, 10, 8, mode)
    horizontal = t.pattern_key([0, 1, 2])
    vertical = t.pattern_key([0, 10, 20])
    assert horizontal != vertical
    assert t.pattern_key([]) == ()


def test_pattern_key_is_none_for_a_pattern_spanning_half_the_torus():
    t = TorusTranslations("square", 10, 8)
    assert t.pattern_key([0, 5]) is None  # columns 0 and 5 on a width of 10: ambiguous
    assert t.pattern_key([0, 4]) is not None
    assert t.pattern_key([0, 40]) is None  # rows 0 and 4 on a height of 8
    assert t.pattern_key([0, 30]) is not None


def test_place_keeps_a_hexagon_pattern_on_even_columns_and_a_triangle_its_orientation():
    hexagon = TorusTranslations("hexagon", 10, 8)
    key = hexagon.pattern_key([0, 1])
    assert hexagon.place(key, 2, 4).tolist() == sorted(hexagon.place(key, 2, 4).tolist())
    with pytest.raises(ValueError, match="even column"):
        hexagon.place(key, 2, 3)
    triangle = TorusTranslations("triangle", 10, 8, "edge")
    down = triangle.pattern_key([1])  # the cell (0, 1): a down triangle
    assert down == ((0, 1),)
    assert triangle.place(down, 2, 2).tolist() == [2 * 10 + 3]
    assert triangle.place(down, 3, 3).tolist() == [3 * 10 + 4]
    with pytest.raises(ValueError, match="orientation"):
        triangle.place(down, 2, 3)


def test_key_extent():
    assert key_extent(()) == (0, 0, -1, -1)
    assert key_extent(((0, 0), (-1, 2), (3, 1))) == (-1, 0, 3, 2)


@pytest.mark.parametrize(("shape", "mode"), SHAPES)
def test_a_translated_pattern_evolves_as_the_original_translated(shape, mode):
    """The property the object census relies on, end to end through LatticeGrid.step."""
    width, height = 12, 10
    t = TorusTranslations(shape, width, height, mode)
    rule = Rule(birth={2, 3}, survive={1, 2, 3})
    rng = np.random.default_rng(5)
    vector = (3, 5)  # dr + dc even, so it is a triangle translation too

    def run(initial):
        grid = LatticeGrid([Lifeform(1, rule)], shape, width, height, True, mode)
        grid.occupancy[:] = initial
        for _ in range(8):
            grid.step(rng)
        return grid.occupancy.copy()

    initial = (rng.random(t.n_cells) < 0.4).astype(np.int8)
    moved = np.zeros_like(initial)
    moved[t.permutation(vector)] = initial
    assert np.array_equal(run(initial)[np.argsort(t.permutation(vector))], run(moved))


def test_wrap_is_required():
    width, height = min_wrap_size("hexagon")
    TorusTranslations("hexagon", width, height)
    with pytest.raises(ValueError):
        TorusTranslations("hexagon", 5, 4)  # odd width cannot wrap
