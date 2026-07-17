import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from latticelife.core.geometries import build_adjacency, max_degree

# shape, triangle_mode, expected wrap-mode degree (§6.1 degree table)
SHAPES = [
    ("square", "edge+vertex", 8),
    ("hexagon", "edge+vertex", 6),
    ("triangle", "edge", 3),
    ("triangle", "edge+vertex", 12),
]


@pytest.mark.parametrize(("shape", "triangle_mode", "expected_degree"), SHAPES)
def test_max_degree_matches_table(shape, triangle_mode, expected_degree):
    assert max_degree(shape, triangle_mode) == expected_degree


@pytest.mark.parametrize(("shape", "triangle_mode", "expected_degree"), SHAPES)
def test_wrap_degree_matches_table_for_every_cell(shape, triangle_mode, expected_degree):
    adjacency = build_adjacency(shape, width=10, height=10, wrap=True, triangle_mode=triangle_mode)
    assert adjacency.shape[1] == expected_degree
    assert np.all((adjacency >= 0).sum(axis=1) == expected_degree)


@pytest.mark.parametrize(("shape", "triangle_mode", "expected_degree"), SHAPES)
def test_clamp_never_exceeds_wrap_degree(shape, triangle_mode, expected_degree):
    adjacency = build_adjacency(
        shape, width=10, height=10, wrap=False, triangle_mode=triangle_mode
    )
    assert adjacency.shape[1] == expected_degree
    assert np.all((adjacency >= 0).sum(axis=1) <= expected_degree)
    # Some interior cell should still reach full degree even without wrap.
    assert np.any((adjacency >= 0).sum(axis=1) == expected_degree)


def _assert_symmetric(shape: str, triangle_mode: str, wrap: bool, width: int, height: int) -> None:
    adjacency = build_adjacency(shape, width, height, wrap, triangle_mode)
    n = width * height
    for i in range(n):
        for slot in adjacency[i]:
            if slot < 0:
                continue
            j = int(slot)
            assert i in adjacency[j], f"{shape}/{triangle_mode} wrap={wrap}: {j}->{i} not mutual"


@pytest.mark.parametrize(("shape", "triangle_mode", "_deg"), SHAPES)
@pytest.mark.parametrize("wrap", [True, False])
def test_adjacency_symmetry(shape, triangle_mode, _deg, wrap):
    _assert_symmetric(shape, triangle_mode, wrap, width=8, height=8)


@given(st.sampled_from(SHAPES), st.sampled_from([True, False]))
def test_adjacency_symmetry_property(shape_data, wrap):
    shape, triangle_mode, _degree = shape_data
    _assert_symmetric(shape, triangle_mode, wrap, width=6, height=6)
