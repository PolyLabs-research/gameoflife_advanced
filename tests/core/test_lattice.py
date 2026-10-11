import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from latticelife.core.lattice import LIFESPAN_MAX, LatticeGrid, clone_grid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.session.metrics import compute_lifeform_metrics

CONWAY = Rule(birth={3}, survive={2, 3})

# shape, triangle_mode, wrap-mode degree — mirrors tests/core/test_geometries.py's table.
SHAPES = [
    ("square", "edge+vertex", 8),
    ("hexagon", "edge+vertex", 6),
    ("triangle", "edge", 3),
    ("triangle", "edge+vertex", 12),
]


def test_empty_grid_is_a_fixpoint():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 10, 10, wrap=True)
    rng = np.random.default_rng(0)

    result = grid.step(rng)

    assert result.changed_cells == 0
    assert np.all(grid.occupancy == 0)
    assert np.all(grid.lifespans == 0)
    assert result.alive[1] == 0
    assert result.births[1] == 0
    assert result.deaths[1] == 0


def _run_trajectory(seed: int, steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    parent = np.random.default_rng(seed)
    init_rng, step_rng = parent.spawn(2)

    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 12, 12, wrap=True)
    alive = init_rng.random(grid.n_cells) < 0.4
    grid.occupancy[alive] = 1
    grid.lifespans[alive] = 1

    for _ in range(steps):
        grid.step(step_rng)

    return grid.occupancy.copy(), grid.lifespans.copy()


def test_single_species_determinism_same_seed_identical_trajectory():
    occ_a, life_a = _run_trajectory(seed=42)
    occ_b, life_b = _run_trajectory(seed=42)

    assert np.array_equal(occ_a, occ_b)
    assert np.array_equal(life_a, life_b)


def test_lifespans_increase_under_survival_and_dtype_is_int32():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 10, 10, wrap=True)
    assert grid.lifespans.dtype == np.int32

    block = [(4, 4), (4, 5), (5, 4), (5, 5)]  # still-life under B3/S23
    for r, c in block:
        grid.occupancy[r * grid.width + c] = 1
        grid.lifespans[r * grid.width + c] = 1

    rng = np.random.default_rng(0)
    for gen in range(1, 51):
        result = grid.step(rng)
        assert result.changed_cells == 0
        for r, c in block:
            assert grid.lifespans[r * grid.width + c] == gen + 1

    assert grid.lifespans.dtype == np.int32


def test_glider_translates_by_1_1_with_period_4_on_wrapped_square():
    width = height = 20
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", width, height, wrap=True)

    glider = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    origin_r, origin_c = 5, 5
    for dr, dc in glider:
        r, c = origin_r + dr, origin_c + dc
        grid.occupancy[r * width + c] = 1
        grid.lifespans[r * width + c] = 1

    rng = np.random.default_rng(0)
    for _ in range(4):
        grid.step(rng)

    expected = {
        ((origin_r + 1 + dr) % height) * width + ((origin_c + 1 + dc) % width)
        for dr, dc in glider
    }
    alive_cells = set(np.nonzero(grid.occupancy == 1)[0].tolist())
    assert alive_cells == expected


@pytest.mark.parametrize(("shape", "triangle_mode", "degree"), SHAPES)
def test_rule_values_beyond_max_degree_are_unreachable(shape, triangle_mode, degree):
    """A survive count one past the geometry's max degree can never be satisfied — neighbor
    counts are bounded by adjacency degree, so out-of-range rule values are naturally clamped
    without any explicit clamping code."""
    unreachable_rule = Rule(birth=set(), survive={degree + 1})
    grid = LatticeGrid(
        [Lifeform(1, unreachable_rule)],
        shape,
        width=8,
        height=8,
        wrap=True,
        triangle_mode=triangle_mode,
    )
    grid.occupancy[:] = 1
    grid.lifespans[:] = 1

    rng = np.random.default_rng(0)
    result = grid.step(rng)

    assert result.alive[1] == 0


@given(st.integers(min_value=0, max_value=1000))
def test_determinism_property_across_seeds(seed):
    occ_a, life_a = _run_trajectory(seed=seed, steps=5)
    occ_b, life_b = _run_trajectory(seed=seed, steps=5)
    assert np.array_equal(occ_a, occ_b)
    assert np.array_equal(life_a, life_b)


def test_clone_grid_shares_adjacency_and_copies_state():
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 6, 6, wrap=True)
    grid.occupancy[0] = 1
    grid.lifespans[0] = 5

    twin = clone_grid(grid)

    assert twin.adjacency is grid.adjacency  # shared read-only, never rebuilt or deep-copied
    assert twin.occupancy is not grid.occupancy
    assert twin.lifespans is not grid.lifespans
    assert np.array_equal(twin.occupancy, grid.occupancy)
    assert np.array_equal(twin.lifespans, grid.lifespans)

    twin.occupancy[1] = 1  # mutating the twin must not affect the original
    assert grid.occupancy[1] == 0


def test_lifespans_saturate_instead_of_wrapping_negative():
    """F70: a cell surviving 2^31-1 generations used to wrap to -2^31, dropping out of the
    static count and dragging avg_lifespan hugely negative."""
    grid = LatticeGrid([Lifeform(1, CONWAY)], "square", 5, 5, wrap=True)
    block = [6, 7, 11, 12]  # 2x2 still life
    grid.occupancy[block] = 1
    grid.lifespans[block] = LIFESPAN_MAX - 1

    rng = np.random.default_rng(0)
    for _ in range(3):
        result = grid.step(rng)
        metrics = compute_lifeform_metrics(1, result, grid, 4, 4)
        assert np.all(grid.lifespans[block] == LIFESPAN_MAX)
        assert result.static[1] == result.alive[1] == 4
        assert metrics.avg_lifespan == float(LIFESPAN_MAX)

    assert grid.lifespans.dtype == np.int32


# --- multi-species semantics (§6.3 steps 2 and 4) ---------------------------------------------


def _place(grid: LatticeGrid, cells: list[tuple[int, int]], lifeform_id: int) -> None:
    for r, c in cells:
        grid.occupancy[r * grid.width + c] = lifeform_id
        grid.lifespans[r * grid.width + c] = 1


@pytest.mark.parametrize("order", [(1, 2), (2, 1)])
def test_contested_birth_tie_break_is_uniform(order):
    """F62: a cell with exactly 3 neighbours of each of two Conway species is a birth for both;
    the winner must be uniform, not biased towards the first-listed or lowest-id species."""
    grid = LatticeGrid([Lifeform(i, CONWAY) for i in order], "square", 11, 11, wrap=True)
    wins = {1: 0, 2: 0}
    trials = 400
    for seed in range(trials):
        grid.occupancy[:] = 0
        grid.lifespans[:] = 0
        _place(grid, [(4, 4), (4, 5), (4, 6)], 1)
        _place(grid, [(6, 4), (6, 5), (6, 6)], 2)
        grid.step(np.random.default_rng(seed))
        wins[int(grid.occupancy[5 * 11 + 5])] += 1

    assert wins[1] + wins[2] == trials
    assert 0.35 * trials < wins[1] < 0.65 * trials


INERT = Rule(birth=set(), survive=set(range(13)))  # never born, never dies on any shape


@pytest.mark.parametrize("order", [(1, 2), (2, 1)])
@pytest.mark.parametrize(("shape", "triangle_mode", "degree"), SHAPES)
def test_death_surrounded_by_a_foreign_species_is_credited_to_it(
    shape, triangle_mode, degree, order
):
    """F12: a cell whose every neighbour is foreign has more neighbours than it survives among
    (max(S) = 1) and dies; the surrounding species is credited. Legacy-style attribution, which
    measured overcrowding by the victim's own count (0 here), credited nothing."""
    rules = {1: Rule(birth=set(), survive={1}), 2: INERT}
    lifeforms = [Lifeform(i, rules[i]) for i in order]
    grid = LatticeGrid(lifeforms, shape, 12, 12, wrap=True, triangle_mode=triangle_mode)
    victim = 5 * 12 + 6
    grid.occupancy[grid.adjacency[victim]] = 2
    grid.occupancy[victim] = 1
    grid.lifespans[grid.occupancy > 0] = 1

    result = grid.step(np.random.default_rng(0))

    assert result.deaths[1] == 1 and result.deaths[2] == 0
    assert result.kills[2] == 1
    assert result.kills[1] == 0


def _centre_victim(victim_rule: Rule, ring: list[int]) -> np.ndarray:
    """A 9x9 wrapped square grid: species 1 (`victim_rule`) alone at the centre, its 8 Moore
    neighbours set from `ring` (0 = empty, 2 and 3 = inert species). Returns kills after one
    step; nothing but the centre can change, since no species has a birth condition."""
    lifeforms = [Lifeform(1, victim_rule), Lifeform(2, INERT), Lifeform(3, INERT)]
    grid = LatticeGrid(lifeforms, "square", 9, 9, wrap=True)
    centre = 4 * 9 + 4
    grid.occupancy[grid.adjacency[centre]] = ring
    grid.occupancy[centre] = 1
    grid.lifespans[grid.occupancy > 0] = 1

    result = grid.step(np.random.default_rng(0))

    assert grid.occupancy[centre] == 0 and result.deaths[1] == 1  # every case kills the centre
    return result.kills


def test_isolation_death_is_never_credited():
    """F12: overcrowding means more live neighbours than the victim's species survives among.
    A Conway cell with 3 foreign neighbours died of isolation (it would survive 3 of its own),
    so nothing is credited even though one species has every neighbour."""
    conway_no_birth = Rule(birth=set(), survive={2, 3})
    assert _centre_victim(conway_no_birth, [2, 2, 2, 0, 0, 0, 0, 0]).tolist() == [0, 0, 0, 0]
    assert _centre_victim(conway_no_birth, [2, 2, 2, 2, 0, 0, 0, 0]).tolist() == [0, 0, 1, 0]


def test_empty_survive_set_deaths_are_never_credited():
    """F12: with S = {} a cell dies whatever surrounds it; legacy attribution (max(S) = -1)
    credited every such death to a foreign majority."""
    no_survival = Rule(birth=set(), survive=set())
    assert _centre_victim(no_survival, [2, 2, 2, 0, 0, 0, 0, 0]).tolist() == [0, 0, 0, 0]
    assert _centre_victim(no_survival, [2] * 8).tolist() == [0, 0, 0, 0]


def test_overcrowding_without_a_foreign_majority_is_not_credited():
    victim = Rule(birth=set(), survive={1})
    assert _centre_victim(victim, [2, 2, 2, 2, 3, 3, 3, 3]).tolist() == [0, 0, 0, 0]  # 4 of 8
    assert _centre_victim(victim, [2, 2, 2, 3, 3, 0, 0, 0]).tolist() == [0, 0, 1, 0]  # 3 of 5


def _reference_kills(grid: LatticeGrid, after: np.ndarray, size: int) -> np.ndarray:
    """Per-cell transcription of §6.3 step 4 as implemented (see core/lattice.py)."""
    rules = {lf.id: lf.rule for lf in grid.lifeforms}
    kills = np.zeros(size, dtype=np.int64)
    for cell, species in enumerate(grid.occupancy.tolist()):
        if species == 0 or after[cell] != 0 or not rules[species].survive:
            continue
        neighbours = [grid.occupancy[j] for j in grid.adjacency[cell] if j >= 0]
        per_species = {s: neighbours.count(s) for s in rules}
        total = sum(per_species.values())
        if total <= max(rules[species].survive):
            continue
        killer, count = max(
            ((s, n) for s, n in per_species.items() if s != species), key=lambda item: item[1]
        )
        if 2 * count > total:
            kills[killer] += 1
    return kills


@pytest.mark.parametrize(("shape", "triangle_mode", "_degree"), SHAPES)
@pytest.mark.parametrize("wrap", [True, False])
def test_kills_match_the_per_cell_reference_on_random_grids(shape, triangle_mode, _degree, wrap):
    lifeforms = [
        Lifeform(1, CONWAY),
        Lifeform(2, Rule(birth={2}, survive={3, 4})),
        Lifeform(3, Rule(birth={3, 6}, survive={1, 2, 5})),
        Lifeform(4, Rule(birth={1}, survive=set())),
        Lifeform(5, Rule(birth={2}, survive={0})),  # max(S) = 0: kills possible at degree 3
    ]
    grid = LatticeGrid(lifeforms, shape, 16, 12, wrap, triangle_mode)
    rng = np.random.default_rng(11)
    grid.occupancy[:] = rng.integers(0, 6, grid.n_cells)
    grid.lifespans[grid.occupancy > 0] = 1
    total_kills = 0
    for _ in range(8):
        before = clone_grid(grid)
        result = grid.step(rng)
        assert result.kills.tolist() == _reference_kills(before, grid.occupancy, 6).tolist()
        total_kills += int(result.kills.sum())
    assert total_kills > 0


def test_common_rules_record_kills():
    """F12: under legacy attribution three Conway species on a square grid could never record
    a kill (it needed degree >= 2*max(S)+3 = 9)."""
    grid = LatticeGrid([Lifeform(i, CONWAY) for i in (1, 2, 3)], "square", 48, 48, wrap=True)
    rng = np.random.default_rng(3)
    grid.occupancy[:] = rng.integers(0, 4, grid.n_cells)
    kills = sum(int(grid.step(rng).kills.sum()) for _ in range(30))
    assert kills > 0
