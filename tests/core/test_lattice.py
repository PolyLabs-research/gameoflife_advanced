import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from latticelife.core.geometries import build_adjacency
from latticelife.core.lattice import LIFESPAN_MAX, LatticeGrid, TieBreak, clone_grid
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


# --- the birth tie-break (core.lattice.TieBreak) --------------------------------------------------

HIGHLIFE = Rule(birth={3, 6}, survive={2, 3})
SPROUT = Rule(birth={0}, survive=set(range(13)))  # born from nothing, never dies
DETERMINISTIC = [TieBreak("standoff"), TieBreak("strongest"), TieBreak("strongest", radius=3)]


def _contest(tie_break: TieBreak | None, extra: tuple = ()) -> LatticeGrid:
    """Three cells of species 1 above (5, 5) and three of species 2 below it: one contested
    cell. `extra` places more cells, as (row, column, species)."""
    species = [Lifeform(1, CONWAY), Lifeform(2, CONWAY)]
    grid = LatticeGrid(species, "square", 11, 11, wrap=True, tie_break=tie_break)
    _place(grid, [(4, 4), (4, 5), (4, 6)], 1)
    _place(grid, [(6, 4), (6, 5), (6, 6)], 2)
    for r, c, lifeform_id in extra:
        _place(grid, [(r, c)], lifeform_id)
    return grid


def _soup(
    tie_break, rules=(CONWAY, HIGHLIFE, CONWAY), side=24, density=0.6, seed=3, shape="square"
) -> LatticeGrid:
    """A dense mixed grid of several species: plenty of contested cells."""
    species = [Lifeform(i + 1, rule) for i, rule in enumerate(rules)]
    grid = LatticeGrid(species, shape, side, side, wrap=True, tie_break=tie_break)
    rng = np.random.default_rng(seed)
    live = rng.random(grid.n_cells) < density
    grid.occupancy[live] = rng.integers(1, len(rules) + 1, int(live.sum()))
    grid.lifespans[live] = 1
    return grid


def test_the_default_tie_break_is_the_random_draw_and_counts_the_contest():
    grid = _contest(None)
    assert grid.tie_break == TieBreak() == TieBreak("random", 2, 1.0)
    assert not grid.tie_break.deterministic
    result = grid.step(np.random.default_rng(0))
    assert result.contested == 1 and grid.occupancy[5 * 11 + 5] in (1, 2)
    # each species is also born, uncontested, on the far side of its own row
    assert grid.occupancy[3 * 11 + 5] == 1 and grid.occupancy[7 * 11 + 5] == 2


def test_standoff_leaves_a_contested_cell_empty():
    grid = _contest(TieBreak("standoff"))
    result = grid.step(np.random.default_rng(0))
    assert result.contested == 1 and grid.occupancy[5 * 11 + 5] == 0
    assert grid.occupancy[3 * 11 + 5] == 1 and grid.occupancy[7 * 11 + 5] == 2
    assert result.births[1] == result.births[2] == 1


@pytest.mark.parametrize(
    ("radius", "extra", "winner"),
    [
        (2, (), 0),  # three cells each within two steps: the top is shared, nobody is born
        (2, ((3, 3, 1),), 1),  # a fourth cell of species 1 two steps away
        (2, ((7, 7, 2),), 2),
        (2, ((3, 3, 1), (7, 7, 2)), 0),  # four each
        (1, ((3, 3, 1),), 0),  # at radius 1 only the neighbours count: three each
        (3, ((2, 2, 1),), 1),  # three steps away, seen at radius 3 ...
        (2, ((2, 2, 1),), 0),  # ... and not at radius 2
    ],
)
def test_strongest_gives_the_cell_to_the_species_with_more_cells_nearby(radius, extra, winner):
    grid = _contest(TieBreak("strongest", radius=radius), extra)
    result = grid.step(np.random.default_rng(0))
    assert result.contested == 1 and grid.occupancy[5 * 11 + 5] == winner


@pytest.mark.parametrize("tie_break", DETERMINISTIC)
def test_a_deterministic_tie_break_never_draws(tie_break):
    """The next grid is a function of the grid: two different Generators give one trajectory,
    and neither is advanced."""
    assert tie_break.deterministic
    first, second = _soup(tie_break), _soup(tie_break)
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(2)
    before = rng_a.bit_generator.state
    contested = 0
    for _ in range(25):
        contested += first.step(rng_a).contested
        second.step(rng_b)
        assert np.array_equal(first.occupancy, second.occupancy)
        assert np.array_equal(first.lifespans, second.lifespans)
    assert rng_a.bit_generator.state == before
    assert contested > 20 and 0 < np.count_nonzero(first.occupancy)


def test_the_random_draw_depends_on_the_generator():
    first, second = _soup(TieBreak("random")), _soup(TieBreak("random"))
    first.step(np.random.default_rng(1))
    second.step(np.random.default_rng(2))
    assert not np.array_equal(first.occupancy, second.occupancy)


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_weighted_with_power_zero_is_the_random_draw_draw_for_draw(radius):
    plain, weighted = _soup(TieBreak("random")), _soup(TieBreak("weighted", radius, 0.0))
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(25):
        a, b = plain.step(rng_a), weighted.step(rng_b)
        assert np.array_equal(plain.occupancy, weighted.occupancy)
        assert (a.contested, a.births.tolist()) == (b.contested, b.births.tolist())
    assert rng_a.bit_generator.state == rng_b.bit_generator.state


@pytest.mark.parametrize(("power", "chance"), [(1.0, 4 / 7), (2.0, 16 / 25), (8.0, 0.909)])
def test_weighted_draws_in_proportion_to_strength_to_the_power(power, chance):
    """Species 1 has four cells within two steps of the contested cell and species 2 three, so
    species 1 wins with chance 4^p / (4^p + 3^p). 3,000 draws: five standard deviations."""
    trials, wins = 3000, 0
    for seed in range(trials):
        grid = _contest(TieBreak("weighted", 2, power), ((3, 3, 1),))
        grid.step(np.random.default_rng(seed))
        wins += int(grid.occupancy[5 * 11 + 5] == 1)
    spread = (trials * chance * (1 - chance)) ** 0.5
    assert abs(wins - trials * chance) < 5 * spread


def test_contenders_with_no_cell_nearby_are_level():
    """Two species that are born from nothing contest every cell of an empty grid, each with
    strength 0. The random draws share the grid; `weighted` gives equal chances when every
    weight is 0; `strongest` and `standoff` give the cell to nobody."""
    for kind in ("random", "weighted", "strongest", "standoff"):
        species = [Lifeform(1, SPROUT), Lifeform(2, SPROUT)]
        grid = LatticeGrid(species, "square", 20, 20, wrap=True, tie_break=TieBreak(kind))
        result = grid.step(np.random.default_rng(4))
        assert result.contested == 400
        if kind in ("random", "weighted"):
            assert np.count_nonzero(grid.occupancy) == 400
            assert 150 < np.count_nonzero(grid.occupancy == 1) < 250
        else:
            assert np.count_nonzero(grid.occupancy) == 0


@pytest.mark.parametrize("tie_break", DETERMINISTIC)
@pytest.mark.parametrize("shape", ["square", "hexagon"])
def test_no_species_is_favoured_by_its_id_or_its_place_in_the_list(tie_break, shape):
    """Swap the two species' ids, or list them the other way round: the same run, relabelled."""
    rules = (CONWAY, HIGHLIFE) if shape == "square" else (Rule({2}, {3, 4}), Rule({2, 4}, {3}))
    plain = _soup(tie_break, rules, shape=shape)
    swapped = _soup(tie_break, rules[::-1], shape=shape)
    swapped.occupancy = np.array([0, 2, 1], dtype=np.int8)[plain.occupancy]
    listed = LatticeGrid(plain.lifeforms[::-1], shape, 24, 24, wrap=True, tie_break=tie_break)
    listed.occupancy, listed.lifespans = plain.occupancy.copy(), plain.lifespans.copy()
    rng = np.random.default_rng(0)
    contested = 0
    for _ in range(20):
        contested += plain.step(rng).contested
        swapped.step(rng)
        listed.step(rng)
        relabelled = np.array([0, 2, 1], dtype=np.int8)[swapped.occupancy]
        assert np.array_equal(relabelled, plain.occupancy)
        assert np.array_equal(listed.occupancy, plain.occupancy)
    assert contested > 0


@pytest.mark.parametrize(("shape", "triangle_mode", "_degree"), SHAPES)
def test_one_species_runs_the_same_under_every_tie_break(shape, triangle_mode, _degree):
    rule = Rule(birth={2}, survive={1, 2}) if _degree == 3 else Rule(birth={3}, survive={2, 3})
    runs = []
    for kind in ("random", "standoff", "strongest", "weighted"):
        grid = LatticeGrid(
            [Lifeform(1, rule)], shape, 16, 16, True, triangle_mode, tie_break=TieBreak(kind)
        )
        grid.occupancy[np.random.default_rng(5).random(256) < 0.4] = 1
        contested = sum(grid.step(np.random.default_rng(6)).contested for _ in range(12))
        runs.append((grid.occupancy.copy(), grid.lifespans.copy()))
        assert contested == 0
    for occupancy, lifespans in runs[1:]:
        assert np.array_equal(occupancy, runs[0][0]) and np.array_equal(lifespans, runs[0][1])


def test_every_kind_counts_the_same_contests_from_the_same_grid():
    counts = {kind: _soup(TieBreak(kind)).step(np.random.default_rng(0)).contested
              for kind in ("random", "standoff", "strongest", "weighted")}  # fmt: skip
    assert len(set(counts.values())) == 1 and counts["random"] > 10
    grid = _soup(TieBreak("standoff"))
    before = grid.occupancy.copy()
    result = grid.step(np.random.default_rng(0))
    # by hand: dead cells with a count in B for two or more species
    neighbours = grid.adjacency
    eligible = sum(
        np.isin((before[neighbours] == lf.id).sum(axis=1), list(lf.rule.birth)).astype(int)
        for lf in grid.lifeforms
    )
    assert result.contested == int(np.count_nonzero((before == 0) & (eligible > 1)))
    assert not grid.occupancy[(before == 0) & (eligible > 1)].any()


def _breadth_first(adjacency: np.ndarray, cell: int, radius: int) -> set[int]:
    seen, frontier = {cell}, {cell}
    for _ in range(radius):
        frontier = {int(j) for i in frontier for j in adjacency[i] if j >= 0} - seen
        seen |= frontier
    return seen - {cell}


@pytest.mark.parametrize(("shape", "triangle_mode", "degree"), SHAPES)
@pytest.mark.parametrize("wrap", [True, False])
@pytest.mark.parametrize("radius", [1, 2, 3])
def test_the_cells_within_a_radius_are_those_a_search_reaches(
    shape, triangle_mode, degree, wrap, radius
):
    grid = LatticeGrid([Lifeform(1, CONWAY)], shape, 12, 12, wrap, triangle_mode)
    cells = np.array([0, 5, 13, 77, 143])
    near = grid._within(cells, radius)
    for cell, row in zip(cells.tolist(), near.tolist(), strict=True):
        found = [j for j in row if j >= 0]
        assert len(found) == len(set(found))
        assert set(found) == _breadth_first(grid.adjacency, cell, radius)
    if wrap and shape == "square":
        assert (near >= 0).sum(axis=1).tolist() == [(2 * radius + 1) ** 2 - 1] * 5
    if wrap and shape == "hexagon":
        assert (near >= 0).sum(axis=1).tolist() == [3 * radius * (radius + 1)] * 5


def test_strength_counts_each_cell_once_on_a_folded_grid():
    """A dilated adjacency on a small grid lists a cell twice; a strength counts it once."""
    adjacency = build_adjacency("square", 8, 8, True, dilation=4)
    species = [Lifeform(1, CONWAY), Lifeform(2, CONWAY)]
    tie_break = TieBreak("strongest", 1)
    grid = LatticeGrid(species, "square", 8, 8, True, adjacency=adjacency, tie_break=tie_break)
    grid.occupancy[4] = 1  # (0, 4): four steps along the row from (0, 0), from either side
    assert sorted(set(adjacency[0].tolist())) == [4, 32, 36]
    assert grid._strength(np.array([0])).tolist() == [[1], [0]]
    assert (grid._within(np.array([0]), 1) >= 0).sum() == 3


def test_a_tie_break_is_validated_and_a_cloned_grid_keeps_it():
    for bad in (
        {"kind": "coin"},
        {"radius": 0},
        {"radius": 5},
        {"radius": 2.0},
        {"radius": True},
        {"power": -1.0},
        {"power": float("inf")},
        {"power": float("nan")},
        {"power": "1"},
    ):
        with pytest.raises(ValueError, match="tie-break"):
            TieBreak(**bad)
    assert TieBreak("weighted", 3, 0).power == 0 and TieBreak("standoff").deterministic
    grid = _soup(TieBreak("strongest", 3))
    assert clone_grid(grid).tie_break is grid.tie_break
