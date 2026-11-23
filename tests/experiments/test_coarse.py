"""Zooming out (experiments.coarse): the blocks, the projections, and the closure test on the
rules whose answer is known."""

import numpy as np
import pytest

from latticelife.core.geometries import build_adjacency
from latticelife.experiments import coarse
from latticelife.experiments.coarse import (
    ClosureTask,
    Diagram,
    Projection,
    World,
    block_cells,
    block_counts,
    closure,
    closure_point,
    coarse_grid,
    leading_species,
    project,
    run_closure,
    sample_cells,
    uniform_start,
)
from latticelife.experiments.harness import grid_dimensions, read_jsonl

PARITY = [
    ("square", "B1357/S1357"),
    ("square", "B1357/S02468"),
    ("hexagon", "B135/S135"),
    ("hexagon", "B135/S0246"),
]
OTHER = {
    "B1357/S1357": "B1357/S02468",
    "B1357/S02468": "B1357/S1357",
    "B135/S135": "B135/S0246",
    "B135/S0246": "B135/S135",
}
BLOCKS = [
    ("square", 12, 2),
    ("square", 12, 3),
    ("square", 12, 4),
    ("square", "24x12", 4),
    ("hexagon", 12, 2),
    ("hexagon", 12, 3),
    ("hexagon", 24, 4),
    ("hexagon", "24x12", 2),
    ("hexagon", "24x18", 3),
]


def _rows(array) -> list[list[int]]:
    return [sorted(row) for row in np.asarray(array).tolist()]


# --- blocks --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("shape", "grid", "block"), BLOCKS)
def test_the_blocks_partition_the_grid(shape, grid, block):
    width, height = grid_dimensions(grid)
    coarse_width, coarse_height = grid_dimensions(coarse_grid(shape, grid, block))
    assert (coarse_width, coarse_height) == (width // block, height // block)
    cells = block_cells(shape, grid, block)
    assert cells.shape == (block * block, coarse_width * coarse_height)
    assert sorted(cells.ravel().tolist()) == list(range(width * height))
    assert np.array_equal(cells[0], sample_cells(shape, grid, block))


@pytest.mark.parametrize(("shape", "grid", "block"), BLOCKS)
def test_a_supercells_neighbours_are_a_block_away_on_the_fine_grid(shape, grid, block):
    """The sampled cells are a grid of the same shape: what the coarse grid's own adjacency
    calls neighbours are `block` lattice steps apart on the fine one."""
    width, height = grid_dimensions(grid)
    coarse_width, coarse_height = grid_dimensions(coarse_grid(shape, grid, block))
    sampled = sample_cells(shape, grid, block, (block - 1, block // 2))
    coarse_neighbours = build_adjacency(shape, coarse_width, coarse_height, True)
    far = build_adjacency(shape, width, height, True, dilation=block)
    assert _rows(sampled[coarse_neighbours]) == _rows(far[sampled])


def test_blocks_that_do_not_fit_are_refused():
    with pytest.raises(ValueError, match="does not divide"):
        coarse_grid("square", 12, 5)
    with pytest.raises(ValueError, match="cannot wrap.*even width"):
        coarse_grid("hexagon", 12, 4)  # 3 x 3: a wrapped hexagon grid has an even width
    with pytest.raises(ValueError, match="cannot wrap"):
        coarse_grid("square", 12, 6)  # 2 x 2: narrower than the neighbourhood
    with pytest.raises(ValueError, match="no block geometry"):
        coarse_grid("triangle", 12, 2)
    with pytest.raises(ValueError, match="outside a block"):
        sample_cells("square", 12, 2, (2, 0))
    with pytest.raises(ValueError, match="at least 1"):
        coarse_grid("square", 12, 0)
    assert coarse_grid("square", "24x12", 4) == "6x3" and coarse_grid("square", (12, 12), 3) == 4


# --- projections ---------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", coarse.SHAPES)
@pytest.mark.parametrize("kind", ["decimate", "parity"])
def test_zooming_out_twice_is_zooming_out_once(shape, kind):
    cells = uniform_start(shape, 24, 7)
    once = project(shape, 24, Projection(kind, 4), cells)
    half = project(shape, 24, Projection(kind, 2), cells)
    assert np.array_equal(project(shape, 12, Projection(kind, 2), half), once)
    assert 0 < once.sum() < once.size and once.dtype == np.int8


def test_the_projections_by_hand():
    cells = np.zeros(36, dtype=np.int8)
    cells[[0, 1, 6, 15, 35]] = 1  # (0,0) (0,1) (1,0) | (2,3) | (5,5)
    assert project("square", 6, Projection("identity"), cells).tolist() == cells.tolist()
    assert project("square", 6, Projection("decimate", 2), cells).tolist() == [1, 0, 0] + [0] * 6
    assert project("square", 6, Projection("decimate", 2, (1, 1)), cells).tolist() == [0] * 8 + [1]
    # block (0,0) holds three live cells, block (1,1) one, block (2,2) one
    assert project("square", 6, Projection("parity", 2), cells).tolist() == [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
    ]
    with pytest.raises(ValueError, match="expected 36 cells"):
        project("square", 6, Projection("parity", 2), cells[:-1])
    with pytest.raises(ValueError, match="unknown projection"):
        Projection("majority", 2)
    with pytest.raises(ValueError, match="no identity projection"):
        Projection("identity", 2)


def test_species_counts_and_the_leading_species_by_hand():
    occupancy = np.zeros(36, dtype=np.int8)
    occupancy[[0, 1, 6]] = 1  # block (0,0): three of species 1 ...
    occupancy[7] = 2  # ... and one of species 2
    occupancy[[2, 3]] = [1, 2]  # block (0,1): one each
    occupancy[[14, 15, 20]] = 2  # block (1,1): three of species 2
    counts = block_counts("square", 6, 2, occupancy, (1, 2))
    assert counts.tolist() == [[3, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 3, 0, 0, 0, 0]]
    empty = block_counts("square", 6, 2, occupancy, (0,))
    assert (counts.sum(axis=0) + empty[0] == 4).all()
    # level at the top, or empty: no leader
    assert leading_species(counts, (1, 2)).tolist() == [1, 0, 0, 0, 2, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="one row of counts per species"):
        leading_species(counts, (0, 1))
    hexagon = block_counts("hexagon", 12, 3, np.ones(144, dtype=np.int8), (1,))
    assert hexagon.tolist() == [[9] * 16]


# --- the known answers ---------------------------------------------------------------------------


@pytest.mark.parametrize(("shape", "rule"), PARITY)
@pytest.mark.parametrize(("grid", "k"), [(12, 2), (12, 8), (16, 8), (24, 16), ("24x12", 4)])
def test_two_to_the_k_generations_are_one_generation_with_neighbours_that_far(shape, rule, grid, k):
    """Frobenius. On 16 x 16 at k = 8 opposite neighbours land on one cell and are counted
    twice; a parity rule on a torus of side 2^e dies or stands still, so its count is zero for
    a dull reason."""
    diagram = Diagram(
        World(shape, rule, grid), World(shape, rule, grid, k), Projection("identity"), k
    )
    (result,), _ = closure([diagram], uniform_start(shape, grid, 11), 3)
    assert result.mismatches == (0, 0, 0)
    if grid != 16:
        assert all(0 < live < result.supercells for live in result.live)


@pytest.mark.parametrize(("shape", "rule"), PARITY)
@pytest.mark.parametrize(
    ("grid", "block"), [(20, 2), (28, 2), (40, 4), ("28x20", 2), ("40x20", 4), (12, 2)]
)
def test_zooming_out_by_a_power_of_two_is_exact(shape, rule, grid, block):
    """On coarse grids of side 5, 10 and 14 every one of these rules stays alive, so the zeros
    compare live grids. (On 6 x 6, the last case, B1357/S02468 can die out at once: its
    polynomial vanishes at the cube roots of unity.)"""
    fine, small = World(shape, rule, grid), World(shape, rule, coarse_grid(shape, grid, block))
    diagrams = [
        Diagram(fine, small, projection, block)
        for projection in (
            Projection("decimate", block),
            Projection("decimate", block, (block - 1, block // 2)),
            Projection("parity", block),
        )
    ]
    results, _ = closure(diagrams, uniform_start(shape, grid, 12), 4)
    for result in results:
        assert result.mismatches == (0, 0, 0, 0) and result.supercells == small.cells
        assert result.live[0] > 0 and (grid == 12 or all(result.live))


def _mismatches(shape, rule, grid, block, clock, coarse_rule, seed) -> int:
    task = ClosureTask(rule, shape, grid, block, clock, seed, 1, coarse_rule=coarse_rule)
    return closure_point(task).mismatches[0]


@pytest.mark.parametrize(("shape", "rule"), PARITY)
def test_a_block_of_three_is_a_fair_coin_per_supercell(shape, rule):
    """64 supercells, 40 starts: 2,560 fair coins, so 1,280 expected, 25.3 standard deviations
    wide; six of them either side."""
    total = sum(_mismatches(shape, rule, 24, 3, 3, None, seed) for seed in range(1, 41))
    assert 1128 <= total <= 1432


@pytest.mark.parametrize(("shape", "rule"), PARITY)
def test_wrong_clocks_and_the_wrong_rule_are_seen(shape, rule):
    """One generation early at block 2 leaves one constraint on the coins, their sum: every
    count is even. The other parity rule on the coarse grid differs by the sampled cell
    itself, so the count is the number of live sampled cells, even or odd."""
    early = [_mismatches(shape, rule, 24, 2, 1, None, seed) for seed in range(1, 21)]
    assert all(count % 2 == 0 for count in early) and min(early) > 40
    for seed in range(1, 6):
        wrong = _mismatches(shape, rule, 24, 2, 2, OTHER[rule], seed)
        sampled = uniform_start(shape, 24, seed)[sample_cells(shape, 24, 2)]
        assert wrong == int(sampled.sum())


def test_a_rule_that_is_not_linear_does_not_close():
    """Conway's Life is not its own coarse rule: the two routes soon disagree."""
    task = ClosureTask("B3/S23", "square", 48, 2, 2, 1, 3)
    record = closure_point(task)
    assert min(record.mismatches) > 0 and record.supercells == 576


# --- records -------------------------------------------------------------------------------------

# Five records of lattice-lab's registered calibration (examples/scale-up-linear, run on
# 2026-11-23 through its own copy of this instrument): task, seed, then live, mismatches, the
# start's sha256 and the fine grid's.
REGISTERED = [
    (
        ClosureTask("B1357/S1357", "square", 10, 2, 2, 1, 8),
        [10, 12, 10, 10, 12, 10, 10, 12],
        [0] * 8,
        "70ebf34d7d96e2980b69c6c38fd54c9e74bab74d80b49e67cb081cfd063567c1",
        "293f55e0627362b5067b1e2d9850318c8557e0a85011722bff3c6f8e86346cae",
    ),
    (
        ClosureTask("B135/S0246", "hexagon", 20, 2, 2, 2, 8, projection="parity"),
        [54, 58, 54, 54, 52, 50, 50, 42],
        [0] * 8,
        "3cbe0a8d17cbb94fc45613b538ce8a3b49d79f4bf3b6f9e813fc7b37fc7f6c61",
        "c5e3ef4bc4ad2393a82b7f0c272d682b04b4be4b6a384081545239294726f8ea",
    ),
    (
        ClosureTask("B135/S135", "hexagon", 48, 1, 32, 3, 3, "identity", coarse_dilation=32),
        [1110, 1110, 1110],
        [0, 0, 0],
        "f81c3413be027fa2831295637abe1738016647485465bcf13b7847267ec37300",
        "88456532f2e012158ec0f2bfdca5f6637aeaccfb0dc11f1ec4778b0cdaa7ee21",
    ),
    (
        ClosureTask("B1357/S02468", "square", 60, 3, 3, 1, 1),
        [188],
        [204],
        "05a1c71bfb201e5e8428ff8426d335ec8a194678e095cc4bd27d18ff687c1c91",
        "8d871037b7da05c05c6d8d935e5b19e2247cd35ded11e68f08cd374df3d73cd1",
    ),
    (
        ClosureTask("B135/S135", "hexagon", 80, 4, 3, 5, 1),
        [202],
        [200],
        "de41b885526cc556ee0c1c48134363f8da1afedbf60e8567a3c02e5f5182e95b",
        "cf0f6b767f00f54f7fea90847504c7534970c6d7a1886f49d636c395140c1f11",
    ),
]


@pytest.mark.parametrize(("task", "live", "mismatches", "start", "fine"), REGISTERED)
def test_registered_records_are_reproduced_to_the_last_bit(task, live, mismatches, start, fine):
    record = closure_point(task)
    assert (record.live, record.mismatches) == (live, mismatches)
    assert (record.start_sha256, record.fine_sha256) == (start, fine)


def test_a_record_names_everything_that_identifies_its_test():
    task = ClosureTask("B1357/S1357", "square", "24x12", 2, 2, 3, 2, "decimate", (1, 0))
    record = closure_point(task)
    assert record.to_json() == {
        "rule": "B1357/S1357",
        "shape": "square",
        "grid": "24x12",
        "block": 2,
        "clock": 2,
        "projection": "decimate",
        "offset": [1, 0],
        "coarse_rule": "B1357/S1357",
        "coarse_grid": "12x6",
        "coarse_dilation": 1,
        "seed": 3,
        "steps": 2,
        "supercells": 72,
        "mismatches": [0, 0],
        "live": record.live,
        "start_sha256": coarse.digest(uniform_start("square", "24x12", 3)),
        "fine_sha256": record.fine_sha256,
    }
    assert closure_point(task) == record
    assert set(coarse.RECORD_KEY) < set(record.to_json())


def test_a_start_is_uniform_and_depends_on_the_shape_the_grid_and_the_seed():
    starts = {
        key: uniform_start(*key)
        for key in [
            ("square", 24, 1),
            ("square", 24, 2),
            ("hexagon", 24, 1),
            ("square", "24x12", 1),
        ]
    }
    assert starts["square", 24, 1].dtype == np.int8 and set(starts["square", 24, 1]) == {0, 1}
    assert len({coarse.digest(cells) for cells in starts.values()}) == 4
    assert np.array_equal(uniform_start("square", "24x24", 1), starts["square", 24, 1])
    assert abs(float(uniform_start("square", 200, 1).mean()) - 0.5) < 0.01


def test_tests_run_together_share_one_fine_world_and_a_diagram_must_fit():
    fine = World("square", "B1357/S1357", 12)
    with pytest.raises(ValueError, match="share their fine world"):
        closure(
            [
                Diagram(fine, World("square", "B1357/S1357", 6), Projection("decimate", 2), 2),
                Diagram(fine, World("square", "B1357/S1357", 4), Projection("decimate", 3), 3),
            ],
            uniform_start("square", 12, 1),
            1,
        )
    with pytest.raises(ValueError, match="sends the fine grid 12 to 6"):
        Diagram(fine, World("square", "B1357/S1357", 4), Projection("decimate", 2), 2)
    with pytest.raises(ValueError, match="share a shape"):
        Diagram(fine, World("hexagon", "B135/S135", 6), Projection("decimate", 2), 2)
    with pytest.raises(ValueError, match="0 and 1"):
        coarse.Run(fine, np.full(144, 2, dtype=np.int8))
    with pytest.raises(ValueError, match="no block geometry"):
        World("triangle", "B13/S13", 12)
    with pytest.raises(ValueError, match="even width"):
        World("hexagon", "B135/S135", "11x12")


def test_run_closure_resumes_and_skips_a_task_named_twice(tmp_path):
    out = tmp_path / "closure.jsonl"
    tasks = [ClosureTask("B1357/S1357", "square", 12, 2, 2, seed, 2) for seed in (1, 2)]
    first = list(run_closure(tasks + tasks[:1], str(out)))
    more = tasks + [
        ClosureTask("B1357/S1357", "square", 12, 2, 2, 3, 2),
        ClosureTask("B1357/S1357", "square", 12, 2, 2, 1, 2, "parity"),
        ClosureTask("B1357/S1357", "square", 12, 2, 2, 1, 2, coarse_rule="B1357/S1357"),
    ]
    again = list(run_closure(more, str(out)))
    assert len(first) == 2 and len(again) == 2 and len(read_jsonl(out)) == 4
    assert {r["projection"] for r in read_jsonl(out)} == {"decimate", "parity"}
