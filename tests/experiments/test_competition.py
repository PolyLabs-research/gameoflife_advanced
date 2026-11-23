"""Several species without the app (experiments.competition): the start, the record, and the
tie-break a run is told to use."""

import numpy as np
import pytest

from latticelife.core.geometries import build_adjacency
from latticelife.experiments import competition
from latticelife.experiments.coarse import digest
from latticelife.experiments.competition import (
    CompetitionTask,
    competition_point,
    pair_counts,
    run_competition,
    species_start,
)
from latticelife.experiments.harness import read_jsonl

LIFE, HIGHLIFE = "B3/S23", "B36/S23"
INERT = "B/S012345678"  # never born, never dies: a start that stays as it is
DOOMED = "B/S"  # never born, never survives


# --- the start -----------------------------------------------------------------------------------


def test_a_start_has_the_density_and_the_shares_asked_for():
    """200 x 200: 40,000 cells, 12,000 expected live, and of those a tenth, three tenths and six
    tenths; each within five standard deviations."""
    start = species_start("square", 200, 0.3, 1, shares=(1, 3, 6))
    assert start.dtype == np.int8 and set(np.unique(start)) == {0, 1, 2, 3}
    live = int(np.count_nonzero(start))
    assert abs(live - 12000) < 5 * (40000 * 0.3 * 0.7) ** 0.5
    for kind, share in ((1, 0.1), (2, 0.3), (3, 0.6)):
        count = int(np.count_nonzero(start == kind))
        assert abs(count - live * share) < 5 * (live * share * (1 - share)) ** 0.5
    equal = species_start("square", 200, 0.5, 1, species=4)
    assert [abs(int(np.count_nonzero(equal == k)) - 5000) < 350 for k in (1, 2, 3, 4)] == [True] * 4


def test_a_start_is_named_by_the_shape_the_grid_the_seed_and_the_shares():
    base = species_start("square", 24, 0.5, 1, species=2)
    assert np.array_equal(base, species_start("square", "24x24", 0.5, 1, shares=(1, 1)))
    others = [
        species_start("square", 24, 0.5, 2, species=2),
        species_start("hexagon", 24, 0.5, 1, species=2),
        species_start("square", "24x12", 0.5, 1, species=2),
        species_start("square", 24, 0.5, 1, shares=(1, 2)),
    ]
    assert len({digest(base), *(digest(other) for other in others)}) == 5
    assert not species_start("square", 24, 0.0, 1, species=2).any()
    assert set(np.unique(species_start("square", 24, 1.0, 1, shares=(0, 1)))) == {2}
    assert set(np.unique(species_start("triangle", 24, 1.0, 1, species=1))) == {1}


def test_a_bad_start_is_refused():
    for bad, message in (
        ({"shares": (1, -1)}, "shares must be numbers"),
        ({"shares": (0, 0)}, "not all 0"),
        ({"shares": (1, float("nan"))}, "shares must be numbers"),
        ({"species": 11}, "1 to 10 species"),
        ({"species": 0}, "1 to 10 species"),
        ({}, "the number of species"),
    ):
        with pytest.raises(ValueError, match=message):
            species_start("square", 24, 0.5, 1, **bad)
    with pytest.raises(ValueError, match="density"):
        species_start("square", 24, 1.5, 1, species=2)
    with pytest.raises(ValueError, match="shape"):
        species_start("pentagon", 24, 0.5, 1, species=2)


# --- pairs ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "mode"), [("square", "edge+vertex"), ("hexagon", "edge+vertex"), ("triangle", "edge")]
)
def test_pairs_are_counted_once_each(shape, mode):
    adjacency = build_adjacency(shape, 12, 12, True, mode)
    occupancy = species_start(shape, 12, 0.6, 3, species=3)
    unlike = like = 0
    for i, row in enumerate(adjacency.tolist()):
        for j in row:
            if j > i and occupancy[i] and occupancy[j]:
                unlike += occupancy[i] != occupancy[j]
                like += occupancy[i] == occupancy[j]
    assert pair_counts(occupancy, adjacency) == (unlike, like) and unlike > 0 and like > 0
    one = np.where(occupancy > 0, 1, 0).astype(np.int8)
    assert pair_counts(one, adjacency) == (0, unlike + like)


def test_pairs_on_a_grid_with_edges():
    adjacency = build_adjacency("square", 3, 3, False)
    occupancy = np.array([1, 2, 0, 1, 0, 0, 0, 0, 2], dtype=np.int8)
    # (0,0)-(0,1) and (0,1)-(1,0) are unlike; (0,0)-(1,0) is like; the far corner touches none
    assert pair_counts(occupancy, adjacency) == (2, 1)


# --- the record ----------------------------------------------------------------------------------


def test_a_grid_of_cells_that_never_change_is_recorded_as_it_stands():
    task = CompetitionTask((INERT, INERT), "square", 16, 0.5, 4, 6, sample_every=2)
    record = competition_point(task)
    start = species_start("square", 16, 0.5, 4, species=2)
    counts = [int(np.count_nonzero(start == kind)) for kind in (1, 2)]
    assert record.generations == [0, 2, 4, 6] and record.cells == 256
    assert record.alive == [[counts[0]] * 4, [counts[1]] * 4]
    assert record.births == [[0] * 4, [0] * 4] and record.contested == [0] * 4
    pairs = pair_counts(start, build_adjacency("square", 16, 16, True))
    assert (record.unlike, record.like) == ([pairs[0]] * 4, [pairs[1]] * 4)
    assert record.extinct_at == [None, None]
    assert record.start_sha256 == record.final_sha256 == digest(start)


def test_extinction_is_the_first_generation_without_a_cell():
    record = competition_point(CompetitionTask((DOOMED, INERT, LIFE), "square", 16, 0.5, 4, 5))
    assert record.extinct_at[0] == 1 and record.extinct_at[1] is None
    assert record.alive[0] == [record.alive[0][0], 0] and record.alive[0][0] > 0
    absent = competition_point(CompetitionTask((LIFE, LIFE), "square", 16, 0.5, 4, 5, (1, 0)))
    assert absent.extinct_at[1] == 0 and absent.alive[1] == [0, 0] and absent.shares == [1.0, 0.0]


def test_a_species_alone_runs_as_it_would_with_the_grid_to_itself():
    """With the other species absent, the record's counts are the one-species run's."""
    alone = competition_point(CompetitionTask((LIFE,), "square", 24, 0.4, 7, 30, sample_every=5))
    paired = competition_point(
        CompetitionTask((LIFE, HIGHLIFE), "square", 24, 0.4, 7, 30, (1, 0), sample_every=5)
    )
    assert paired.alive[0] == alone.alive[0] and paired.like == alone.like
    assert paired.unlike == [0] * 7 and sum(paired.contested) == 0


@pytest.mark.parametrize("kind", ["random", "standoff", "strongest", "weighted"])
def test_samples_add_up_whatever_the_stride(kind):
    def run(stride):
        task = CompetitionTask(
            (LIFE, HIGHLIFE, LIFE), "square", 24, 0.6, 5, 24, tie_break=kind, sample_every=stride
        )
        return competition_point(task)

    every, sixth, ends = run(1), run(6), run(0)
    assert every.generations == list(range(25)) and sixth.generations == [0, 6, 12, 18, 24]
    assert ends.generations == [0, 24]
    assert every.final_sha256 == sixth.final_sha256 == ends.final_sha256
    for i in range(3):
        assert sum(every.births[i]) == sum(sixth.births[i]) == ends.births[i][1]
        assert [every.alive[i][g] for g in sixth.generations] == sixth.alive[i]
    assert sum(every.contested) == sum(sixth.contested) == ends.contested[1] > 0
    assert [every.unlike[g] for g in sixth.generations] == sixth.unlike
    assert run(6) == sixth


def test_the_tie_break_named_is_the_one_used():
    """From one start, every kind meets the same contested cells in the first generation; a
    standoff then leaves them empty, so it has that many fewer births than a draw."""
    records = {
        kind: competition_point(
            CompetitionTask((LIFE, LIFE), "square", 32, 0.6, 2, 1, tie_break=kind, sample_every=1)
        )
        for kind in ("random", "standoff", "strongest", "weighted")
    }
    contested = {record.contested[1] for record in records.values()}
    assert len(contested) == 1 and min(contested) > 5
    assert len({record.start_sha256 for record in records.values()}) == 1

    def born(kind):
        return sum(series[1] for series in records[kind].births)

    assert born("random") == born("weighted") == born("standoff") + min(contested)
    assert born("standoff") <= born("strongest") <= born("random")
    assert records["standoff"].tie_break == "standoff" and records["random"].tie_break == "random"


def test_a_record_names_its_task_and_runs_again_the_same():
    task = CompetitionTask(
        ("B3/S23", "b36/s23"), "hexagon", "24x12", 0.5, 9, 10, (3, 1), "weighted", 3, 2, 5
    )
    record = competition_point(task)
    assert record == competition_point(task)
    data = record.to_json()
    assert {key: data[key] for key in competition.RECORD_KEY} == {
        "rules": ["B3/S23", "B36/S23"],
        "shape": "hexagon",
        "triangle_mode": None,
        "grid": "24x12",
        "density": 0.5,
        "shares": [0.75, 0.25],
        "tie_break": "weighted",
        "tie_break_radius": 3,
        "tie_break_power": 2.0,
        "seed": 9,
        "steps": 10,
        "sample_every": 5,
    }
    assert data["cells"] == 288 and data["generations"] == [0, 5, 10]
    triangle = competition_point(CompetitionTask(("B2/S12",), "triangle", 12, 0.5, 1, 3))
    assert triangle.triangle_mode == "edge+vertex" and triangle.shares == [1.0]


def test_a_bad_task_is_refused():
    good = {
        "rules": (LIFE, LIFE),
        "shape": "square",
        "grid": 16,
        "density": 0.5,
        "seed": 1,
        "steps": 5,
    }
    for change, message in (
        ({"rules": ()}, "at least one rule"),
        ({"rules": (LIFE,) * 11}, "1 to 10 species"),
        ({"steps": 0}, "steps must be at least 1"),
        ({"sample_every": -1}, "sample_every"),
        ({"shares": (1, 2, 3)}, "2 species need 2 shares"),
        ({"tie_break": "coin"}, "tie-break"),
        ({"tie_break_radius": 9}, "tie-break radius"),
        ({"rules": (LIFE, "nonsense")}, "not a B/S rule"),
        ({"grid": "11x12", "shape": "hexagon"}, "even width"),
    ):
        with pytest.raises(ValueError, match=message):
            competition_point(CompetitionTask(**{**good, **change}))


def test_run_competition_resumes_and_skips_a_task_named_twice(tmp_path):
    out = tmp_path / "competition.jsonl"
    tasks = [CompetitionTask((LIFE, HIGHLIFE), "square", 16, 0.5, seed, 8) for seed in (1, 2)]
    first = list(run_competition(tasks + tasks[:1], str(out)))
    more = tasks + [
        CompetitionTask((LIFE, HIGHLIFE), "square", 16, 0.5, 3, 8),
        CompetitionTask((LIFE, HIGHLIFE), "square", 16, 0.5, 1, 8, tie_break="standoff"),
        CompetitionTask((LIFE, HIGHLIFE), "square", 16, 0.5, 1, 8, (1, 1)),  # the first again
    ]
    again = list(run_competition(more, str(out)))
    assert len(first) == 2 and len(again) == 2 and len(read_jsonl(out)) == 4
    assert {r["tie_break"] for r in read_jsonl(out)} == {"random", "standoff"}
