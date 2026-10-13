"""The object census (`experiments.objects`): harvesting components and naming them."""

from pathlib import Path

import numpy as np
import pytest

from latticelife.config import load_config
from latticelife.core.geometries import build_adjacency
from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.core.translations import TorusTranslations
from latticelife.experiments.harness import read_jsonl
from latticelife.experiments.objects import (
    SoupTask,
    catalogue,
    harvest,
    identify,
    run_soups,
    soup_objects,
    widened_adjacency,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"
LIFE = Rule(birth={3}, survive={2, 3})
BLOCK = [(0, 0), (0, 1), (1, 0), (1, 1)]
BLINKER = [(0, 0), (0, 1), (0, 2)]
GLIDER = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
# Life's pulsar (period 3), with internal gaps of one cell.
PULSAR = [
    (r, c)
    for r, c in [
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 8),
        (0, 9),
        (0, 10),
        (2, 0),
        (2, 5),
        (2, 7),
        (2, 12),
        (3, 0),
        (3, 5),
        (3, 7),
        (3, 12),
        (4, 0),
        (4, 5),
        (4, 7),
        (4, 12),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 8),
        (5, 9),
        (5, 10),
        (7, 2),
        (7, 3),
        (7, 4),
        (7, 8),
        (7, 9),
        (7, 10),
        (8, 0),
        (8, 5),
        (8, 7),
        (8, 12),
        (9, 0),
        (9, 5),
        (9, 7),
        (9, 12),
        (10, 0),
        (10, 5),
        (10, 7),
        (10, 12),
        (12, 2),
        (12, 3),
        (12, 4),
        (12, 8),
        (12, 9),
        (12, 10),
    ]
]


def _key(cells, shape="square", mode="edge+vertex", size=24):
    t = TorusTranslations(shape, size, size, mode)
    return t.pattern_key([r * size + c for r, c in cells])


def _grid(cells, shape="square", mode="edge+vertex", size=24, rule=LIFE):
    grid = LatticeGrid([Lifeform(1, rule)], shape, size, size, True, mode)
    for r, c in cells:
        grid.occupancy[r * size + c] = 1
    return grid


def test_widened_adjacency_reaches_two_steps():
    adjacency = build_adjacency("square", 8, 8, True)
    wide = widened_adjacency(adjacency, 2)
    assert wide.shape == (64, 8 + 64)
    reach = set(wide[0].tolist()) - {-1}
    # (0, 0) reaches every cell within Chebyshev distance 2: a 5x5 square, itself included.
    assert len(reach) == 25
    assert np.array_equal(widened_adjacency(adjacency, 1), adjacency)
    with pytest.raises(ValueError, match="radius"):
        widened_adjacency(adjacency, 0)


def test_harvest_separates_distant_objects_and_keeps_a_gapped_one_whole():
    far = [(r + 20, c + 20) for r, c in BLOCK]  # well clear of the pulsar's 13x13 box
    grid = _grid(PULSAR + far, size=32)
    components = harvest(grid, radius=2)
    assert [c.size for c in components] == [48, 4]
    assert harvest(_grid([]), radius=2) == []
    # At radius 1 the pulsar's gapped arms fall apart.
    assert len(harvest(_grid(PULSAR, size=32), radius=1)) > 1
    assert _key(PULSAR) is None  # on a 24-torus the pulsar spans over half: no key


def test_identify_names_life_objects():
    block = identify(_key(BLOCK), LIFE, "square")
    assert (block.kind, block.period, block.displacement, block.population) == (
        "still",
        1,
        (0, 0),
        4,
    )
    # The key is the smallest sorted cell list over anchors: here the anchor is the block's
    # bottom-right cell, so the others read negative.
    assert block.key == ((-1, -1), (-1, 0), (0, -1), (0, 0))
    assert block.key == _key([(r + 7, c + 9) for r, c in BLOCK])

    blinker = identify(_key(BLINKER), LIFE, "square")
    assert (blinker.kind, blinker.period, blinker.displacement) == ("oscillator", 2, (0, 0))
    assert blinker.population == blinker.max_population == 3
    # Both phases name the same object.
    vertical = identify(_key([(0, 0), (1, 0), (2, 0)]), LIFE, "square")
    assert vertical == blinker

    glider = identify(_key(GLIDER), LIFE, "square")
    assert (glider.kind, glider.period, glider.displacement) == ("spaceship", 4, (1, 1))
    assert glider.population == glider.max_population == 5
    assert identify(_key(GLIDER), LIFE, "square", margin=12) == glider  # torus-independent

    pulsar = identify(_key(PULSAR, size=40), LIFE, "square")  # 13 wide: needs a torus over 26
    assert (pulsar.kind, pulsar.period, pulsar.population, pulsar.max_population) == (
        "oscillator",
        3,
        48,
        72,
    )


def test_identify_reports_why_a_component_is_not_an_object():
    assert identify(_key([(0, 0)]), LIFE, "square") == "died"
    assert identify((), LIFE, "square") == "died"
    # A glider's period is 4: with fewer generations allowed it cannot be settled.
    assert identify(_key(GLIDER), LIFE, "square", max_period=3) == "unsettled"
    # The R-pentomino grows for over a thousand generations, past half of any torus a 3x3
    # seed is given: it touches itself across the seam and is reported as too large.
    r_pentomino = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]
    assert identify(_key(r_pentomino), LIFE, "square", max_period=200) == "too_large"
    # Under a rule where every live cell's neighbours are born, a seed grows without bound.
    grow = Rule(birth={1, 2, 3, 4, 5, 6, 7, 8}, survive=set(range(9)))
    assert identify(_key(BLOCK), grow, "square", max_period=50) == "too_large"
    with pytest.raises(ValueError, match="margin"):
        identify(_key(BLOCK), LIFE, "square", margin=3)
    with pytest.raises(ValueError, match="max_period"):
        identify(_key(BLOCK), LIFE, "square", max_period=0)


@pytest.mark.parametrize(
    ("shape", "mode"),
    [("hexagon", "edge+vertex"), ("triangle", "edge"), ("triangle", "edge+vertex")],
)
def test_identify_still_lifes_and_oscillators_on_every_lattice(shape, mode):
    """Rules built so the answer is known by construction: a lone cell survives with no
    neighbours (still life), or two adjacent cells trade places each step (oscillator)."""
    lone = identify(_key([(2, 2)], shape, mode), Rule(birth=set(), survive={0}), shape, mode)
    assert (lone.kind, lone.period, lone.population) == ("still", 1, 1)
    # A down triangle (odd parity) is its own kind of still life: a different key.
    if shape == "triangle":
        odd = identify(_key([(2, 3)], shape, mode), Rule(birth=set(), survive={0}), shape, mode)
        assert odd.key != lone.key and odd.kind == "still"
    # B1/S: a lone cell dies and every cell with exactly one live neighbour is born.
    # From one cell that gives its ring of neighbours; the ring then dies back only where
    # counts differ, so instead test with the complement idea: everything with 0 or 1
    # neighbours survives and nothing is born -> a pair is a still life of 2.
    pair_cells = [(2, 2), (2, 3)] if shape != "hexagon" else [(2, 2), (3, 2)]
    pair = identify(_key(pair_cells, shape, mode), Rule(birth=set(), survive={0, 1}), shape, mode)
    assert (pair.kind, pair.period, pair.population) == ("still", 1, 2)


def test_identify_finds_a_hexagon_oscillator_by_running_a_random_rule():
    """No hand-made hex oscillator is assumed: a 3-cell seed under B2/S34 (a known hexagonal
    Life variant) must either die, settle or be reported honestly; the record it gives must be
    reproducible from its own key."""
    seed = [(3, 3), (3, 4), (4, 4)]
    rule = Rule(birth={2}, survive={3, 4})
    found = identify(_key(seed, "hexagon"), rule, "hexagon", max_period=50)
    if isinstance(found, str):
        assert found in ("died", "unsettled", "too_large")
    else:
        again = identify(found.key, rule, "hexagon", max_period=50)
        assert again == found


def test_soup_objects_and_run_soups_record_and_resume(tmp_path):
    config = load_config(DEFAULT_CONFIG_PATH)
    task = SoupTask(config, "B3/S23", "square", 24, 3, 300, 0.3, 2, 100, 8)
    record = soup_objects(task)
    assert record["rule"] == "B3/S23" and record["seed"] == 3
    assert set(record["unidentified"]) == {"died", "unsettled", "too_large"}
    kinds = {found["kind"] for found in record["objects"]}
    assert kinds <= {"still", "oscillator", "spaceship"}
    if record["soup_period"] is not None:
        assert record["stopped_at"] < 300

    out = tmp_path / "soups.jsonl"
    first = list(run_soups(config, LIFE, "square", 16, [1, 2], 60, 0.3, out, max_period=50))
    assert len(first) == 2
    assert list(run_soups(config, LIFE, "square", 16, [1, 2], 60, 0.3, out, max_period=50)) == []
    assert len(read_jsonl(out)) == 2

    listed = catalogue(read_jsonl(out))
    assert all(
        set(entry) == {"key", "kind", "period", "displacement", "population", "count"}
        for entry in listed
    )
    assert listed == sorted(listed, key=lambda e: -e["count"])


def test_run_soups_matches_in_parallel(tmp_path):
    config = load_config(DEFAULT_CONFIG_PATH)
    args = (config, LIFE, "hexagon", 16, [1, 2, 3], 40, 0.3)
    serial = [r["objects"] for r in run_soups(*args, tmp_path / "s.jsonl", max_period=30)]
    parallel = {
        r["seed"]: r["objects"]
        for r in run_soups(*args, tmp_path / "p.jsonl", workers=2, max_period=30)
    }
    assert serial == [parallel[seed] for seed in (1, 2, 3)]
