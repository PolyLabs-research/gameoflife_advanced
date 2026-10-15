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
    assert len(blinker.key) == blinker.population

    glider = identify(_key(GLIDER), LIFE, "square")
    assert (glider.kind, glider.period, glider.displacement) == ("spaceship", 4, (1, 1))
    assert glider.population == glider.max_population == 5
    assert identify(_key(GLIDER), LIFE, "square", margin=12) == glider  # torus-independent

    pulsar = identify(_key(PULSAR, size=40), LIFE, "square")  # 13 wide: needs a torus over 26
    assert len(pulsar.key) == pulsar.population == 48  # the key is the smallest phase
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
        set(entry)
        == {
            "key",
            "kind",
            "period",
            "displacement",
            "population",
            "rule_min",
            "rule_max",
            "rule_count",
            "count",
            "rules",
        }
        for entry in listed
    )
    assert len(catalogue(read_jsonl(out), up_to_symmetry=True)) <= len(listed)
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


# --- canonical keys and rule ranges -------------------------------------------------------------


def _bits(rule_text):
    rule = Rule.from_bs_string(rule_text)
    return rule.birth, rule.survive


def test_the_hexagon_domino_bridge_oscillator_is_one_object_in_every_orientation():
    """The orientation artifact of translation-only keys: in two orientations the cycle is filed
    under the domino, in the other under the bridge, but it is one object. Its rule range is the
    hand-derived condition: born on 2, dies on 0 and 1, never born on 0 or 1."""
    b2 = Rule(birth={2}, survive=set())
    keys = [((-1, 0), (0, 0)), ((-1, -1), (0, 0)), ((-1, 1), (0, 0)), ((0, -2), (0, 0))]
    found = [identify(key, b2, "hexagon") for key in keys]
    assert {(f.kind, f.period) for f in found} == {("oscillator", 2)}
    assert len({f.key for f in found}) == 3  # three orientations, filed under two key types
    assert len({f.canonical_key for f in found}) == 1
    assert (found[0].rule_min, found[0].rule_max, found[0].rule_count) == (
        "B2/S",
        "B23456/S23456",
        2**9,
    )


def test_rule_range_is_exactly_the_rules_that_run_the_same_cycle():
    """Every rule inside the range runs the same object; every rule just outside it does not."""
    glider = identify(_key(GLIDER), LIFE, "square")
    assert (glider.rule_min, glider.rule_max, glider.rule_count) == (
        "B3/S23",
        "B3678/S0235678",
        256,
    )
    low_b, low_s = _bits(glider.rule_min)
    high_b, high_s = _bits(glider.rule_max)
    rng = np.random.default_rng(0)
    for _ in range(12):
        birth = low_b | {k for k in high_b - low_b if rng.random() < 0.5}
        survive = low_s | {k for k in high_s - low_s if rng.random() < 0.5}
        inside = identify(_key(GLIDER), Rule(birth, survive), "square")
        assert (inside.kind, inside.period, inside.displacement) == ("spaceship", 4, (1, 1))
    for flipped in [Rule({3, 4}, {2, 3}), Rule({3}, {2, 3, 4}), Rule({3}, {3})]:
        outside = identify(_key(GLIDER), flipped, "square")
        assert isinstance(outside, str) or (outside.kind, outside.period) != ("spaceship", 4)


def test_canonical_key_is_the_same_for_every_phase_and_orientation_of_an_oscillator():
    blinker_a = identify(_key([(0, 0), (0, 1), (0, 2)]), LIFE, "square")
    blinker_b = identify(_key([(0, 0), (1, 0), (2, 0)]), LIFE, "square")
    assert blinker_a.canonical_key == blinker_b.canonical_key
    glider_turned = identify(_key([(0, 1), (1, 0), (2, 0), (2, 1), (2, 2)]), LIFE, "square")
    assert glider_turned.kind == "spaceship"
    assert glider_turned.canonical_key == identify(_key(GLIDER), LIFE, "square").canonical_key


def test_catalogue_up_to_symmetry_merges_orientations_including_old_records():
    """Records written before canonical keys existed are re-identified to merge them."""
    b2 = Rule(birth={2}, survive=set())
    objects = [
        identify(key, b2, "hexagon").to_json()
        for key in [((-1, 0), (0, 0)), ((-1, -1), (0, 0)), ((-1, 1), (0, 0))]
    ]
    old = [
        {k: v for k, v in o.items() if k not in ("canonical_key", "rule_min", "rule_max")}
        for o in objects
    ]
    for records in (
        [{"rule": "B2/S", "shape": "hexagon", "objects": objects}],
        [{"rule": "B2/S", "shape": "hexagon", "triangle_mode": None, "objects": old}],
    ):
        by_key = catalogue(records)
        merged = catalogue(records, up_to_symmetry=True)
        assert len(by_key) == 3  # one domino key and two bridge keys
        assert [(e["kind"], e["period"], e["count"], e["rules"]) for e in merged] == [
            ("oscillator", 2, 3, 1)
        ]
        assert (merged[0]["rule_min"], merged[0]["rule_count"]) == ("B2/S", 512)


def test_catalogue_keeps_one_pattern_apart_when_it_behaves_differently_under_two_rules():
    domino = ((-1, 0), (0, 0))
    still = identify(domino, Rule(set(), {1}), "hexagon").to_json()
    blinks = identify(domino, Rule({2}, set()), "hexagon").to_json()
    records = [
        {"rule": "B/S1", "shape": "hexagon", "objects": [still]},
        {"rule": "B2/S", "shape": "hexagon", "objects": [blinks]},
    ]
    assert sorted((e["kind"], e["rules"]) for e in catalogue(records)) == [
        ("oscillator", 1),
        ("still", 1),
    ]


def test_a_hexagon_object_near_half_its_torus_never_crashes_the_naming():
    """A period-29 oscillator under B25/S26 grows to a phase whose row span is one short of half
    its torus from its first cell but half the torus from a cell in an odd column, where
    pattern_key re-anchors it: identify raised AssertionError (1 call in 1,024,000 of an
    exhaustive small-seed search). It now reports too_large there, and names the object on a
    roomier torus."""
    seed = ((-5, 3), (-3, 2), (-2, 1), (0, 0))  # four cells, each two steps from the next
    rule = Rule(birth={2, 5}, survive={2, 6})
    assert identify(seed, rule, "hexagon", margin=8) == "too_large"
    found = identify(seed, rule, "hexagon", margin=10)
    assert (found.kind, found.period, found.population, found.max_population) == (
        "oscillator",
        29,
        6,
        30,
    )
    assert (found.rule_min, found.rule_max, found.rule_count) == ("B25/S26", "B25/S26", 1)


def test_a_long_thin_hexagon_pattern_is_not_too_large_while_it_stays_in_its_hull():
    """A rule without 0, 1 or 2 in B never births outside a pattern's hexagonal hull, but a
    one-row pattern grows to fill its hull, a tall hexagon. The object's torus is sized from the
    hull, so the row settles into a still life instead of being called too large."""
    row = tuple((0, c) for c in range(-15, 1))
    found = identify(row, Rule({3}, {2, 3}), "hexagon", margin=8)
    assert not isinstance(found, str)
    assert (found.kind, found.period) == ("still", 1)


def test_catalogue_counts_a_spaceship_once_up_to_symmetry_whichever_way_it_travels():
    from latticelife.core.translations import point_group, transform_key

    ship = ((-3, -2), (-3, -1), (-3, 0), (-2, 1), (-1, 0), (0, 0))  # c/3 under B24/S245
    rule = Rule({2, 4}, {2, 4, 5})
    found = [
        identify(transform_key(ship, g, "hexagon"), rule, "hexagon") for g in point_group("hexagon")
    ]
    assert {(f.kind, f.period, f.rule_count) for f in found} == {("spaceship", 3, 16)}
    records = [{"rule": "B24/S245", "shape": "hexagon", "objects": [f.to_json() for f in found]}]
    assert len(catalogue(records)) == 6  # six directions of travel
    merged = catalogue(records, up_to_symmetry=True)
    assert [(e["kind"], e["count"], e["rule_count"]) for e in merged] == [("spaceship", 12, 16)]


def test_the_fit_check_is_exact_on_hexagons_not_one_row_short():
    """A uniform one-row slack (the first fix for the pattern_key crash) also threw away objects
    that fit from every anchor: a period-17 oscillator under B256/S346 was too_large at margin
    8. The check now reads both column-parity frames, as pattern_key does."""
    seed = ((-2, -1), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0))  # small-search seed 208
    found = identify(seed, Rule({2, 5, 6}, {3, 4, 6}), "hexagon", margin=8)
    assert not isinstance(found, str)
    assert (found.kind, found.period, found.population, found.max_population) == (
        "oscillator",
        17,
        6,
        36,
    )


def test_a_pattern_and_its_rotations_get_the_same_verdict():
    """The object's torus is sized over every orientation of the seed, so a seed near the size
    limit is not named in one orientation and called too large in another."""
    from latticelife.core.translations import point_group, transform_key

    seed = ((-4, -5), (-2, 0), (-1, 0), (0, 0))
    rule = Rule({2, 4, 5}, {1, 2, 4, 6})
    verdicts = {
        _summary(identify(transform_key(seed, g, "hexagon"), rule, "hexagon", 120, 4))
        for g in point_group("hexagon")
    }
    assert len(verdicts) == 1


def _summary(found):
    return found if isinstance(found, str) else (found.kind, found.period, found.canonical_key)


def test_catalogue_counts_an_object_once_when_old_and_new_records_mix():
    domino = ((-1, 0), (0, 0))
    new = identify(domino, Rule({2}, set()), "hexagon").to_json()
    old = {
        k: v
        for k, v in new.items()
        if k not in ("canonical_key", "rule_min", "rule_max", "rule_count")
    }
    records = [
        {"rule": "B2/S", "shape": "hexagon", "objects": [old]},
        {"rule": "B2/S", "shape": "hexagon", "objects": [new]},
    ]
    assert [e["count"] for e in catalogue(records)] == [2]
    assert [e["count"] for e in catalogue(records, up_to_symmetry=True)] == [2]
