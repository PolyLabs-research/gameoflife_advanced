"""Rule presets and example patterns (`ui/presets.py`): every example does what its note says
under its rule, every preset parses and is offered only where its counts fit, and examples are
placed on the grid intact."""

import numpy as np
import pytest

from latticelife.core.geometries import max_degree
from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.translations import TorusTranslations
from latticelife.experiments.objects import identify
from latticelife.ui.presets import (
    EXAMPLES,
    RULE_PRESETS,
    example_size,
    examples_for,
    place_cells,
    rule_presets,
)

# What each example is: (kind, period, cells moved per period in either coordinate).
EXPECTED = {
    "Glider": ("spaceship", 4, (1, 1)),
    "Lightweight spaceship": ("spaceship", 4, (0, -2)),
    "Pulsar": ("oscillator", 3, (0, 0)),
    "Bays' c/5 ship": ("spaceship", 5, (1, 0)),
    "Bays' glider": ("spaceship", 10, (-1, 2)),
    "c/3 ship": ("spaceship", 3, (0, -1)),
    "Diagonal ship": ("spaceship", 3, (0, -2)),
    "Period-48 oscillator": ("oscillator", 48, (0, 0)),
}


def _example(name):
    return next(e for e in EXAMPLES if e.name == name)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_example_is_the_object_its_note_names(name):
    example = _example(name)
    record = identify(example.cells, example.as_rule(), example.shape, max_period=60, margin=12)
    assert not isinstance(record, str), record
    assert (record.kind, record.period, record.displacement) == EXPECTED[name]
    assert record.population == len(example.cells)  # the stored phase is a smallest one


def test_the_gun_adds_a_glider_every_30_generations():
    gun = _example("Gosper glider gun")
    grid = LatticeGrid([Lifeform(1, gun.as_rule())], "square", 120, 120, True)
    grid.occupancy[place_cells(gun.cells, "square", 120, 120)] = 1
    rng = np.random.default_rng(0)
    populations = []
    for generation in range(121):
        if generation % 30 == 0:
            populations.append(int(np.count_nonzero(grid.occupancy)))
        grid.step(rng)
    assert populations == [36, 41, 46, 51, 56]


def test_the_r_pentomino_is_still_busy_after_two_hundred_generations():
    r = _example("R-pentomino")
    grid = LatticeGrid([Lifeform(1, r.as_rule())], "square", 64, 64, True)
    grid.occupancy[place_cells(r.cells, "square", 64, 64)] = 1
    rng = np.random.default_rng(0)
    for _ in range(200):
        grid.step(rng)
    assert int(np.count_nonzero(grid.occupancy)) > 50


def test_every_example_is_checked_here():
    assert {e.name for e in EXAMPLES} == set(EXPECTED) | {"Gosper glider gun", "R-pentomino"}


@pytest.mark.parametrize("preset", RULE_PRESETS, ids=lambda p: p.label)
def test_each_preset_parses_and_is_offered_only_where_it_fits(preset):
    rule = preset.as_rule()  # parses
    for shape, mode in (("square", "edge+vertex"), ("hexagon", "edge+vertex"),
                        ("triangle", "edge"), ("triangle", "edge+vertex")):  # fmt: skip
        offered = preset in rule_presets(shape, mode)
        fits = max(rule.birth | rule.survive, default=0) <= max_degree(shape, mode)
        meant = preset.shapes is None or shape in preset.shapes
        assert offered == (fits and meant)


def test_every_shape_has_presets_and_labels_are_unique():
    for shape, mode in (("square", "edge+vertex"), ("hexagon", "edge+vertex"),
                        ("triangle", "edge"), ("triangle", "edge+vertex")):  # fmt: skip
        labels = [p.label for p in rule_presets(shape, mode)]
        assert labels and len(labels) == len(set(labels))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.name)
@pytest.mark.parametrize("size", [(64, 64), (50, 40), (80, 80)])
def test_an_example_is_placed_whole_and_unchanged(example, size):
    width, height = size
    cells = place_cells(example.cells, example.shape, width, height)
    assert len(cells) == len(example.cells)
    torus = TorusTranslations(example.shape, width, height)
    # Up to translation, what is placed is the pattern itself (hexagons keep their columns'
    # parity, so the shape is kept, not just the offsets).
    assert torus.pattern_key(cells) == torus.pattern_key(torus.place(example.cells, 20, 20))


def test_an_example_keeps_a_grid_that_holds_it_and_otherwise_gets_its_own():
    glider = _example("Glider")
    assert example_size(glider, 100, 70, True, "edge+vertex") == (100, 70)
    assert example_size(glider, 30, 30, True, "edge+vertex") == glider.size
    ship = _example("c/3 ship")
    assert example_size(ship, 65, 64, True, "edge+vertex") == ship.size  # odd: not a hexagon torus
    assert example_size(ship, 66, 64, True, "edge+vertex") == (66, 64)


def test_examples_can_be_listed_by_shape():
    assert {e.shape for e in examples_for("hexagon")} == {"hexagon"}
    assert len(examples_for()) == len(EXAMPLES)
