"""SettingsPanel rule notation and Apply validation (F1, F2, F43), driven headless under the SDL
dummy driver."""

from dataclasses import replace

import pygame
import pytest

from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.ui import panel as panel_module
from latticelife.ui.panel import SettingsPanel

CONWAY = Rule(birth={3}, survive={2, 3})


@pytest.fixture
def font():
    pygame.init()
    return pygame.font.Font(None, 16)


def _grid_config(config, **grid):
    return replace(config, grid=replace(config.grid, **grid))


def _make_panel(config, lifeforms, font):
    applied: list[tuple] = []
    panel = SettingsPanel(config, lifeforms, font, lambda c, lfs: applied.append((c, lfs)))
    return panel, applied


def _click_apply(panel: SettingsPanel) -> None:
    surface = pygame.Surface((400, 900))
    panel.draw(surface, 16, 16, 300)
    pos = panel.apply_button.rect.center
    panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))


def test_rule_fields_show_comma_separated_counts(config, font):
    panel, _ = _make_panel(config, [Lifeform(1, CONWAY)], font)
    birth, survive = panel.rule_inputs[0]
    assert (birth.text, survive.text) == ("3", "2,3")


def test_counts_of_ten_or_more_survive_apply(config, font):
    """F2: a lifeform with birth {11} and survive {2, 12} used to be applied as {1}/{1, 2}."""
    triangle = _grid_config(config, shape="triangle", triangle_mode="edge+vertex")
    rule = Rule(birth={11}, survive={2, 12})
    panel, applied = _make_panel(triangle, [Lifeform(1, rule)], font)

    _click_apply(panel)

    assert panel.error is None
    assert applied[0][1] == [Lifeform(1, rule)]


def test_randomize_then_apply_keeps_the_generated_rule(config, font, monkeypatch):
    """F2 repro: R produced B{2,5,8,10}/S{0,8,12}, shown as '25810'/'0812', applied as a B0
    rule."""
    triangle = _grid_config(config, shape="triangle", triangle_mode="edge+vertex")
    generated = Rule(birth={2, 5, 8, 10}, survive={0, 8, 12})
    monkeypatch.setattr(panel_module.Rule, "random", staticmethod(lambda *a, **k: generated))
    panel, applied = _make_panel(triangle, [Lifeform(1, CONWAY)], font)

    panel._randomize_rule(0)
    birth, survive = panel.rule_inputs[0]
    assert (birth.text, survive.text) == ("2,5,8,10", "0,8,12")
    _click_apply(panel)

    assert applied[0][1][0].rule == generated


def test_unpatched_randomizer_round_trips_through_apply(config, font):
    triangle = _grid_config(config, shape="triangle", triangle_mode="edge+vertex")
    panel, applied = _make_panel(triangle, [Lifeform(1, CONWAY)], font)
    for _ in range(50):
        panel._randomize_rule(0)
        birth, survive = panel.rule_inputs[0]
        shown = (birth.text, survive.text)
        _click_apply(panel)
        rule = applied[-1][1][0].rule
        assert ",".join(str(v) for v in sorted(rule.birth)) == shown[0]
        assert ",".join(str(v) for v in sorted(rule.survive)) == shown[1]


def test_typed_separators_are_normalised_after_apply(config, font):
    panel, applied = _make_panel(config, [Lifeform(1, CONWAY)], font)
    birth, survive = panel.rule_inputs[0]
    birth.text, survive.text = " 3 ", "3; 2"

    _click_apply(panel)

    assert applied[0][1][0].rule == CONWAY
    assert (birth.text, survive.text) == ("3", "2,3")


@pytest.mark.parametrize(
    ("shape", "width", "height", "message"),
    [
        ("hexagon", 51, 50, "even width"),
        ("triangle", 50, 51, "even width and height"),
        ("triangle", 51, 50, "even width and height"),
    ],
)
def test_wrapped_sizes_that_do_not_tile_are_refused(config, font, shape, width, height, message):
    """F1: the panel refuses (never snaps) an odd hexagon/triangle size with wrap on."""
    assert config.grid.wrap is True
    panel, applied = _make_panel(config, [Lifeform(1, CONWAY)], font)
    panel.shape_dropdown.selected = shape
    panel.width_input.text = str(width)
    panel.height_input.text = str(height)

    _click_apply(panel)

    assert applied == []
    assert panel.error is not None and message in panel.error
    assert (panel.width_input.value, panel.height_input.value) == (width, height)


def test_odd_size_is_accepted_with_wrap_off(config, font):
    panel, applied = _make_panel(_grid_config(config, wrap=False), [Lifeform(1, CONWAY)], font)
    panel.shape_dropdown.selected = "hexagon"
    panel.width_input.text = "51"

    _click_apply(panel)

    assert panel.error is None
    assert applied[0][0].grid.width == 51


@pytest.mark.parametrize("text", ["²", "3,３"])
def test_non_ascii_digits_are_refused_without_crashing(config, font, text):
    """F43: '²' (AZERTY) used to crash Apply with int('²')."""
    panel, applied = _make_panel(config, [Lifeform(1, CONWAY)], font)
    panel.rule_inputs[0][0].text = text

    _click_apply(panel)

    assert applied == []
    assert panel.error == f"Species 1 birth: '{text[-1]}' is not a digit 0-9."


def test_counts_above_the_shape_degree_are_refused(config, font):
    panel, applied = _make_panel(config, [Lifeform(1, CONWAY), Lifeform(2, CONWAY)], font)
    panel.rule_inputs[1][1].text = "2,9"

    _click_apply(panel)

    assert applied == []
    assert panel.error == "Species 2 survive: 9 exceeds the 8 neighbours this shape has."


def test_count_limit_follows_the_selected_shape(config, font):
    panel, applied = _make_panel(config, [Lifeform(1, CONWAY)], font)
    panel.rule_inputs[0][0].text = "12"
    _click_apply(panel)
    assert applied == []  # square: 8 neighbours

    panel.shape_dropdown.selected = "triangle"
    _click_apply(panel)
    assert panel.error is None
    assert applied[0][1][0].rule.birth == {12}


def test_error_is_drawn_above_apply_and_cleared_by_a_valid_apply(config, font):
    panel, applied = _make_panel(config, [Lifeform(1, CONWAY)], font)
    surface = pygame.Surface((400, 900))
    panel.draw(surface, 16, 16, 300)
    apply_y_without_error = panel.apply_button.rect.y

    panel.rule_inputs[0][0].text = "abc"
    _click_apply(panel)
    assert panel.error is not None
    panel.draw(surface, 16, 16, 300)
    assert panel.apply_button.rect.y > apply_y_without_error

    error_height = panel.apply_button.rect.y - apply_y_without_error
    error_rows = surface.subsurface(pygame.Rect(16, apply_y_without_error, 300, error_height))
    red = [
        error_rows.get_at((x, y))
        for x in range(error_rows.get_width())
        for y in range(error_rows.get_height())
    ]
    assert any(c.r > 200 and c.g < 160 and c.b < 160 for c in red)

    panel.rule_inputs[0][0].text = "3"
    _click_apply(panel)
    assert panel.error is None
    assert len(applied) == 1


def test_set_lifeforms_rewrites_the_species_rows(config, font):
    panel, _ = _make_panel(config, [Lifeform(1, CONWAY)], font)
    panel.error = "stale"
    new = [
        Lifeform(1, Rule({1}, {2})),
        Lifeform(2, Rule({0, 8}, set())),
        Lifeform(3, Rule(set(), {4, 5})),
    ]

    panel.set_lifeforms(new)

    assert panel.species_count_input.value == 3
    assert [(b.text, s.text) for b, s in panel.rule_inputs] == [
        ("1", "2"),
        ("0,8", ""),
        ("", "4,5"),
    ]
    assert len(panel.randomize_buttons) == 3
    assert panel.error is None
