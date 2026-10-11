"""SettingsPanel rule notation and Apply validation (F1, F2, F43), and its event dispatch,
collapsing and scrolling (F0, F41, F42, F44, F45, F49, F55, F60), driven headless under the SDL
dummy driver."""

from dataclasses import replace

import numpy as np
import pygame
import pytest

from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.ui import panel as panel_module
from latticelife.ui import widgets
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


# --- event dispatch, collapsing and scrolling, driven through the real App ------------------


def _open_shape_list(app, u):
    u.click(app.settings_panel.shape_dropdown.rect.center)
    u.frame()  # the open list is laid out when drawn
    return dict(app.settings_panel.shape_dropdown._option_rects)


def test_an_open_shape_list_takes_the_click_for_itself(make_app, ui):
    """F0: choosing 'triangle' also toggled the Wrap button under it, and 'hexagon' focused
    Height."""
    app = make_app()
    u = ui(app)
    u.frame()
    panel = app.settings_panel

    options = _open_shape_list(app, u)
    assert options["triangle"].colliderect(panel.wrap_button.rect)
    u.click(options["triangle"].center)
    assert panel.shape_dropdown.selected == "triangle"
    assert panel.config.grid.wrap is True

    options = _open_shape_list(app, u)
    assert options["hexagon"].colliderect(panel.height_input.rect)
    u.click(options["hexagon"].center)
    assert panel.shape_dropdown.selected == "hexagon"
    assert not panel.height_input.active

    _open_shape_list(app, u)
    session = app.session
    u.click(app.new_button.rect.center)  # outside the list: only closes it
    assert not panel.shape_dropdown.active and app.session is session


def test_shrinking_the_count_then_clicking_a_removed_row_does_nothing(make_app, ui):
    """F41: committing a smaller count by clicking a now-removed R button raised IndexError."""
    app = make_app()
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    u.click(panel.species_count_input.rect.center)
    u.clear_field()
    u.type("3")
    u.key(pygame.K_RETURN, "\r")
    assert len(panel.rule_inputs) == 3
    removed_r = panel.randomize_buttons[2].rect.center
    first_rule = (panel.rule_inputs[0][0].text, panel.rule_inputs[0][1].text)

    u.click(panel.species_count_input.rect.center)
    u.clear_field()
    u.type("1")
    u.click(removed_r)  # the blur commits 1 first; the R it was aimed at is gone

    assert len(panel.rule_inputs) == len(panel.randomize_buttons) == 1
    assert (panel.rule_inputs[0][0].text, panel.rule_inputs[0][1].text) == first_rule
    panel._randomize_rule(2)  # a stale index is ignored, not an IndexError


def test_collapsed_sections_get_no_clicks(make_app, ui):
    """F42: widgets in a collapsed section kept their rects and took clicks meant for the
    controls that moved up into their place."""
    app = make_app()
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    old_shape = panel.shape_dropdown.rect.copy()

    u.click(panel.grid_section.header_rect.center)  # collapse Grid
    assert panel.wrap_button.rect.size == (0, 0) and panel.height_input.rect is None
    u.click(old_shape.center)  # now the Speed header: no ghost Shape list
    assert not panel.shape_dropdown.active
    u.click(panel.speed_section.header_rect.center)  # re-expand Speed
    u.click(panel.species_section.header_rect.center)
    assert panel.config.grid.wrap is True  # hidden Wrap untouched
    u.click(panel.fps_slider.rect.center)
    assert not panel.height_input.active and not panel.width_input.active

    u.click(panel.grid_section.header_rect.center)  # expand Grid again
    u.click(panel.species_section.header_rect.center)  # expand Species again
    fps = panel.fps_slider.value
    u.click(panel.speed_section.header_rect.center)  # collapse Speed
    assert panel.fps_slider.rect is None
    header = panel.species_section.header_rect
    u.click((header.right - 14, header.centery))
    assert panel.fps_slider.value == fps


def test_collapsing_a_section_commits_and_blurs_its_focused_field(make_app, ui):
    app = make_app()
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    u.click(panel.height_input.rect.center)
    u.clear_field()
    u.type("40")
    u.click(panel.grid_section.header_rect.center)
    assert not panel.height_input.active and panel.height_input.value == 40
    u.key(pygame.K_SPACE, " ")  # shortcuts work again
    assert app.session.paused


def test_ten_species_scroll_and_apply_stays_reachable(make_app, ui):
    """F44: with 7+ species the panel ran off the window and Apply could not be clicked."""
    app = make_app(simulation={"n_lifeforms": 10})
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    window = app.screen.get_rect()
    assert window.contains(panel.apply_button.rect)
    assert panel.max_scroll > 0
    last_r = panel.randomize_buttons[9]
    assert not panel._content_clip.contains(last_r.rect)

    u.wheel(-1, over=(150, 400))
    assert panel.scroll_y > 0
    u.wheel(-20)
    assert panel.scroll_y == panel.max_scroll  # clamped at the bottom
    assert panel._content_clip.contains(last_r.rect)
    before = panel.rule_inputs[9][0].text, panel.rule_inputs[9][1].text
    for _ in range(5):
        u.click(last_r.rect.center)
        if (panel.rule_inputs[9][0].text, panel.rule_inputs[9][1].text) != before:
            break
    assert (panel.rule_inputs[9][0].text, panel.rule_inputs[9][1].text) != before

    # The Grid header is now scrolled up behind the stat cards: a click there is not its.
    hidden = panel.grid_section.header_rect
    assert hidden.bottom <= panel._content_clip.top
    u.click(hidden.center)
    assert panel.grid_section.expanded

    u.click(panel.apply_button.rect.center)
    assert len(app.session.lifeforms) == 10
    u.wheel(30, over=(150, 400))
    assert panel.scroll_y == 0  # clamped at the top


def test_the_wheel_outside_the_panel_does_not_scroll_it(make_app, ui):
    app = make_app(simulation={"n_lifeforms": 10})
    u = ui(app)
    u.frame()
    u.wheel(-3, over=app.layout.grid_rect().center)
    assert app.settings_panel.scroll_y == 0


def test_apply_reads_width_height_shape_and_rules_into_the_right_fields(config, font):
    """F60: a swapped width/height or birth/survive in Apply went unnoticed by every test."""
    for shape in ("square", "hexagon", "triangle"):
        panel, applied = _make_panel(config, [Lifeform(1, CONWAY), Lifeform(2, CONWAY)], font)
        panel.shape_dropdown.selected = shape
        panel.width_input.text, panel.height_input.text = "30", "20"
        panel.rule_inputs[1][0].text, panel.rule_inputs[1][1].text = "1", "4,5"
        _click_apply(panel)
        new_config, lifeforms = applied[-1]
        assert (new_config.grid.shape, new_config.grid.width, new_config.grid.height) == (
            shape,
            30,
            20,
        )
        assert new_config.simulation.n_lifeforms == 2
        assert lifeforms[1] == Lifeform(2, Rule(birth={1}, survive={4, 5}))


def test_wrap_is_staged_until_apply(config, font):
    panel, applied = _make_panel(config, [Lifeform(1, CONWAY)], font)
    panel._toggle_wrap()
    assert applied == []
    assert panel.wrap_button.text == "Wrap: Off"
    _click_apply(panel)
    assert applied[0][0].grid.wrap is False


def test_raising_the_count_adds_rows_with_placeholders(config, font):
    panel, _ = _make_panel(config, [Lifeform(1, CONWAY)], font)
    panel.species_count_input.text = "4"
    panel.species_count_input.commit()
    assert len(panel.rule_inputs) == len(panel.randomize_buttons) == 4
    assert [(b.placeholder, s.placeholder) for b, s in panel.rule_inputs] == [
        ("birth", "survive")
    ] * 4


def test_more_than_ten_lifeforms_are_shown_and_applied_as_ten(config, font):
    """F49: the count showed 12 while Apply silently cut the rows to 10."""
    lifeforms = [Lifeform(i, CONWAY) for i in range(1, 13)]
    panel, applied = _make_panel(config, lifeforms, font)
    assert panel.species_count_input.text == "10"
    assert len(panel.rule_inputs) == 10
    _click_apply(panel)
    assert len(applied[0][1]) == 10


def test_the_r_button_is_reproducible_from_the_seed(config, font):
    """F55: R sampled from a fresh unseeded generator on every press."""
    seeded = replace(config, simulation=replace(config.simulation, seed=42))
    texts = []
    for _ in range(2):
        panel, _ = _make_panel(seeded, [Lifeform(1, CONWAY)], font)
        sequence = []
        for _ in range(4):
            panel._randomize_rule(0)
            sequence.append((panel.rule_inputs[0][0].text, panel.rule_inputs[0][1].text))
        texts.append(sequence)
    assert texts[0] == texts[1]
    assert len(set(texts[0])) > 1


def test_fps_slider_reports_live_and_shows_its_value(config, font):
    """F45: the slider had no callback and never showed a value."""
    seen = []
    panel = SettingsPanel(
        config, [Lifeform(1, CONWAY)], font, lambda c, lfs: None, on_fps_change=seen.append
    )
    surface = pygame.Surface((400, 900))
    panel.draw(surface, 16, 16, 300)
    slider = panel.fps_slider
    value_area = pygame.Rect(slider.rect.right, slider.rect.y, 316 - slider.rect.right, 28)
    before = pygame.surfarray.array3d(surface.subsurface(value_area)).copy()

    end = (slider.rect.right - 1, slider.rect.centery)  # the last pixel reads 60, not 59
    panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=end, button=1))
    panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=end, button=1))
    surface.fill((0, 0, 0))
    panel.draw(surface, 16, 16, 300)

    assert seen == [60] and panel.config.ui.fps == 60
    assert not np.array_equal(before, pygame.surfarray.array3d(surface.subsurface(value_area)))


def test_a_click_that_blurs_a_field_still_reaches_apply_and_the_action_buttons(make_app, ui):
    """Commit 0b5d48d: a focused field that commits on blur must not swallow the click meant
    for another control — the typed value is committed first, then that control acts."""
    app = make_app()
    u = ui(app)
    u.frame(3)
    panel = app.settings_panel

    u.click(panel.width_input.rect.center)
    u.clear_field()
    u.type("30")
    u.click(panel.apply_button.rect.center)  # no Enter first
    assert not panel.width_input.active
    assert app.session.grid.width == 30

    u.frame(3)
    u.click(panel.rule_inputs[0][1].rect.center)
    u.click(app.reset_button.rect.center)
    assert not panel.rule_inputs[0][1].active
    assert app.session.generation == 0


@pytest.mark.parametrize(
    ("shape", "width", "height", "wrap"),
    [("triangle", 6, 4, True), ("square", 3, 3, True), ("hexagon", 600, 2, False)],
)
def test_any_valid_grid_size_is_shown_and_applied_unchanged(
    config, font, shape, width, height, wrap
):
    """The fields used to clamp to 5..500, so a valid 6x4 triangle config showed 6x5 and Apply
    then refused the odd height it had made up."""
    sized = _grid_config(config, shape=shape, width=width, height=height, wrap=wrap)
    panel, applied = _make_panel(sized, [Lifeform(1, CONWAY)], font)
    assert (panel.width_input.text, panel.height_input.text) == (str(width), str(height))
    _click_apply(panel)
    assert panel.error is None
    assert (applied[0][0].grid.width, applied[0][0].grid.height) == (width, height)


def test_a_wrapped_size_below_the_shape_minimum_is_refused_with_the_minimum(config, font):
    panel, applied = _make_panel(
        _grid_config(config, shape="triangle"), [Lifeform(1, CONWAY)], font
    )
    panel.width_input.text, panel.height_input.text = "4", "2"
    _click_apply(panel)
    assert applied == []
    assert "at least 6 x 4" in panel.error


@pytest.mark.parametrize(
    ("shape", "width", "height", "flagged"),
    [
        ("hexagon", 51, 50, ["width"]),
        ("triangle", 50, 51, ["height"]),
        ("triangle", 51, 51, ["width", "height"]),
        ("triangle", 4, 50, ["width"]),
    ],
)
def test_a_refused_size_outlines_the_field_at_fault(config, font, shape, width, height, flagged):
    """F1, UI side: the inline error names the problem and the field(s) at fault are outlined."""
    panel, applied = _make_panel(config, [Lifeform(1, CONWAY)], font)
    panel.shape_dropdown.selected = shape
    panel.width_input.text, panel.height_input.text = str(width), str(height)
    _click_apply(panel)
    assert applied == []
    assert [name for name in ("width", "height") if getattr(panel, f"{name}_input").invalid] == (
        flagged
    )

    panel.width_input.text, panel.height_input.text = "50", "50"
    _click_apply(panel)
    assert applied and not panel.width_input.invalid and not panel.height_input.invalid


def test_a_refused_rule_is_outlined_and_scrolled_into_view(make_app, ui):
    """F2, UI side: with 10 species the bad row may be scrolled away or its section collapsed;
    a failed Apply opens the section, scrolls the row into view and outlines that field."""
    app = make_app(simulation={"n_lifeforms": 10})
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    bad = panel.rule_inputs[9][1]
    bad.text = "2,9"
    u.click(panel.species_section.header_rect.center)  # collapse Species
    u.click(panel.apply_button.rect.center)
    u.frame()

    assert panel.error == "Species 10 survive: 9 exceeds the 8 neighbours this shape has."
    assert panel.species_section.expanded
    assert bad.invalid and not panel.rule_inputs[9][0].invalid
    assert panel._content_clip.contains(bad.rect)
    red = pygame.surfarray.array3d(app.screen)[
        bad.rect.left, bad.rect.top + 8 : bad.rect.bottom - 8
    ]
    assert np.all(red == widgets.ERROR_COLOR)


# --- focus and scrolling, error text ----------------------------------------------------------


def test_a_focused_field_scrolled_out_of_view_commits_and_frees_the_shortcuts(make_app, ui):
    """A focused field the wheel scrolled away kept focus: typing edited it unseen and Space,
    N, R, S and V did nothing."""
    app = make_app(simulation={"n_lifeforms": 10})
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    width = panel.width_input
    u.click(width.rect.center)
    u.clear_field()
    u.type("40")

    u.wheel(-20, over=(150, 400))
    assert not panel._content_clip.colliderect(width.rect)
    assert not width.active and width.value == 40  # committed, as a click elsewhere would
    assert panel.focused_rect() is None
    u.key(pygame.K_SPACE, " ")
    assert app.session.paused
    u.type("9")
    assert width.text == "40"


def test_a_focused_field_a_smaller_window_squeezes_out_of_view_commits(make_app, ui):
    app = make_app(simulation={"n_lifeforms": 10})
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    birth = panel.rule_inputs[5][0]
    assert panel._content_clip.contains(birth.rect)
    u.click(birth.rect.center)
    u.type("4")

    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=1200, h=500, size=(1200, 500)))
    u.frame()

    assert not panel._content_clip.colliderect(birth.rect)
    assert not birth.active and birth.text == "34"
    u.key(pygame.K_SPACE, " ")
    assert app.session.paused


def test_keys_in_the_same_frame_as_the_scroll_reach_the_shortcuts(make_app, ui):
    app = make_app(simulation={"n_lifeforms": 10})
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    u.click(panel.width_input.rect.center)
    assert panel.width_input.active

    pygame.event.post(
        pygame.event.Event(pygame.MOUSEMOTION, pos=(150, 400), rel=(0, 0), buttons=(0, 0, 0))
    )
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-20, flipped=False))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, unicode=" ", mod=0))
    pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text="9"))
    u.frame()

    assert app.session.paused
    assert not panel.width_input.active and panel.width_input.text == "50"


def test_a_click_in_the_same_frame_as_the_scroll_meets_the_scrolled_widgets(make_app, ui):
    app = make_app(simulation={"n_lifeforms": 10})
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    spot = panel.rule_inputs[4][0].rect.center  # species 6's birth field after one notch

    pygame.event.post(
        pygame.event.Event(pygame.MOUSEMOTION, pos=(150, 400), rel=(0, 0), buttons=(0, 0, 0))
    )
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1, flipped=False))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=spot, button=1))
    u.frame()

    assert panel.scroll_y == panel_module.WHEEL_STEP
    assert panel.rule_inputs[5][0].active and not panel.rule_inputs[4][0].active


def test_only_the_visible_part_of_a_field_counts_as_on_it(make_app, ui):
    """A click above the viewport, over the stat cards, landed in a half-scrolled field's
    hidden part and so did not blur it."""
    app = make_app(simulation={"n_lifeforms": 10})
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    width = panel.width_input
    u.click(width.rect.center)
    clip = panel._content_clip
    panel.scroll_y = width.rect.top - clip.top + 10  # the field straddles the viewport top
    u.frame()
    assert width.rect.top < clip.top < width.rect.bottom
    assert width.active  # partly visible: still being typed in
    assert panel.focused_rect() == width.rect.clip(clip)

    u.click((width.rect.centerx, clip.top - 4))
    assert not width.active


def _any_red(pixels: np.ndarray) -> bool:
    """True if any pixel of an (..., 3) array is error-text red (or antialiased toward it)."""
    return bool(np.any((pixels[..., 0] > 150) & (pixels[..., 1] < 140) & (pixels[..., 2] < 140)))


def test_a_long_error_is_drawn_inside_the_panel_and_apply_stays_put(config, font):
    """An error echoing a long digit run drew one unbroken line past the panel into the gutter;
    a long enough one would push Apply off the window."""
    panel, _ = _make_panel(config, [Lifeform(1, CONWAY)], font)
    panel.error = "Species 1 birth: " + "1" * 5000 + " exceeds the 8 neighbours this shape has."
    surface = pygame.Surface((800, 700))
    panel.draw(surface, 16, 16, 300, 600)

    pixels = pygame.surfarray.array3d(surface)  # indexed [x, y]
    assert not _any_red(pixels[316:])
    error_height = panel_module.MAX_ERROR_LINES * font.get_linesize() + panel_module.CONTROL_GAP
    apply_top = panel.apply_button.rect.top
    assert apply_top - panel._content_clip.bottom == error_height
    assert _any_red(pixels[16:316, apply_top - error_height : apply_top])
    assert panel.apply_button.rect.bottom <= 616

    # Clipped to the column even where a single character is wider than it.
    panel.error = "W" * 3
    surface.fill((0, 0, 0))
    panel.draw(surface, 16, 16, 4, 600)
    assert not _any_red(pygame.surfarray.array3d(surface)[20:])


def test_a_long_rule_is_refused_with_a_short_error_drawn_inside_the_panel(make_app, ui):
    app = make_app()
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    birth = panel.rule_inputs[0][0]
    u.click(birth.rect.center)
    u.type("1" * 60)
    u.click(panel.apply_button.rect.center)
    u.frame()

    assert panel.error == "Species 1 birth: 31111…11111 exceeds the 8 neighbours this shape has."
    gutter = pygame.Rect(panel._content_clip.right, panel._content_clip.bottom, 0, 0)
    gutter.width = app.layout.grid_rect().left - gutter.left
    gutter.height = panel.apply_button.rect.top - gutter.top
    assert not _any_red(pygame.surfarray.array3d(app.screen.subsurface(gutter)))


def test_wrap_text_breaks_an_over_long_word_and_caps_the_lines(font):
    text = "Species 1 birth: " + "1" * 200 + " exceeds the 8 neighbours"
    lines = panel_module._wrap_text(text, font, 120)
    assert all(font.size(line)[0] <= 120 for line in lines)
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
    assert lines[0] == "Species 1 birth:"

    capped = panel_module._wrap_text(text, font, 120, max_lines=3)
    assert capped[:2] == lines[:2] and len(capped) == 3
    assert capped[2].endswith("…") and font.size(capped[2])[0] <= 120
    assert panel_module._wrap_text("short text", font, 120, max_lines=3) == ["short text"]
