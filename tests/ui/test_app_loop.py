"""Frame timing, speed control, keyboard handling and New, driven through the real App."""

import time

import numpy as np
import pygame
import pytest

from latticelife.core.geometries import max_degree
from latticelife.core.rules import format_count_list
from latticelife.session.alerts import Alert, AlertKind
from latticelife.ui import app as app_module
from latticelife.ui.app import UI_FPS, App


def test_input_is_handled_at_the_ui_rate_even_at_one_generation_per_second(config):
    """F46: the loop ticked at the simulation FPS, so at 1 fps every click and key took up to a
    second to register, even while paused."""
    from dataclasses import replace

    app = App(replace(config, ui=replace(config.ui, fps=1)))
    app.session.paused = True
    start = time.perf_counter()
    for _ in range(10):
        app.step_frame()
    assert time.perf_counter() - start < 2.0  # was 10 s

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, unicode=" ", mod=0))
    app.step_frame()
    assert app.session.paused is False


def test_the_loop_ticks_at_the_ui_rate_and_steps_at_the_simulation_rate(make_app, ui):
    app = make_app(ui={"fps": 10})
    app.clock.ms = 1000 / 60  # a 60 Hz display
    ui(app).frame(61)  # one second after the first frame
    assert set(app.clock.framerates) == {UI_FPS}
    assert 9 <= app.session.generation <= 11


def test_a_long_frame_does_not_replay_its_backlog(make_app, ui):
    app = make_app(ui={"fps": 60})
    app.clock.ms = 5000  # e.g. the window was dragged for five seconds
    u = ui(app)
    u.frame()
    u.frame()
    assert app.session.generation <= app_module.MAX_STEPS_PER_FRAME


def test_fps_slider_changes_speed_live_without_restarting(make_app, ui):
    """F45: the slider only took effect through Apply, which threw the running session away."""
    app = make_app()
    u = ui(app)
    u.frame(11)
    session, generation = app.session, app.session.generation
    slider = app.settings_panel.fps_slider

    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=slider.rect.midleft, button=1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=slider.rect.midleft, button=1))
    u.frame()

    assert app.config.ui.fps == 1
    assert app.settings_panel.config.ui.fps == 1
    assert app.session is session and app.session.generation >= generation
    generation = app.session.generation
    u.frame(28)  # with the slider's own frame, 29 frames at 30 a second: under one second
    assert app.session.generation == generation
    u.frame(2)
    assert app.session.generation == generation + 1


def test_new_gives_every_species_a_fresh_random_rule_and_the_panel_shows_it(make_app, ui):
    """F27: N dropped the applied species and rules, falling back to the config's Conway."""
    app = make_app(grid={"shape": "triangle"})
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    panel.species_count_input.set_value(3)
    panel._resize_species(3)
    for (birth, survive), (b, s) in zip(
        panel.rule_inputs, [("2", "3,4"), ("3,5", "2"), ("2,4", "1,3")], strict=True
    ):
        birth.text, survive.text = b, s
    u.click(panel.apply_button.rect.center)
    applied = list(app.session.lifeforms)
    assert len(applied) == 3
    grid_before = app.session.grid.occupancy.copy()

    u.key(pygame.K_n, "n")

    lifeforms = app.session.lifeforms
    assert [lf.id for lf in lifeforms] == [1, 2, 3]
    assert lifeforms != applied
    degree = max_degree("triangle", app.config.grid.triangle_mode)
    for lf in lifeforms:
        assert max(lf.rule.birth | lf.rule.survive, default=0) <= degree
    assert [(b.text, s.text) for b, s in panel.rule_inputs] == [
        (format_count_list(lf.rule.birth), format_count_list(lf.rule.survive)) for lf in lifeforms
    ]
    assert not np.array_equal(app.session.grid.occupancy, grid_before)
    assert app.session.generation == 0

    # Apply now re-applies exactly the rules New made (counts of 10-12 included).
    u.click(panel.apply_button.rect.center)
    assert app.session.lifeforms == lifeforms

    session = app.session
    u.click(app.new_button.rect.center)
    assert app.session is not session and len(app.session.lifeforms) == 3


def test_reset_keeps_the_rules_and_reseeds_the_grid(make_app, ui):
    app = make_app()
    u = ui(app)
    u.key(pygame.K_n, "n")
    lifeforms = app.session.lifeforms
    grid_before = app.session.grid.occupancy.copy()
    u.key(pygame.K_r, "r")
    assert app.session.lifeforms == lifeforms
    assert not np.array_equal(app.session.grid.occupancy, grid_before)


def test_a_nonzero_seed_makes_new_and_r_reproducible(make_app, ui):
    """F55: R drew from an unseeded generator; New must follow the same seeded stream."""
    runs = []
    for _ in range(2):
        app = make_app(simulation={"seed": 42})
        u = ui(app)
        u.click(app.settings_panel.randomize_buttons[0].rect.center)
        typed = [(b.text, s.text) for b, s in app.settings_panel.rule_inputs]
        u.key(pygame.K_n, "n")
        first = (app.session.lifeforms, app.session.grid.occupancy.copy())
        u.key(pygame.K_n, "n")
        second = (app.session.lifeforms, app.session.grid.occupancy.copy())
        runs.append((typed, first, second))
        pygame.quit()

    (typed_a, first_a, second_a), (typed_b, first_b, second_b) = runs
    assert typed_a == typed_b
    assert first_a[0] == first_b[0] and np.array_equal(first_a[1], first_b[1])
    assert second_a[0] == second_b[0] and np.array_equal(second_a[1], second_b[1])
    assert not np.array_equal(first_a[1], second_a[1])  # each New is a new grid


@pytest.mark.parametrize(
    ("key", "unicode"),
    [(223, "?"), (pygame.K_COMMA, "?"), (pygame.K_QUOTE, "?"), (pygame.K_PLUS, "?")],
)
def test_question_mark_opens_help_on_any_keyboard_layout(make_app, ui, key, unicode):
    """F73: only the US Shift+/ key opened help; German, French, Spanish and Swedish layouts
    put '?' elsewhere."""
    app = make_app()
    ui(app).key(key, unicode, pygame.KMOD_LSHIFT)
    assert app.hints_visible


def test_f1_opens_help_too(make_app, ui):
    app = make_app()
    ui(app).key(pygame.K_F1)
    assert app.hints_visible


@pytest.mark.parametrize("mod", [pygame.KMOD_LCTRL, pygame.KMOD_LALT, pygame.KMOD_LGUI])
def test_modified_letters_are_not_shortcuts(make_app, ui, mod):
    app = make_app()
    u = ui(app)
    u.frame(3)
    session = app.session
    u.key(pygame.K_n, "n", mod)
    u.key(pygame.K_r, "r", mod)
    u.key(pygame.K_SPACE, " ", mod)
    assert app.session is session and session.generation > 3 and not session.paused


def test_typed_text_arrives_as_textinput_and_is_never_doubled(make_app, ui):
    """F75: fields ignored TEXTINPUT, so input-method and on-screen-keyboard text was lost."""
    app = make_app()
    u = ui(app)
    u.frame()
    panel = app.settings_panel
    birth = panel.rule_inputs[0][0]
    u.click(birth.rect.center)
    u.type("5")
    assert birth.text == "35"
    u.key(pygame.K_6, "6")  # the KEYDOWN half of a key press types nothing by itself
    assert birth.text == "35"
    u.type(",1２")  # the full-width digit is not a rule character
    assert birth.text == "35,1"

    width = panel.width_input
    u.click(width.rect.center)
    u.clear_field()
    u.type("８")  # full-width: int() would read it, so the field refuses it
    assert width.text == ""
    u.type("80")
    u.key(pygame.K_KP_ENTER)
    assert (width.value, width.active) == (80, False)


def test_typing_keeps_the_shortcuts_quiet(make_app, ui):
    app = make_app()
    u = ui(app)
    u.frame(2)
    session = app.session
    u.click(app.settings_panel.rule_inputs[0][1].rect.center)
    u.key(pygame.K_n, "n")
    u.key(pygame.K_SPACE, " ")
    u.key(pygame.K_SLASH, "?", pygame.KMOD_SHIFT)
    assert app.session is session and not session.paused and not app.hints_visible


def test_key_repeat_is_on_only_while_a_field_has_focus(make_app, ui):
    """F76: repeat was never enabled, so holding Backspace deleted one character."""
    app = make_app()
    u = ui(app)
    u.frame()
    assert pygame.key.get_repeat() == (0, 0)
    u.click(app.settings_panel.height_input.rect.center)
    assert pygame.key.get_repeat() == app_module.TEXT_KEY_REPEAT
    u.key(pygame.K_RETURN, "\r")
    assert pygame.key.get_repeat() == (0, 0)


def test_the_alert_card_stays_inside_its_card_and_the_full_message_is_shown(make_app, ui):
    """F47: the full alert message was drawn in the card, spilling past the panel."""
    app = make_app()
    app.session.paused = True
    message = "Sudden death: gen 1234 drop 12345->1"
    app.session.alert = Alert(AlertKind.SUDDEN_DEATH, message)
    ui(app).frame()
    assert dict(app_module._stat_values(app.session))["Alert"] == "Sudden death"

    panel_right = app.layout.panel_width
    strip = pygame.surfarray.array3d(app.screen)[panel_right : panel_right + 16, 64:196]
    assert np.all(strip == app_module.BACKGROUND_COLOR)

    font = app.fonts.get("lg")
    assert font.size(app_module._fit_text(font, message, 100))[0] <= 100
    chart = app.layout.chart_rect()
    banner = pygame.surfarray.array3d(app.screen.subsurface(chart.x, chart.y, chart.w, 30))
    red = np.all(np.abs(banner.astype(int) - app_module.ACCENT_DANGER) <= 40, axis=-1)
    assert red.sum() > 50
