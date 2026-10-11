from dataclasses import replace

import pygame
import pytest

from latticelife.ui.app import App


def test_app_runs_100_frames_without_crashing(config):
    app = App(config)  # the real clock
    for _ in range(100):
        assert app.step_frame() is True
    assert app.session.generation > 0


def test_quit_event_stops_the_loop(config):
    app = App(config)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert app.step_frame() is False


def test_one_generation_per_frame_at_the_configured_speed(make_app, ui):
    app = make_app()
    u = ui(app)
    u.frame()  # the first frame draws generation 0
    assert app.session.generation == 0
    u.frame(5)
    assert app.session.generation == 5


def test_space_toggles_pause(make_app, ui):
    app = make_app()
    assert app.session.paused is False
    ui(app).key(pygame.K_SPACE, " ")
    assert app.session.paused is True


def test_r_resets_and_the_frame_shows_generation_zero(make_app, ui):
    app = make_app()
    u = ui(app)
    u.frame(6)
    assert app.session.generation == 5

    u.key(pygame.K_r, "r")
    assert app.session.generation == 0  # drawn before any step
    u.frame()
    assert app.session.generation == 1


def _restart(u, app, action: str) -> None:
    """Reset (R), New (N) or Apply (the panel's button), as a person does them."""
    if action == "reset":
        u.key(pygame.K_r, "r")
    elif action == "new":
        u.key(pygame.K_n, "n")
    else:
        u.click(app.settings_panel.apply_button.rect.center)


@pytest.mark.parametrize("action", ["reset", "new", "apply"])
def test_a_paused_run_stays_paused_on_generation_zero_after_reset_new_and_apply(
    make_app, ui, action
):
    """F48: Reset, New and Apply always unpaused, so a paused user could neither inspect nor
    single-step generation 0."""
    app = make_app()
    u = ui(app)
    u.frame(6)
    u.key(pygame.K_SPACE, " ")
    assert app.session.paused is True

    _restart(u, app, action)
    u.frame(3)

    assert app.session.paused is True
    assert app.session.generation == 0
    u.key(pygame.K_s, "s")
    assert app.session.generation == 1
    assert app.session.paused is True


@pytest.mark.parametrize("action", ["reset", "new", "apply"])
def test_reset_new_and_apply_lift_a_pause_the_sudden_death_alert_imposed(make_app, ui, action):
    """The auto-pause belongs to the run that crashed: like `SimulationSession.reset()`, New
    and Apply start the next run going, while a pause the user chose is kept (above)."""
    app = make_app()
    u = ui(app)
    u.frame()
    session = app.session
    for _ in range(session.config.alerts.sudden_death_window):
        session.history_alive.append(10**6)
    session.grid.occupancy[:] = 0
    u.frame()
    assert session.paused_by_alert is True

    _restart(u, app, action)
    assert app.session.paused is False
    assert app.session.generation == 0  # drawn before any step
    u.frame(2)
    assert app.session.generation == 2


def test_s_steps_once_only_when_paused(make_app, ui):
    app = make_app()
    app.session.paused = True
    u = ui(app)
    u.frame(3)
    assert app.session.generation == 0

    u.key(pygame.K_s, "s")

    assert app.session.generation == 1
    assert app.session.paused is True


def test_s_does_nothing_while_running(make_app, ui):
    app = make_app()
    u = ui(app)
    u.frame(3)
    gen_before = app.session.generation

    u.key(pygame.K_s, "s")

    # Not paused: S is a no-op, but the frame's own unpaused step still advances by one.
    assert app.session.generation == gen_before + 1


def test_question_mark_toggles_hints(make_app, ui):
    app = make_app()
    assert app.hints_visible is False
    ui(app).key(pygame.K_SLASH, "?", pygame.KMOD_SHIFT)
    assert app.hints_visible is True


@pytest.mark.parametrize("shape", ["square", "hexagon", "triangle"])
def test_each_shape_renders_without_crashing(config, shape):
    shaped_config = replace(config, grid=replace(config.grid, shape=shape, width=10, height=10))
    app = App(shaped_config)
    for _ in range(10):
        app.step_frame()
