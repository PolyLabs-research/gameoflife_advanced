from dataclasses import replace

import pygame
import pytest

from latticelife.ui.app import App


def test_app_runs_100_frames_without_crashing(config):
    app = App(config)
    for _ in range(100):
        assert app.step_frame() is True
    assert app.session.generation > 0


def test_quit_event_stops_the_loop(config):
    app = App(config)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert app.step_frame() is False


def test_space_toggles_pause(config):
    app = App(config)
    assert app.session.paused is False
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    app.step_frame()
    assert app.session.paused is True


def test_r_resets_then_the_same_frame_steps_once(config):
    app = App(config)
    for _ in range(5):
        app.step_frame()
    assert app.session.generation > 1

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    app.step_frame()  # reset() sets generation=0, then the frame's own step() makes it 1
    assert app.session.generation == 1


def test_s_steps_once_only_when_paused(config):
    app = App(config)
    app.session.paused = True
    gen_before = app.session.generation

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))
    app.step_frame()

    assert app.session.generation == gen_before + 1
    assert app.session.paused is True


def test_s_does_nothing_while_running(config):
    app = App(config)
    assert app.session.paused is False
    gen_before = app.session.generation

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))
    app.step_frame()

    # Not paused: S is a no-op, but the frame's own unpaused step still advances by one.
    assert app.session.generation == gen_before + 1


def test_question_mark_toggles_hints(config):
    app = App(config)
    assert app.hints_visible is False
    pygame.event.post(
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SLASH, mod=pygame.KMOD_SHIFT)
    )
    app.step_frame()
    assert app.hints_visible is True


def test_v_toggle_does_not_crash_regardless_of_torus_extra_availability(config):
    app = App(config)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v))
    app.step_frame()
    assert app.view_mode in ("2D", "3D")
    if app.view_mode == "2D":
        assert app.torus_status is not None


def test_click_toggles_exactly_one_cell(config):
    app = App(config)
    app.session.paused = True
    app.session.grid.occupancy[:] = 0

    grid_rect = app.layout.grid_rect()
    pos = (grid_rect.x + 5, grid_rect.y + 5)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    app.step_frame()

    assert int(app.session.grid.occupancy.sum()) == 1


@pytest.mark.parametrize("shape", ["square", "hexagon", "triangle"])
def test_each_shape_renders_without_crashing(config, shape):
    shaped_config = replace(config, grid=replace(config.grid, shape=shape, width=10, height=10))
    app = App(shaped_config)
    for _ in range(10):
        app.step_frame()
