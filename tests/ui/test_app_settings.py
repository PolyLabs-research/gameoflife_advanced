"""The settings panel's expanded controls, driven through the real App with real events: the
live controls do what their keys do (V, C, '?'), plus alert pausing, torus rotation and camera
reset; the staged grid settings (neighbourhood, density, seed) reach the session on Apply, and
Apply says when something is pending; preset rules and examples load; and a drag on the grid
paints."""

import sys
import types

import numpy as np
import pygame
import pytest

from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.ui.presets import EXAMPLES
from latticelife.ui.torus.camera import DEFAULT_DISTANCE, OrbitalCamera

VIEW_MODULE = "latticelife.ui.torus.view"


class FakeView:
    """Stands in for `TorusView` (no GL): a real camera, draws that fill the area."""

    def __init__(self, background, ctx=None):
        self.camera = OrbitalCamera()

    def draw(self, surface, area, grid, palette):
        surface.fill((1, 2, 3), area)

    def handle_event(self, event, area):
        return self.camera.handle_event(event, area)

    def release(self):
        pass


class FakeContext:
    def release(self):
        pass


@pytest.fixture
def fake_3d(monkeypatch):
    module = types.SimpleNamespace(TorusView=FakeView, create_offscreen_context=FakeContext)
    monkeypatch.setitem(sys.modules, VIEW_MODULE, module)


def _show(app, u, widget):
    """Scroll the panel until `widget` is wholly in view."""
    panel = app.settings_panel
    for _ in range(40):
        u.frame()
        rect = widget.rect if not hasattr(widget, "header_rect") else widget.header_rect
        if rect is not None and rect.width and panel._content_clip.contains(rect):
            return rect
        u.wheel(-1, over=(150, 400))
    raise AssertionError(f"{widget} never came into view")


def _choose(app, u, dropdown, label):
    u.click(_show(app, u, dropdown).center)
    u.frame()  # the open list is laid out when drawn
    options = dict(dropdown._option_rects)
    for _ in range(20):
        if label in options:
            break
        u.wheel(-1)  # scroll the list itself
        u.frame()
        options = dict(dropdown._option_rects)
    u.click(options[label].center)


def _centre(app, index):
    layout = app._grid_layout()
    x, y = layout.cell_center(*divmod(index, app.session.grid.width))
    return int(round(x)), int(round(y))


# --- live controls ---------------------------------------------------------------------------


def test_clear_empties_the_grid_and_pauses_by_button_and_by_key(make_app, ui):
    app = make_app(grid={"width": 30, "height": 30})
    u = ui(app)
    u.frame(5)
    button = _show(app, u, app.settings_panel.clear_button)
    generation = app.session.generation
    u.click(button.center)
    assert not app.session.grid.occupancy.any() and app.session.paused
    assert app.session.generation == generation  # an edit, not a new run

    u.key(pygame.K_SPACE, " ")
    app.session.set_cell(5, 1)
    u.key(pygame.K_c, "c")
    assert not app.session.grid.occupancy.any() and app.session.paused


def test_the_help_button_opens_the_shortcuts(make_app, ui):
    app = make_app()
    u = ui(app)
    u.click(_show(app, u, app.settings_panel.help_button).center)
    assert app.hints_visible
    u.click((600, 400))
    assert not app.hints_visible


def test_the_alert_control_switches_alert_pausing_and_survives_apply(make_app, ui):
    app = make_app()
    u = ui(app)
    panel = app.settings_panel
    assert app.session.auto_pause and panel.alert_button.text == "Auto-pause: On"
    u.click(_show(app, u, panel.alert_button).center)
    u.frame()
    assert not app.session.auto_pause and panel.alert_button.text == "Auto-pause: Off"
    u.click(panel.apply_button.rect.center)
    assert not app.session.auto_pause  # the new session keeps the choice


def test_the_view_button_does_what_v_does(make_app, ui, monkeypatch):
    monkeypatch.setitem(sys.modules, VIEW_MODULE, None)  # no [torus] extra
    app = make_app()
    u = ui(app)
    u.click(_show(app, u, app.settings_panel.view_button).center)
    assert app.view_mode == "2D" and "unavailable" in app.torus_status


def test_view_rotation_and_camera_controls_drive_the_torus(make_app, ui, fake_3d):
    app = make_app()
    u = ui(app)
    panel = app.settings_panel
    u.frame()
    assert panel.view_button.text == "3D view (V)" and not panel.camera_button.enabled
    u.click(_show(app, u, panel.camera_button).center)  # disabled in 2D: nothing happens

    u.click(_show(app, u, panel.view_button).center)
    assert app.view_mode == "3D"
    u.frame()
    assert panel.view_button.text == "2D view (V)" and panel.camera_button.enabled
    camera = app.torus_view.camera
    assert camera.auto_rotate and panel.rotate_button.text == "Rotate: On"

    u.click(_show(app, u, panel.rotate_button).center)
    u.frame()
    assert not camera.auto_rotate and panel.rotate_button.text == "Rotate: Off"

    camera.handle_scroll(3)
    camera.azimuth, camera.elevation = 1.0, -0.5
    u.click(_show(app, u, panel.camera_button).center)
    assert camera.distance == DEFAULT_DISTANCE
    assert (camera.azimuth, camera.elevation) == (0.0, 0.3)

    u.key(pygame.K_v, "v")  # back to 2D: rotation stays off for the next 3D view
    assert app.view_mode == "2D" and not app.auto_rotate
    u.click(_show(app, u, panel.view_button).center)
    assert app.view_mode == "3D" and not app.torus_view.camera.auto_rotate


def test_a_drag_on_the_torus_that_stops_rotation_shows_in_the_panel(make_app, ui, fake_3d):
    app = make_app()
    u = ui(app)
    u.key(pygame.K_v, "v")
    area = app.layout.grid_rect()
    for event_type, extra in (
        (pygame.MOUSEBUTTONDOWN, {"button": 1}),
        (pygame.MOUSEMOTION, {"rel": (5, 0), "buttons": (1, 0, 0)}),
        (pygame.MOUSEBUTTONUP, {"button": 1}),
    ):
        pygame.event.post(pygame.event.Event(event_type, pos=area.center, **extra))
    u.frame(2)
    assert app.settings_panel.rotate_button.text == "Rotate: Off"


# --- staged grid settings --------------------------------------------------------------------


def test_apply_reads_density_and_seed_and_a_seed_repeats_the_grid(make_app, ui):
    app = make_app(grid={"width": 40, "height": 40})
    u = ui(app)
    panel = app.settings_panel
    panel.density_slider.value = 20
    seed = panel.seed_input
    u.click(_show(app, u, seed).center)
    u.clear_field()
    u.type("1234")
    u.key(pygame.K_RETURN, "\r")
    assert panel.apply_button.text == "Apply changes"
    u.click(panel.apply_button.rect.center)
    simulation = app.session.config.simulation
    assert (simulation.initial_alive_percentage, simulation.seed) == (0.2, 1234)
    assert int(np.count_nonzero(app.session.grid.occupancy)) == 40 * 40 * 20 // 100
    first = app.session.grid.occupancy.copy()

    u.click(panel.apply_button.rect.center)  # same settings again: the same grid
    assert np.array_equal(app.session.grid.occupancy, first)
    assert panel.apply_button.text == "Apply"


def test_a_loaded_density_is_kept_exactly_until_the_slider_moves(make_app, ui):
    app = make_app(simulation={"initial_alive_percentage": 0.333})
    u = ui(app)
    u.frame()
    assert app.settings_panel.apply_button.text == "Apply"
    u.click(app.settings_panel.apply_button.rect.center)
    assert app.session.config.simulation.initial_alive_percentage == 0.333


def test_the_neighbourhood_row_shows_for_triangles_and_limits_the_rules(make_app, ui):
    app = make_app()
    u = ui(app)
    panel = app.settings_panel
    u.frame()
    assert panel.triangle_dropdown.rect is None  # not a triangle grid

    _choose(app, u, panel.shape_dropdown, "triangle")
    u.frame()
    assert panel.triangle_dropdown.rect is not None
    _choose(app, u, panel.triangle_dropdown, "edge")
    birth = panel.rule_inputs[0][0]
    birth.text = "4"  # an edge triangle has 3 neighbours
    u.click(panel.apply_button.rect.center)
    assert panel.error is not None and "3" in panel.error
    birth.text = "2"
    panel.width_input.set_value(40)
    panel.height_input.set_value(40)
    u.click(panel.apply_button.rect.center)
    assert panel.error is None
    grid = app.session.config.grid
    assert (grid.shape, grid.triangle_mode) == ("triangle", "edge")


def test_apply_reads_apply_changes_only_while_something_is_staged(make_app, ui):
    app = make_app()
    u = ui(app)
    panel = app.settings_panel
    u.frame()
    assert panel.apply_button.text == "Apply" and not panel.apply_button.primary
    u.click(_show(app, u, panel.width_input).center)
    u.clear_field()
    u.type("60")
    assert panel.apply_button.text == "Apply changes" and panel.apply_button.primary
    u.key(pygame.K_ESCAPE, "\x1b")  # put back as it was
    assert panel.apply_button.text == "Apply"
    panel.rule_inputs[0][0].text = "3,6"
    u.frame()
    assert panel.apply_button.text == "Apply changes"
    panel.rule_inputs[0][0].text = "3"
    u.frame()
    assert panel.apply_button.text == "Apply"


# --- presets and examples --------------------------------------------------------------------


def test_a_preset_fills_the_chosen_species_and_waits_for_apply(make_app, ui):
    app = make_app(simulation={"n_lifeforms": 2})
    u = ui(app)
    panel = app.settings_panel
    u.key(pygame.K_2, "2")
    _choose(app, u, panel.preset_dropdown, "HighLife · B36/S23")
    assert (panel.rule_inputs[1][0].text, panel.rule_inputs[1][1].text) == ("3,6", "2,3")
    assert (panel.rule_inputs[0][0].text, panel.rule_inputs[0][1].text) == ("3", "2,3")
    assert app.session.lifeforms[1].rule == Rule(birth={3}, survive={2, 3})  # staged only
    u.click(panel.apply_button.rect.center)
    assert app.session.lifeforms[1].rule == Rule(birth={3, 6}, survive={2, 3})


def test_presets_follow_the_staged_shape(make_app, ui):
    app = make_app()
    u = ui(app)
    panel = app.settings_panel
    u.frame()
    assert "HighLife · B36/S23" in panel.preset_dropdown.options
    _choose(app, u, panel.shape_dropdown, "hexagon")
    u.frame()
    assert "c/3 ship · B24/S245" in panel.preset_dropdown.options
    assert "HighLife · B36/S23" not in panel.preset_dropdown.options


@pytest.mark.parametrize("name", ["Glider", "c/3 ship"])
def test_an_example_loads_paused_on_a_clear_grid_and_moves_on_play(make_app, ui, name):
    example = next(e for e in EXAMPLES if e.name == name)
    app = make_app(simulation={"n_lifeforms": 3})
    u = ui(app)
    panel = app.settings_panel
    _choose(app, u, panel.example_dropdown, f"{example.name} · {example.shape}")
    session = app.session
    assert session.paused and session.generation == 0
    assert session.config.grid.shape == example.shape and session.config.grid.wrap
    assert session.lifeforms == [Lifeform(1, example.as_rule())]
    start = np.flatnonzero(session.grid.occupancy)
    assert len(start) == len(example.cells)
    u.frame()
    assert panel.apply_button.text == "Apply" and len(panel.rule_inputs) == 1
    assert panel.shape_dropdown.selected == example.shape

    u.key(pygame.K_SPACE, " ")
    period = {"Glider": 4, "c/3 ship": 3}[name]
    while session.generation < 4 * period:
        u.frame()
    moved = np.flatnonzero(session.grid.occupancy)
    assert len(moved) == len(start) and not np.array_equal(moved, start)


# --- painting --------------------------------------------------------------------------------


def _paused_empty(make_app, ui):
    app = make_app(grid={"width": 30, "height": 30})
    app.session.paused = True
    app.session.clear()
    ui(app).frame()
    return app


def _drag(app, u, indices, buttons=(1, 0, 0)):
    first, *rest = indices
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=_centre(app, first), button=1))
    for index in rest:
        pygame.event.post(
            pygame.event.Event(
                pygame.MOUSEMOTION, pos=_centre(app, index), rel=(0, 0), buttons=buttons
            )
        )
    u.frame()


def test_a_drag_paints_every_cell_it_crosses_and_one_from_a_live_cell_erases(make_app, ui):
    app = _paused_empty(make_app, ui)
    u = ui(app)
    width = app.session.grid.width
    row = 10 * width
    _drag(app, u, [row + 2, row + 12])  # one jump of ten cells: no gaps
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEBUTTONUP, pos=_centre(app, row + 12), button=1)
    )
    u.frame()
    assert np.flatnonzero(app.session.grid.occupancy).tolist() == list(range(row + 2, row + 13))

    _drag(app, u, [row + 5, row + 8])  # starts on a live cell: erases
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=_centre(app, row + 8), button=1))
    u.frame()
    alive = np.flatnonzero(app.session.grid.occupancy).tolist()
    assert alive == [*range(row + 2, row + 5), *range(row + 9, row + 13)]


def test_a_drag_ends_when_the_button_is_found_released(make_app, ui):
    app = _paused_empty(make_app, ui)
    u = ui(app)
    width = app.session.grid.width
    _drag(app, u, [5 * width + 5])
    u.move(_centre(app, 5 * width + 15), buttons=(0, 0, 0))  # released outside the window
    assert np.flatnonzero(app.session.grid.occupancy).tolist() == [5 * width + 5]
    assert app._stroke is None


def test_a_drag_holds_a_running_simulation_until_released(make_app, ui):
    app = make_app(grid={"width": 30, "height": 30})
    u = ui(app)
    u.frame(3)
    generation = app.session.generation
    _drag(app, u, [100, 101])
    u.frame(5)
    assert app.session.generation == generation
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=_centre(app, 101), button=1))
    u.frame(30)
    assert app.session.generation > generation
