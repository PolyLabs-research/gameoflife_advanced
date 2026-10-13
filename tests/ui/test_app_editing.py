"""Grid drawing and editing through the real App: clicks map to the drawn cell on every shape
(F26), edits go through the session (F80) and show at once (F81, F82), any species can be
placed (F83), a focused field does not eat the click (F72), and the help overlay is modal
(F74)."""

import numpy as np
import pygame
import pytest

from latticelife.session.alerts import AlertKind
from latticelife.ui import app as app_module
from latticelife.ui import render

SIZES = {"square": (20, 16), "hexagon": (20, 16), "triangle": (20, 16)}


def _paused_empty(make_app, ui, **sections):
    app = make_app(**sections)
    app.session.paused = True
    app.session.grid.occupancy[:] = 0
    app.session.grid.lifespans[:] = 0
    ui(app).frame()
    return app


def _centre(app, index):
    layout = app._grid_layout()
    x, y = layout.cell_center(*divmod(index, app.session.grid.width))
    return int(round(x)), int(round(y))


@pytest.mark.parametrize("shape", ["square", "hexagon", "triangle"])
def test_a_click_toggles_exactly_the_cell_drawn_under_it(make_app, ui, shape):
    """F26: hexagon and triangle clicks used square maths and hit the wrong cell or none."""
    width, height = SIZES[shape]
    app = _paused_empty(make_app, ui, grid={"shape": shape, "width": width, "height": height})
    u = ui(app)
    n = width * height
    for index in [0, width - 1, n // 2 + 3, n - width, n - 1, 5 * width + 7]:
        u.click(_centre(app, index))
        assert np.flatnonzero(app.session.grid.occupancy).tolist() == [index]
        # ...and the cell is drawn alive where it was clicked.
        assert tuple(app.screen.get_at(_centre(app, index)))[:3] == render.ALIVE_COLORS[0]
        u.click(_centre(app, index))
        assert not app.session.grid.occupancy.any()


@pytest.mark.parametrize("shape", ["hexagon", "triangle"])
def test_a_click_on_any_pixel_of_a_cell_s_colour_edits_that_cell(make_app, ui, shape):
    """pygame drew the polygons up to a pixel up-left of the geometry clicks were mapped with:
    of a lone live cell's red pixels, 4-5 picked its left neighbour, so a click on the red
    cell's edge placed a second cell instead of clearing it."""
    app = _paused_empty(make_app, ui, grid={"shape": shape, "width": 50, "height": 50})
    u = ui(app)
    target = 25 * 50 + 25
    app.session.set_cell(target, 1)
    u.frame()

    grid_rect = app.layout.grid_rect()
    pixels = pygame.surfarray.array3d(app.screen.subsurface(grid_rect))
    xs, ys = np.nonzero(np.all(pixels == render.ALIVE_COLORS[0], axis=-1))
    red = [(grid_rect.x + int(x), grid_rect.y + int(y)) for x, y in zip(xs, ys, strict=True)]
    assert len(red) > 30
    layout = app._grid_layout()
    assert {layout.cell_at(x, y) for x, y in red} == {target}

    # A click on its red pixel farthest from the centre, on its edge, clears exactly it.
    cx, cy = layout.cell_center(25, 25)
    u.click(max(red, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2))
    assert not app.session.grid.occupancy.any()


@pytest.mark.parametrize("shape", ["hexagon", "triangle"])
def test_a_paused_grid_stays_drawn(make_app, ui, shape):
    """F25: from the second paused frame on, a hex/triangle grid was pure background."""
    app = make_app(grid={"shape": shape})
    app.session.paused = True
    u = ui(app)
    grid_rect = app.layout.grid_rect()
    shots = []
    for _ in range(3):
        u.frame()
        shots.append(pygame.surfarray.array3d(app.screen.subsurface(grid_rect)).copy())
    assert np.array_equal(shots[0], shots[2])
    dead = int(np.flatnonzero(app.session.grid.occupancy == 0)[0])
    alive = int(np.flatnonzero(app.session.grid.occupancy == 1)[0])
    assert tuple(app.screen.get_at(_centre(app, dead)))[:3] == render.DEAD_CELL_COLOR
    assert tuple(app.screen.get_at(_centre(app, alive)))[:3] == render.ALIVE_COLORS[0]


@pytest.mark.parametrize("shape", ["square", "hexagon", "triangle"])
@pytest.mark.parametrize("size", [(6, 4), (50, 50), (400, 400)])
def test_the_grid_never_paints_outside_its_area(make_app, ui, shape, size):
    """F28/F29: hex/triangle cells were drawn up-left over the panel and chart, and tall hex
    grids ran off the bottom of the window."""
    width, height = size
    app = make_app(grid={"shape": shape, "width": width, "height": height})
    app.session.grid.occupancy[:] = 1
    app.screen.fill((1, 2, 3))
    app._draw_grid()
    pixels = pygame.surfarray.array3d(app.screen).copy()
    grid_rect = app.layout.grid_rect()
    inside = pixels[grid_rect.left : grid_rect.right, grid_rect.top : grid_rect.bottom].copy()
    pixels[grid_rect.left : grid_rect.right, grid_rect.top : grid_rect.bottom] = (1, 2, 3)
    assert np.all(pixels == (1, 2, 3))
    assert np.any(np.all(np.abs(inside.astype(int) - render.ALIVE_COLORS[0]) <= 3, axis=-1))


def test_clicks_go_through_the_session_so_an_edit_is_no_sudden_death(make_app, ui):
    """F80: erasing the grid by hand used to raise a false SUDDEN_DEATH and re-pause for the
    whole alert window."""
    app = make_app(grid={"width": 20, "height": 20}, simulation={"seed": 1234})
    u = ui(app)
    u.frame(31)
    assert app.session.generation == 30 and app.session.alert is None
    u.key(pygame.K_SPACE, " ")
    grid = app.session.grid
    for index in np.flatnonzero(grid.occupancy).tolist():
        app._toggle_cell(_centre(app, index))
    for index in (5 * 20 + 5, 5 * 20 + 6, 5 * 20 + 7):  # a blinker
        app._toggle_cell(_centre(app, index))
    assert int(np.count_nonzero(grid.occupancy)) == 3

    u.key(pygame.K_SPACE, " ")
    u.frame(15)

    assert app.session.generation > 40
    assert app.session.paused is False
    edited = [r for r in app.session.history_records if r.generation > 30]
    assert all(r.alert is None or r.alert.kind is not AlertKind.SUDDEN_DEATH for r in edited)
    first = edited[0].lifeform_metrics[1]
    assert first.growth_rate == pytest.approx((first.alive - 3) / 3)


def test_a_cell_placed_while_running_is_drawn_before_the_rule_acts(make_app, ui):
    """F81: the click was applied and stepped in the same frame, so an isolated new cell died
    before it was ever drawn."""
    app = make_app(grid={"width": 20, "height": 20})
    u = ui(app)
    u.frame(3)
    grid = app.session.grid
    grid.occupancy[:] = 0
    grid.lifespans[:] = 0
    generation = app.session.generation
    index = 10 * 20 + 10

    u.click(_centre(app, index))

    assert app.session.generation == generation  # the click frame does not step
    assert tuple(app.screen.get_at(_centre(app, index)))[:3] == render.ALIVE_COLORS[0]
    frames_held = int(app_module.EDIT_HOLD_SECONDS * app.config.ui.fps)
    u.frame(frames_held - 1)
    assert app.session.generation == generation
    assert grid.occupancy[index] == 1
    u.frame(3)
    assert app.session.generation > generation


def test_population_card_shows_the_edited_grid_while_paused(make_app, ui):
    """F82: the Population card showed the pre-edit count until the next step."""
    app = make_app()
    u = ui(app)
    u.frame(5)
    u.key(pygame.K_SPACE, " ")
    grid = app.session.grid
    for index in np.flatnonzero(grid.occupancy)[:40].tolist():
        u.click(_centre(app, index))
    values = dict(app_module._stat_values(app.session))
    assert values["Population"] == str(int(np.count_nonzero(grid.occupancy)))


def test_number_keys_and_panel_colours_choose_the_species_a_click_places(make_app, ui):
    """F83: with several species, a click could only ever place species 1."""
    app = _paused_empty(make_app, ui, simulation={"n_lifeforms": 3})
    u = ui(app)
    grid = app.session.grid
    a, b = 3 * 50 + 3, 9 * 50 + 9

    u.key(pygame.K_2, "2")
    u.click(_centre(app, a))
    assert grid.occupancy[a] == 2
    assert "species 2" in app._footer_hint()

    u.key(pygame.K_KP3)
    u.click(_centre(app, a))  # another species' cell takes the chosen species
    assert grid.occupancy[a] == 3
    u.click(_centre(app, a))  # the chosen species' own cell is cleared
    assert grid.occupancy[a] == 0

    u.key(pygame.K_5, "5")  # there is no species 5
    assert app.selected_species == 3

    panel = app.settings_panel
    u.click(panel._species_label_rects[0].center)
    assert app.selected_species == 1 and panel.selected_species == 1
    u.click(_centre(app, b))
    assert grid.occupancy[b] == 1


def test_number_keys_choose_the_species_by_the_digit_typed_on_any_layout(make_app, ui):
    """F83 follow-up: on AZERTY the top-row digits are typed with Shift and the keys can report
    the keycodes of &, é, \", ... so only the keypad chose a species there."""
    app = _paused_empty(make_app, ui, simulation={"n_lifeforms": 10})
    u = ui(app)
    grid = app.session.grid
    a = 3 * 50 + 3

    u.key(pygame.K_QUOTEDBL, "3", pygame.KMOD_LSHIFT)  # AZERTY Shift + top-row 3
    assert app.selected_species == 3 and app.settings_panel.selected_species == 3
    u.click(_centre(app, a))
    assert grid.occupancy[a] == 3

    u.key(pygame.K_AMPERSAND, "1", pygame.KMOD_LSHIFT)  # AZERTY Shift + top-row 1
    assert app.selected_species == 1
    u.key(ord("à"), "0", pygame.KMOD_LSHIFT)  # AZERTY Shift + top-row 0: species 10
    assert app.selected_species == 10

    # No digit typed: the keycode decides (US Shift+2, AZERTY top-row 7 unshifted as SDL
    # reports it, the keypad with Num Lock off).
    u.key(pygame.K_2, "@", pygame.KMOD_LSHIFT)
    assert app.selected_species == 2
    u.key(pygame.K_7, "è")
    assert app.selected_species == 7
    u.key(pygame.K_KP5)
    assert app.selected_species == 5

    u.key(pygame.K_QUOTEDBL, "3", pygame.KMOD_LCTRL)  # Ctrl+3 is not a shortcut
    u.key(pygame.K_SLASH, "/")  # a key with no digit chooses nothing
    assert app.selected_species == 5


def test_a_focused_field_does_not_swallow_a_grid_click(make_app, ui):
    """F72: the first click on the grid only blurred the focused field."""
    app = _paused_empty(make_app, ui)
    u = ui(app)
    panel = app.settings_panel
    for field in (panel.rule_inputs[0][0], panel.width_input):
        u.click(field.rect.center)
        assert field.active
        index = int(np.flatnonzero(app.session.grid.occupancy == 0)[0])
        u.click(_centre(app, index))
        assert app.session.grid.occupancy[index] == 1
        assert not field.active


def test_the_help_overlay_is_modal(make_app, ui):
    """F74: clicks through the overlay toggled cells and pressed New, Wrap and Apply."""
    app = _paused_empty(make_app, ui)
    u = ui(app)
    session = app.session
    panel = app.settings_panel

    u.key(pygame.K_SLASH, "?", pygame.KMOD_SHIFT)
    assert app.hints_visible
    u.key(pygame.K_SPACE, " ")
    u.key(pygame.K_n, "n")
    u.wheel(-3, over=(150, 500))
    assert app.session is session and session.paused and panel.scroll_y == 0

    u.click(app.screen.get_rect().center)  # over the grid: closes, toggles nothing
    assert not app.hints_visible
    assert not session.grid.occupancy.any()

    for target in (app.new_button.rect.center, panel.wrap_button.rect.center):
        u.key(pygame.K_F1)
        assert app.hints_visible
        u.click(target)
        assert not app.hints_visible
    assert app.session is session
    assert panel.config.grid.wrap is True

    u.key(pygame.K_SLASH, "?", pygame.KMOD_SHIFT)
    u.key(pygame.K_ESCAPE, "\x1b")
    assert not app.hints_visible
