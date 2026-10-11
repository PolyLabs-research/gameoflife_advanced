"""The app's 3D view (V key, §6.14) — F30-F34, F39, F40.

Most tests swap `ui.torus.view` for a stand-in (no GL needed), to pin down the app side: when
3D starts or refuses to, where the torus is drawn, what reaches the camera, and that the view is
released. The last tests run the real view on an offscreen OpenGL context, and are skipped
when the [torus] extra or OpenGL 3.3 is missing."""

import os
import subprocess
import sys
import types
from dataclasses import replace

import numpy as np
import pygame
import pytest

from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.session.alerts import Alert, AlertKind
from latticelife.ui.app import ACCENT_DANGER, STATUS_SECONDS
from latticelife.ui.torus.camera import OrbitalCamera

VIEW_MODULE = "latticelife.ui.torus.view"
TORUS_FILL = (1, 2, 3)  # what the stand-in draws over the grid area


class FakeView:
    """Stands in for `TorusView`: records draws, routes input to a real camera, and fills the
    area it is given so tests can see where the torus went."""

    instances: list["FakeView"] = []
    fail_init: Exception | None = None
    fail_draw: Exception | None = None

    def __init__(self, background):
        if FakeView.fail_init is not None:
            raise FakeView.fail_init
        self.background = background
        self.camera = OrbitalCamera()
        self.draws: list[tuple[pygame.Rect, object, np.ndarray]] = []
        self.released = 0
        FakeView.instances.append(self)

    @property
    def dragging(self):
        return self.camera.dragging

    def draw(self, surface, area, grid, palette):
        if FakeView.fail_draw is not None:
            raise FakeView.fail_draw
        self.draws.append((pygame.Rect(area), grid, palette.copy()))
        surface.fill(TORUS_FILL, area)

    def handle_event(self, event, area):
        return self.camera.handle_event(event, area)

    def release(self):
        self.released += 1


@pytest.fixture
def fake_view(monkeypatch):
    FakeView.instances = []
    FakeView.fail_init = FakeView.fail_draw = None
    monkeypatch.setitem(sys.modules, VIEW_MODULE, types.SimpleNamespace(TorusView=FakeView))
    return FakeView


def _enter_3d(app, ui):
    ui(app).key(pygame.K_v, "v")
    assert app.view_mode == "3D", app.torus_status
    return app.torus_view


def test_the_app_never_imports_the_torus_package_at_load():
    """§6.14: the [torus] extra is imported lazily, inside the V handler only."""
    code = (
        "import sys, latticelife.ui.app; "
        "print(sorted(m for m in sys.modules if m == 'moderngl' or 'torus' in m))"
    )
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert result.stdout.strip() == "[]"


# --- V without 3D ----------------------------------------------------------------------------


def test_v_without_the_extra_says_how_to_install_it(make_app, ui, monkeypatch):
    monkeypatch.setitem(sys.modules, VIEW_MODULE, None)  # import fails as if not installed
    app = make_app()
    ui(app).key(pygame.K_v, "v")
    assert app.view_mode == "2D" and app.torus_view is None
    assert "[torus] extra" in app.torus_status


def test_v_without_opengl_says_why_and_stays_2d(make_app, ui, fake_view):
    """F40: the old smoke test passed whether or not 3D could ever start."""
    fake_view.fail_init = RuntimeError("no OpenGL 3.3 context (egl: no device)\ndetails")
    app = make_app()
    ui(app).key(pygame.K_v, "v")
    assert app.view_mode == "2D" and app.torus_view is None
    assert app.torus_status == "3D view unavailable: no OpenGL 3.3 context (egl: no device)"


def test_the_status_sits_between_the_chart_and_the_grid_and_expires(make_app, ui, fake_view):
    """F39: the message was drawn over the bottom grid rows and never went away."""
    fake_view.fail_init = RuntimeError("no OpenGL")
    app = make_app()
    u = ui(app)
    u.key(pygame.K_v, "v")
    _, rect = app._torus_status_text()
    chart = app.layout.chart_rect()
    assert chart.bottom <= rect.top and rect.bottom <= app.layout.grid_rect().top
    assert chart.left <= rect.left and rect.right <= chart.right

    seconds_per_frame = app.clock.ms / 1000
    u.frame(int(STATUS_SECONDS / seconds_per_frame) - 2)
    assert app.torus_status is not None
    u.frame(3)
    assert app.torus_status is None


def test_the_status_and_an_alert_message_share_the_strip_side_by_side(make_app, ui, fake_view):
    """The alert message moved off the chart into the status strip; with a 3D status up too,
    the alert takes the left half and the status the right, neither over the other."""
    fake_view.fail_init = RuntimeError("no OpenGL 3.3 context (egl: no device) " + "x" * 200)
    app = make_app()
    app.session.paused = True
    u = ui(app)
    u.key(pygame.K_v, "v")
    app.session.alert = Alert(AlertKind.SUDDEN_DEATH, "Sudden death: gen 1234 drop 12345->1")
    u.frame()

    strip = app._status_strip()
    _, rect = app._torus_status_text()
    assert strip.contains(rect) and rect.left >= strip.centerx
    shown = pygame.surfarray.array3d(app.screen.subsurface(strip))
    red = np.all(np.abs(shown.astype(int) - ACCENT_DANGER) <= 40, axis=-1)
    left_half = red[: rect.left - strip.left - 1]
    assert left_half.sum() > 50  # the alert message, left of the status
    assert red[rect.left - strip.left :].sum() > 50  # the status itself


def test_reset_new_and_apply_clear_the_status(make_app, ui, fake_view):
    fake_view.fail_init = RuntimeError("no OpenGL")
    app = make_app()
    u = ui(app)
    for action in (pygame.K_r, pygame.K_n):
        u.key(pygame.K_v, "v")
        assert app.torus_status is not None
        u.key(action, pygame.key.name(action))
        assert app.torus_status is None
    u.key(pygame.K_v, "v")
    u.click(app.settings_panel.apply_button.rect.center)
    assert app.torus_status is None


# --- in 3D -------------------------------------------------------------------------------------


def test_v_draws_the_torus_in_the_grid_area_and_v_again_releases_it(make_app, ui, fake_view):
    """F30 (app side) and F32: leaving 3D dropped the renderer without releasing it."""
    app = make_app()
    u = ui(app)
    view = _enter_3d(app, ui)
    assert app.torus_status is None

    u.frame()
    area, grid, palette = view.draws[-1]
    assert area == app.layout.grid_rect()
    assert grid is app.session.grid
    np.testing.assert_array_equal(palette, app.palette)
    assert app.screen.get_at(area.center)[:3] == TORUS_FILL
    panel_point = (app.layout.panel_width // 2, app.layout.window_height // 2)
    assert app.screen.get_at(panel_point)[:3] != TORUS_FILL  # the panel is still drawn
    assert "Drag to rotate" in app._footer_hint()

    u.key(pygame.K_v, "v")
    assert app.view_mode == "2D" and app.torus_view is None
    assert view.released == 1
    u.frame()
    assert app.screen.get_at(area.center)[:3] != TORUS_FILL  # the 2D grid is back


def test_apply_and_new_in_3d_reach_the_view(make_app, ui, fake_view):
    """F31: Apply kept a renderer built for the old grid size and species count."""
    app = make_app()
    u = ui(app)
    view = _enter_3d(app, ui)
    conway = Rule(birth={3}, survive={2, 3})
    lifeforms = [Lifeform(i, conway) for i in (1, 2, 3)]
    config = replace(
        app.config,
        grid=replace(app.config.grid, width=40, shape="hexagon"),
        simulation=replace(app.config.simulation, n_lifeforms=3),
    )
    app._apply_settings(config, lifeforms)
    u.frame()
    _, grid, palette = view.draws[-1]
    assert (grid.shape, grid.width) == ("hexagon", 40)
    assert len(palette) == 7  # dead + 3 alive + 3 static shades

    u.key(pygame.K_n, "n")
    assert view.draws[-1][1] is app.session.grid
    assert app.view_mode == "3D" and app.torus_view is view


def test_drag_and_wheel_on_the_torus_move_the_camera_not_cells(make_app, ui, fake_view):
    """F33: mouse input never reached the camera."""
    app = make_app()
    app.session.paused = True
    u = ui(app)
    camera = _enter_3d(app, ui).camera
    before = app.session.grid.occupancy.copy()
    centre = app.layout.grid_rect().center
    azimuth, elevation, distance = camera.azimuth, camera.elevation, camera.distance

    end = (centre[0] + 60, centre[1] + 30)
    for event in (
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=centre, button=1),
        pygame.event.Event(pygame.MOUSEMOTION, pos=end, rel=(60, 30), buttons=(1, 0, 0)),
        pygame.event.Event(pygame.MOUSEBUTTONUP, pos=end, button=1),
    ):
        pygame.event.post(event)
    u.frame()
    assert camera.auto_rotate is False
    assert camera.azimuth == pytest.approx(azimuth - 0.6)
    assert camera.elevation == pytest.approx(elevation + 0.3)
    np.testing.assert_array_equal(app.session.grid.occupancy, before)  # no cell was toggled

    u.wheel(1, over=centre)
    assert camera.distance < distance
    zoomed = camera.distance
    u.wheel(1, over=(app.layout.panel_width // 2, 400))  # over the panel: not a zoom
    assert camera.distance == zoomed


def test_resizing_in_3d_draws_into_the_new_grid_area(make_app, ui, fake_view):
    """F34 (app side): the torus must follow the grid area through a window resize."""
    app = make_app()
    u = ui(app)
    view = _enter_3d(app, ui)
    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=900, h=700, size=(900, 700)))
    u.frame()
    assert view.draws[-1][0] == app.layout.grid_rect()
    assert app.layout.grid_rect().bottom <= 700


def test_a_failing_draw_returns_to_2d_with_the_reason(make_app, ui, fake_view):
    app = make_app()
    u = ui(app)
    view = _enter_3d(app, ui)
    fake_view.fail_draw = ValueError("a 99999 x 3 grid exceeds this GPU's 16384-cell limit")
    u.frame()  # does not raise
    assert app.view_mode == "2D" and app.torus_view is None and view.released == 1
    assert app.torus_status.startswith("3D view stopped: a 99999 x 3 grid")


def test_closing_the_app_releases_the_3d_view(make_app, ui, fake_view):
    app = make_app()
    view = _enter_3d(app, ui)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run()
    assert view.released == 1 and app.torus_view is None


# --- the real view on OpenGL ---------------------------------------------------------------------


@pytest.fixture
def real_3d(make_app, ui):
    """`real_3d(**sections)`: an app (as `make_app`) already in 3D on the real view."""
    pytest.importorskip("moderngl", exc_type=ImportError)

    def make(**sections):
        app = make_app(**sections)
        ui(app).key(pygame.K_v, "v")
        if app.view_mode != "3D":
            pytest.skip(f"3D view unavailable here: {app.torus_status}")
        return app

    return make


def _species_pixels(app, species_id):
    """Fraction of the grid area showing a species' alive colour (any shade)."""
    rect = app.layout.grid_rect()
    image = pygame.surfarray.array3d(app.screen.subsurface(rect)).astype(float)
    color = app.palette[species_id].astype(float)
    total = image.sum(axis=2, keepdims=True).clip(min=1)
    match = np.abs(image / total - color / color.sum()).max(axis=2) < 0.08
    return (match & (total[..., 0] > 60)).mean()


@pytest.mark.parametrize(
    "grid", [{}, {"shape": "hexagon", "width": 40, "height": 30}, {"shape": "triangle"}]
)
def test_the_real_3d_view_shows_the_live_grid(real_3d, ui, grid):
    """F30: pressing V always fell back to '3D view unavailable'."""
    app = real_3d(grid=grid)
    u = ui(app)
    app.session.paused = True  # a full grid would die in one step
    app.session.grid.occupancy[:] = 1
    u.frame()
    assert _species_pixels(app, 1) > 0.1
    view = app.torus_view
    renderer = view.renderer

    u.key(pygame.K_v, "v")
    assert app.view_mode == "2D"
    assert renderer.program is None and renderer.texture is None and view.ctx is None


def test_apply_resize_and_new_work_in_the_real_3d_view(real_3d, ui):
    """F31 and F34 on real GL: Apply with a new size and species count crashed the app, and
    a resize left the camera's aspect ratio behind."""
    app = real_3d()
    u = ui(app)
    panel = app.settings_panel
    panel.width_input.text, panel.height_input.text = "40", "30"
    panel.species_count_input.text = "3"
    panel.species_count_input.commit()
    u.click(panel.apply_button.rect.center)
    assert len(app.session.lifeforms) == 3, panel.error
    app.session.paused = True
    app.session.grid.occupancy[:] = 3
    u.frame(3)
    assert app.view_mode == "3D", app.torus_status
    assert app.torus_view.renderer.texture.size == (40, 30)
    assert _species_pixels(app, 3) > 0.1

    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=900, h=900, size=(900, 900)))
    u.frame()
    rect = app.layout.grid_rect()
    assert app.torus_view.camera.aspect_ratio == pytest.approx(rect.width / rect.height)

    u.key(pygame.K_n, "n")
    u.frame()
    assert app.view_mode == "3D", app.torus_status
