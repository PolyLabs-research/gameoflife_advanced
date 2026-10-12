"""The app's 3D view (V key, §6.14) — F30-F34, F39, F40.

Most tests swap `ui.torus.view` for a stand-in (no GL needed), to pin down the app side: when
3D starts or refuses to, where the torus is drawn, what reaches the camera, and that the view is
released. The last tests run the real view on an offscreen OpenGL context, and are skipped
when the [torus] extra or OpenGL 3.3 is missing."""

import ctypes
import json
import os
import subprocess
import sys
import threading
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


class FakeContext:
    """Stands in for `create_offscreen_context()`'s GL context: counts what made and freed it."""

    instances: list["FakeContext"] = []
    fail: Exception | None = None

    def __init__(self):
        if FakeContext.fail is not None:
            raise FakeContext.fail
        self.released = 0
        self.threads = {threading.current_thread()}  # every thread that made or freed it
        FakeContext.instances.append(self)

    def release(self):
        self.released += 1
        self.threads.add(threading.current_thread())


class FakeView:
    """Stands in for `TorusView`: records draws, routes input to a real camera, and fills the
    area it is given so tests can see where the torus went."""

    instances: list["FakeView"] = []
    fail_init: Exception | None = None
    fail_draw: Exception | None = None

    def __init__(self, background, ctx=None):
        if FakeView.fail_init is not None:
            raise FakeView.fail_init
        self.background = background
        self.ctx = ctx
        self.camera = OrbitalCamera()
        self.draws: list[tuple[pygame.Rect, object, np.ndarray]] = []
        self.released = 0
        self.threads = {threading.current_thread()}  # every thread that made, drew or freed it
        FakeView.instances.append(self)

    @property
    def dragging(self):
        return self.camera.dragging

    def draw(self, surface, area, grid, palette):
        self.threads.add(threading.current_thread())
        if FakeView.fail_draw is not None:
            raise FakeView.fail_draw
        self.draws.append((pygame.Rect(area), grid, palette.copy()))
        surface.fill(TORUS_FILL, area)

    def handle_event(self, event, area):
        return self.camera.handle_event(event, area)

    def release(self):
        self.released += 1
        self.threads.add(threading.current_thread())


@pytest.fixture
def fake_view(monkeypatch):
    FakeView.instances, FakeContext.instances = [], []
    FakeView.fail_init = FakeView.fail_draw = FakeContext.fail = None
    module = types.SimpleNamespace(TorusView=FakeView, create_offscreen_context=FakeContext)
    monkeypatch.setitem(sys.modules, VIEW_MODULE, module)
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


@pytest.mark.parametrize("failing", [FakeContext, FakeView])
def test_v_without_opengl_says_why_and_stays_2d(make_app, ui, fake_view, failing):
    """F40: the old smoke test passed whether or not 3D could ever start. The context, or the
    view on it, fails; either way no context is kept for the next V to inherit."""
    error = RuntimeError("no OpenGL 3.3 context (egl: no device)\ndetails")
    if failing is FakeContext:
        FakeContext.fail = error
    else:
        fake_view.fail_init = error
    app = make_app()
    ui(app).key(pygame.K_v, "v")
    assert app.view_mode == "2D" and app.torus_view is None
    assert app.torus_status == "3D view unavailable: no OpenGL 3.3 context (egl: no device)"
    assert all(ctx.released == 1 for ctx in FakeContext.instances)


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


def test_every_v_draws_on_one_context_freed_when_the_app_closes(make_app, ui, fake_view):
    """Each V made (and later destroyed) a context of its own, and the driver keeps part of
    every destroyed context: memory and threads grew with each toggle (F32 follow-up)."""
    app = make_app()
    u = ui(app)
    for _ in range(3):
        _enter_3d(app, ui)
        u.frame()
        u.key(pygame.K_v, "v")
        assert app.view_mode == "2D"
    assert len(fake_view.instances) == 3 and all(v.released == 1 for v in fake_view.instances)
    [context] = FakeContext.instances
    assert all(view.ctx is context for view in fake_view.instances)
    assert context.released == 0  # leaving 3D keeps it for the next V

    app.close()
    assert context.released == 1
    app.close()  # closing twice frees it once
    assert context.released == 1


def test_the_app_sets_no_sdl_hints(make_app, monkeypatch):
    """The app set SDL_FRAMEBUFFER_ACCELERATION=0 for every run, to keep SDL from presenting
    the window through a GL context of its own. That is the only way SDL 2's Wayland and
    KMSDRM drivers have to give a window a surface, so there the window could fail to open at
    all. How SDL presents the window is SDL's choice; the 3D view keeps out of its way on a
    thread."""
    for name in [name for name in os.environ if name.startswith("SDL_")]:
        monkeypatch.setenv(name, os.environ[name])  # recorded: restored after the test
    monkeypatch.setenv("SDL_FRAMEBUFFER_ACCELERATION", "")
    monkeypatch.delenv("SDL_FRAMEBUFFER_ACCELERATION")
    before = {name: value for name, value in os.environ.items() if name.startswith("SDL_")}
    make_app()
    assert {name: value for name, value in os.environ.items() if name.startswith("SDL_")} == before


def test_every_gl_call_runs_on_one_thread_of_its_own(make_app, ui, fake_view):
    """SDL may present the window through a GL context it keeps current on the main thread
    (GLX on x11, EGL on Wayland and KMSDRM). Made on that thread, the 3D view's context took
    SDL's place (or, under libglvnd, could not be made current over GLX at all). Its context
    and views are now made, drawn and freed on one other thread, which close() stops."""
    app = make_app()
    u = ui(app)
    for _ in range(2):
        _enter_3d(app, ui)
        u.frame(2)
        u.key(pygame.K_v, "v")
    _enter_3d(app, ui)
    u.frame()
    app.close()

    touched = [*fake_view.instances, *FakeContext.instances]
    assert all(obj.released == 1 for obj in touched)
    [thread] = set().union(*(obj.threads for obj in touched))
    assert thread is not threading.main_thread()
    assert not thread.is_alive() and app._gl_thread is None

    _enter_3d(app, ui)  # a V after close() starts a new thread
    assert fake_view.instances[-1].threads != {thread}


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
    # The failure may have broken the context: it is freed, and the next V makes a new one.
    [broken] = FakeContext.instances
    assert broken.released == 1
    fake_view.fail_draw = None
    assert _enter_3d(app, ui).ctx is not broken
    assert len(FakeContext.instances) == 2


def test_closing_the_app_releases_the_3d_view(make_app, ui, fake_view):
    app = make_app()
    view = _enter_3d(app, ui)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run()
    assert view.released == 1 and app.torus_view is None
    assert [ctx.released for ctx in FakeContext.instances] == [1]


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


def test_the_real_3d_view_reuses_one_context_until_the_app_closes(real_3d, ui):
    """A fresh EGL context per V grew memory and driver threads with every toggle; one
    context now serves every 3D view, survives leaving 3D, and is freed by close()."""
    moderngl = pytest.importorskip("moderngl", exc_type=ImportError)
    app = real_3d()
    u = ui(app)
    context = app.torus_view.ctx

    def use_context():
        with context:
            context.buffer(b"\0" * 16).release()

    for _ in range(3):
        u.frame()
        u.key(pygame.K_v, "v")
        assert app.view_mode == "2D"
        app._on_gl_thread(use_context)  # still usable between 3D sessions, on its thread
        u.key(pygame.K_v, "v")
        assert app.view_mode == "3D", app.torus_status
        assert app.torus_view.ctx is context

    app.close()
    assert app.torus_view is None and app._gl_thread is None
    assert isinstance(context.mglo, moderngl.InvalidObject)


def test_the_real_3d_view_leaves_the_main_threads_gl_context_current(make_app, ui):
    """How SDL presents the window on Wayland and KMSDRM: an EGL context it keeps current on
    the main thread (here a stand-in). The 3D view made its own context current there and,
    leaving `with ctx:`, left none, so SDL would draw into nothing and the window froze."""
    moderngl = pytest.importorskip("moderngl", exc_type=ImportError)
    try:
        egl = ctypes.CDLL("libEGL.so.1")
    except OSError:
        pytest.skip("no libEGL")
    egl.eglGetCurrentContext.restype = egl.eglGetCurrentDisplay.restype = ctypes.c_void_p
    egl.eglMakeCurrent.argtypes = [ctypes.c_void_p] * 4
    try:  # made current on this thread, as SDL's renderer context is
        sdl_context = moderngl.create_standalone_context(require=330, backend="egl")
    except Exception as exc:
        pytest.skip(f"no EGL context: {exc}")
    try:
        current = egl.eglGetCurrentContext()
        assert current
        app = make_app()
        u = ui(app)
        u.key(pygame.K_v, "v")
        if app.view_mode != "3D":
            pytest.skip(f"3D view unavailable here: {app.torus_status}")
        u.frame(2)
        assert egl.eglGetCurrentContext() == current
        u.key(pygame.K_v, "v")
        u.key(pygame.K_v, "v")
        u.frame()
        app.close()
        assert egl.eglGetCurrentContext() == current
        sdl_context.buffer(b"\0" * 16).release()  # still current: no `with` needed
    finally:
        egl.eglMakeCurrent(egl.eglGetCurrentDisplay(), None, None, None)
        sdl_context.release()


# The app on SDL's offscreen driver, which (as Wayland and KMSDRM must) presents the window
# through a GL renderer texture, with SDL's EGL context current on the main thread. `probe`
# reports whether plain pygame presents that way here; `app` reports what the app does.
_SDL_GL_WINDOW_SCRIPT = """
import ctypes, json, sys
import numpy as np
import pygame

def report(**result):
    print(json.dumps(result))
    sys.exit()

try:
    egl, gl = ctypes.CDLL("libEGL.so.1"), ctypes.CDLL("libOpenGL.so.0")
except OSError as exc:
    report(gl=False, why=str(exc))
egl.eglGetCurrentContext.restype = ctypes.c_void_p

def shown(app):
    # What SDL presented (read back with its context) equals what the app drew.
    width, height = app.screen.get_size()
    pixels = (ctypes.c_ubyte * (width * height * 4))()
    gl.glFinish()
    gl.glReadPixels(0, 0, width, height, 0x1908, 0x1401, pixels)  # RGBA, unsigned bytes
    presented = np.frombuffer(pixels, np.uint8).reshape(height, width, 4)[::-1, :, :3]
    return bool(np.array_equal(presented, pygame.surfarray.array3d(app.screen).swapaxes(0, 1)))

def press_v(app, frames=3):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v, unicode="v", mod=0))
    for _ in range(frames):
        app.step_frame()

if sys.argv[1] == "probe":
    pygame.init()
    pygame.display.set_mode((64, 64))
    pygame.display.flip()
    report(gl=bool(egl.eglGetCurrentContext()))

from latticelife.config import DEFAULT_CONFIG_PATH, load_config
from latticelife.ui.app import App

app = App(load_config(DEFAULT_CONFIG_PATH))
app.step_frame()
sdl_context = egl.eglGetCurrentContext()
if not sdl_context:
    report(sdl_gl=False)
result = dict(sdl_gl=True, shown_2d=shown(app))
press_v(app)
result.update(mode=app.view_mode, status=app.torus_status)
if app.view_mode == "3D":
    result.update(kept_3d=egl.eglGetCurrentContext() == sdl_context, shown_3d=shown(app))
    press_v(app)
    result.update(shown_back_in_2d=shown(app))
app.close()
result.update(kept_after_close=egl.eglGetCurrentContext() == sdl_context)
report(**result)
"""


def _run_on_sdl_gl_window(mode: str) -> dict:
    env = {name: value for name, value in os.environ.items() if not name.startswith("SDL_")}
    env.update(SDL_VIDEODRIVER="offscreen", SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    result = subprocess.run(
        [sys.executable, "-c", _SDL_GL_WINDOW_SCRIPT, mode],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_3d_view_keeps_a_window_sdl_presents_through_gl_working():
    """End to end, on a window SDL presents through its own GL context (Wayland's and KMSDRM's
    only way): the app forced SDL off that path with SDL_FRAMEBUFFER_ACCELERATION=0, which on
    Wayland and KMSDRM can stop the window opening; and where SDL took that path anyway,
    entering 3D took over SDL's context, so the window showed black and stayed frozen even back
    in 2D."""
    pytest.importorskip("moderngl", exc_type=ImportError)
    if not _run_on_sdl_gl_window("probe")["gl"]:
        pytest.skip("SDL's offscreen driver does not present through GL here")
    result = _run_on_sdl_gl_window("app")
    assert result["sdl_gl"] and result["shown_2d"]  # the app left SDL's choice alone
    if result["mode"] != "3D":
        pytest.skip(f"3D view unavailable here: {result['status']}")
    assert result["kept_3d"] and result["shown_3d"], result
    assert result["shown_back_in_2d"] and result["kept_after_close"], result
