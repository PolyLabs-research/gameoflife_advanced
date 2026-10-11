"""The 3D torus view inside the app window (§6.14, V key).

The window stays an ordinary pygame surface: the panel, chart, stat cards and overlays are drawn
exactly as in 2D. The torus is rendered offscreen — a standalone OpenGL 3.3 context (EGL first,
then the platform's default), a framebuffer the size of the grid area — read back, and blitted
into the grid area each frame. So the 3D view needs no OpenGL window, never fights pygame for
the display, and runs the same with or without a desktop.

`TorusView.draw` follows the grid it is given every frame (shape, size, palette, static
threshold), so a new session from Apply or New, or a resized window, needs no rebuild: the
texture, framebuffer and camera aspect follow on the next draw. Mouse input reaches the camera
through `handle_event`. `release()` frees every GL object and the context; the app calls it when
it leaves 3D or exits.

Contexts. moderngl draws with whichever context is current on the thread, so each public method
makes the view's context current for its whole body (`with self.ctx:`) — once, never nested:
leaving that block can leave no context current at all (moderngl's EGL backend does).
"""

from __future__ import annotations

import moderngl
import pygame

from latticelife.core.lattice import LatticeGrid
from latticelife.ui.torus.camera import OrbitalCamera
from latticelife.ui.torus.torus_renderer import TorusRenderer

GL_VERSION = 330
# Multisampling for smooth torus edges, when the GPU offers it.
SAMPLES = 4


class TorusUnavailable(RuntimeError):
    """No OpenGL 3.3 context could be created."""


def create_offscreen_context() -> moderngl.Context:
    """A standalone (windowless) OpenGL 3.3 context: EGL first (headless and Wayland friendly),
    then the platform default (GLX, WGL or CGL). Raises TorusUnavailable if neither works."""
    errors = []
    for settings in ({"backend": "egl"}, {}):
        try:
            return moderngl.create_standalone_context(require=GL_VERSION, **settings)
        except Exception as exc:  # each backend fails in its own way (missing library, no GPU)
            errors.append(f"{settings.get('backend', 'default')}: {_first_line(exc)}")
    raise TorusUnavailable("no OpenGL 3.3 context (" + "; ".join(errors) + ")")


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__


class TorusView:
    """Renders the grid on a torus into the grid area of a pygame surface."""

    def __init__(
        self,
        background: tuple[int, int, int] = (0, 0, 0),
        ctx: moderngl.Context | None = None,
        samples: int = SAMPLES,
    ) -> None:
        """`ctx` defaults to a new offscreen context, which the view then owns and releases."""
        self._owns_ctx = ctx is None
        self.ctx: moderngl.Context | None = create_offscreen_context() if ctx is None else ctx
        self.camera = OrbitalCamera()
        self.background = background
        self._target: moderngl.Framebuffer | None = None  # rendered into (multisampled)
        self._resolved: moderngl.Framebuffer | None = None  # read back from
        try:
            with self.ctx:
                self.renderer = TorusRenderer(self.ctx)
                self.samples = samples if 0 < samples <= self.ctx.max_samples else 0
        except Exception:
            if self._owns_ctx:
                self.ctx.release()
            self.ctx = None
            raise

    @property
    def dragging(self) -> bool:
        return self.camera.dragging

    def _framebuffers(self, size: tuple[int, int]) -> tuple[moderngl.Framebuffer, ...]:
        if self._target is None or self._target.size != size:
            self._release_framebuffers()
            self._target = self.ctx.simple_framebuffer(size, components=4, samples=self.samples)
            if self.samples:
                self._resolved = self.ctx.framebuffer(self.ctx.renderbuffer(size, components=4))
            else:
                self._resolved = self._target
        assert self._resolved is not None
        return self._target, self._resolved

    def draw(self, surface: pygame.Surface, area: pygame.Rect, grid: LatticeGrid, palette) -> None:
        """Advance the camera one frame and draw the torus over `area` of `surface`."""
        if self.ctx is None:
            raise RuntimeError("the 3D view has been released")
        area = pygame.Rect(area)
        if area.width < 1 or area.height < 1:
            return
        self.camera.update()
        with self.ctx:
            target, resolved = self._framebuffers(area.size)
            self.renderer.update_cells(grid, palette)
            self.renderer.render(target, self.camera, self.background)
            if resolved is not target:
                self.ctx.copy_framebuffer(resolved, target)
            pixels = resolved.read(components=3, alignment=1)
        # GL rows run bottom-up; flipped=True puts the first row read at the bottom.
        surface.blit(pygame.image.frombytes(pixels, area.size, "RGB", True), area.topleft)

    def handle_event(self, event: pygame.event.Event, area: pygame.Rect) -> bool:
        """Camera input over the grid area `area` (`OrbitalCamera.handle_event`): drag to
        rotate, wheel to zoom. True if the event was used."""
        return self.camera.handle_event(event, area)

    def _release_framebuffers(self) -> None:
        for fbo in {id(f): f for f in (self._target, self._resolved) if f is not None}.values():
            for attachment in (*fbo.color_attachments, fbo.depth_attachment):
                if attachment is not None:
                    attachment.release()
            fbo.release()
        self._target = self._resolved = None

    def release(self) -> None:
        """Free the framebuffers, the renderer's GL objects and (if the view created it) the
        context. Safe to call twice."""
        if self.ctx is None:
            return
        with self.ctx:
            self._release_framebuffers()
            self.renderer.release()
        if self._owns_ctx:
            self.ctx.release()
        self.ctx = None
