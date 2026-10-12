"""Draws a grid's cells on a 3D torus with moderngl (§6.14), into a framebuffer it is handed.

`TorusRenderer` owns only the GL objects it creates (program, mesh buffers, grid texture) and
works with any moderngl context and target framebuffer — the app's offscreen view
(`view.TorusView`) or a test's standalone context. It never touches the window, and it expects
its context to be current when its methods run (`with ctx:` around them; `TorusView` does
that). How cells map onto the surface is described in `shaders.py`; `release()` frees
everything it created and is safe to call twice.

Part of the `[torus]` optional extra: `ui/app.py` imports this package lazily, only inside its
V-toggle handler, so a missing `moderngl` install degrades to a status message.
"""

from __future__ import annotations

import moderngl
import numpy as np

from latticelife.core.lattice import LatticeGrid
from latticelife.ui import render
from latticelife.ui.torus.camera import OrbitalCamera, gl_bytes
from latticelife.ui.torus.geometry import generate_torus_mesh
from latticelife.ui.torus.shaders import FRAGMENT_SHADER, SHAPE_CODES, VERTEX_SHADER
from latticelife.ui.torus.texture_mapper import cell_colors, surface_window

MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.4
MAJOR_SEGMENTS = 128
MINOR_SEGMENTS = 64

_GL_OBJECTS = ("vao", "vbo", "ibo", "texture", "program")


class TorusRenderer:
    """The torus mesh and shader program, and the grid's texture, on one moderngl context."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.program: moderngl.Program | None = None
        self.vbo: moderngl.Buffer | None = None
        self.ibo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None
        self.texture: moderngl.Texture | None = None
        try:
            self.program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
            vertices, indices = generate_torus_mesh(
                MAJOR_RADIUS, MINOR_RADIUS, MAJOR_SEGMENTS, MINOR_SEGMENTS
            )
            self.vbo = ctx.buffer(vertices.tobytes())
            self.ibo = ctx.buffer(indices.tobytes())
            self.vao = ctx.vertex_array(
                self.program,
                [(self.vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
                self.ibo,
            )
            self.program["line_color"].value = tuple(c / 255 for c in render.GRID_LINE_COLOR)
            self.program["min_outline_px"].value = render.MIN_OUTLINE_SIZE
            self.program["cells"].value = 0
        except Exception:
            self.release()  # nothing half-built stays allocated
            raise

    def update_cells(self, grid: LatticeGrid, palette: np.ndarray) -> None:
        """Upload the grid's current colours; the texture and the cell lookup follow the grid's
        size, shape and wrap."""
        size = (grid.width, grid.height)
        if self.texture is None or self.texture.size != size:
            if self.texture is not None:
                self.texture.release()
                self.texture = None
            limit = self.ctx.info["GL_MAX_TEXTURE_SIZE"]
            if max(size) > limit:
                raise ValueError(
                    f"a {grid.width} x {grid.height} grid exceeds this GPU's {limit}-cell limit"
                )
            self.texture = self.ctx.texture(size, 3, dtype="f1")
        self.texture.write(np.ascontiguousarray(cell_colors(grid, palette)).tobytes())
        self.program["grid_size"].value = size
        self.program["shape"].value = SHAPE_CODES[grid.shape]
        self.program["wrap"].value = bool(grid.wrap)
        origin, span = surface_window(grid.shape, grid.width, grid.height, grid.wrap)
        self.program["lattice_origin"].value = origin
        self.program["lattice_span"].value = span

    def render(
        self,
        framebuffer: moderngl.Framebuffer,
        camera: OrbitalCamera,
        background: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """Draw the torus over a `background` clear of `framebuffer` (which needs a depth
        attachment); beyond a bounded grid's edges the surface shows `background` too. The
        camera's aspect ratio is matched to the framebuffer first, so a resized target is never
        drawn stretched."""
        if self.texture is None:
            raise RuntimeError("update_cells() must run before render()")
        width, height = framebuffer.size
        camera.on_resize(width, height)
        framebuffer.use()
        self.ctx.viewport = (0, 0, width, height)
        red, green, blue = (c / 255 for c in background)
        framebuffer.clear(red, green, blue, 1.0, depth=1.0)
        self.program["outside_color"].value = (red, green, blue)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.program["view"].write(gl_bytes(camera.view_matrix))
        self.program["projection"].write(gl_bytes(camera.projection_matrix))
        self.program["camera_position"].value = tuple(float(c) for c in camera.position)
        self.texture.use(0)
        self.vao.render(moderngl.TRIANGLES)

    def release(self) -> None:
        """Free every GL object this renderer created (the context is the caller's)."""
        for name in _GL_OBJECTS:
            obj = getattr(self, name)
            if obj is not None:
                obj.release()
                setattr(self, name, None)
