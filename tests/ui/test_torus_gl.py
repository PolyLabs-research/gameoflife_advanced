"""3D torus rendering on a real OpenGL context (§6.14): F30 (it draws), F31 (it follows grid
changes), F32 (it frees its GL objects), F34 (resizes keep the shape), F35 (orientation), F37
(static threshold) and the shape-aware cell lookup.

These need the [torus] extra and an OpenGL 3.3 context (EGL works without a display); they are
skipped cleanly otherwise. Pixels are read back from the offscreen render, top row first."""

import numpy as np
import pygame
import pytest

moderngl = pytest.importorskip("moderngl", exc_type=ImportError)

from latticelife.core.lattice import LatticeGrid  # noqa: E402
from latticelife.core.lifeform import Lifeform  # noqa: E402
from latticelife.core.rules import Rule  # noqa: E402
from latticelife.ui import render  # noqa: E402
from latticelife.ui.torus.shaders import CELL_LOOKUP, SHAPE_CODES  # noqa: E402
from latticelife.ui.torus.texture_mapper import surface_cells, surface_window  # noqa: E402
from latticelife.ui.torus.view import (  # noqa: E402
    TorusUnavailable,
    TorusView,
    create_offscreen_context,
)

BACKGROUND = (18, 18, 24)
CONWAY = Rule(birth={3}, survive={2, 3})


@pytest.fixture(scope="module")
def gl_ctx():
    try:
        ctx = create_offscreen_context()
    except TorusUnavailable as exc:
        pytest.skip(f"no OpenGL 3.3 context: {exc}")
    yield ctx
    ctx.release()


def _grid(shape, width, height, n_species=1, wrap=True, static_threshold=10):
    lifeforms = [Lifeform(id=i, rule=CONWAY) for i in range(1, n_species + 1)]
    grid = LatticeGrid(lifeforms, shape, width, height, wrap, static_threshold=static_threshold)
    return grid, render.build_palette(lifeforms)


def _front_view(ctx, **camera):
    """A view with a still camera looking straight at the grid's middle (azimuth 0)."""
    view = TorusView(BACKGROUND, ctx=ctx, samples=0)
    view.camera.auto_rotate = False
    view.camera.elevation = camera.get("elevation", 0.0)
    view.camera.distance = camera.get("distance", 3.0)
    return view


def _draw(view, grid, palette, size=(480, 360)) -> np.ndarray:
    surface = pygame.Surface(size)
    surface.fill((255, 0, 255))
    view.draw(surface, pygame.Rect((0, 0), size), grid, palette)
    return pygame.surfarray.array3d(surface).swapaxes(0, 1).astype(int)  # (h, w, 3)


def _mask(image, color, tolerance=0.08):
    """Pixels with `color`'s hue: each channel's share of the pixel's brightness matches. Lit
    and shaded pixels of one cell colour all pass; other colours and the background do not."""
    color = np.asarray(color, dtype=float)
    total = image.sum(axis=2, keepdims=True).clip(min=1)
    share = image / total
    return (np.abs(share - color / color.sum()).max(axis=2) < tolerance) & (total[..., 0] > 60)


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    assert len(xs), "colour not found on screen"
    return xs.mean(), ys.mean()


# --- the cell lookup ------------------------------------------------------------------------

_FLAT_VS = """
#version 330 core
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""
# Red and green: the texel (col, row) a pixel shows, 255 for none; blue: its outline weight.
_FLAT_FS = (
    "\n#version 330 core\nin vec2 v_uv;\nout vec4 out_color;\n"
    + CELL_LOOKUP
    + """
void main() {
    vec2 p = lattice_point(v_uv);
    ivec2 cell;
    ivec2 across;
    float edge;
    float extent;
    find_cell(p, dFdx(p), dFdy(p), cell, across, edge, extent);
    ivec2 texel = grid_texel(cell);
    vec2 shown = texel.x >= 0 ? vec2(texel) / 255.0 : vec2(1.0);
    out_color = vec4(shown, outline_weight(cell, across, edge, extent), 1.0);
}
"""
)


def _flat_lookup(ctx, shape, width, height, wrap, n=256) -> np.ndarray:
    """The flat lookup program over the whole uv square, n x n pixels; image row i is at
    v = (i + 0.5) / n (GL rows run bottom-up) and column j at u = (j + 0.5) / n."""
    with ctx:
        program = ctx.program(vertex_shader=_FLAT_VS, fragment_shader=_FLAT_FS)
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        vao = ctx.vertex_array(program, [(quad, "2f", "in_pos")])
        fbo = ctx.simple_framebuffer((n, n))
        program["grid_size"].value = (width, height)
        program["shape"].value = SHAPE_CODES[shape]
        program["wrap"].value = wrap
        origin, span = surface_window(shape, width, height, wrap)
        program["lattice_origin"].value = origin
        program["lattice_span"].value = span
        program["min_outline_px"].value = render.MIN_OUTLINE_SIZE
        fbo.use()
        fbo.clear()
        vao.render(moderngl.TRIANGLE_STRIP)
        image = np.frombuffer(fbo.read(components=3), np.uint8).reshape(n, n, 3)
        for obj in (vao, quad, program, fbo):
            obj.release()
    return image


@pytest.mark.parametrize(
    "shape, width, height, wrap",
    [
        ("square", 50, 50, True),
        ("square", 7, 5, False),
        ("hexagon", 40, 30, True),
        ("hexagon", 7, 5, False),  # odd sizes: bounded grids only
        ("hexagon", 8, 5, False),
        ("triangle", 60, 30, True),
        ("triangle", 5, 3, False),
        ("triangle", 7, 6, False),
    ],
)
def test_shader_cell_lookup_matches_its_numpy_twin(gl_ctx, shape, width, height, wrap):
    """Every pixel of the surface's uv square shows the cell `surface_cells` names — the same
    hexagons and triangles the 2D view draws, each beside its own neighbours (tested in
    test_torus_mapping) — or, beyond a bounded grid's edges, none."""
    n = 256
    image = _flat_lookup(gl_ctx, shape, width, height, wrap, n)
    centres = (np.arange(n) + 0.5) / n
    u, v = np.meshgrid(centres, centres)
    rows, cols = surface_cells(shape, width, height, u, v, wrap=wrap)
    mismatched = (image[..., 0] != cols % 256) | (image[..., 1] != rows % 256)
    assert mismatched.mean() < 0.001  # float32 vs float64 may differ on an exact edge
    if not wrap and shape != "square":
        assert (rows < 0).mean() > 0.05  # the notches along the edges are in the test


@pytest.mark.parametrize("shape, width, height", [("hexagon", 7, 5), ("triangle", 5, 3)])
def test_outlines_follow_cell_edges_only(gl_ctx, shape, width, height):
    """Outlines are drawn on every edge of a cell, as in 2D, but not on an edge between two
    places beyond a bounded grid's edges (the level edges between a triangle grid's notches),
    where 2D draws nothing."""
    image = _flat_lookup(gl_ctx, shape, width, height, False)
    shows_cell = image[..., 0] != 255
    outlined = image[..., 2] > 64
    assert outlined[shows_cell].mean() > 0.02  # cells this large are outlined
    near_a_cell = shows_cell.copy()
    for _ in range(2):  # grow by 2 px each way
        near_a_cell = (
            near_a_cell
            | np.roll(near_a_cell, 1, 0)
            | np.roll(near_a_cell, -1, 0)
            | np.roll(near_a_cell, 1, 1)
            | np.roll(near_a_cell, -1, 1)
        )
    assert not (outlined & ~near_a_cell).any()


# --- what the torus shows ---------------------------------------------------------------------


def test_a_bounded_grid_shows_the_background_beyond_its_edges(gl_ctx):
    """A bounded hexagon grid is drawn whole, with the background in the notches along its
    edges (on the seams, which a camera above the torus sees); a wrapped one has no notches."""
    seen = {}
    for wrap in (False, True):
        grid, palette = _grid("hexagon", 8, 6, wrap=wrap)
        grid.occupancy[:] = 1
        grid.lifespans[:] = 1
        view = TorusView((0, 0, 200), ctx=gl_ctx, samples=0)
        view.camera.auto_rotate = False
        view.camera.elevation = 1.2
        try:
            image = _draw(view, grid, palette)
        finally:
            view.release()
        cleared = (image == (0, 0, 200)).all(axis=2)
        seen[wrap] = (_mask(image, (0, 0, 200)) & ~cleared).mean()  # lit, so not the clear colour
    assert seen[False] > 0.002
    assert seen[True] == 0


def test_alive_cells_light_up_in_their_species_colour(gl_ctx):
    """F30: the torus is drawn, and live cells show their species colour."""
    grid, palette = _grid("square", 50, 50, n_species=2)
    view = _front_view(gl_ctx)
    try:
        empty = _draw(view, grid, palette)
        grid.occupancy[: grid.n_cells // 2] = 1  # rows 0-24 red, rows 25-49 green
        grid.occupancy[grid.n_cells // 2 :] = 2
        grid.lifespans[:] = 1
        full = _draw(view, grid, palette)
    finally:
        view.release()

    assert (empty[0, 0] == BACKGROUND).all()  # the corner shows the background
    assert not (empty == (255, 0, 255)).all(axis=2).any()  # every pixel was drawn
    red, green = _mask(full, palette[1]), _mask(full, palette[2])
    assert not _mask(empty, palette[1]).any()
    assert red.mean() > 0.1 and green.mean() > 0.1
    # The front of the tube (rows near 25) shows both, red above green.
    assert _centroid(red)[1] < _centroid(green)[1]


@pytest.mark.parametrize(
    "shape, width, height", [("square", 40, 20), ("hexagon", 40, 20), ("triangle", 40, 20)]
)
def test_rows_run_down_and_columns_right_as_in_2d(gl_ctx, shape, width, height):
    """F35: the torus showed the grid mirrored top-to-bottom (rows running up the wall)."""
    grid, palette = _grid(shape, width, height, n_species=3)
    mid_row, mid_col = height // 2, width // 2

    def put(row, col, species):
        grid.occupancy[row * width + col] = species

    for col in range(mid_col - 6, mid_col + 4):
        put(mid_row - 3, col, 1)  # red band above the middle
        put(mid_row + 3, col, 2)  # green band below it
    for row in range(mid_row - 2, mid_row + 3):
        put(row, mid_col + 6, 3)  # blue bar to the right
    grid.lifespans[:] = 1

    view = _front_view(gl_ctx)
    try:
        image = _draw(view, grid, palette)
    finally:
        view.release()
    red_x, red_y = _centroid(_mask(image, palette[1]))
    green_x, green_y = _centroid(_mask(image, palette[2]))
    blue_x, _ = _centroid(_mask(image, palette[3]))
    assert red_y < green_y  # the lower row index is higher on screen
    assert blue_x > max(red_x, green_x)  # the higher column is further right


@pytest.mark.parametrize("threshold, expect_static", [(50, False), (10, True)])
def test_cells_use_the_grids_static_threshold(gl_ctx, threshold, expect_static):
    """F37: the torus used a fixed threshold of 10, so lifespan 20 looked static at 50."""
    grid, palette = _grid("square", 10, 10, static_threshold=threshold)
    grid.occupancy[:] = 1
    grid.lifespans[:] = 20
    view = _front_view(gl_ctx)
    try:
        image = _draw(view, grid, palette)
    finally:
        view.release()
    alive, static = _mask(image, palette[1]).mean(), _mask(image, palette[2]).mean()
    assert (static > 0.1 and alive == 0) if expect_static else (alive > 0.1 and static == 0)


def test_the_view_follows_grid_size_shape_and_species_changes(gl_ctx):
    """F31: Apply with a new size or species count crashed the stale renderer."""
    view = _front_view(gl_ctx)
    try:
        grid, palette = _grid("square", 50, 50)
        _draw(view, grid, palette)
        grid, palette = _grid("hexagon", 40, 30, n_species=3)
        grid.occupancy[:] = 3
        image = _draw(view, grid, palette)
        assert view.renderer.texture.size == (40, 30)
        assert _mask(image, palette[3]).mean() > 0.1
    finally:
        view.release()


def _torus_box(image):
    drawn = np.abs(image - BACKGROUND).max(axis=2) > 6
    ys, xs = np.nonzero(drawn)
    return xs.min(), ys.min(), xs.max(), ys.max()


def _box_ratio(box):
    left, top, right, bottom = box
    return (right - left) / (bottom - top)


def test_resizing_the_area_keeps_the_torus_undistorted(gl_ctx):
    """F34: a resized view kept the old aspect ratio and drew the torus squashed."""
    grid, palette = _grid("square", 20, 20)
    view = _front_view(gl_ctx, elevation=1.2, distance=4.0)  # from above: a round ring
    try:
        wide = _box_ratio(_torus_box(_draw(view, grid, palette, size=(600, 400))))
        square = _box_ratio(_torus_box(_draw(view, grid, palette, size=(400, 400))))
        assert view.camera.aspect_ratio == 1.0
    finally:
        view.release()
    assert square == pytest.approx(wide, rel=0.03)


@pytest.mark.parametrize("size", [(868, 552), (400, 400), (300, 600)])
def test_the_whole_torus_fits_any_area_at_the_default_distance(gl_ctx, size):
    grid, palette = _grid("square", 20, 20)
    view = TorusView(BACKGROUND, ctx=gl_ctx, samples=0)
    view.camera.auto_rotate = False
    try:
        for elevation in (0.0, 0.3, 1.2):
            view.camera.elevation = elevation
            left, top, right, bottom = _torus_box(_draw(view, grid, palette, size=size))
            assert left > 0 and top > 0 and right < size[0] - 1 and bottom < size[1] - 1
    finally:
        view.release()


# --- resources ----------------------------------------------------------------------------------


def _gl_objects(view):
    renderer = view.renderer
    objects = [renderer.program, renderer.vbo, renderer.ibo, renderer.vao, renderer.texture]
    for fbo in {id(f): f for f in (view._target, view._resolved)}.values():
        objects += [fbo, *fbo.color_attachments, fbo.depth_attachment]
    return [obj for obj in objects if obj is not None]  # the resolve target has no depth


@pytest.mark.parametrize("samples", [0, 4])
def test_release_frees_every_gl_object(gl_ctx, samples):
    """F32: leaving 3D dropped the renderer without releasing its buffers, program and
    texture (about 96 MiB per V cycle on a 2000 x 2000 grid)."""
    grid, palette = _grid("square", 30, 30)
    view = TorusView(BACKGROUND, ctx=gl_ctx, samples=samples)
    _draw(view, grid, palette)
    objects = _gl_objects(view)
    assert all(not isinstance(obj.mglo, moderngl.InvalidObject) for obj in objects)

    view.release()

    assert all(isinstance(obj.mglo, moderngl.InvalidObject) for obj in objects)
    assert view.renderer.texture is None and view.renderer.program is None
    view.release()  # a second release is harmless
    with pytest.raises(RuntimeError):
        _draw(view, grid, palette)
    with gl_ctx:  # the shared context itself is still usable
        gl_ctx.buffer(b"\0" * 16).release()


def test_a_view_releases_the_context_it_created():
    try:
        view = TorusView(BACKGROUND)
    except TorusUnavailable as exc:
        pytest.skip(f"no OpenGL 3.3 context: {exc}")
    ctx = view.ctx
    view.release()
    assert view.ctx is None
    assert isinstance(ctx.mglo, moderngl.InvalidObject)
