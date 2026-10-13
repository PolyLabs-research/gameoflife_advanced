"""GLSL 3.30 shaders for the 3D torus view (§6.14).

The grid reaches the GPU as a width x height RGB texture, one texel per cell (texel (col, row)
holds cell (row, col)'s colour, from the same palette and alive/static fold as the 2D view). The
fragment shader decides which cell a surface point shows with the grid's own geometry, so
hexagons and triangles keep their shape and neighbours on the torus:

* The surface coordinates uv (`geometry.py`) become a point on the shape's lattice: one trip
  around the ring covers the grid's columns and one trip around the tube its rows, over the
  window `texture_mapper.surface_window` gives (uniforms `lattice_origin`, `lattice_span`).
  Columns run to the right and rows down the outer wall, as in the 2D view, and the middle of
  the grid is on the outer equator facing a camera at azimuth 0 (u = 0.5, v = 0.5).
* The seams (u = 0 at the back of the ring, v = 0 on the inner equator, where the mesh repeats
  its vertices) are the grid's edges, so every cell sits at its own place on the lattice —
  never a whole grid width or height away, which for an odd size would flip a hex column's
  half-cell shift or a triangle's up/down, and show it beside cells it does not touch.
* square: one unit per column and row. hexagon: odd-q hexagons of size 1 (centres 1.5 apart
  across columns, sqrt 3 down a column, odd columns half a cell lower), the cell is the nearest
  centre, as `render._hex_at`. triangle: half a cell per column and one per row; the cell is the
  up or down triangle that contains the point, as `render._triangle_at`.
* A wrapped grid's window is one period of its lattice: cells beyond an edge are the cells the
  grid wraps them to, and the surface continues across the seams with no break. A bounded
  grid's window is its 2D footprint: every cell is whole, beyond its edges there are no cells,
  and the notches between the edge cells of a hexagon or triangle grid show the background
  (`outside_color`), as in 2D. Its edges meet at the seams, but nothing joins them.

Hexagon and triangle cells get a thin outline once they are at least `min_outline_px` pixels on
screen, as in 2D: on every edge of a cell, not between two places beyond a bounded grid's edges.
Lighting is a headlight (light from the camera), so the side of the tube that faces the viewer
is always lit, whatever the camera's elevation.

`CELL_LOOKUP` is self-contained, so tests can compile it into a flat program and compare its
answer with `texture_mapper.surface_cells`, its numpy twin.
"""

SHAPE_CODES = {"square": 0, "hexagon": 1, "triangle": 2}

VERTEX_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 v_position;
out vec3 v_normal;
out vec2 v_uv;

uniform mat4 view;
uniform mat4 projection;

void main() {
    v_position = in_position;
    v_normal = in_normal;
    v_uv = in_uv;
    gl_Position = projection * view * vec4(in_position, 1.0);
}
"""

CELL_LOOKUP = """
uniform ivec2 grid_size;      // (width, height) in cells
uniform int shape;            // SHAPE_CODES
uniform bool wrap;            // whether the grid wraps (else there are no cells beyond its edges)
uniform vec2 lattice_origin;  // texture_mapper.surface_window: the lattice point at u = 0, v = 1
uniform vec2 lattice_span;    // and the lattice distance one trip around the ring / tube covers
uniform float min_outline_px; // cells this many pixels across or more get an outline

const float SQRT3 = 1.7320508075688772;

// Surface coordinates -> a point on the shape's (unbounded) lattice, x across columns.
vec2 lattice_point(vec2 uv) {
    return lattice_origin + vec2(uv.x, 1.0 - uv.y) * lattice_span;
}

// i mod n in 0..n-1 for any sign of i, in integers (GLSL leaves % undefined for negative
// operands, and float division can land a hair below a whole number).
int wrap_index(int i, int n) {
    return i >= 0 ? i % n : (n - 1) - (-i - 1) % n;
}

// Pixels from the lattice point to the straight edge where f = 0, for f linear with lattice
// gradient g; px and py are the lattice point's screen derivatives.
float edge_pixels(float f, vec2 g, vec2 px, vec2 py) {
    return f / max(length(vec2(dot(g, px), dot(g, py))), 1e-6);
}

// The cell (col, row) on the unbounded lattice at point p, the cell across its edge nearest p,
// the distance in pixels from p to that edge, and the cell's smaller on-screen extent in pixels.
void find_cell(
    vec2 p, vec2 px, vec2 py, out ivec2 cell, out ivec2 across, out float edge, out float extent
) {
    float across_px = 1.0 / max(length(vec2(px.x, py.x)), 1e-6);  // pixels per lattice x unit
    float down_px = 1.0 / max(length(vec2(px.y, py.y)), 1e-6);    // pixels per lattice y unit
    if (shape == 1) {
        int q0 = int(round(p.x / 1.5));
        float best = 1e30;
        float second = 1e30;
        vec2 best_centre = vec2(0.0);
        vec2 second_centre = vec2(0.0);
        cell = ivec2(q0, 0);
        across = cell;
        for (int dq = -1; dq <= 1; ++dq) {
            int q = q0 + dq;
            float shift = 0.5 * float(q & 1);
            int r0 = int(round(p.y / SQRT3 - shift));
            for (int dr = -1; dr <= 1; ++dr) {
                int r = r0 + dr;
                vec2 centre = vec2(1.5 * float(q), SQRT3 * (float(r) + shift));
                vec2 offset = p - centre;
                float d2 = dot(offset, offset);
                if (d2 < best) {
                    second = best;
                    second_centre = best_centre;
                    across = cell;
                    best = d2;
                    best_centre = centre;
                    cell = ivec2(q, r);
                } else if (d2 < second) {
                    second = d2;
                    second_centre = centre;
                    across = ivec2(q, r);
                }
            }
        }
        // The nearest edge is the bisector with the second-nearest centre.
        vec2 apart = best_centre - second_centre;
        float gap = length(apart);
        edge = edge_pixels((second - best) / (2.0 * gap), apart / gap, px, py);
        extent = min(2.0 * across_px, SQRT3 * down_px);
    } else if (shape == 2) {
        int r = int(floor(p.y + 0.5));
        float depth = p.y + 0.5 - float(r);  // 0 at the row's top edge, 1 at its bottom
        int c = int(floor(p.x));
        bool up = ((r + c) & 1) == 0;
        float margin = (up ? depth : 1.0 - depth) - (p.x - float(c));
        if (margin < 0.0) {  // the point is in the next triangle along, which points the other way
            c += 1;
            up = !up;
            margin = -margin;
        }
        cell = ivec2(c, r);
        bool left = p.x < float(c);
        float slanted = edge_pixels(margin, vec2(left ? 1.0 : -1.0, up ? 1.0 : -1.0), px, py);
        // An up triangle's level edge is its base, at the bottom; a down triangle's is on top.
        float level = edge_pixels(up ? 1.0 - depth : depth, vec2(0.0, up ? -1.0 : 1.0), px, py);
        edge = min(slanted, level);
        across = slanted < level ? ivec2(left ? c - 1 : c + 1, r) : ivec2(c, up ? r + 1 : r - 1);
        extent = min(2.0 * across_px, down_px);
    } else {
        cell = ivec2(floor(p));
        across = cell;
        edge = 1e6;  // square cells have no outline, as in 2D
        extent = min(across_px, down_px);
    }
}

// The texel (col, row) of the grid cell a lattice cell shows: the cell itself, or the one a
// wrapped grid wraps it to; (-1, -1) beyond a bounded grid's edges, where there is none.
ivec2 grid_texel(ivec2 cell) {
    if (wrap) {
        return ivec2(wrap_index(cell.x, grid_size.x), wrap_index(cell.y, grid_size.y));
    }
    bool inside = all(greaterThanEqual(cell, ivec2(0))) && all(lessThan(cell, grid_size));
    return inside ? cell : ivec2(-1);
}

// How much of the outline colour a pixel `edge` pixels from its cell's nearest edge gets: a line
// about a pixel wide along every edge of a grid cell at least `min_outline_px` across. An edge
// between two places beyond a bounded grid's edges belongs to no cell, so it is not drawn.
float outline_weight(ivec2 cell, ivec2 across, float edge, float extent) {
    if (grid_texel(cell).x < 0 && grid_texel(across).x < 0) {
        return 0.0;
    }
    float shown = clamp(extent - min_outline_px + 1.0, 0.0, 1.0);
    return shown * (1.0 - smoothstep(0.25, 0.75, edge));
}
"""

FRAGMENT_SHADER = (
    """
#version 330 core

in vec3 v_position;
in vec3 v_normal;
in vec2 v_uv;

out vec4 out_color;

uniform sampler2D cells;
uniform vec3 camera_position;
uniform vec3 line_color;
uniform vec3 outside_color;  // beyond a bounded grid's edges: the view's background
"""
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
    vec3 color = texel.x >= 0 ? texelFetch(cells, texel, 0).rgb : outside_color;
    color = mix(color, line_color, outline_weight(cell, across, edge, extent));

    vec3 normal = normalize(v_normal);
    vec3 to_eye = normalize(camera_position - v_position);
    float shade = 0.4 + 0.6 * abs(dot(normal, to_eye));
    out_color = vec4(color * shade, 1.0);
}
"""
)
