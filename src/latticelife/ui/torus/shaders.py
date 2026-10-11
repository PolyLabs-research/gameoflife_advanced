"""GLSL 3.30 shaders for the 3D torus view (§6.14).

The grid reaches the GPU as a width x height RGB texture, one texel per cell (texel (col, row)
holds cell (row, col)'s colour, from the same palette and alive/static fold as the 2D view). The
fragment shader decides which cell a surface point shows with the grid's own geometry, so
hexagons and triangles keep their shape and neighbours on the torus:

* The surface coordinates uv (`geometry.py`) become a point on the shape's lattice, scaled so
  one trip around the ring covers the grid's width and one trip around the tube its height.
  u = 0, v = 0 (the outer equator facing a camera at azimuth 0) is the middle of the grid;
  columns run to the right and rows run down the outer wall, as in the 2D view.
* square: one unit per column and row. hexagon: odd-q hexagons of size 1 (centres 1.5 apart
  across columns, sqrt 3 down a column, odd columns half a cell lower), the cell is the nearest
  centre, as `render._hex_at`. triangle: half a cell per column and one per row; the cell is the
  up or down triangle that contains the point, as `render._triangle_at`.
* Cell indices wrap around the grid, so a wrapped grid's opposite edges meet seamlessly. A
  bounded grid (wrap off) is drawn on the same closed surface; its edges simply meet at the seam.

Hexagon and triangle cells get a thin outline once they are at least `min_outline_px` pixels on
screen, as in 2D. Lighting is a headlight (light from the camera), so the side of the tube that
faces the viewer is always lit, whatever the camera's elevation.

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
uniform ivec2 grid_size;  // (width, height) in cells
uniform int shape;        // SHAPE_CODES

const float SQRT3 = 1.7320508075688772;

// Surface coordinates -> a point on the shape's (unbounded) lattice, x across columns.
vec2 lattice_point(vec2 uv) {
    vec2 front = vec2(uv.x + 0.5, 0.5 - uv.y);
    vec2 size = vec2(grid_size);
    if (shape == 1) {
        return front * vec2(1.5 * size.x, SQRT3 * size.y);
    }
    return front * size;
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

// The cell (col, row) on the unbounded lattice at point p, the distance in pixels from p to the
// cell's nearest edge, and the cell's smaller on-screen extent in pixels.
void find_cell(vec2 p, vec2 px, vec2 py, out ivec2 cell, out float edge, out float extent) {
    float across = 1.0 / max(length(vec2(px.x, py.x)), 1e-6);  // pixels per lattice x unit
    float down = 1.0 / max(length(vec2(px.y, py.y)), 1e-6);    // pixels per lattice y unit
    if (shape == 1) {
        int q0 = int(round(p.x / 1.5));
        float best = 1e30;
        float second = 1e30;
        vec2 best_centre = vec2(0.0);
        vec2 second_centre = vec2(0.0);
        cell = ivec2(q0, 0);
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
                    best = d2;
                    best_centre = centre;
                    cell = ivec2(q, r);
                } else if (d2 < second) {
                    second = d2;
                    second_centre = centre;
                }
            }
        }
        // The nearest edge is the bisector with the second-nearest centre.
        vec2 apart = best_centre - second_centre;
        float gap = length(apart);
        edge = edge_pixels((second - best) / (2.0 * gap), apart / gap, px, py);
        extent = min(2.0 * across, SQRT3 * down);
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
        float side = p.x < float(c) ? 1.0 : -1.0;
        float slanted = edge_pixels(margin, vec2(side, up ? 1.0 : -1.0), px, py);
        float level = edge_pixels(up ? 1.0 - depth : depth, vec2(0.0, up ? -1.0 : 1.0), px, py);
        edge = min(slanted, level);
        extent = min(2.0 * across, down);
    } else {
        cell = ivec2(floor(p));
        edge = 1e6;  // square cells have no outline, as in 2D
        extent = min(across, down);
    }
}

ivec2 grid_texel(ivec2 cell) {
    return ivec2(wrap_index(cell.x, grid_size.x), wrap_index(cell.y, grid_size.y));
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
uniform float min_outline_px;
"""
    + CELL_LOOKUP
    + """
void main() {
    vec2 p = lattice_point(v_uv);
    ivec2 cell;
    float edge;
    float extent;
    find_cell(p, dFdx(p), dFdy(p), cell, edge, extent);
    vec3 color = texelFetch(cells, grid_texel(cell), 0).rgb;

    float outline = clamp(extent - min_outline_px + 1.0, 0.0, 1.0);
    color = mix(color, line_color, outline * (1.0 - smoothstep(0.25, 0.75, edge)));

    vec3 normal = normalize(v_normal);
    vec3 to_eye = normalize(camera_position - v_position);
    float shade = 0.4 + 0.6 * abs(dot(normal, to_eye));
    out_color = vec4(color * shade, 1.0);
}
"""
)
