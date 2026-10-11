# Adjacency

Geometry is data: every lattice shape is expressed as an adjacency structure built by
`core.geometries.build_adjacency(shape, width, height, wrap, triangle_mode) -> np.ndarray[int32]`
of shape `(n_cells, max_degree)`, padded with `-1`. Cell index = `r * width + c`. There is exactly
one update rule (`core.lattice`); it consumes this adjacency uniformly regardless of shape.

## Degree table (wrap, all cells)

| shape    | mode           | degree |
|----------|----------------|--------|
| square   | —              | 8 (Moore) |
| hexagon  | odd-q offset   | 6 |
| triangle | edge           | 3 |
| triangle | edge+vertex    | 12 |

`wrap=True` uses modulo indexing; `wrap=False` omits out-of-range neighbors (padded with `-1`,
so non-wrapped edge/corner cells have degree < the table above). Square gets real clamp support
for the first time in this rebuild — the legacy `square_grid.py` hardcoded toroidal wrap via
`convolve2d(..., boundary='wrap')` regardless of the `wrap` parameter (defect D-6).

Symmetry invariant, property-tested per shape and both wrap modes: for all `i, j`,
`j ∈ N(i) ⇔ i ∈ N(j)`.

## Sizes that can wrap

`validate_dimensions(shape, width, height, wrap, triangle_mode)` rejects (ValueError) any size
whose wrapped adjacency would be wrong; `build_adjacency` calls it, and so do config loading and
the settings panel's Apply. Odd sizes are refused, never rounded.

- Hex offsets depend on column parity and triangle orientation on (row + column) parity, so a
  wrapped hexagon grid needs an even width and a wrapped triangle grid an even width and height.
  With an odd size the seam joins cells of the same parity: one-way links (breaking symmetry),
  or, for edge-mode triangles, two same-orientation triangles counted as edge neighbours.
- Along each axis a wrapped grid must be larger than the stencil's span (offsets plus the cell
  itself), or modulo indexing folds two offsets onto one cell: a cell becomes its own neighbour
  or one neighbour is counted twice. Minimum wrapped sizes (width x height, from
  `min_wrap_size`): square 3 x 3, hexagon 4 x 3, triangle edge 4 x 2, triangle edge+vertex 6 x 4.
- Clamped (`wrap=False`) grids accept any size of at least 1 x 1.

A test sweeps every size up to 12 x 12 per shape and checks the rejected sizes are exactly the
ones whose unvalidated adjacency has a self-neighbour, a repeated neighbour, a one-way link or a
same-orientation triangle edge.

## Legacy baseline (carried over from `DEV_NOTES.md`, for reference)

- Neighbor counts:
  - Square: 8 (Moore) via 3×3 convolution kernel with center removed in `square_grid.py:update`.
  - Hex: 6 offsets in `hexagon_grid_numpy.py:_create_neighbor_map` (odd-q layout).
  - Triangle: configurable; default `edge+vertex` gives 12 neighbors per cell (orientation-consistent
    sets) in `triangle_grid_numpy.py:_build_neighbor_offsets`; `edge` mode retains 3-neighbor
    edge-only adjacency.

- Hardcoded N=8 assumptions:
  - Previously in `game.py:randomise_lifeforms` and rule logging; replaced with `get_max_neighbors`.
  - `rule_discovery.py` used `get_max_neighbors` for genome length and seeding (moot in this
    rebuild — the gene-string codec is discarded; discovery operates on `Rule` objects directly).

- Boundary conditions (legacy):
  - Square: toroidal wrap only, via `convolve2d(..., boundary='wrap')` — `wrap=False` was silently
    ignored (D-6).
  - Hex: configurable; wrap by default (modulo indexing in neighbor map), clamp when `wrap=False`.
  - Triangle: configurable; wrap by default (modulo indexing in neighbor map), clamp when `wrap=False`.

## Neighbor counting (this rebuild)

Generalizes the legacy hex/tri gather to every shape: pad the flat grid with one sentinel slot,
remap `-1` adjacency indices to the sentinel, `padded[adjacency]` then reduce. `convolve2d` and
scipy are gone (§6.1). Connected-component clustering (§6.6) reuses the same adjacency array, as
the edge list of a vectorised union-find (`session.metrics.component_roots`: hook roots, then
pointer-jump), so it is geometry-correct and wrap-correct by construction for all three shapes —
`scipy.ndimage` is not used since its 4/8-connectivity has no hex/tri notion and no torus
support. Union-find replaced the spec's min-label propagation, which needed one full-grid pass
per cell along a cluster's longest internal path (seconds per call on large winding clusters).
