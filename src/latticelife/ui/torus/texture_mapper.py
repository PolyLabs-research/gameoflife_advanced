"""RGB texture mapper for the torus surface, adapted from the legacy
`view/torus/texture_mapper.py`. The legacy version derived colors from
`Lifeform.color_alive`/`color_static`; those attributes don't exist on the new domain-pure
`core.lifeform.Lifeform` (colors live in `ui/render.py`'s palette instead), so this now takes
the same `(2L+1, 3)` palette and reuses `ui.render`'s occupancy-folding scheme. Mesh/camera/
shader math elsewhere in this package is untouched.
"""

from __future__ import annotations

import numpy as np


class GridTextureMapper:
    """Converts grid state to an RGB texture for the torus surface."""

    def __init__(
        self, grid_width: int, grid_height: int, palette: np.ndarray, static_threshold: int = 10
    ) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.palette = palette
        self.static_threshold = static_threshold
        self.texture_data = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    def update(self, occupancy: np.ndarray, lifespans: np.ndarray) -> np.ndarray:
        """Convert grid state to an RGB texture via the shared palette-index scheme."""
        n_species = (self.palette.shape[0] - 1) // 2
        static_mask = lifespans > self.static_threshold
        state = occupancy.astype(np.int32) + n_species * static_mask.astype(np.int32)
        self.texture_data[:] = self.palette[state].reshape(self.grid_height, self.grid_width, 3)
        return self.texture_data
