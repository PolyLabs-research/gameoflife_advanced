"""UpdateResult — the sole return type from LatticeGrid.step(). No 8-tuples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UpdateResult:
    births: np.ndarray  # (L+1,) int64, index by species id, [0] unused
    deaths: np.ndarray  # (L+1,)
    kills: np.ndarray  # (L+1,)
    alive: np.ndarray  # (L+1,)
    static: np.ndarray  # (L+1,) lifespan > static_threshold
    changed_cells: int
