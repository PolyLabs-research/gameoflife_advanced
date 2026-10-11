"""Headless experiment runner (§6.12), rebuilt on SimulationSession with injected RNG. The
legacy `experiments/runner.py` reseeded the global stdlib and numpy RNGs at the top of
`make_grid()` and `run_damage_spreading()` — global state mutated from whatever thread happened
to run an experiment, which is exactly the D-3 contamination this rebuild closes. Timeseries and
freeze-detection semantics, and the damage-spreading method (perturbed-twin Hamming trajectory),
are preserved exactly; only the RNG plumbing changed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from latticelife.core.lattice import clone_grid
from latticelife.core.lifeform import Lifeform
from latticelife.experiments.criticality import BURN_IN_FRAC
from latticelife.session.session import SimulationSession


@dataclass(frozen=True)
class TimeseriesResult:
    rho: np.ndarray
    activity: np.ndarray
    frozen_at: int | None


def run_timeseries(
    session: SimulationSession,
    lifeform_id: int,
    steps: int,
    burn_in_frac: float = BURN_IN_FRAC,
    epsilon: float = 1e-6,
    freeze_window: int = 100,
) -> TimeseriesResult:
    """Run `steps` generations forward from `session`'s current state, driven by
    `session.step()` (session owns the RNG — no seeding here). rho/activity/freeze-detection
    semantics match the legacy `run_timeseries` exactly: quiet steps are counted from
    int(steps * burn_in_frac) on, so a freeze is only detectable when
    int(steps * burn_in_frac) + freeze_window <= steps (`discover` refuses a config that breaks
    this: `config.validate_discovery_config`)."""
    total_cells = session.grid.n_cells
    rho = np.zeros(steps, dtype=float)
    activity = np.zeros(steps, dtype=float)
    frozen_at: int | None = None
    quiet_streak = 0
    burn_in = int(steps * burn_in_frac)

    rho[0] = np.count_nonzero(session.grid.occupancy == lifeform_id) / total_cells

    for step in range(1, steps):
        session.step()
        rho[step] = np.count_nonzero(session.grid.occupancy == lifeform_id) / total_cells
        activity[step] = abs(rho[step] - rho[step - 1])

        if step >= burn_in and activity[step] <= epsilon:
            quiet_streak += 1
            if quiet_streak >= freeze_window and frozen_at is None:
                frozen_at = step
        else:
            quiet_streak = 0

    return TimeseriesResult(rho=rho, activity=activity, frozen_at=frozen_at)


def run_damage_spreading(
    session: SimulationSession,
    steps: int,
    rng: np.random.Generator,
    flip_cells: int = 1,
) -> np.ndarray:
    """Classic damage spreading: evolve two copies of `session`'s current grid, one perturbed
    by `flip_cells` random cell flips (an empty cell becomes species `session.lifeforms[0]`, an
    occupied one becomes empty). Returns H(t) = mean disagreement over time. The twins share
    `session.grid`'s adjacency array read-only (`core.lattice.clone_grid`) — only
    occupancy/lifespans are cloned.

    Common random numbers: both twins step with identical copies of one stream spawned from
    `rng`, so the only difference between them is the perturbation. `LatticeGrid.step` draws
    the same amount of randomness whatever the grid state, so the copies stay in lockstep and
    multi-species tie-breaks resolve identically wherever the twins agree; with independent
    streams, unperturbed two-species twins drifted apart to H ~ 0.26 on tie-break noise alone.
    """
    if not session.lifeforms:
        raise ValueError("damage spreading needs at least one species")
    grid_a = session.grid
    grid_b = clone_grid(grid_a)
    lifeform: Lifeform = session.lifeforms[0]

    perturb_rng, step_rng_a = rng.spawn(2)
    step_rng_b = copy.deepcopy(step_rng_a)

    total = grid_b.n_cells
    flip_n = min(flip_cells, total)
    flat_indices = perturb_rng.choice(total, size=flip_n, replace=False)
    for idx in flat_indices:
        idx = int(idx)
        if grid_b.occupancy[idx] == 0:
            grid_b.occupancy[idx] = lifeform.id
            grid_b.lifespans[idx] = 1
        else:
            grid_b.occupancy[idx] = 0
            grid_b.lifespans[idx] = 0

    h = np.zeros(steps, dtype=float)
    h[0] = np.mean(grid_a.occupancy != grid_b.occupancy)

    for t in range(1, steps):
        grid_a.step(step_rng_a)
        grid_b.step(step_rng_b)
        h[t] = np.mean(grid_a.occupancy != grid_b.occupancy)

    return h
