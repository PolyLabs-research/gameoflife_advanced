"""SimulationSession: headless orchestration (§6.8). Owns LatticeGrid, config, RNG, history ring
buffers, metrics, and alerts. Both the GUI and every headless path (autorun, experiments,
discovery evaluations) drive this one class. Zero pygame imports.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from latticelife.config import Config
from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.results import UpdateResult
from latticelife.core.rules import Rule
from latticelife.session import alerts as alerts_module
from latticelife.session.alerts import Alert, AlertKind
from latticelife.session.metrics import (
    ClusterMetrics,
    GridMetrics,
    LifeformMetrics,
    compute_cluster_metrics,
    compute_grid_metrics,
    compute_lifeform_metrics,
)


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    result: UpdateResult
    grid_metrics: GridMetrics
    lifeform_metrics: dict[int, LifeformMetrics]
    cluster_metrics: dict[int, ClusterMetrics] | None
    alert: Alert | None


def _default_lifeforms(config: Config) -> list[Lifeform]:
    conway = Rule(birth={3}, survive={2, 3})
    return [Lifeform(id=i, rule=conway) for i in range(1, config.simulation.n_lifeforms + 1)]


class SimulationSession:
    def __init__(
        self, config: Config, lifeforms: list[Lifeform], seed: int | None = None
    ) -> None:
        self.config = config
        self.lifeforms = lifeforms

        session_seed = seed if seed is not None else (config.simulation.seed or None)
        self.rng = np.random.default_rng(session_seed)

        self.grid = LatticeGrid(
            lifeforms,
            config.grid.shape,
            config.grid.width,
            config.grid.height,
            config.grid.wrap,
            config.grid.triangle_mode,
            config.simulation.static_threshold,
        )

        self.generation = 0
        self.paused = False
        self.alert: Alert | None = None
        self._previous_alive: dict[int, int] = dict.fromkeys((lf.id for lf in lifeforms), 0)

        history_limit = config.simulation.history_limit
        self.history_alive: deque[int] = deque(maxlen=history_limit)
        self.history_volatility: deque[float] = deque(maxlen=history_limit)
        self.history_records: deque[GenerationRecord] = deque(maxlen=history_limit)

        self.randomise_lifeforms()

    @classmethod
    def from_config(
        cls, config: Config, lifeforms: list[Lifeform] | None = None, seed: int | None = None
    ) -> SimulationSession:
        return cls(config, lifeforms if lifeforms is not None else _default_lifeforms(config), seed)

    def spawn_rng(self) -> np.random.Generator:
        """A child Generator for a component that needs its own stream (§6.4)."""
        return self.rng.spawn(1)[0]

    def randomise_lifeforms(self) -> None:
        init_rng = self.spawn_rng()
        n = self.grid.n_cells
        num_alive = int(n * self.config.simulation.initial_alive_percentage)

        self.grid.occupancy[:] = 0
        self.grid.lifespans[:] = 0

        if num_alive > 0 and self.lifeforms:
            indices = init_rng.choice(n, size=num_alive, replace=False)
            ids = np.array([lf.id for lf in self.lifeforms], dtype=self.grid.occupancy.dtype)
            chosen = ids[init_rng.integers(0, len(ids), size=num_alive)]
            self.grid.occupancy[indices] = chosen
            self.grid.lifespans[indices] = 1

    def reset(self) -> None:
        self.generation = 0
        self.paused = False
        self.alert = None
        self._previous_alive = dict.fromkeys((lf.id for lf in self.lifeforms), 0)
        self.history_alive.clear()
        self.history_volatility.clear()
        self.history_records.clear()
        self.randomise_lifeforms()

    def step(self, force_clusters: bool = False) -> UpdateResult:
        result = self.grid.step(self.rng)
        self.generation += 1

        grid_metrics = compute_grid_metrics(result)
        total_alive = grid_metrics.total_alive
        occupancy_fraction = total_alive / max(self.grid.n_cells, 1)

        lifeform_metrics: dict[int, LifeformMetrics] = {}
        for lf in self.lifeforms:
            previous_alive = self._previous_alive.get(lf.id, 0)
            lifeform_metrics[lf.id] = compute_lifeform_metrics(
                lf.id, result, self.grid, total_alive, previous_alive
            )
            self._previous_alive[lf.id] = int(result.alive[lf.id])

        stride = max(self.config.metrics.cluster_stride, 1)
        cluster_metrics: dict[int, ClusterMetrics] | None = None
        if force_clusters or self.generation % stride == 0:
            cluster_metrics = {
                lf.id: compute_cluster_metrics(self.grid, lf.id) for lf in self.lifeforms
            }

        alert = alerts_module.evaluate(
            self.history_alive,
            total_alive,
            occupancy_fraction,
            grid_metrics.volatility,
            self.config.alerts,
            self.generation,
        )
        self.alert = alert
        if alert is not None and alert.kind is AlertKind.SUDDEN_DEATH:
            self.paused = True

        self.history_alive.append(total_alive)
        self.history_volatility.append(grid_metrics.volatility)
        self.history_records.append(
            GenerationRecord(
                self.generation, result, grid_metrics, lifeform_metrics, cluster_metrics, alert
            )
        )

        return result

    def run(self, n: int) -> list[UpdateResult]:
        results: list[UpdateResult] = []
        for i in range(n):
            if self.paused:
                break
            results.append(self.step(force_clusters=(i == n - 1)))
        return results
