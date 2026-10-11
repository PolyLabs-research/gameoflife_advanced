"""SimulationSession: headless orchestration (§6.8). Owns LatticeGrid, config, RNG, history ring
buffers, metrics, and alerts. Both the GUI and every headless path (autorun, experiments,
discovery evaluations) drive this one class. Zero pygame imports.

Alerts are always evaluated and exposed (`alert`, and each record's `alert`); `auto_pause` only
decides whether a sudden-death alert also pauses the session (§6.7). The interactive app keeps
the default; headless batch runs pass `auto_pause=False` so every planned generation is recorded.
The pause comes when a sudden-death alert is raised, not on every generation it stays raised:
the condition compares against the population `sudden_death_window` generations back, so after
a crash it holds for up to a window of generations, and pausing on each one made every resume
advance a single generation. A later crash, after the alert has cleared, pauses again.

`paused` belongs to whoever set it last. An auto-pause is marked (`paused_by_alert`) until
anything else sets `paused`; `reset()` keeps a pause the user chose, so generation 0 can be
inspected and single-stepped, and lifts one the alert imposed, together with the alert.

`history_alive` starts with the seeded generation-0 total, so it holds one entry more than
`history_records` until both ring buffers are full: sudden death can fire from generation
`sudden_death_window` on, measured against generation 0 (an extinction in the first generation
is reported), and the population is known before the first step. Growth rates (§6.6) measure
each species against its count on the grid just before the step, so generation 1 is measured
against the seeded grid and any edit made between steps is never counted as growth.

Manual cell edits go through `set_cell()`, never straight into the grid arrays, so the
sudden-death window describes the edited grid rather than the pre-edit one: it restarts with the
edited grid as its generation 0, so an extinction in the first generation after an edit is
reported too.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
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
        self,
        config: Config,
        lifeforms: list[Lifeform],
        seed: int | None = None,
        auto_pause: bool = True,
        paused: bool = False,
    ) -> None:
        self.config = config
        self.lifeforms = lifeforms
        self.auto_pause = auto_pause

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
        self._paused = bool(paused)
        self._paused_by_alert = False
        self.alert: Alert | None = None

        history_limit = config.simulation.history_limit
        self.history_alive: deque[int] = deque(maxlen=history_limit)
        self.history_volatility: deque[float] = deque(maxlen=history_limit)
        self.history_records: deque[GenerationRecord] = deque(maxlen=history_limit)
        # The sudden-death window after a manual edit (see `restart_alert_window`):
        # `_alert_horizon` counts the generations stepped since the edit, whose history_alive
        # entries the check may use, or is None when the whole history is usable again;
        # `_alert_baseline` is the edited grid's total, the restarted window's generation 0.
        self._alert_horizon: int | None = None
        self._alert_baseline = 0

        self.randomise_lifeforms()
        self._start_history()

    @classmethod
    def from_config(
        cls,
        config: Config,
        lifeforms: list[Lifeform] | None = None,
        seed: int | None = None,
        auto_pause: bool = True,
        paused: bool = False,
    ) -> SimulationSession:
        chosen = lifeforms if lifeforms is not None else _default_lifeforms(config)
        return cls(config, chosen, seed, auto_pause=auto_pause, paused=paused)

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        # Any outside change (Space, Play, a caller) makes the pause state the user's own.
        self._paused = bool(value)
        self._paused_by_alert = False

    @property
    def paused_by_alert(self) -> bool:
        """True while the session is paused because a sudden-death alert paused it and nothing
        has set `paused` since. A new session meant to carry the user's run state over (the
        app's New and Apply) takes `paused=old.paused and not old.paused_by_alert`, as
        `reset()` does."""
        return self._paused_by_alert

    def spawn_rng(self) -> np.random.Generator:
        """A child Generator for a component that needs its own stream (§6.4)."""
        return self.rng.spawn(1)[0]

    def randomise_lifeforms(self) -> None:
        """Re-seed the grid at the configured density. Mid-run this is an edit of every cell,
        so the sudden-death window restarts as for `set_cell`."""
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
        self.restart_alert_window()

    def _start_history(self) -> None:
        """Generation 0 opens the alive history (the first sudden-death baseline) and the
        alert window spans the whole history again."""
        self.history_alive.append(int(np.count_nonzero(self.grid.occupancy)))
        self._alert_horizon = None

    def reset(self) -> None:
        """Back to generation 0 on a freshly seeded grid, with the histories and the alert
        cleared. A pause the user chose is kept; a pause the sudden-death alert imposed is
        lifted along with the alert."""
        if self._paused_by_alert:
            self.paused = False
        self.generation = 0
        self.alert = None
        self.history_alive.clear()
        self.history_volatility.clear()
        self.history_records.clear()
        self.randomise_lifeforms()
        self._start_history()

    def set_cell(self, index: int, lifeform_id: int) -> bool:
        """Manually set cell `index` to `lifeform_id` (0 = empty) between steps, keeping the
        session consistent: a newly placed cell gets lifespan 1 and a cleared one 0, and the
        sudden-death window restarts, so an edit is never reported (and auto-paused on) as a
        collapse. The next step's growth_rate measures only what the rule did, because growth
        is always measured against the grid as the step finds it. History records and the
        current alert are left as they were — they describe generations that really ran.
        Returns False, changing nothing, when the cell already holds `lifeform_id`. Raises
        IndexError for an index outside the grid, ValueError for an unknown lifeform id."""
        grid = self.grid
        index, lifeform_id = int(index), int(lifeform_id)
        if not 0 <= index < grid.n_cells:
            raise IndexError(f"cell index {index} outside a grid of {grid.n_cells} cells")
        if lifeform_id != 0 and lifeform_id not in {lf.id for lf in self.lifeforms}:
            raise ValueError(f"unknown lifeform id {lifeform_id}")
        if int(grid.occupancy[index]) == lifeform_id:
            return False

        grid.occupancy[index] = lifeform_id
        grid.lifespans[index] = 1 if lifeform_id else 0
        self.restart_alert_window()
        return True

    def restart_alert_window(self) -> None:
        """Start the sudden-death window afresh from the current generation: alive counts
        recorded before now are kept in `history_alive` (the chart reads them) but no longer
        compared against. The edited grid becomes the window's generation 0, as the seeded grid
        is for a new session: its total is taken when the next step starts, after every edit
        made in between, so a population edited down to cells that die in that step is reported
        a full window later, and nothing earlier fires."""
        self._alert_horizon = 0

    def _alert_history(self) -> Sequence[int]:
        """The alive counts sudden death may compare against, oldest first."""
        horizon = self._alert_horizon
        if horizon is None:
            return self.history_alive
        # At most a window's worth of entries, read from the deque's end.
        return [self._alert_baseline, *(self.history_alive[i] for i in range(-horizon, 0))]

    def step(self, force_clusters: bool = False) -> UpdateResult:
        # Growth baselines (§6.6 alive_i(t-1)) are read off the grid the step starts from, so
        # they are right after seeding, reset, set_cell or any other write to the grid.
        previous_alive = {
            lf.id: int(np.count_nonzero(self.grid.occupancy == lf.id)) for lf in self.lifeforms
        }
        if self._alert_horizon == 0:
            # The first step since an edit: the grid it finds opens the restarted window.
            self._alert_baseline = int(np.count_nonzero(self.grid.occupancy))
        result = self.grid.step(self.rng)
        self.generation += 1

        grid_metrics = compute_grid_metrics(result)
        total_alive = grid_metrics.total_alive
        occupancy_fraction = total_alive / max(self.grid.n_cells, 1)

        lifeform_metrics: dict[int, LifeformMetrics] = {
            lf.id: compute_lifeform_metrics(
                lf.id, result, self.grid, total_alive, previous_alive[lf.id]
            )
            for lf in self.lifeforms
        }

        previous_alert = self.alert
        alert = alerts_module.evaluate(
            self._alert_history(),
            total_alive,
            occupancy_fraction,
            grid_metrics.volatility,
            self.config.alerts,
            self.generation,
        )
        self.alert = alert
        raised = (
            alert is not None
            and alert.kind is AlertKind.SUDDEN_DEATH
            and (previous_alert is None or previous_alert.kind is not AlertKind.SUDDEN_DEATH)
        )
        pausing = self.auto_pause and raised and not self._paused
        if pausing:
            self._paused = True
            self._paused_by_alert = True

        # A generation that auto-pauses may be the last one a run() records (§6.6: cluster
        # metrics are always present on a run's final generation).
        stride = max(self.config.metrics.cluster_stride, 1)
        cluster_metrics: dict[int, ClusterMetrics] | None = None
        if force_clusters or pausing or self.generation % stride == 0:
            cluster_metrics = {
                lf.id: compute_cluster_metrics(self.grid, lf.id) for lf in self.lifeforms
            }

        self.history_alive.append(total_alive)
        if self._alert_horizon is not None:
            self._alert_horizon += 1
            if self._alert_horizon >= self.config.alerts.sudden_death_window:
                # The entry a window back now lies after the edit, so the whole history gives
                # the same comparison (config keeps history_limit >= sudden_death_window).
                self._alert_horizon = None
        self.history_volatility.append(grid_metrics.volatility)
        self.history_records.append(
            GenerationRecord(
                self.generation, result, grid_metrics, lifeform_metrics, cluster_metrics, alert
            )
        )

        return result

    def run(self, n: int) -> int:
        """Step up to `n` generations, stopping early once the session is paused, and return
        how many generations were stepped. The last of them carries cluster metrics (§6.6).
        Nothing is accumulated here: a run of any length holds only the bounded
        `history_records`, and a caller wanting every generation steps itself."""
        stepped = 0
        for i in range(n):
            if self._paused:
                break
            self.step(force_clusters=(i == n - 1))
            stepped += 1
        return stepped
