# Changelog

## The stable interface

Other projects drive the experiments as a library (the law-discovery loop, `lattice-lab`, does).
These names, their parameters, and the fields of the records and rows they return are kept
stable; `tests/experiments/test_stable_api.py` pins them, and any change to them gets a note
here, under the version that makes it.

- `experiments.harness`: `single_species_session`, `torus_config`, `parse_grid`,
  `grid_dimensions`, `grid_sort_key`, `read_jsonl`, `append_jsonl`, `resume_keys`, `map_tasks`
- `experiments.sweeps`: `SweepTask`, `SweepRecord`, `sweep_point`, `sweep_session`,
  `run_sweep`, `survival_table`, `critical_densities`, `RECORD_KEY`
- `experiments.census`: `CensusTask`, `RuleRecord`, `classify_rule`, `classify_session`,
  `run_census`, `RECORD_KEY`
- `experiments.damage`: `DamageTask`, `DamageRecord`, `damage_point`, `run_damage`,
  `RECORD_KEY`
- `experiments.observables`: `median_transient`, `survival`, `density_series`,
  `damage_spread`, `event_times`, `settling_time`, `no_repeat_before`, `first_repeat_time`,
  `extinction_outcome`, `kaplan_meier_median`, `damage_curves`, `object_sizes`,
  `bootstrap_interval`, `wilson_interval`, and the rows `TransientRow`, `SurvivalRow`,
  `DensityRow`, `DamageRow`, `EventTimeRow`, `CurveRow`, `ObjectSizeRow`
- `experiments.cost`: `CostHint`, `cost_hint`, `task_cost`, `measure_seconds_per_step`
- `experiments.series`: `count_series`
- `experiments.local`: `LocalTask`, `LocalRecord`, `local_point`, `run_local`, `RECORD_KEY`
- `experiments.competition`: `CompetitionTask`, `CompetitionRecord`, `competition_point`,
  `run_competition`, `species_start`, `pair_counts`, `RECORD_KEY`
- `experiments.coarse`: `coarse_grid`, `sample_cells`, `block_cells`, `Projection`, `project`,
  `block_counts`, `leading_species`, `World`, `Diagram`, `Closure`, `closure`, `uniform_start`,
  `ClosureTask`, `ClosureRecord`, `closure_point`, `run_closure`, `RECORD_KEY`

Record files are append-only JSONL. A new field may be added to a record; an existing field is
never renamed, retyped or given a new meaning. Readers treat a missing new field as its
documented default (see `observables.no_repeat_before`).

## Unreleased

- **Several species without the app** (`experiments.competition`): the first experiment that
  runs more than one species. `species_start` seeds a wrapped grid with a chosen density and
  chosen shares (every cell independently); `competition_point` runs it under a named
  tie-break and records, at the start, every `sample_every`-th generation and the end, each
  species' live cells, the adjacent pairs of two species and of one (how mixed they are),
  each species' births and the cells the tie-break settled since the sample before, and the
  generation each species died out. `run_competition` writes resumable JSONL records. A run is
  reproducible from its task; under `standoff` or `strongest` it is a function of its start.
  New stable names: `CompetitionTask`, `CompetitionRecord`, `competition_point`,
  `run_competition`, `species_start`, `pair_counts`.
- **The birth tie-break is a setting** (`core.lattice.TieBreak`; `simulation.tie_break`,
  `tie_break_radius`, `tie_break_power`; `ui --tie-break`, `autorun --tie-break`). With several
  species a dead cell can be due a birth of more than one. `random`, the default, is the rule
  as it always was, draw for draw. `standoff` leaves the cell empty. `strongest` gives it to
  the eligible species with the most cells within `tie_break_radius` lattice steps, and to
  nobody if the top is shared. `weighted` draws with chances in proportion to that count
  raised to `tie_break_power`: power 0 is `random` draw for draw. With `standoff` or
  `strongest` the update never draws, so the next grid is a function of the grid whatever
  the number of species. No kind favours a species by its id, and one species runs the same
  under all four. The three config keys are optional: a config without them keeps `random`.
- **Contested cells are counted**: `UpdateResult.contested` and `GridMetrics.contested`, the
  dead cells more than one species was eligible for in a generation, under every tie-break.
- **Recorder schema v3.** A new database records each session's tie-break
  (`session_metadata.tie_break`, `tie_break_radius`, `tie_break_power`) and each generation's
  contested cells (`generation_metrics.contested`). A schema v2 database is not converted: it
  is recorded into as before, and since every session in it drew at random and it has no
  column to say otherwise, it refuses a session with another tie-break (record that one into a
  new file). `observables.density_series` reads both.
- **Zooming out** (`experiments.coarse`): block projections of a wrapped square or hexagon
  grid, and the closure test. A block of b sends a W x H grid to the (W / b) x (H / b) grid of
  the same shape (rhombi in axial coordinates on the hexagon, so a supercell's neighbours are
  b steps away on the fine grid). `project` reads a coarse configuration off a one-species
  one (a sampled cell, or a block's parity); `block_counts` and `leading_species` read a grid
  of several species. `closure` runs the fine world then projects, or projects then runs a
  coarse world, and counts the supercells where the two differ. `closure_point` and
  `run_closure` give resumable JSONL records, each naming its start by seed (`uniform_start`:
  every configuration equally likely) and carrying the sha256 of the start and of the fine
  grid it ended on. A parity rule is its own coarse rule at every power of two, which the
  tests use as the known answer; lattice-lab's `examples/scale-up-linear` registered and
  confirmed it, and five of its records are reproduced in the tests to the last bit.
- **Dilated adjacency** (`core.geometries.build_adjacency(..., dilation=k)`): the same lattice
  with every neighbour k lattice steps away, on a wrapped square or hexagon grid. Where a far
  neighbour wraps onto another it is listed twice and counted twice. The default, 1, is the
  adjacency as before.
- **Wrapped grids are built without per-cell loops.** `build_adjacency` now fills a wrapped
  grid's array in whole-array operations: the same array as before, cell for cell and in the
  same neighbour order (the tests compare the two on every shape), about fifteen times faster
  at 512 x 512, and without the per-cell Python lists, which a grid of several thousand cells a
  side would need gigabytes for. Clamped grids are built as before.
- **Local transients** (`experiments.local`): when each b x b block of the grid enters the
  soup's final cycle. A second pass compares every state with the one a period earlier; a
  block's local transient is one past the last generation it differed. The largest is the
  soup's own transient, checked on every run. For extreme-value accounts of settling times,
  which treat the global transient as the largest of many local ones. New stable names:
  `LocalTask`, `LocalRecord`, `local_point`, `run_local`.
- **Counts over time** (`experiments.series.count_series`): live cells, cumulative births and
  deaths, and live and dead cells by live-neighbour count at every generation of one run, for a
  search for conserved quantities. Four bookkeeping identities hold by construction and are
  tested.
- **Replicate curves and size samples** (`observables.damage_curves`, `observables.object_sizes`;
  `latticelife observe damage_curves|object_sizes`): every seed's whole damage curve, for a law
  over time with seed-to-seed error bars, and every object's population and each distinct
  object's count, for a distribution fit. New stable names: `damage_curves`, `object_sizes`,
  `CurveRow`, `ObjectSizeRow`.
- **Event times** (`observables.event_times`, `latticelife observe first_repeat|extinction`):
  each seed's time to its first repeated state (transient + period, the quantity a cap really
  bounds) or to extinction, exact or a lower bound, for fitting with a censored likelihood. Runs
  that settle on a living cycle are counted apart as never dying. Rows give a Kaplan-Meier
  median. New stable names: `event_times`, `first_repeat_time`, `extinction_outcome`,
  `kaplan_meier_median`, `EventTimeRow`.
- **Observables** (`experiments.observables`, `latticelife observe`): records become numbers
  with uncertainty and stated limits. `median_transient` (median settling time per torus, a
  seeded bootstrap 95% interval, seeds and unsettled seeds), `survival` (share alive per size
  and density, Wilson interval), `density_series` (mean density over time from an `autorun`
  database) and `damage_spread`. numpy only.
- **What "unsettled" means.** Census, sweep and soup records gain `no_repeat_before`: set only
  when no state repeated within the cap, it is the last generation reached, and bounds
  transient + period together, not the transient alone. A settling-time row with such a seed is
  not `complete`, has no `median`, and names the seeds to rerun with a larger cap. Older files
  without the field are read from `stopped_at` (sweep) or `steps - 1` (census).
- **Rectangular tori.** `--grid` and every experiment function take `L` or `WxH` (`24x26`).
  Records name a square torus by its int side as before and a rectangle by `"WxH"`;
  `survival_table` keys follow suit and sort by cell count. The wrap rules still apply:
  hexagon tori need an even width, triangle tori an even width and height.
- **Damage spreading as a command**: `latticelife damage` (`experiments.damage`), one JSONL
  record per run with the number of differing cells at every generation, exact `healed_at`,
  `--flips` and `--warmup`.
- **Cost hints** (`experiments.cost`): generations x cells per task, and seconds from a
  measured time per step.
