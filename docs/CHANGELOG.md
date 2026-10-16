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

Record files are append-only JSONL. A new field may be added to a record; an existing field is
never renamed, retyped or given a new meaning. Readers treat a missing new field as its
documented default (see `observables.no_repeat_before`).

## Unreleased

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
