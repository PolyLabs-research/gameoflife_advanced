# Mathematics experiments

Four headless commands turn the simulator into an instrument for exact, exhaustive results on
the non-square lattices, where prior work is thin, and `observe` turns their records into
numbers with uncertainty. Each runs one species on wrapped tori, so the
dynamics are deterministic given the seed and every period reported is exact (a state seen
twice), never inferred from a density trace.

```bash
uv run python -m latticelife.cli census --shape hexagon --rules dual --grid 32 --steps 500
uv run python -m latticelife.cli sweep  --shape hexagon --rule B2/S34 --grid 16 32 64 --seeds 20
uv run python -m latticelife.cli soup   --shape hexagon --rule B2/S34 --grid 64 --seeds 50
uv run python -m latticelife.cli damage --shape hexagon --rule B2/S34 --grid 32 --seeds 20
uv run python -m latticelife.cli observe transient out/sweep-hexagon-B2S34.jsonl
```

`--grid` takes `L` for a square torus or `WxH` for a rectangle (`--grid 24x26`), so sizes between
squares can be sampled where a quantity changes fast. Records name a square torus by its side
(`32`) and a rectangle by `"24x26"`. The wrap rules still apply: a hexagon torus needs an even
width, a triangle torus an even width and height.

All four append one JSON record per task to a JSONL file as the task finishes (default
`out/<command>-<shape>[-<rule>].jsonl` under the config's project root, or `--out PATH`), skip
the tasks already in that file when started again, report progress on stderr every 100 records,
and print a JSON summary on stdout. Ctrl+C, SIGTERM or SIGHUP stop a run with everything finished
so far written (exit status 130). `--workers N` runs tasks in N processes; records then arrive in
completion order, which changes nothing about their content. `--config` supplies the triangle
neighbourhood (`grid.triangle_mode`), the static threshold and alert settings a session needs,
and `simulation.seed` for `--rules random:N`; the grid shape, size and wrap come from the
command itself. Argument problems that depend on each other (a count above the lattice's degree,
a torus size the shape cannot wrap) are one line on stderr with exit status 2.

The modules behind the commands (`latticelife.experiments.census`, `sweeps`, `objects`,
`damage`, `observables`, `cost`, `periodicity`, and `latticelife.core.translations`) are
importable for scripting and notebooks, with the same functions the commands call. The ones
other projects rely on are listed, and kept stable, in [CHANGELOG.md](CHANGELOG.md).

### When a run does not repeat

A run that reaches its cap without a repeated state records `no_repeat_before`: the last
generation it reached, with no state among generations 0..`no_repeat_before` seen twice. That
bounds the transient and the period *together* (transient + period > `no_repeat_before`), not
the transient alone: the run may have settled long ago onto a cycle longer than the cap. So
"never repeated within the cap" is not "a long transient", and a settling time measured on such a
run is not a measurement. Settled records leave the field null. It is set by `census`
(`outcome: unsettled`), `sweep` (a run neither repeated nor absorbed into the empty grid) and
`soup` (the soup itself).

## Rule spaces

An outer-totalistic rule is a birth set and a survival set, each any subset of the neighbour
counts 0..degree, written `B3/S23` (see the README for counts of 10 and above).

| lattice                 | degree | rules            | `census --rules all` |
|-------------------------|--------|------------------|----------------------|
| hexagon                 | 6      | 2^14 = 16,384    | yes                  |
| triangle, edge          | 3      | 2^8 = 256        | yes                  |
| square (Moore)          | 8      | 2^18 = 262,144   | yes, but long        |
| triangle, edge+vertex   | 12     | 2^26 = 67 M      | refused: sample      |

Every rule has a **dual**, its complement under swapping alive and dead (`Rule.complement`):
`B' = {n-k : k not in S}`, `S' = {n-k : k not in B}`. Running the dual on the complemented grid
keeps the grids complementary for ever, so a census of one rule per pair (`--rules dual`) covers
the whole space at half the cost; each record names its `dual`.

## `census`: terminal behaviour of rules

For each (rule, torus size, seed): a soup at `--density` is run for `--steps` generations with
cycle detection and classified.

| outcome     | meaning                                                              |
|-------------|----------------------------------------------------------------------|
| `extinct`   | no live cell at the end                                              |
| `fixed`     | a state repeated after one step (a still configuration)              |
| `periodic`  | a state repeated after more than one step: `period` and `transient` are exact |
| `unsettled` | no state repeated within `steps`: chaotic, or a transient longer than the run |

A torus-level period is the least common multiple of everything on the torus: two blinkers and
a period-3 oscillator give 6, and a spaceship adds the time it takes to come round the torus.
Fields: `rho_final`, `rho_min`, `rho_max`, `activity_final` are over the last `--tail`
generations (of the periodically extended series when a cycle was found); `changed_final` is the
fraction of cells changing per step over the last stepped generations, which tells a chaotic
rule with a steady density from a quiet one, as `activity_final` (|delta rho|) cannot;
`cluster_count` and
`max_cluster_size` are the connected components of live cells at the end; `frozen_at` is the
runner's density-based freeze detection, kept for comparison with discovery.

Two quantities come from the rule alone. `langton_lambda` is the fraction of the 2^(n+1)
(own state, count) configurations, weighted by the C(n, k) neighbourhoods having count k, that
map to a live cell. `mean_field` lists the stable fixed points of the mean-field density map
`rho -> (1-rho) P[Bin(n,rho) in B] + rho P[Bin(n,rho) in S]`, which knows only the degree; its
disagreement with `rho_final` measures how much the spatial arrangement matters for that rule
(rules that freeze are invisible to it, since it assumes the grid is reshuffled every step).

`--rules` takes `all`, `dual`, `random:N` or a list of B/S strings. Several `--grid` sizes in
one run give the period spectrum as a function of torus size.

## `sweep`: survival against initial density

For one rule: at each (torus size, initial density, seed), whether a soup is still alive at
generation `steps - 1`. A run stops as soon as its fate is settled: when the grid is empty and
the empty grid is absorbing (0 not in B), or when a state repeats and the population at the last
generation can be read off the cycle. Records carry `survived`, `extinction_step`, `rho_final`,
`period`, `transient` and `stopped_at`.

`--densities` takes numbers and `LO:HI:STEP` ranges. The summary lists, per torus size, the
densities at which P(survive) crosses 1/2 over every record in the file for that rule and step
count; a rule can have two (too sparse to start, too dense to survive overcrowding).
`sweeps.survival_table`, `sweeps.critical_densities` and `sweeps.fit_finite_size` (a fit of
`rho*(L) = rho_inf + a L^(-b)` across sizes) are the analysis helpers.

## `soup`: the objects soups leave behind

For one rule: each soup runs for `--steps` generations (stopping early if the whole torus repeats
exactly), its live cells are split into components (cells within `--radius` lattice steps of
each other belong together), and each component is put alone on a torus of its own and run until
it repeats up to translation. One record per soup: the soup's own period and transient, every
object found, and how many components were not objects, by reason.

| kind         | meaning                                                   |
|--------------|-----------------------------------------------------------|
| `still`      | period 1, no displacement                                 |
| `oscillator` | period above 1, no displacement                           |
| `spaceship`  | returns displaced: speed is `displacement` per `period`   |
| `died`       | nothing left                                              |
| `unsettled`  | no repeat within `--max-period` generations               |
| `too_large`  | grew to span half its torus (self-interaction across the seam); a larger `--margin` gives more room |

An object's `key` is its identity: the cells of its smallest phase relative to one of them, the
smallest such list over every choice of anchor, so two objects have the same key exactly when
one is a translate of the other, whatever torus or soup they came from. Rotations and
reflections are not identified (a glider heading each way is its own entry): the lattices have
different point groups, and relating objects under them is analysis, not census. `population`
and `max_population` are the smallest and largest phase. The summary lists the ten most common
objects over every record in the file for that rule and torus size; `objects.catalogue` lists
them all.

Two objects closer than `--radius` are harvested as one component and identified together, as a
compound with the least common multiple of their periods; an object with internal gaps wider
than the radius is split, and each part identified alone (where it usually dies). Radius 2
keeps Life's pulsar whole and separates most of its ash.

### Translations and keys

`core.translations.TorusTranslations` is the translation group of a wrapped lattice acting on
flat cell indices. A translation is named by where it sends the cell at row 0, column 0, reduced
modulo the torus. On the hexagon lattice (odd-q offset) a one-column move from an even column
keeps the row while the same move from an odd column also goes one row down, so the name is the
move as column 0 sees it and the arithmetic runs through axial coordinates; two one-column moves
are the name `(1, 2)`, not `(0, 2)`. On the triangle lattice only moves with `dr + dc` even are
translations: up and down triangles are not translates of each other. The test suite checks that
every translation is an automorphism of the adjacency array for every shape over a range of
sizes, which is the property the object census relies on.

## `damage`: how far one flipped cell's effect spreads

For one rule: at each (torus size, initial density, seed), a soup runs `--warmup` generations,
is copied, `--flips` random cells of the copy are flipped, and both twins are stepped for
`--steps` generations side by side (the method discovery scores rules with). Each record holds
`differing`, the number of cells in which the twins differ at every generation, its `final` and
`peak` fractions, and `healed_at`, the first generation at which the twins agree again: they then
agree for ever, since single-species dynamics are deterministic. The summary gives, per torus
size, the mean final damage and the share of runs that healed.

## `observe`: numbers with uncertainty

`latticelife observe QUANTITY FILE... [--rule R]` prints one JSON row per setting
(`experiments.observables`):

| quantity    | from          | per                     | value and interval                         |
|-------------|---------------|-------------------------|--------------------------------------------|
| `transient` | census, sweep | rule, torus, density    | median settling time, bootstrap 95%        |
| `survival`  | sweep         | + steps                 | share alive at `steps - 1`, Wilson 95%     |
| `damage`    | damage        | + steps, flips, warmup  | mean final damage, bootstrap 95%           |
| `first_repeat` | census, sweep | rule, torus, density | each seed's time, exact or a bound; Kaplan-Meier median |
| `extinction`   | sweep         | rule, torus, density | the same, and the seeds that can never die |
| `damage_curves` | damage       | + steps, flips, warmup | every seed's damage curve (replicates)   |
| `object_sizes`  | soup         | rule, torus, density, steps | every object's population; each object's count |

A seed's settling time is its exact transient (for a soup that dies on a rule with 0 not in B,
the generation it died). Records of one seed at different caps are merged: runs are
deterministic, so a seed that settled under any cap is settled, and two settled records that
disagree are an error. A `transient` row with a seed that never repeated is not `complete`: its
`median` is null, `median_at_least` and `median_at_most` bound what the median could be, and
`unsettled_seeds` lists the seeds to rerun with a larger `--steps`. Rows carry `size`, the
square root of the cell count, so square and rectangular tori share one axis. The bootstrap is
seeded: the same records always give the same rows. `observables.density_series` reads the mean
density over time of single-species sessions from an `autorun` database.

`first_repeat` and `extinction` are for fitting a law to times some of which are only bounds (a
censored likelihood, such as Kepler's `likelihood="censored"`), rather than to a median that a
cap may have turned into a guess. A run that hit its cap bounds transient + period, not the
transient: so the first repeated state (generation transient + period) is the time a cap
censors exactly. Each row lists every seed's `times` with `is_censored` (True: the time exceeds
the number given). For `extinction`, the first generation with no live cell, a run that settled
on a cycle with live cells can never die: it is listed in `never_seeds`, not as a bound. The
Kaplan-Meier median (`km_median`) is null when the censoring, or the runs that never die, leave
it unknown; `median_above` then says what it exceeds.

`damage_curves` keeps each seed's curve whole. Points along one curve are correlated, so a law
over time fitted to the mean curve should take its error bars from resampling whole seeds (as
Kepler's `fit_replicates` does), never from the scatter about the mean. `object_sizes` pools
every soup's objects: `populations` for the distribution of sizes, `abundances` (how often each
distinct object appeared, most common first) for the distribution of abundances.

`experiments.series.count_series` records, for one run, the live cells, cumulative births and
deaths, and live and dead cells by live-neighbour count at every generation: integer columns for
a search for conserved quantities. Its bookkeeping identities (live = live(0) + births - deaths,
and three more) are what such a search should find first on any rule.

`experiments.cost` prices a task before it runs: generations times cells (a cap, so an upper
bound), and seconds from `measure_seconds_per_step` on this machine.

## Caveats

- Everything here is single-species. With several species the birth tie-break is random, a
  repeated state proves nothing, and `run_timeseries(detect_cycles=True)` refuses.
- `unsettled` is relative to the run length: a transient of 1,000 generations on a 32x32 torus
  is not chaos. Compare several `--steps` before calling a rule chaotic, and read
  `no_repeat_before` as a bound on transient + period, never on the transient alone.
- The smallest tori can mislead about growth with size: on the settling-time prototype they
  alone gave a false "logarithmic" verdict for every rule. Check a size dependence with the
  smallest sizes left out.
- A census at one torus size sees that size's cycles. Spaceships and gliders show up as long
  torus-level periods; the `soup` command names them.
- Mean-field fixed points describe an infinite, well-mixed system; a finite torus can sit at a
  density mean field calls unstable.
