# LatticeLife

A lattice cellular automaton research platform. Supersedes `gameoflife_advanced`.

![Square, hexagon and triangle grids, each after 120 generations of three competing species](docs/images/banner.png)

**Design thesis:** geometry is data. There is exactly one update rule in this codebase; a
lattice geometry is expressed purely as an adjacency structure handed to that rule. See
`docs/LOCAL_RULE_DETERMINISM_POSITION_PAPER.md` for the motivating argument.

See `latticelife_rebuild_spec_v1.0.md` (in the legacy repo) for the full rebuild specification.

## What it does

- **Three lattices, one rule.** Square grids with 8 neighbours, hexagon grids with 6, and
  triangle grids with 3 (edge) or 12 (edge+vertex), wrapped into a torus or bounded. A shape is
  an adjacency array (`docs/adjacency.md`); the update rule never knows which one it was given.
- **Up to ten species** on one grid, each with its own outer-totalistic `B/S` rule, competing
  for cells.
- **An interactive app** with live editing by clicking, a population chart per species, alerts
  for a sudden population crash and for stasis, and a 3D view of the wrapped grid on its torus.
- **Headless instruments.** `autorun` records every generation of batches of sessions to
  SQLite, `discover` runs a (mu+lambda) search for rules near criticality, and `census`,
  `sweep` and `soup` give exact periods, phase diagrams and object catalogues per lattice.

## Screenshots

<table>
  <tr>
    <td width="50%">
      <img src="docs/images/ui-hexagon.png" alt="Three species on a wrapped 50 by 50 hexagon grid, with the settings panel and the population chart">
      <br><sub>Three species on a wrapped 50 x 50 hexagon grid, each with its own <code>B/S</code> rule. The chart tracks every species' population.</sub>
    </td>
    <td width="50%">
      <img src="docs/images/ui-torus.png" alt="The same hexagon grid drawn on a 3D torus">
      <br><sub>The same grid on its torus (<code>V</code>): drag to rotate, wheel to zoom. Needs the <code>[torus]</code> extra.</sub>
    </td>
  </tr>
  <tr>
    <td>
      <img src="docs/images/ui-triangle.png" alt="Three species on a wrapped 60 by 50 triangle grid in edge+vertex mode">
      <br><sub>A 60 x 50 triangle grid in edge+vertex mode, where every cell has 12 neighbours.</sub>
    </td>
    <td>
      <img src="docs/images/ui-square.png" alt="Conway's Life and HighLife competing on a square grid">
      <br><sub>Conway's Life (<code>B3/S23</code>) against HighLife (<code>B36/S23</code>) on a square grid. Lighter cells have been alive longer than the static threshold.</sub>
    </td>
  </tr>
  <tr>
    <td>
      <img src="docs/images/ui-help.png" alt="The keyboard shortcuts overlay">
      <br><sub>The shortcuts overlay (<code>?</code> or <code>F1</code>).</sub>
    </td>
    <td>
      <img src="docs/images/ui-alert.png" alt="The stasis alert over a grid that is almost full and not changing">
      <br><sub>The stasis alert: over 80% of the grid occupied and almost nothing changing. A population crash raises a sudden-death alert and pauses the run.</sub>
    </td>
  </tr>
</table>

Every image is rendered by `scripts/screenshots.py` without a display, from seeded runs, so
`uv run python scripts/screenshots.py` re-renders them after a change to the app.

## Usage

```bash
uv run python -m latticelife.cli ui                                    # interactive pygame app
uv run python -m latticelife.cli autorun --sessions 10 --generations 200 --shape random
uv run python -m latticelife.cli discover --minutes 30                 # (mu+lambda) rule search
uv run python -m latticelife.cli census --shape hexagon --rules dual   # every rule's exact fate
uv run python -m latticelife.cli sweep --shape hexagon --rule B2/S34 --grid 16 32 64
uv run python -m latticelife.cli soup --shape hexagon --rule B2/S34    # objects soups leave
```

`census`, `sweep` and `soup` are the mathematics experiments: exact periods, phase diagrams and
object catalogues per lattice, written as JSONL and resumable. See `docs/experiments.md`.

3D torus rendering (the `V` key in `ui`) needs the optional extra: `uv sync --extra torus`.

Requires Python 3.12 or 3.13: the locked pygame and moderngl releases ship wheels for those only.

## Configuration

Every tunable lives in `config/default.toml` (pass `--config PATH` for another file). Each key is
checked when the file loads, and a bad value stops the command with an error naming the key. With
wrap on, hexagon grids need an even width and triangle grids an even width and height.

Relative output paths in a config (`recording.db_path`, `discovery.checkpoint_path`,
`discovery.discovered_path`, `logging.dir`) resolve against the project root (the nearest folder
above the config file that holds `pyproject.toml`), not the directory you run from. Set
`LATTICELIFE_LOG_DIR` to send the JSONL logs somewhere else.

An installed (non-editable) `latticelife` command has no checkout: it uses the copy of
`config/default.toml` shipped inside the package and writes its outputs under
`$XDG_DATA_HOME/latticelife/out` (default `~/.local/share/latticelife/out`), whichever directory
it is run from.

Rules are written `B3/S23`. Once any neighbour count in a rule reaches 10 (12-neighbour
triangles), all of its counts are comma-separated: `B2,5,8,10/S0,8,12`, `B1,3,4,5,9/S7,9,12`. A
single count of 10 or more keeps a trailing comma (`B12,/S2,3`) so it cannot be read as two
digits.

## Recording and discovery

`autorun` records every generation of every session (a sudden-death alert does not end a
session early). Ctrl+C or SIGTERM stops it with everything recorded so far written and exit
status 130. It will not write into a `--db` file that is not a LatticeLife schema v2 database.

`discover` keeps one checkpoint per search configuration (shape, triangle mode, `grid_size`,
`steps`, `freeze_window` and the `[criticality]` settings) under `discovery.checkpoint_path`, so
changing any of them starts a separate lineage instead of resuming another one. A second
`discover` on a configuration that is already running refuses to start. Each rule is added to
`discovery.discovered_path` once per lineage, when its running mean first reaches
`save_threshold`. `discover` refuses to start unless `discovery.steps` leaves room to detect a
freeze: `int(0.4 * steps) + freeze_window <= steps`. `ui` and `autorun` do not check this,
because they never use those settings.

## Development

```bash
uv sync
uv run ruff check .
uv run pytest -q
```

`scripts/smoke_ui.py` drives the app for a number of frames per shape without a display,
`scripts/bench_render.py` times drawing and stepping, and `scripts/screenshots.py` renders the
images above.
