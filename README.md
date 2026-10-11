# LatticeLife

A lattice cellular automaton research platform. Supersedes `gameoflife_advanced`.

**Design thesis:** geometry is data. There is exactly one update rule in this codebase; a
lattice geometry is expressed purely as an adjacency structure handed to that rule. See
`docs/LOCAL_RULE_DETERMINISM_POSITION_PAPER.md` for the motivating argument.

See `latticelife_rebuild_spec_v1.0.md` (in the legacy repo) for the full rebuild specification.

## Usage

```bash
uv run python -m latticelife.cli ui                                    # interactive pygame app
uv run python -m latticelife.cli autorun --sessions 10 --generations 200 --shape random
uv run python -m latticelife.cli discover --minutes 30                 # (mu+lambda) rule search
```

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

Rules are written `B3/S23`. Once a neighbour count reaches 10 (12-neighbour triangles), counts
are comma-separated: `B2,5,8,10/S0,8,12`. A single count of 10 or more keeps a trailing comma
(`B12,/S2,3`) so it cannot be read as two digits.

## Recording and discovery

`autorun` records every generation of every session (a sudden-death alert does not end a
session early). Ctrl+C or SIGTERM stops it with everything recorded so far written and exit
status 130. It will not write into a `--db` file that is not a LatticeLife schema v2 database.

`discover` keeps one checkpoint per search configuration (shape, triangle mode, `grid_size`,
`steps`, `freeze_window` and the `[criticality]` settings) under `discovery.checkpoint_path`, so
changing any of them starts a separate lineage instead of resuming another one. A second
`discover` on a configuration that is already running refuses to start. Each rule is added to
`discovery.discovered_path` once per lineage, when its running mean first reaches
`save_threshold`. `discovery.steps` must leave room to detect a freeze:
`int(0.4 * steps) + freeze_window <= steps`.

## Development

```bash
uv sync
uv run ruff check .
uv run pytest -q
```
