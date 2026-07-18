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

## Development

```bash
uv sync
uv run ruff check .
uv run pytest -q
```
