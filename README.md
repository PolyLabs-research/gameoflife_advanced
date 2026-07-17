# LatticeLife

A lattice cellular automaton research platform. Supersedes `gameoflife_advanced`.

**Design thesis:** geometry is data. There is exactly one update rule in this codebase; a
lattice geometry is expressed purely as an adjacency structure handed to that rule. See
`docs/LOCAL_RULE_DETERMINISM_POSITION_PAPER.md` for the motivating argument.

See `latticelife_rebuild_spec_v1.0.md` (in the legacy repo) for the full rebuild specification.

## Development

```bash
uv sync
uv run ruff check .
uv run pytest -q
```
