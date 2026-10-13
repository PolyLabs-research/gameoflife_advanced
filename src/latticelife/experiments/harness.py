"""Shared plumbing for the headless experiments (`census`, `sweeps`, `objects`): one-species
torus sessions built from a loaded config, JSONL output that resumes, and a worker pool that a
Ctrl+C stops cleanly.

Every experiment runs a single species on a wrapped `size x size` grid: the dynamics are then
deterministic given the seed (`experiments.periodicity`), and the torus has a translation group
(`core.translations`). Grid shape and size come from the experiment's own arguments, not from
`[grid]`; the config supplies `triangle_mode`, the alert and static-threshold settings a session
needs, and the output root.

Output is one JSON object per line, appended as each task finishes and flushed at once, so a run
stopped at any point keeps everything it finished. A run started again on the same file skips the
tasks whose records are already there (`resume_keys`), matched on every field that identifies a
task, so a record from a different experiment never stands in for one of this run's.

`map_tasks` hands tasks to a `multiprocessing` pool in which SIGINT is ignored: a Ctrl+C then
reaches only the parent, which stops the pool and re-raises, instead of every worker dying
mid-task and the pool hanging on their results.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import signal
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from latticelife.config import Config
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.session.session import SimulationSession


def torus_config(config: Config, shape: str, size: int, density: float) -> Config:
    """`config` set up for one species on a wrapped `size x size` grid of `shape`, seeded at
    `density`, with per-generation cluster metrics off (the experiments compute clusters
    themselves, once, where they need them)."""
    return replace(
        config,
        grid=replace(config.grid, shape=shape, width=size, height=size, wrap=True),
        simulation=replace(config.simulation, initial_alive_percentage=density, n_lifeforms=1),
        metrics=replace(config.metrics, cluster_stride=2**31 - 1),
    )


def single_species_session(
    config: Config, rule: Rule, shape: str, size: int, density: float, seed: int
) -> SimulationSession:
    """A fresh, unpaused session of `rule` alone on a `size x size` torus, reproducible from
    `seed` (which must not be 0: a session seed of 0 means fresh entropy)."""
    if seed == 0:
        raise ValueError("seed 0 means fresh entropy to SimulationSession; use seeds from 1")
    session_config = torus_config(config, shape, size, density)
    return SimulationSession(
        session_config, [Lifeform(id=1, rule=rule)], seed=seed, auto_pause=False
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Every record in a JSONL file (none for a missing file). A trailing partial line — a
    run killed mid-write — is skipped, not an error."""
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def resume_keys(path: str | Path, fields: tuple[str, ...]) -> set[tuple]:
    """The identities (`fields` of each record, as a tuple) already recorded in `path`."""
    keys = set()
    for record in read_jsonl(path):
        try:
            keys.add(tuple(_hashable(record[field]) for field in fields))
        except KeyError:
            continue
    return keys


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def map_tasks(fn: Callable[[Any], Any], tasks: Iterable[Any], workers: int) -> Iterator[Any]:
    """`fn(task)` for every task, yielded as they finish: in order with `workers <= 1` (in this
    process), otherwise in completion order from a pool of `workers` processes. `fn` and the
    tasks must be picklable (a module-level function and plain data)."""
    tasks = list(tasks)
    if workers <= 1:
        for task in tasks:
            yield fn(task)
        return
    context = mp.get_context()
    with context.Pool(workers, initializer=_ignore_sigint) as pool:
        yield from pool.imap_unordered(fn, tasks, chunksize=1)
