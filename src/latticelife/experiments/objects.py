"""Object census: what a soup leaves behind, named. After a soup has run, its live cells are
split into components (`harvest`), each component is put alone on a torus of its own and run
until it repeats up to translation (`identify`), and the result is a catalogue of still lifes,
oscillators and spaceships with their exact periods and displacements — per rule, per lattice.

Identity. An object is named by its `key` (`core.translations.pattern_key`): the cells of its
smallest phase, relative to one of them, so two objects get the same key exactly when one is a
translate of the other, whatever torus or soup they came from. Its `canonical_key` also forgets
orientation: the smallest key, under the lattice's point group (`core.translations.point_group`),
of any of its fewest-cell phases. Outer-totalistic rules commute with rotations and reflections,
so a rotated object is the same object, and taking every fewest-cell phase matters: a hexagonal
domino that alternates with a two-cell "bridge" is filed under the domino's key in one
orientation and the bridge's in another, but has one canonical key. `catalogue(...,
up_to_symmetry=True)` counts objects that way, and recovers canonical keys for records written
before they were recorded by re-identifying each object.

Rule range. Along its cycle every cell, live or dead, takes some (state, neighbour count) to
next-state transition, and the rules that make exactly those transitions are exactly the rules
under which the object runs through the same cycle: a subcube of the rule space, from
`rule_min` (the birth and survival counts the cycle needs) to `rule_max` (every count it does
not forbid), holding `rule_count` rules. It is how many rules support the object, and which.

Kinds. `still`: period 1, no displacement. `oscillator`: period above 1, no displacement.
`spaceship`: a displacement per period; its `displacement` is the translation's name
(`core.translations`, the move of an even column 0 cell) reduced to the shortest form, and
its speed is that displacement over `period` generations. A component that is not an object
is reported with a reason instead: `died` (nothing left), `unsettled` (no repeat within
`max_period` generations: a chaotic blob, or a period longer than the limit), `too_large`
(it grew to span half its torus, so it was interacting with itself across the seam; a bigger
`margin` gives it more room).

Separation. Two objects closer than `radius` lattice steps are harvested as one component and
identified together, as a compound with the least common multiple of their periods; one object
with internal gaps wider than `radius` is split and each part identified alone (where it usually
dies). The default radius of 2 keeps Life's pulsar whole and separates most of its ash.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from latticelife.config import Config
from latticelife.core.geometries import max_degree, min_wrap_size
from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.core.translations import (
    PatternKey,
    TorusTranslations,
    canonical_key,
    key_extent,
)
from latticelife.experiments.harness import (
    append_jsonl,
    map_tasks,
    resume_keys,
    single_species_session,
)
from latticelife.experiments.periodicity import CycleDetector, TranslationCycleDetector
from latticelife.session.metrics import component_roots

KINDS: tuple[str, ...] = ("still", "oscillator", "spaceship")
UNIDENTIFIED: tuple[str, ...] = ("died", "unsettled", "too_large")
RECORD_KEY: tuple[str, ...] = ("rule", "shape", "triangle_mode", "grid", "seed", "steps", "density")


@dataclass(frozen=True)
class ObjectRecord:
    key: PatternKey  # the smallest phase, as pattern_key gives it
    kind: str
    period: int
    displacement: tuple[int, int]
    population: int  # cells in the smallest phase
    max_population: int  # cells in the largest phase
    canonical_key: PatternKey  # up to translation, rotation, reflection and phase
    rule_min: str  # the smallest rule the cycle runs under
    rule_max: str  # the largest
    rule_count: int  # how many rules lie between them: all run the same cycle

    def to_json(self) -> dict:
        data = asdict(self)
        data["key"] = [list(cell) for cell in self.key]
        data["canonical_key"] = [list(cell) for cell in self.canonical_key]
        data["displacement"] = list(self.displacement)
        return data


def widened_adjacency(adjacency: np.ndarray, radius: int) -> np.ndarray:
    """An adjacency array joining every cell to all cells within `radius` steps of it (with
    repeats and -1 padding, which union-find does not mind). Radius 1 is the array itself."""
    if radius < 1:
        raise ValueError(f"radius must be at least 1 (got {radius})")
    n, degree = adjacency.shape
    padded = np.vstack([adjacency, -np.ones((1, degree), dtype=adjacency.dtype)])
    frontier = adjacency
    reach = [adjacency]
    for _ in range(radius - 1):
        gather = np.where(frontier >= 0, frontier, n)
        frontier = padded[gather].reshape(n, -1)
        reach.append(frontier)
    return np.hstack(reach)


def harvest(grid: LatticeGrid, lifeform_id: int = 1, radius: int = 2) -> list[np.ndarray]:
    """The components of `lifeform_id`'s live cells, cells within `radius` lattice steps of
    each other joined, each as a sorted array of flat indices, largest first."""
    cells = np.flatnonzero(grid.occupancy == lifeform_id)
    if cells.size == 0:
        return []
    roots = component_roots(widened_adjacency(grid.adjacency, radius), cells)
    components: dict[int, list[int]] = {}
    for cell, root in zip(cells.tolist(), roots.tolist(), strict=True):
        components.setdefault(root, []).append(cell)
    return sorted(
        (np.array(group, dtype=np.int64) for group in components.values()),
        key=lambda group: (-group.size, group[0]),
    )


def _own_torus(
    key: PatternKey, shape: str, triangle_mode: str, margin: int
) -> tuple[TorusTranslations, int, int]:
    """A torus with room for `key` to move and grow: twice its extent plus margins each way
    (so the fit check below has a whole half-torus of slack), even-sized, at least the shape's
    smallest wrappable size; and the origin that places the key `margin` cells from the top
    left, on a cell the key's frame allows (an even column; even row + column on triangles)."""
    min_r, min_c, max_r, max_c = key_extent(key)
    height = 2 * (max_r - min_r + 1 + margin)
    width = 2 * (max_c - min_c + 1 + margin)
    min_width, min_height = min_wrap_size(shape, triangle_mode)
    height, width = max(height, min_height), max(width, min_width)
    height += height % 2
    width += width % 2
    row, column = margin - min_r, margin - min_c
    if shape == "hexagon" and column % 2:
        column += 1
    if shape == "triangle" and (row + column) % 2:
        column += 1
    return TorusTranslations(shape, width, height, triangle_mode), row, column


def _fits(translations: TorusTranslations, cells: np.ndarray) -> bool:
    """Whether `cells` lie within a window smaller than half the torus each way, read from the
    first cell: then they do not reach round the torus to touch themselves."""
    width, height = translations.width, translations.height
    rows = (cells // width - cells[0] // width) % height
    cols = (cells % width - cells[0] % width) % width
    rows = np.where(rows > height // 2, rows - height, rows)
    cols = np.where(cols > width // 2, cols - width, cols)
    return bool(rows.max() - rows.min() < height // 2 and cols.max() - cols.min() < width // 2)


def identify(
    key: PatternKey,
    rule: Rule,
    shape: str,
    triangle_mode: str = "edge+vertex",
    max_period: int = 200,
    margin: int = 8,
) -> ObjectRecord | str:
    """Run the pattern `key` alone under `rule` on a torus of its own until it repeats up to
    translation, and name it; or one of `UNIDENTIFIED` saying why it is not an object. `margin`
    must be even (so the placement keeps the key's geometry) and `max_period` at least 1."""
    if margin < 2 or margin % 2:
        raise ValueError(f"margin must be an even number of at least 2 (got {margin})")
    if max_period < 1:
        raise ValueError(f"max_period must be at least 1 (got {max_period})")
    if not key:
        return "died"
    translations, row, column = _own_torus(key, shape, triangle_mode, margin)
    grid = LatticeGrid(
        [Lifeform(1, rule)], shape, translations.width, translations.height, True, triangle_mode
    )
    grid.occupancy[translations.place(key, row, column)] = 1
    detector = TranslationCycleDetector(translations, grid.adjacency)
    rng = np.random.default_rng(0)  # one species: the draws decide nothing
    detector.observe(grid.occupancy, 0)
    repeat = None
    for step in range(1, max_period + 1):
        grid.step(rng)
        cells = np.flatnonzero(grid.occupancy)
        if cells.size == 0:
            return "died"
        if not _fits(translations, cells):
            return "too_large"
        repeat = detector.observe(grid.occupancy, step)
        if repeat is not None:
            break
    if repeat is None:
        return "unsettled"
    period, vector = repeat
    # The grid is on its cycle: read every phase for the smallest one and the populations, and
    # every cell's (state, neighbour count) -> next state transition for the rule range.
    phases: list[PatternKey] = []
    populations: list[int] = []
    gather = np.where(grid.adjacency >= 0, grid.adjacency, grid.n_cells)
    used: dict[tuple[bool, bool], set[int]] = {
        (alive, after): set() for alive in (False, True) for after in (False, True)
    }
    for _ in range(period):
        alive = grid.occupancy != 0
        cells = np.flatnonzero(alive)
        phase = translations.pattern_key(cells)
        assert phase is not None  # every phase fitted
        phases.append(phase)
        populations.append(int(cells.size))
        counts = np.append(alive, False)[gather].sum(axis=1)
        grid.step(rng)
        after = grid.occupancy != 0
        for was in (False, True):
            for now in (False, True):
                used[(was, now)].update(np.unique(counts[(alive == was) & (after == now)]).tolist())
    displacement = translations.centred(vector)
    if displacement != (0, 0):
        kind = "spaceship"
    elif period == 1:
        kind = "still"
    else:
        kind = "oscillator"
    # The key is the smallest phase (fewest cells; the smallest cell list among equals), so
    # `population` is the key's own cell count.
    smallest = min(range(period), key=lambda i: (populations[i], phases[i]))
    fewest = {phases[i] for i in range(period) if populations[i] == populations[smallest]}
    n = grid.adjacency.shape[1]
    every = set(range(n + 1))
    born, stays_dead = used[(False, True)], used[(False, False)]
    survives, dies = used[(True, True)], used[(True, False)]
    assert not born & stays_dead and not survives & dies  # one rule made every transition
    constrained = len(born | stays_dead) + len(survives | dies)
    return ObjectRecord(
        key=phases[smallest],
        kind=kind,
        period=period,
        displacement=displacement,
        population=populations[smallest],
        max_population=max(populations),
        canonical_key=min(canonical_key(phase, shape, triangle_mode) for phase in fewest),
        rule_min=Rule(birth=born, survive=survives).to_bs_string(n),
        rule_max=Rule(birth=every - stays_dead, survive=every - dies).to_bs_string(n),
        rule_count=2 ** (2 * (n + 1) - constrained),
    )


@dataclass(frozen=True)
class SoupTask:
    config: Config
    rule: str
    shape: str
    grid: int
    seed: int
    steps: int
    density: float
    radius: int
    max_period: int
    margin: int


def soup_objects(task: SoupTask) -> dict:
    """One soup: run it (stopping early if the whole torus repeats exactly, since nothing new
    appears after that), harvest its components and identify each. The record holds the
    soup's own period and transient when it had one, every object found, and how many
    components were not objects, by reason."""
    rule = Rule.from_bs_string(task.rule)
    triangle_mode = task.config.grid.triangle_mode
    n = max_degree(task.shape, triangle_mode)
    session = single_species_session(
        task.config, rule, task.shape, task.grid, task.density, task.seed
    )
    detector = CycleDetector()
    detector.observe(session.grid.occupancy, 0)
    period = transient = None
    stopped_at = 0
    for step in range(1, task.steps):
        session.step()
        stopped_at = step
        repeat = detector.observe(session.grid.occupancy, step)
        if repeat is not None:
            period, transient = repeat
            break
    translations = TorusTranslations(task.shape, task.grid, task.grid, triangle_mode)
    objects: list[dict] = []
    unidentified = dict.fromkeys(UNIDENTIFIED, 0)
    cache: dict[PatternKey, ObjectRecord | str] = {}
    for component in harvest(session.grid, 1, task.radius):
        key = translations.pattern_key(component)
        if key is None:
            unidentified["too_large"] += 1
            continue
        if key not in cache:
            cache[key] = identify(
                key, rule, task.shape, triangle_mode, task.max_period, task.margin
            )
        found = cache[key]
        if isinstance(found, str):
            unidentified[found] += 1
        else:
            objects.append({"cells": int(component.size), **found.to_json()})
    return {
        "rule": rule.to_bs_string(n),
        "shape": task.shape,
        "triangle_mode": triangle_mode if task.shape == "triangle" else None,
        "grid": task.grid,
        "seed": task.seed,
        "steps": task.steps,
        "density": task.density,
        "soup_period": period,
        "soup_transient": transient,
        "stopped_at": stopped_at,
        "objects": objects,
        "unidentified": unidentified,
    }


def run_soups(
    config: Config,
    rule: Rule,
    shape: str,
    grid: int,
    seeds: Sequence[int],
    steps: int,
    density: float,
    out_path: str,
    workers: int = 1,
    radius: int = 2,
    max_period: int = 200,
    margin: int = 8,
) -> Iterator[dict]:
    """One record per soup seed not already in `out_path`, appended as it finishes and yielded."""
    n = max_degree(shape, config.grid.triangle_mode)
    rule_text = rule.to_bs_string(n)
    triangle_mode = config.grid.triangle_mode if shape == "triangle" else None
    done = resume_keys(out_path, RECORD_KEY)
    tasks = [
        SoupTask(config, rule_text, shape, grid, seed, steps, density, radius, max_period, margin)
        for seed in seeds
        if (rule_text, shape, triangle_mode, grid, seed, steps, density) not in done
    ]
    for record in map_tasks(soup_objects, tasks, workers):
        append_jsonl(out_path, record)
        yield record


def _as_key(cells: Iterable[Iterable[int]]) -> PatternKey:
    return tuple((int(r), int(c)) for r, c in cells)


def catalogue(records: Iterable[dict], up_to_symmetry: bool = False) -> list[dict]:
    """The objects seen across soup records, most common first: one entry per distinct object
    with its kind, period, displacement, population, how many times it appeared (`count`) and
    under how many distinct rules (`rules`). An object is its key and its behaviour (kind, period
    and displacement), since one pattern can be a still life under one rule and oscillate under
    another. With `up_to_symmetry`, objects are merged under rotation and reflection by their
    canonical key, which a record written before canonical keys were recorded gets by
    re-identifying the object under its rule (once per rule and key)."""
    counts: Counter[tuple] = Counter()
    rules: dict[tuple, set[str]] = {}
    details: dict[tuple, dict] = {}
    recomputed: dict[tuple, PatternKey] = {}
    for record in records:
        for found in record["objects"]:
            if not up_to_symmetry:
                identity = _as_key(found["key"])
            elif "canonical_key" in found:
                identity = _as_key(found["canonical_key"])
            else:
                cache_key = (record["rule"], _as_key(found["key"]))
                if cache_key not in recomputed:
                    shape = record["shape"]
                    again = identify(
                        _as_key(found["key"]),
                        Rule.from_bs_string(record["rule"]),
                        shape,
                        record.get("triangle_mode") or "edge+vertex",
                        max_period=int(found["period"]),
                    )
                    assert isinstance(again, ObjectRecord), (cache_key, again)
                    recomputed[cache_key] = again.canonical_key
                identity = recomputed[cache_key]
            entry = (identity, found["kind"], found["period"], tuple(found["displacement"]))
            counts[entry] += 1
            rules.setdefault(entry, set()).add(record["rule"])
            details.setdefault(
                entry,
                {
                    "key": [list(cell) for cell in identity],
                    "kind": found["kind"],
                    "period": found["period"],
                    "displacement": found["displacement"],
                    "population": found["population"],
                },
            )
    return [
        {**details[entry], "count": count, "rules": len(rules[entry])}
        for entry, count in counts.most_common()
    ]
