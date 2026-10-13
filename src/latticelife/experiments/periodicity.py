"""Exact periodicity for deterministic dynamics. A single-species grid has one successor state
(the birth tie-break draws random numbers but has nothing to choose between), so the first state
seen twice closes a cycle: every later state is determined, and the grid's period and transient
are exact — not a density heuristic like the freeze detection in `experiments.runner`. With
several species the tie-break is random and a repeated state proves nothing, so the detectors
here are for single-species grids only.

`CycleDetector` finds an exact repeat of the whole grid: a torus-level period, which for a
spaceship includes the time it takes to come round the torus. `TranslationCycleDetector` finds a
repeat up to translation (`core.translations`), the right notion for one object on its own
torus: a still life or oscillator repeats in place, a spaceship repeats displaced. Candidate
earlier states are filtered by a translation-invariant fingerprint, the histogram of live
neighbour counts over live and over dead cells, so the translation search only runs on states
that could match.

States are kept as packed bits (one bit per cell: n/8 bytes per generation), so a 64x64 torus
run for 10,000 generations holds about 5 MB; comparison is exact, never by hash alone.
"""

from __future__ import annotations

import numpy as np

from latticelife.core.translations import TorusTranslations, Vector


def pack_state(occupancy: np.ndarray) -> bytes:
    """One bit per cell, live or not (species identity is dropped: see the module docstring)."""
    return np.packbits(np.asarray(occupancy) != 0).tobytes()


def unpack_state(packed: bytes, n_cells: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:n_cells].astype(bool)


class CycleDetector:
    """Feed every generation's occupancy, in order, to `observe`; the first time a state
    repeats it returns `(period, first_seen)`: the state at `generation` equals the one at
    `first_seen = generation - period`, so the cycle starts at `first_seen` (the transient) and
    every state from then on recurs with that period."""

    def __init__(self) -> None:
        self._first_seen: dict[bytes, int] = {}

    def observe(self, occupancy: np.ndarray, generation: int) -> tuple[int, int] | None:
        key = pack_state(occupancy)
        first = self._first_seen.get(key)
        if first is not None:
            return generation - first, first
        self._first_seen[key] = generation
        return None

    def __len__(self) -> int:
        return len(self._first_seen)


class TranslationCycleDetector:
    """Like `CycleDetector`, but a state also matches an earlier one that is its translate:
    `observe` returns `(period, vector)` with `vector` the name (`core.translations`) of the
    translation carrying the earlier state onto this one, reduced modulo the torus. `(p, (0, 0))`
    is an exact repeat."""

    def __init__(self, translations: TorusTranslations, adjacency: np.ndarray) -> None:
        if adjacency.shape[0] != translations.n_cells:
            raise ValueError("adjacency and translations describe different grids")
        self.translations = translations
        n = translations.n_cells
        # Padded gather, as in LatticeGrid._neighbor_counts: -1 slots read a sentinel False.
        self._gather = np.where(adjacency >= 0, adjacency, n)
        self._degree = adjacency.shape[1]
        self._seen: dict[bytes, list[tuple[int, bytes]]] = {}

    def fingerprint(self, alive: np.ndarray) -> bytes:
        """Translation-invariant summary of a state: how many live and how many dead cells have
        each number of live neighbours. Any adjacency automorphism preserves it."""
        padded = np.append(alive, False)
        counts = padded[self._gather].sum(axis=1)
        live = np.bincount(counts[alive], minlength=self._degree + 1)
        dead = np.bincount(counts[~alive], minlength=self._degree + 1)
        return np.concatenate([live, dead]).astype(np.int64).tobytes()

    def observe(self, occupancy: np.ndarray, generation: int) -> tuple[int, Vector] | None:
        alive = np.asarray(occupancy) != 0
        fingerprint = self.fingerprint(alive)
        earlier = self._seen.setdefault(fingerprint, [])
        for seen_at, packed in earlier:
            vector = self.translations.find(unpack_state(packed, self.translations.n_cells), alive)
            if vector is not None:
                return generation - seen_at, vector
        earlier.append((generation, np.packbits(alive).tobytes()))
        return None
