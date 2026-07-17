"""multiprocessing.Process wrapper around the (μ+λ) ES loop (§6.11). Runs as a real OS process,
not a thread and not a daemon: the legacy discovery thread's global stdlib/numpy RNG reseeding
(D-3) could corrupt the main simulation because they shared one process's global RNG state. A
separate process makes that structurally impossible — there is no global state left to share.
Clean shutdown via `multiprocessing.Event`; checkpoints every generation so a stopped or killed
worker resumes without losing progress, and stop-responsiveness is checked between each
candidate evaluation so a `stop()` call lands within about one evaluation's wall time.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from latticelife.config import Config
from latticelife.discovery import es
from latticelife.logging import log_event


class DiscoveryWorker:
    def __init__(self, config: Config, shape: str = "square") -> None:
        self.config = config
        self.shape = shape
        self.checkpoint_path = Path(config.discovery.checkpoint_path) / "checkpoint.json"
        self.discovered_path = Path("out/discovered_rules.jsonl")
        self._stop_event: mp.synchronize.Event = mp.Event()
        self._process: mp.Process | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._process = mp.Process(
            target=_run_loop,
            args=(
                self.config,
                self.shape,
                self.checkpoint_path,
                self.discovered_path,
                self._stop_event,
            ),
            daemon=False,
        )
        self._process.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._process is not None:
            self._process.join(timeout)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()


def _load_checkpoint(checkpoint_path: Path) -> es.ESState | None:
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open() as f:
        data = json.load(f)
    return es.from_checkpoint_dict(data)


def _save_checkpoint(checkpoint_path: Path, state: es.ESState) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(es.to_checkpoint_dict(state), f)
    tmp_path.replace(checkpoint_path)


def _append_discovered(discovered_path: Path, shape: str, candidate: es.Candidate) -> None:
    discovered_path.parent.mkdir(parents=True, exist_ok=True)
    all_counts = candidate.rule.birth | candidate.rule.survive
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "shape": shape,
        "rule": candidate.rule.to_bs_string(max(all_counts, default=0)),
        "score": candidate.mean_score,
        "evals": candidate.eval_count,
    }
    with discovered_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _gen_rng(entropy: int, generation: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([entropy, generation]))


def _run_loop(
    config: Config,
    shape: str,
    checkpoint_path: Path,
    discovered_path: Path,
    stop_event: mp.synchronize.Event,
) -> None:
    state = _load_checkpoint(checkpoint_path)
    if state is None:
        entropy = int(np.random.SeedSequence().entropy % (2**63))
        population = es.initial_population(
            shape, config.grid.triangle_mode, config.discovery, _gen_rng(entropy, 0)
        )
        state = es.ESState(generation=0, population=population, shape=shape, entropy=entropy)

    log_event("discovery", level="info", event="worker_started", generation=state.generation)

    while not stop_event.is_set():
        next_generation = state.generation + 1
        gen_rng = _gen_rng(state.entropy, next_generation)

        offspring = es.mutate_population(state.population, shape, config, gen_rng)
        pool = list(state.population) + offspring

        interrupted = False
        for candidate in pool:
            for _ in range(config.discovery.evals_per_candidate):
                if stop_event.is_set():
                    interrupted = True
                    break
                score = es.evaluate_candidate(candidate.rule, shape, config, gen_rng.spawn(1)[0])
                candidate.add_evaluation(score)
            if interrupted:
                break
        if interrupted:
            break

        new_population = es.select_elites(pool, config.discovery.mu)
        state = es.ESState(next_generation, new_population, shape, state.entropy)

        for candidate in state.population:
            if candidate.mean_score >= config.discovery.save_threshold:
                _append_discovered(discovered_path, shape, candidate)

        _save_checkpoint(checkpoint_path, state)
        log_event(
            "discovery",
            level="info",
            event="generation_complete",
            generation=state.generation,
            best_score=state.population[0].mean_score,
        )

    log_event("discovery", level="info", event="worker_stopped", generation=state.generation)
