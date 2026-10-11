"""multiprocessing.Process wrapper around the (μ+λ) ES loop (§6.11). Runs as a real OS process,
not a thread and not a daemon: the legacy discovery thread's global stdlib/numpy RNG reseeding
(D-3) could corrupt the main simulation because they shared one process's global RNG state. A
separate process makes that structurally impossible — there is no global state left to share.
Checkpoints every generation so a stopped or killed worker resumes without losing progress, and
stop-responsiveness is checked between each candidate evaluation so a `stop()` call lands within
about one evaluation's wall time.

Lineages. Each search configuration (`es.search_settings`: shape, triangle neighbourhood, grid,
steps, freeze window, criticality settings) has its own checkpoint file,
`<discovery.checkpoint_path>/checkpoint-<shape>[-<mode>]-<fingerprint>.json`, which also stores
the settings and is checked against them on load. A run resumes only its own configuration's
lineage; changing any of those settings starts (or resumes) a different one. The pre-settings
`checkpoint.json` is never read.

One writer per lineage. The worker holds an exclusive lock on `<checkpoint>.lock` for its whole
life; a second worker for the same lineage logs `checkpoint_locked` and exits with
`EXIT_LOCKED` instead of racing the first. Checkpoints are written to a unique temporary file,
fsynced and renamed into place. A checkpoint that cannot be used is renamed aside to
`<name>.unusable-<UTC time>` with an error event, and the lineage starts fresh.

Discovered rules (`discovery.discovered_path`) get one row per rule per lineage, written when its
running mean first reaches `save_threshold` — not once per generation it stays an elite. Rows
add `triangle_mode`, `grid_size`, `steps` and the lineage fingerprint (`settings`) to the spec's
{ts, shape, rule, score, evals}.

Stopping. The worker stops cleanly (discarding the generation in progress, logging
`worker_stopped`, exit code 0) when `stop()` is called, when its parent process exits, when its
optional time limit passes, or on SIGTERM/SIGHUP. It ignores SIGINT: Ctrl+C reaches the whole
foreground process group and the parent owns it. The stop flag is a lock-free shared byte, and
every signal handler (here and in `cli`) only sets a flag, so no handler can block on a lock its
own process already holds. `start()` creates the process with these signals ignored, so an
early Ctrl+C cannot kill it half-started (under spawn or forkserver, start-up takes a while).

Failure. Any exception in the worker is logged as `worker_crashed` (with the traceback) and
re-raised, so the process exits non-zero and `exitcode` tells the parent.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import numpy as np

from latticelife.config import Config
from latticelife.discovery import es
from latticelife.logging import configure_logging, log_event

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt

# Exit code of a worker that found its lineage locked by another worker.
EXIT_LOCKED = 3

# SIGTERM/SIGHUP ask the worker to stop cleanly; SIGINT is ignored (the parent owns Ctrl+C).
STOP_SIGNALS: tuple[signal.Signals, ...] = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
)
_HANDLED_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGINT, *STOP_SIGNALS)

LEGACY_CHECKPOINT_NAME = "checkpoint.json"


class DiscoveryWorker:
    def __init__(self, config: Config, shape: str = "square") -> None:
        self.config = config
        self.shape = shape
        self.settings = es.search_settings(config, shape)
        self.checkpoint_path = Path(config.discovery.checkpoint_path) / es.checkpoint_name(
            self.settings
        )
        self.discovered_path = Path(config.discovery.discovered_path)
        self._context = mp.get_context()
        # A raw shared byte, not an mp.Event: setting it takes no lock, so stop() is safe from
        # anywhere, including a signal handler that interrupted code holding that lock.
        self._stop_flag = self._context.RawValue(ctypes.c_bool, False)
        self._process: mp.process.BaseProcess | None = None

    def start(self, max_seconds: float | None = None) -> None:
        """Start the worker process. `max_seconds` is a time limit the worker enforces itself,
        as a backstop for a parent that is alive but no longer stopping it."""
        self._stop_flag.value = False
        process = self._context.Process(
            target=_run_loop,
            args=(
                self.config,
                self.shape,
                self.checkpoint_path,
                self.discovered_path,
                self._stop_flag,
                max_seconds,
            ),
            name="latticelife-discovery",
            daemon=False,
        )
        with _ignored_by_new_processes(_HANDLED_SIGNALS):
            process.start()
        self._process = process

    def stop(self) -> None:
        self._stop_flag.value = True

    def join(self, timeout: float | None = None) -> None:
        if self._process is not None:
            self._process.join(timeout)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def kill(self) -> None:
        """SIGKILL the worker (last resort when it does not stop; the checkpoint on disk is
        always a complete generation)."""
        if self.is_alive():
            self._process.kill()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def exitcode(self) -> int | None:
        """None while running (or never started); 0 after a clean stop; `EXIT_LOCKED` when
        another worker holds this lineage; 1 after a crash; -N when killed by signal N."""
        return self._process.exitcode if self._process is not None else None


# --- signals ------------------------------------------------------------------------------------


@contextlib.contextmanager
def _ignored_by_new_processes(signals: tuple[signal.Signals, ...]) -> Iterator[None]:
    """While a process is created inside this block, `signals` are set to SIG_IGN, which every
    start method passes on to the child: fork copies it, and exec keeps an ignored disposition
    (Python then installs no KeyboardInterrupt handler at all). The child is therefore immune
    from its first instruction until `_run_loop` installs the worker's own handlers — under
    spawn and forkserver that start-up takes a noticeable fraction of a second. The signals are
    also blocked meanwhile so this process keeps any that arrive; they are delivered to its
    restored handlers on exit. (A spawn/forkserver start also launches CPython's resource
    tracker, which unblocks SIGINT/SIGTERM itself, so one arriving during that first start
    can be dropped.) Does nothing outside the main thread, where handlers cannot be set."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signals = tuple(sig for sig in signals if signal.getsignal(sig) is not None)
    previous_mask = None
    if hasattr(signal, "pthread_sigmask"):
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in signals}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _install_signal_handlers(received: list[int]) -> Callable[[], None]:
    """Worker signal policy (see module docstring): SIGINT ignored, SIGTERM/SIGHUP append to
    `received`; then unblock them (a forked worker inherits them blocked from `start()`).
    Returns a function restoring the previous state, used when `_run_loop` runs in-process
    rather than as the worker process."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    def request_stop(signum: int, frame: object) -> None:
        received.append(signum)

    previous = {signal.SIGINT: signal.signal(signal.SIGINT, signal.SIG_IGN)}
    for sig in STOP_SIGNALS:
        previous[sig] = signal.signal(sig, request_stop)
    previous_mask = None
    if hasattr(signal, "pthread_sigmask"):
        previous_mask = signal.pthread_sigmask(signal.SIG_UNBLOCK, _HANDLED_SIGNALS)

    def restore() -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    return restore


# --- checkpoint files ---------------------------------------------------------------------------


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def acquire_lineage_lock(checkpoint_path: Path) -> IO[str] | None:
    """Take the exclusive, non-blocking lock on `<checkpoint>.lock`. Returns the open lock file
    (keep it open to hold the lock; the OS releases it when the process exits, however it
    exits), or None when another process holds it."""
    lock_path = checkpoint_path.with_name(checkpoint_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:  # pragma: no cover - Windows
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock_file.close()
        return None
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file


def _set_aside(path: Path, reason: str) -> Path:
    aside = path.with_name(f"{path.name}.{reason}-{_utc_stamp()}")
    path.replace(aside)
    return aside


def _load_checkpoint(checkpoint_path: Path, settings: dict[str, Any]) -> es.ESState | None:
    """The lineage's saved state, or None to start fresh: when there is no checkpoint, or when
    it cannot be used (unreadable JSON, wrong version or shape, counts out of range, settings
    that differ from this run's) — that file is renamed aside and an error event logged."""
    try:
        text = checkpoint_path.read_text()
    except FileNotFoundError:
        return None
    try:
        state = es.from_checkpoint_dict(json.loads(text))
        if state.settings != settings:
            raise ValueError("its search settings differ from this run's")
    except (ValueError, KeyError, TypeError) as exc:
        aside = _set_aside(checkpoint_path, "unusable")
        log_event(
            "discovery",
            level="error",
            event="checkpoint_unusable",
            checkpoint=str(checkpoint_path),
            moved_to=str(aside),
            error=f"{type(exc).__name__}: {exc}",
        )
        return None
    return state


def _save_checkpoint(checkpoint_path: Path, state: es.ESState) -> None:
    """Atomic replace: a unique temporary file in the same directory (concurrent writers can
    never share or truncate each other's), fsynced before the rename, then the directory
    fsynced so the rename itself survives a crash."""
    directory = checkpoint_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f"{checkpoint_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(es.to_checkpoint_dict(state), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, checkpoint_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    with contextlib.suppress(OSError):  # directories cannot be opened for fsync everywhere
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


# --- discovered rules ---------------------------------------------------------------------------


def _saved_rules(discovered_path: Path, lineage: str) -> set[str]:
    """Rule strings this lineage has already written to the discovered-rules file. Lines that
    are not JSON objects, and rows of other lineages or from before rows carried one, are
    skipped."""
    saved: set[str] = set()
    try:
        f = discovered_path.open()
    except FileNotFoundError:
        return saved
    with f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("settings") == lineage:
                rule = row.get("rule")
                if isinstance(rule, str):
                    saved.add(rule)
    return saved


def _append_discovered(
    discovered_path: Path, settings: dict[str, Any], rule_string: str, candidate: es.Candidate
) -> None:
    discovered_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "shape": settings["shape"],
        "triangle_mode": settings["triangle_mode"],
        "grid_size": settings["grid_size"],
        "steps": settings["steps"],
        "settings": es.settings_id(settings),
        "rule": rule_string,
        "score": candidate.mean_score,
        "evals": candidate.eval_count,
    }
    # One write call per row, so concurrent lineages appending to the same file never
    # interleave within a line.
    with discovered_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# --- the loop -----------------------------------------------------------------------------------


def _gen_rng(entropy: int, generation: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([entropy, generation]))


def _run_loop(
    config: Config,
    shape: str,
    checkpoint_path: Path,
    discovered_path: Path,
    stop_flag: Any,
    max_seconds: float | None = None,
) -> None:
    """The worker process body. `stop_flag` is anything with a boolean `.value` (the worker's
    shared byte). Can also be called in-process (tests); it then restores the caller's signal
    handlers and mask on return."""
    received: list[int] = []
    restore_signals = _install_signal_handlers(received)
    configure_logging(config.logging.dir)  # a spawned child does not inherit the parent's
    in_worker_process = mp.parent_process() is not None
    lock_file = None
    try:
        lock_file = acquire_lineage_lock(checkpoint_path)
        if lock_file is None:
            log_event(
                "discovery",
                level="error",
                event="checkpoint_locked",
                checkpoint=str(checkpoint_path),
                pid=os.getpid(),
            )
            if in_worker_process:
                sys.exit(EXIT_LOCKED)
            raise RuntimeError(f"another discovery worker holds {checkpoint_path}")
        _search(config, shape, checkpoint_path, discovered_path, stop_flag, received, max_seconds)
    except Exception as exc:
        log_event(
            "discovery",
            level="error",
            event="worker_crashed",
            pid=os.getpid(),
            checkpoint=str(checkpoint_path),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if lock_file is not None:
            lock_file.close()
        if not in_worker_process:
            restore_signals()


def _search(
    config: Config,
    shape: str,
    checkpoint_path: Path,
    discovered_path: Path,
    stop_flag: Any,
    received_signals: list[int],
    max_seconds: float | None,
) -> None:
    settings = es.search_settings(config, shape)
    lineage = es.settings_id(settings)
    max_count = es.settings_max_degree(settings)
    parent = mp.parent_process()
    deadline = time.monotonic() + max_seconds if max_seconds is not None else None

    def stop_reason() -> str | None:
        if stop_flag.value:
            return "requested"
        if received_signals:
            return f"signal {signal.Signals(received_signals[0]).name}"
        if parent is not None and not parent.is_alive():
            return "parent exited"
        if deadline is not None and time.monotonic() >= deadline:
            return "time limit"
        return None

    legacy = checkpoint_path.parent / LEGACY_CHECKPOINT_NAME
    if legacy.exists():
        log_event(
            "discovery",
            level="warning",
            event="legacy_checkpoint_ignored",
            checkpoint=str(legacy),
            reason="it predates per-settings lineages and records no search settings",
        )

    state = _load_checkpoint(checkpoint_path, settings)
    if state is None:
        entropy = int(np.random.SeedSequence().entropy % (2**63))
        population = es.initial_population(
            shape, config.grid.triangle_mode, config.discovery, _gen_rng(entropy, 0)
        )
        state = es.ESState(
            generation=0, population=population, shape=shape, entropy=entropy, settings=settings
        )
    saved = _saved_rules(discovered_path, lineage)

    log_event(
        "discovery",
        level="info",
        event="worker_started",
        pid=os.getpid(),
        generation=state.generation,
        shape=shape,
        settings=lineage,
        entropy=state.entropy,
        checkpoint=str(checkpoint_path),
    )

    reason = stop_reason()
    while reason is None:
        next_generation = state.generation + 1
        gen_rng = _gen_rng(state.entropy, next_generation)

        offspring = es.mutate_population(state.population, shape, config, gen_rng)
        pool = es.merge_duplicates([*state.population, *offspring])

        for candidate in pool:
            for _ in range(config.discovery.evals_per_candidate):
                reason = stop_reason()
                if reason is not None:
                    break
                score = es.evaluate_candidate(candidate.rule, shape, config, gen_rng.spawn(1)[0])
                candidate.add_evaluation(score)
            if reason is not None:
                break
        if reason is not None:
            break  # the unfinished generation is discarded; the checkpoint is the last full one

        new_population = es.select_elites(pool, config.discovery.mu)
        state = es.ESState(next_generation, new_population, shape, state.entropy, settings)

        for candidate in state.population:
            if candidate.mean_score < config.discovery.save_threshold:
                continue
            rule_string = candidate.rule.to_bs_string(max_count)
            if rule_string not in saved:
                _append_discovered(discovered_path, settings, rule_string, candidate)
                saved.add(rule_string)

        _save_checkpoint(checkpoint_path, state)
        log_event(
            "discovery",
            level="info",
            event="generation_complete",
            pid=os.getpid(),
            generation=state.generation,
            best_score=state.population[0].mean_score,
        )
        reason = stop_reason()

    log_event(
        "discovery",
        level="info",
        event="worker_stopped",
        pid=os.getpid(),
        generation=state.generation,
        reason=reason,
    )
