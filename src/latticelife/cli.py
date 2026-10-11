"""Console entry points (§6.13). `autorun` is fully headless — no pygame dependency anywhere on
its path. Every command is wrapped so the recorder always closes and an unexpected exception is
logged as an `unhandled_exception` JSONL event before it is re-raised (§6.15). `print()`
appears here, and only here in `src/`, for user-facing CLI output.

Exit status: 0 success; 1 the discovery worker failed (crashed, did not stop, or another
`discover` run holds the same lineage); 2 bad arguments, an invalid config, or a `--db` file
that is not a LatticeLife schema v2 database (each reported in one line on stderr, no
traceback); 130 an `autorun` interrupted by Ctrl+C, SIGTERM or SIGHUP, after everything
recorded so far has been flushed (or a `discover` whose worker a second Ctrl+C had killed).

`autorun` streams each generation to the Recorder the moment it is produced, so a session's
recording never depends on the session's bounded history buffers, and it runs sessions with
`auto_pause=False`: sudden-death alerts are still computed, but every planned generation is
recorded (legacy auto-run did the same; stopping at the alert biased the data toward sessions
that never collapse).

`discover` runs the ES worker in the foreground until `--minutes` pass or it is told to stop.
SIGINT (Ctrl+C), SIGTERM and SIGHUP all request a clean stop; the handler only records the
request, and the polling loop then stops the worker, so no handler ever touches a lock.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import signal
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from latticelife.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config, log_dir_for
from latticelife.core.geometries import SHAPES, max_degree, validate_dimensions
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.discovery.worker import EXIT_LOCKED, STOP_SIGNALS, DiscoveryWorker
from latticelife.logging import configure_logging, log_event, log_path
from latticelife.recording.recorder import Recorder, RecorderSchemaError
from latticelife.session.session import SimulationSession

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

# discover: how often the foreground loop checks the worker, how long a stopping worker may
# take to finish its current evaluation before it is killed, and how far past --minutes the
# worker's own backstop time limit lies.
_POLL_SECONDS = 0.2
_STOP_TIMEOUT_SECONDS = 30.0
_WORKER_GRACE_SECONDS = 60.0


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 (got {value})")
    return value


def _positive_minutes(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of minutes, got {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number of minutes (got {text})")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latticelife")
    subparsers = parser.add_subparsers(dest="command", required=True)

    autorun = subparsers.add_parser(
        "autorun", help="Fully headless batch of sessions, recorded to schema v2."
    )
    autorun.add_argument("--sessions", type=_positive_int, required=True)
    autorun.add_argument("--generations", type=_positive_int, required=True)
    autorun.add_argument("--shape", choices=(*SHAPES, "random"), default="random")
    autorun.add_argument("--db", type=Path, default=None)
    autorun.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    discover = subparsers.add_parser(
        "discover", help="Run the (mu+lambda) ES worker in the foreground."
    )
    discover.add_argument("--minutes", type=_positive_minutes, default=60.0)
    discover.add_argument("--shape", choices=SHAPES, default="square")
    discover.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    ui = subparsers.add_parser("ui", help="Interactive pygame app.")
    ui.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    return parser


def _randomise_rules(
    shape: str, triangle_mode: str, n_lifeforms: int, rng: random.Random
) -> list[Lifeform]:
    """Mirrors the legacy `game.py:randomise_lifeforms` sampling exactly: for each lifeform,
    birth/survive are a sorted random sample of 1-4 values (capped at the shape's max degree)
    drawn from the full neighbor-count range."""
    max_n = max_degree(shape, triangle_mode)
    lifeforms = []
    for i in range(1, n_lifeforms + 1):
        birth = set(rng.sample(range(max_n + 1), rng.randint(1, min(4, max_n + 1))))
        survive = set(rng.sample(range(max_n + 1), rng.randint(1, min(4, max_n + 1))))
        lifeforms.append(Lifeform(id=i, rule=Rule(birth=birth, survive=survive)))
    return lifeforms


def _randomise_session_params(
    config: Config, shape_arg: str, rng: random.Random
) -> tuple[Config, list[Lifeform]]:
    """Mirrors the legacy `auto_run_simulations` randomization: shape, initial density, and
    lifeform count/rules vary per session; grid dimensions/wrap/triangle_mode do not — the
    legacy auto-run never varied them either."""
    shape = shape_arg if shape_arg != "random" else rng.choice(SHAPES)
    initial_alive_percentage = rng.uniform(0.01, 0.9)
    n_lifeforms = rng.randint(1, 10)

    session_config = replace(
        config,
        grid=replace(config.grid, shape=shape),
        simulation=replace(
            config.simulation,
            initial_alive_percentage=initial_alive_percentage,
            n_lifeforms=n_lifeforms,
        ),
    )
    lifeforms = _randomise_rules(shape, config.grid.triangle_mode, n_lifeforms, rng)
    return session_config, lifeforms


def _load_config(path: Path) -> Config:
    """Load the config and send this process's JSONL logs to its `logging.dir` (§6.10). A config
    file that fails to load still logs next to itself (`log_dir_for`), not in this copy's own
    `out/logs`: a bad config from another project must not write into this checkout. Only a
    path with no file behind it falls back to the default log directory."""
    try:
        config = load_config(path)
    except ConfigError:
        configure_logging(log_dir_for(path))
        raise
    configure_logging(config.logging.dir)
    return config


def _check_autorun_shapes(config: Config, shape_arg: str) -> None:
    """The config's grid is only validated for its own `grid.shape`; autorun may run it as
    another shape (`--shape`, or every shape with `random`), so check those up front rather than
    failing partway through a batch."""
    grid = config.grid
    for shape in SHAPES if shape_arg == "random" else (shape_arg,):
        try:
            validate_dimensions(shape, grid.width, grid.height, grid.wrap, grid.triangle_mode)
        except ValueError as exc:
            wrapped = "wrapped " if grid.wrap else ""
            raise ConfigError(
                f"--shape {shape_arg}: the configured {wrapped}{grid.width}x{grid.height} grid "
                f"cannot be run as {shape}: {exc}"
            ) from None


@contextlib.contextmanager
def _termination_interrupts() -> Iterator[None]:
    """While active, SIGTERM and SIGHUP raise KeyboardInterrupt like Ctrl+C does, so `kill` or a
    closed terminal unwinds through the Recorder and flushes instead of dropping its buffer."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.signal(sig, signal.default_int_handler) for sig in STOP_SIGNALS}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_autorun(args: argparse.Namespace) -> dict:
    """Run and record the batch. Returns the summary: `sessions` completed, `generations`
    recorded in total, the DB path, and `interrupted: true` when it was stopped early."""
    config = _load_config(args.config)
    _check_autorun_shapes(config, args.shape)
    db_path = args.db if args.db is not None else Path(config.recording.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    top_level_rng = random.Random(config.simulation.seed or None)
    summary: dict = {"sessions": 0, "generations": 0, "db_path": str(db_path)}
    generations = args.generations

    with (
        _termination_interrupts(),
        Recorder(db_path, batch_size=config.recording.batch_size) as recorder,
    ):
        session_id = None
        session_generation = 0
        try:
            for _ in range(args.sessions):
                session_config, lifeforms = _randomise_session_params(
                    config, args.shape, top_level_rng
                )
                seed = top_level_rng.randrange(2**32)
                session = SimulationSession(session_config, lifeforms, seed=seed, auto_pause=False)

                new_id = str(uuid.uuid4())
                session_generation = 0
                recorder.start_session(new_id, session_config, lifeforms, generations, seed)
                session_id = new_id

                for i in range(generations):
                    session.step(force_clusters=(i == generations - 1))
                    # history_limit >= 1 (validated), so the newest record is always here.
                    recorder.record_generation(session_id, session.history_records[-1])
                    session_generation = session.generation
                    summary["generations"] += 1

                summary["sessions"] += 1
                session_id = None
        except KeyboardInterrupt:
            summary["interrupted"] = True
            log_event(
                "app",
                level="warning",
                event="autorun_interrupted",
                completed_sessions=summary["sessions"],
                recorded_generations=summary["generations"],
                partial_session_id=session_id,
                partial_session_generations=session_generation if session_id else None,
                planned_generations=generations,
            )

    return summary


def _discover_summary(worker: DiscoveryWorker) -> dict:
    generation = 0
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        generation = json.loads(worker.checkpoint_path.read_text())["generation"]
    return {
        "shape": worker.shape,
        "generation": generation,
        "checkpoint": str(worker.checkpoint_path),
        "discovered_path": str(worker.discovered_path),
    }


def run_discover(args: argparse.Namespace) -> int:
    """Runs the ES worker in the foreground for up to `--minutes`, with signal-based clean
    shutdown. Returns the exit status: 0 after a clean stop (time up, Ctrl+C, SIGTERM, SIGHUP),
    1 when the worker crashed, was locked out by another run of the same lineage, or had to be
    killed because it did not stop in time, 130 when a second stop signal had it killed."""
    config = _load_config(args.config)
    worker = DiscoveryWorker(config, shape=args.shape)

    stop_requests: list[int] = []

    def request_stop(signum: int, frame: object) -> None:
        # Only record it: worker.stop() and everything else happen in the loop below.
        stop_requests.append(signum)

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, *STOP_SIGNALS)
    }
    kill_reason = None
    forced = False
    try:
        run_seconds = args.minutes * 60
        worker.start(max_seconds=run_seconds + _WORKER_GRACE_SECONDS)
        deadline = time.monotonic() + run_seconds
        while worker.is_alive() and not stop_requests and time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
    finally:
        # The worker stops after its current evaluation. Another stop signal while waiting
        # (a second Ctrl+C) or the timeout kills it; its checkpoint is a whole generation.
        worker.stop()
        requests_before = len(stop_requests)
        stop_deadline = time.monotonic() + _STOP_TIMEOUT_SECONDS
        while (
            worker.is_alive()
            and len(stop_requests) == requests_before
            and time.monotonic() < stop_deadline
        ):
            worker.join(timeout=_POLL_SECONDS)
        if worker.is_alive():
            forced = len(stop_requests) > requests_before
            kill_reason = (
                "stop requested again"
                if forced
                else f"did not stop within {_STOP_TIMEOUT_SECONDS:g} s"
            )
            log_event(
                "app",
                level="error",
                event="discovery_worker_killed",
                pid=worker.pid,
                reason=kill_reason,
            )
            worker.kill()
            worker.join(timeout=10)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    exitcode = worker.exitcode
    if exitcode == 0:
        print(json.dumps(_discover_summary(worker)))
        return EXIT_OK

    if kill_reason is not None:
        message = (
            f"the discovery worker was killed ({kill_reason}); its checkpoint holds the last "
            "complete generation"
        )
        print(f"latticelife discover: {message}", file=sys.stderr)
        return EXIT_INTERRUPTED if forced else EXIT_FAILED
    if exitcode == EXIT_LOCKED:
        message = f"another discover run is already using {worker.checkpoint_path}"
    else:
        message = f"the discovery worker failed (exit code {exitcode}); see {log_path('discovery')}"
    log_event(
        "app",
        level="error",
        event="discovery_worker_failed",
        exitcode=exitcode,
        checkpoint=str(worker.checkpoint_path),
        error=message,
    )
    print(f"latticelife discover: {message}", file=sys.stderr)
    return EXIT_FAILED


def run_ui(args: argparse.Namespace) -> None:
    """Launches the interactive pygame app. Imported lazily so `autorun`/`discover` never pull
    in pygame on their path (§6.13's headless commands must never import pygame)."""
    from latticelife.ui.app import App

    config = _load_config(args.config)
    App(config).run()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "autorun":
            summary = run_autorun(args)
            print(json.dumps(summary))
            return EXIT_INTERRUPTED if summary.get("interrupted") else EXIT_OK
        if args.command == "discover":
            return run_discover(args)
        run_ui(args)
        return EXIT_OK
    except (ConfigError, RecorderSchemaError) as exc:
        log_event("app", level="error", event="invalid_input", command=args.command, error=str(exc))
        print(f"latticelife {args.command}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        log_event("app", level="warning", event="interrupted", command=args.command)
        return EXIT_INTERRUPTED
    except Exception as exc:
        log_event(
            "app",
            level="error",
            event="unhandled_exception",
            command=args.command,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
