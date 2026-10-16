"""Console entry points (§6.13). `autorun` is fully headless — no pygame dependency anywhere on
its path. Every command is wrapped so the recorder always closes and an unexpected exception is
logged as an `unhandled_exception` JSONL event before it is re-raised (§6.15). `print()`
appears here, and only here in `src/`, for user-facing CLI output.

Exit status: 0 success; 1 the discovery worker failed (crashed, did not stop, or another
`discover` run holds the same lineage); 2 bad arguments, an invalid config, or a `--db` path
that is not a LatticeLife schema v2 database, cannot be opened or written (a read-only file, a
folder that cannot be written), or is locked by another process (each reported in one line on
stderr, no traceback); 130 an `autorun` interrupted by Ctrl+C, SIGTERM or SIGHUP, after
everything recorded so far has been flushed (or a `discover` whose worker a second Ctrl+C had
killed).

`autorun` streams each generation to the Recorder the moment it is produced, so a session's
recording never depends on the session's bounded history buffers, and it runs sessions with
`auto_pause=False`: sudden-death alerts are still computed, but every planned generation is
recorded (legacy auto-run did the same; stopping at the alert biased the data toward sessions
that never collapse).

`discover` runs the ES worker in the foreground until `--minutes` pass or it is told to stop.
SIGINT (Ctrl+C), SIGTERM and SIGHUP all request a clean stop; the handler only records the
request, and the polling loop then stops the worker, so no handler ever touches a lock.

`census`, `sweep`, `soup` and `damage` are the headless mathematics experiments
(docs/experiments.md): each appends one JSON record per task to a JSONL file as the task
finishes, skips the tasks that file already holds when started again, reports progress on
stderr, and prints a JSON summary on stdout. Ctrl+C, SIGTERM and SIGHUP stop one with
everything finished so far written (exit 130). `--grid` takes `L` for a square torus or `WxH`
for a rectangle. Their arguments that depend on each other (a rule's counts against the
lattice's degree, a torus size against the shape) are checked before anything runs, and a
problem is one line on stderr with exit status 2, like a bad config. `observe` reads such files
and prints their observables (`experiments.observables`), one JSON row per line.
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

import numpy as np

from latticelife.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    load_config,
    log_dir_for,
    project_root_for,
)
from latticelife.core.geometries import SHAPES, max_degree, validate_dimensions
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.discovery.worker import EXIT_LOCKED, STOP_SIGNALS, DiscoveryWorker
from latticelife.experiments import census, damage, objects, observables, sweeps
from latticelife.experiments.harness import grid_dimensions, parse_grid, read_jsonl
from latticelife.logging import configure_logging, log_event, log_path
from latticelife.recording.recorder import Recorder, RecorderOpenError, RecorderSchemaError
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


def _unit_fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1 (got {text})")
    return value


def _grid_size(text: str) -> int | str:
    """A torus size: `L` for a square torus, `WxH` for a rectangle."""
    try:
        return parse_grid(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0 (got {value})")
    return value


def _even_positive_int(text: str) -> int:
    value = _positive_int(text)
    if value % 2:
        raise argparse.ArgumentTypeError(f"must be an even number (got {value})")
    return value


# How often the experiment commands report progress on stderr.
_PROGRESS_EVERY = 100


_GRID_HELP = "torus sizes: L for a square torus, WxH for a rectangle (default 32)"


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

    census_parser = subparsers.add_parser(
        "census",
        help="Classify rules on one lattice (outcome, exact period, density) into a JSONL file.",
    )
    census_parser.add_argument("--shape", choices=SHAPES, required=True)
    census_parser.add_argument(
        "--rules",
        nargs="+",
        default=["all"],
        metavar="RULE",
        help="'all' (every rule of the lattice), 'dual' (one per complement pair), "
        "'random:N' (N random rules, reproducible with simulation.seed), or B/S strings",
    )
    census_parser.add_argument("--grid", type=_grid_size, nargs="+", default=[32], help=_GRID_HELP)
    census_parser.add_argument("--steps", type=_positive_int, default=500)
    census_parser.add_argument("--seeds", type=_positive_int, default=1, help="soups per rule")
    census_parser.add_argument("--density", type=_unit_fraction, default=0.5)
    census_parser.add_argument(
        "--tail", type=_positive_int, default=50, help="generations averaged at the end"
    )
    _add_experiment_arguments(census_parser)

    sweep_parser = subparsers.add_parser(
        "sweep", help="Survival of one rule over initial densities, torus sizes and seeds."
    )
    sweep_parser.add_argument("--rule", required=True, help="a B/S string, e.g. B3/S23")
    sweep_parser.add_argument("--shape", choices=SHAPES, required=True)
    sweep_parser.add_argument("--grid", type=_grid_size, nargs="+", default=[32], help=_GRID_HELP)
    sweep_parser.add_argument(
        "--densities",
        nargs="+",
        default=["0.05:0.95:0.05"],
        metavar="DENSITY",
        help="initial densities: numbers, or LO:HI:STEP ranges (default 0.05:0.95:0.05)",
    )
    sweep_parser.add_argument("--seeds", type=_positive_int, default=10, help="soups per point")
    sweep_parser.add_argument("--steps", type=_positive_int, default=500)
    _add_experiment_arguments(sweep_parser)

    soup_parser = subparsers.add_parser(
        "soup", help="Catalogue the objects (still lifes, oscillators, spaceships) soups leave."
    )
    soup_parser.add_argument("--rule", required=True, help="a B/S string, e.g. B3/S23")
    soup_parser.add_argument("--shape", choices=SHAPES, required=True)
    soup_parser.add_argument(
        "--grid", type=_grid_size, default=64, help="torus size: L, or WxH (default 64)"
    )
    soup_parser.add_argument("--seeds", type=_positive_int, default=20, help="number of soups")
    soup_parser.add_argument("--steps", type=_positive_int, default=1000)
    soup_parser.add_argument("--density", type=_unit_fraction, default=0.3)
    soup_parser.add_argument(
        "--radius", type=_positive_int, default=2, help="cells closer than this are one object"
    )
    soup_parser.add_argument(
        "--max-period", type=_positive_int, default=200, help="generations to settle an object"
    )
    soup_parser.add_argument(
        "--margin", type=_even_positive_int, default=8, help="room round an object (even)"
    )
    _add_experiment_arguments(soup_parser)

    damage_parser = subparsers.add_parser(
        "damage", help="Damage spreading: how far one flipped cell's effect spreads."
    )
    damage_parser.add_argument("--rule", required=True, help="a B/S string, e.g. B3/S23")
    damage_parser.add_argument("--shape", choices=SHAPES, required=True)
    damage_parser.add_argument("--grid", type=_grid_size, nargs="+", default=[32], help=_GRID_HELP)
    damage_parser.add_argument(
        "--densities",
        nargs="+",
        default=["0.5"],
        metavar="DENSITY",
        help="initial densities: numbers, or LO:HI:STEP ranges (default 0.5)",
    )
    damage_parser.add_argument("--seeds", type=_positive_int, default=10, help="runs per point")
    damage_parser.add_argument(
        "--steps", type=_positive_int, default=200, help="generations after the flip"
    )
    damage_parser.add_argument("--flips", type=_positive_int, default=1, help="cells flipped")
    damage_parser.add_argument(
        "--warmup", type=_non_negative_int, default=0, help="generations run before the flip"
    )
    _add_experiment_arguments(damage_parser)

    observe_parser = subparsers.add_parser(
        "observe", help="Observables (numbers with intervals) from experiment JSONL files."
    )
    observe_parser.add_argument(
        "quantity",
        choices=("transient", "survival", "damage"),
        help="transient: median settling time (census or sweep files); survival: share alive "
        "(sweep); damage: final damage (damage)",
    )
    observe_parser.add_argument("files", nargs="+", type=Path, help="JSONL record files")
    observe_parser.add_argument("--rule", default=None, help="only records of this B/S rule")

    return parser


def _add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=_positive_int, default=1, help="worker processes")
    parser.add_argument(
        "--out", type=Path, default=None, help="JSONL output (default under the project's out/)"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)


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


def _load_config(path: Path, *, for_discovery: bool = False) -> Config:
    """Load the config and send this process's JSONL logs to its `logging.dir` (§6.10). A config
    file that fails to load still logs next to itself (`log_dir_for`), not in this copy's own
    `out/logs`: a bad config from another project must not write into this checkout. Only a
    path with no file behind it falls back to the default log directory. `for_discovery` adds
    the discovery-only checks (`config.validate_discovery_config`), for `discover` alone."""
    try:
        config = load_config(path, for_discovery=for_discovery)
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
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        detail = f"{exc.strerror}: {exc.filename}" if exc.strerror and exc.filename else exc
        raise RecorderOpenError(f"cannot create the folder for {db_path} ({detail})") from None

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
    config = _load_config(args.config, for_discovery=True)
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


# --- the mathematics experiments -------------------------------------------------------------


def _parse_rule(text: str, shape: str, config: Config) -> Rule:
    """A B/S string checked against the lattice: a count above its degree is a ConfigError."""
    try:
        rule = Rule.from_bs_string(text)
    except ValueError as exc:
        raise ConfigError(f"--rule(s): {exc}") from None
    n = max_degree(shape, config.grid.triangle_mode)
    too_big = sorted((rule.birth | rule.survive) - set(range(n + 1)))
    if too_big:
        raise ConfigError(
            f"--rule(s): {text} names neighbour count(s) {too_big}, but {shape} cells have "
            f"{n} neighbours"
        )
    return rule


# The largest rule space `census --rules all` will enumerate: the square lattice's 2^18. The
# edge+vertex triangle lattice has 2^26 (67 million) rules, which no run will get through.
_MAX_ENUMERATED_RULES = 2**18


def _census_rules(specs: list[str], shape: str, config: Config) -> list[Rule]:
    n = max_degree(shape, config.grid.triangle_mode)
    if specs in (["all"], ["dual"]):
        size = census.rule_space_size(n)
        if size > _MAX_ENUMERATED_RULES:
            raise ConfigError(
                f"--rules {specs[0]}: {shape} cells with {n} neighbours have 2^{2 * (n + 1)} = "
                f"{size:,} rules, too many to enumerate; use random:N or list the rules"
            )
        return list(census.enumerate_rules(n, up_to_duality=specs == ["dual"]))
    if len(specs) == 1 and specs[0].startswith("random:"):
        count = _positive_int(specs[0].removeprefix("random:"))
        rng = np.random.default_rng(config.simulation.seed or None)
        return [Rule.random(n, 0.5, 0.5, rng) for _ in range(count)]
    if any(spec in ("all", "dual") or spec.startswith("random:") for spec in specs):
        raise ConfigError("--rules: 'all', 'dual' and 'random:N' stand alone, not in a list")
    return [_parse_rule(spec, shape, config) for spec in specs]


def _check_torus_sizes(shape: str, sizes: list[int | str], config: Config) -> None:
    for size in sizes:
        width, height = grid_dimensions(size)
        try:
            validate_dimensions(shape, width, height, True, config.grid.triangle_mode)
        except ValueError as exc:
            raise ConfigError(f"--grid {size}: {exc}") from None


def _parse_densities(specs: list[str]) -> list[float]:
    """Numbers and LO:HI:STEP ranges (inclusive of HI when STEP divides the range), in order,
    each kept to 6 decimals so a range's values match the numbers a later run would type."""
    densities: list[float] = []
    for spec in specs:
        if ":" in spec:
            parts = spec.split(":")
            if len(parts) != 3:
                raise ConfigError(f"--densities: {spec!r} is not LO:HI:STEP")
            try:
                low, high, step = (float(part) for part in parts)
            except ValueError:
                raise ConfigError(f"--densities: {spec!r} is not LO:HI:STEP") from None
            if step <= 0 or high < low:
                raise ConfigError(f"--densities: {spec!r} needs LO <= HI and STEP > 0")
            count = int(round((high - low) / step)) + 1
            values = [
                round(low + i * step, 6) for i in range(count) if low + i * step <= high + 1e-9
            ]
        else:
            try:
                values = [round(float(spec), 6)]
            except ValueError:
                raise ConfigError(f"--densities: {spec!r} is not a number") from None
        for value in values:
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"--densities: {value} is not between 0 and 1")
            if value not in densities:
                densities.append(value)
    if not densities:
        raise ConfigError("--densities: no densities given")
    return densities


def _rule_slug(rule: Rule, n: int) -> str:
    return rule.to_bs_string(n).replace("/", "").replace(",", "-")


def _default_out(args: argparse.Namespace, name: str) -> Path:
    return args.out if args.out is not None else project_root_for(args.config) / "out" / name


def _progress(label: str, done: int, total: int) -> None:
    if done % _PROGRESS_EVERY == 0 or done == total:
        print(f"latticelife {label}: {done}/{total} records", file=sys.stderr, flush=True)


def _run_experiment(label: str, records, total: int, summary: dict) -> int:
    """Drain a record generator, reporting progress; on an interrupt, note it in the summary
    and the log. Returns the exit status."""
    done = 0
    with _termination_interrupts():
        try:
            for _ in records:
                done += 1
                _progress(label, done, total)
        except KeyboardInterrupt:
            summary["interrupted"] = True
            log_event(
                "app",
                level="warning",
                event="experiment_interrupted",
                command=label,
                completed=done,
                planned=total,
            )
    summary["records"] = done
    return EXIT_INTERRUPTED if summary.get("interrupted") else EXIT_OK


def run_census(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    _check_torus_sizes(args.shape, args.grid, config)
    rules = _census_rules(args.rules, args.shape, config)
    out = _default_out(args, f"census-{args.shape}.jsonl")
    seeds = list(range(1, args.seeds + 1))
    tasks = census.census_tasks(
        config, args.shape, rules, args.grid, seeds, args.steps, args.density, args.tail
    )
    summary: dict = {"out": str(out), "rules": len(rules), "planned": len(tasks)}
    records: list[census.RuleRecord] = []

    def collect():
        for record in census.run_census(
            config,
            args.shape,
            rules,
            args.grid,
            seeds,
            args.steps,
            args.density,
            out,
            workers=args.workers,
            tail=args.tail,
        ):
            records.append(record)
            yield record

    status = _run_experiment("census", collect(), len(tasks), summary)
    summary["skipped"] = len(tasks) - len(records) if not summary.get("interrupted") else None
    summary.update(census.summarise(records))
    print(json.dumps(summary))
    return status


def run_sweep(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    rule = _parse_rule(args.rule, args.shape, config)
    _check_torus_sizes(args.shape, args.grid, config)
    densities = _parse_densities(args.densities)
    n = max_degree(args.shape, config.grid.triangle_mode)
    out = _default_out(args, f"sweep-{args.shape}-{_rule_slug(rule, n)}.jsonl")
    seeds = list(range(1, args.seeds + 1))
    total = len(args.grid) * len(densities) * len(seeds)
    summary: dict = {"out": str(out), "rule": rule.to_bs_string(n), "planned": total}
    records = sweeps.run_sweep(
        config, rule, args.shape, args.grid, densities, seeds, args.steps, out, args.workers
    )
    status = _run_experiment("sweep", records, total, summary)
    # Thresholds from everything in the file, this run's records and earlier ones alike.
    table = sweeps.survival_table(
        r
        for r in read_jsonl(out)
        if r.get("rule") == summary["rule"] and r.get("steps") == args.steps
    )
    summary["critical_densities"] = {
        str(grid): sweeps.critical_densities([(d, p) for d, p, _ in points])
        for grid, points in table.items()
    }
    print(json.dumps(summary))
    return status


def run_soup(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    rule = _parse_rule(args.rule, args.shape, config)
    _check_torus_sizes(args.shape, [args.grid], config)
    n = max_degree(args.shape, config.grid.triangle_mode)
    out = _default_out(args, f"soups-{args.shape}-{_rule_slug(rule, n)}.jsonl")
    seeds = list(range(1, args.seeds + 1))
    summary: dict = {"out": str(out), "rule": rule.to_bs_string(n), "planned": len(seeds)}
    records = objects.run_soups(
        config,
        rule,
        args.shape,
        args.grid,
        seeds,
        args.steps,
        args.density,
        out,
        workers=args.workers,
        radius=args.radius,
        max_period=args.max_period,
        margin=args.margin,
    )
    status = _run_experiment("soup", records, len(seeds), summary)
    listed = objects.catalogue(
        r
        for r in read_jsonl(out)
        if r.get("rule") == summary["rule"] and r.get("grid") == args.grid
    )
    summary["distinct_objects"] = len(listed)
    summary["objects"] = listed[:10]
    print(json.dumps(summary))
    return status


def run_damage(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    rule = _parse_rule(args.rule, args.shape, config)
    _check_torus_sizes(args.shape, args.grid, config)
    densities = _parse_densities(args.densities)
    n = max_degree(args.shape, config.grid.triangle_mode)
    out = _default_out(args, f"damage-{args.shape}-{_rule_slug(rule, n)}.jsonl")
    seeds = list(range(1, args.seeds + 1))
    total = len(args.grid) * len(densities) * len(seeds)
    summary: dict = {"out": str(out), "rule": rule.to_bs_string(n), "planned": total}
    records = damage.run_damage(
        config,
        rule,
        args.shape,
        args.grid,
        densities,
        seeds,
        args.steps,
        out,
        workers=args.workers,
        flips=args.flips,
        warmup=args.warmup,
    )
    status = _run_experiment("damage", records, total, summary)
    summary["by_grid"] = damage.summarise(
        r
        for r in read_jsonl(out)
        if r.get("rule") == summary["rule"]
        and r.get("steps") == args.steps
        and r.get("flips") == args.flips
        and r.get("warmup") == args.warmup
    )
    print(json.dumps(summary))
    return status


def run_observe(args: argparse.Namespace) -> int:
    records = []
    for path in args.files:
        if not path.is_file():
            raise ConfigError(f"{path}: no such file")
        records.extend(read_jsonl(path))
    if args.rule is not None:
        try:
            wanted = Rule.from_bs_string(args.rule)
        except ValueError as exc:
            raise ConfigError(f"--rule: {exc}") from None
        wanted_sets = (wanted.birth, wanted.survive)
        kept = []
        for record in records:
            rule = Rule.from_bs_string(record["rule"])
            if (rule.birth, rule.survive) == wanted_sets:
                kept.append(record)
        records = kept
    try:
        if args.quantity == "transient":
            rows = observables.median_transient(records)
        elif args.quantity == "survival":
            rows = observables.survival(records)
        else:
            rows = observables.damage_spread(records)
    except KeyError as exc:
        raise ConfigError(f"records lack the field {exc} that {args.quantity} needs") from None
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    for row in rows:
        print(json.dumps(row.to_json()))
    return EXIT_OK


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
        if args.command == "census":
            return run_census(args)
        if args.command == "sweep":
            return run_sweep(args)
        if args.command == "soup":
            return run_soup(args)
        if args.command == "damage":
            return run_damage(args)
        if args.command == "observe":
            return run_observe(args)
        run_ui(args)
        return EXIT_OK
    except (ConfigError, RecorderSchemaError, RecorderOpenError) as exc:
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
