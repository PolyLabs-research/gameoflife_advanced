"""Console entry points (§6.13). `autorun` is fully headless — no pygame dependency anywhere on
its path. Every command is wrapped in try/finally guaranteeing the recorder closes and logging an
unhandled-exception JSONL event before re-raising (§6.15). `print()` appears here, and only
here in `src/`, for user-facing CLI output.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

from latticelife.config import Config, load_config
from latticelife.core.geometries import max_degree
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.discovery.worker import DiscoveryWorker
from latticelife.logging import log_event
from latticelife.recording.recorder import Recorder
from latticelife.session.session import SimulationSession

DEFAULT_CONFIG_PATH = Path("config/default.toml")
SHAPES = ("square", "hexagon", "triangle")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latticelife")
    subparsers = parser.add_subparsers(dest="command", required=True)

    autorun = subparsers.add_parser(
        "autorun", help="Fully headless batch of sessions, recorded to schema v2."
    )
    autorun.add_argument("--sessions", type=int, required=True)
    autorun.add_argument("--generations", type=int, required=True)
    autorun.add_argument("--shape", choices=(*SHAPES, "random"), default="random")
    autorun.add_argument("--db", type=Path, default=None)
    autorun.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    discover = subparsers.add_parser(
        "discover", help="Run the (mu+lambda) ES worker in the foreground."
    )
    discover.add_argument("--minutes", type=float, default=60.0)
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


def run_autorun(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    db_path = args.db if args.db is not None else Path(config.recording.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    top_level_rng = random.Random(config.simulation.seed or None)
    summary = {"sessions": 0, "generations": 0, "db_path": str(db_path)}

    with Recorder(db_path, batch_size=config.recording.batch_size) as recorder:
        for _ in range(args.sessions):
            session_config, lifeforms = _randomise_session_params(
                config, args.shape, top_level_rng
            )
            seed = top_level_rng.randrange(2**32)
            session = SimulationSession(session_config, lifeforms, seed=seed)

            session_id = str(uuid.uuid4())
            recorder.start_session(
                session_id, session_config, lifeforms, args.generations, seed
            )

            session.run(args.generations)
            for record in session.history_records:
                recorder.record_generation(session_id, record)

            summary["sessions"] += 1
            summary["generations"] += session.generation

    return summary


def run_discover(args: argparse.Namespace) -> None:
    """Runs the ES worker in the foreground for up to `--minutes`, with signal-based clean
    shutdown: SIGINT sets the worker's stop event instead of killing the process outright."""
    config = load_config(args.config)
    worker = DiscoveryWorker(config, shape=args.shape)

    def handle_sigint(signum: int, frame: object) -> None:
        worker.stop()

    previous_handler = signal.signal(signal.SIGINT, handle_sigint)
    worker.start()
    try:
        deadline = time.monotonic() + args.minutes * 60
        while worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.5)
    finally:
        worker.stop()
        worker.join(timeout=10)
        signal.signal(signal.SIGINT, previous_handler)


def run_ui(args: argparse.Namespace) -> None:
    """Launches the interactive pygame app. Imported lazily so `autorun`/`discover` never pull
    in pygame on their path (§6.13's headless commands must never import pygame)."""
    from latticelife.ui.app import App

    config = load_config(args.config)
    App(config).run()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "autorun":
            print(json.dumps(run_autorun(args)))
        elif args.command == "discover":
            run_discover(args)
        elif args.command == "ui":
            run_ui(args)
        return 0
    except Exception as exc:
        log_event("app", level="error", event="unhandled_exception", error=str(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
