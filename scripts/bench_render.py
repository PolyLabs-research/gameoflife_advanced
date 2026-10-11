"""Benchmark the App's real grid-drawing path at the app's own window, layout and cell size.
Chunk 5's gate requires >= 30 fps at `--shape square --size 200 --frames 300`.

For each shape it reports, separately:
- paused: a full frame draw (`App._draw` + flip) of an unchanging grid — the steady state;
- running: the same draw while the grid changes every frame (hex/triangle redraw changed cells);
- step: `SimulationSession.step()` alone, in generations per second.
Drawing and stepping are timed apart, and the frame-rate cap is bypassed, so the figures are
the costs themselves rather than the UI loop's 60 Hz pacing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from latticelife.config import DEFAULT_CONFIG_PATH, load_config
from latticelife.core.geometries import SHAPES, validate_dimensions
from latticelife.ui.app import App


def _rate(count: int, seconds: float) -> float:
    return count / seconds if seconds > 0 else float("inf")


def bench(config, shape: str, size: int, frames: int) -> tuple[float, float, float]:
    grid = replace(config.grid, shape=shape, width=size, height=size)
    validate_dimensions(shape, size, size, grid.wrap, grid.triangle_mode)
    app = App(replace(config, grid=grid))
    app.session.paused = True
    app._draw()  # the first, full draw is not the steady state

    start = time.perf_counter()
    for _ in range(frames):
        app._draw()
        pygame.display.flip()
    paused = _rate(frames, time.perf_counter() - start)

    draw_seconds = step_seconds = 0.0
    for _ in range(frames):
        t0 = time.perf_counter()
        app.session.step()
        t1 = time.perf_counter()
        app._draw()
        pygame.display.flip()
        draw_seconds += time.perf_counter() - t1
        step_seconds += t1 - t0
    pygame.quit()
    return paused, _rate(frames, draw_seconds), _rate(frames, step_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=[*SHAPES, "all"], default="square")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    shapes = SHAPES if args.shape == "all" else (args.shape,)
    for shape in shapes:
        try:
            paused, running, step = bench(config, shape, args.size, args.frames)
        except ValueError as exc:
            print(f"{shape}: {exc}")
            return 1
        print(
            f"{paused:.1f} fps paused, {running:.1f} fps running, {step:.1f} gen/s step "
            f"({args.frames} frames, shape={shape}, size={args.size})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
