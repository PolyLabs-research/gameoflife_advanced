"""SDL-dummy UI smoke test: drives the App through `--frames` frames per shape without a real
display, asserting it doesn't crash or stop early. Run with `SDL_VIDEODRIVER=dummy`. Exits 1,
naming the shape, if any run ends before its frames are done.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from latticelife.config import DEFAULT_CONFIG_PATH, load_config
from latticelife.core.geometries import SHAPES, validate_dimensions
from latticelife.ui.app import App


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shape", choices=[*SHAPES, "all"], default="all")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    shapes = SHAPES if args.shape == "all" else (args.shape,)
    for shape in shapes:
        grid = replace(config.grid, shape=shape)
        try:
            validate_dimensions(shape, grid.width, grid.height, grid.wrap, grid.triangle_mode)
        except ValueError as exc:
            print(f"SMOKE-FAIL ({shape}): {exc}")
            return 1
        app = App(replace(config, grid=grid))
        frames = 0
        while frames < args.frames and app.step_frame():
            frames += 1
        generation = app.session.generation
        pygame.quit()
        if frames < args.frames:
            print(f"SMOKE-FAIL ({shape}): stopped after {frames} of {args.frames} frames")
            return 1
        print(f"  {shape}: {frames} frames, generation {generation}")

    print(f"SMOKE-OK ({args.frames} frames each: {', '.join(shapes)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
