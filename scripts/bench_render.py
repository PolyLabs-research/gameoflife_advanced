"""Benchmark the grid-rendering path. Prints fps; Chunk 5's gate requires >= 30 fps at
`--shape square --size 200 --frames 300`.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from latticelife.core.lattice import LatticeGrid
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.ui import render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=["square", "hexagon", "triangle"], default="square")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((args.size, args.size))

    lifeform = Lifeform(1, Rule(birth={3}, survive={2, 3}))
    grid = LatticeGrid([lifeform], args.shape, args.size, args.size, wrap=True)
    rng = np.random.default_rng(0)
    alive = rng.random(grid.n_cells) < 0.3
    grid.occupancy[alive] = 1
    grid.lifespans[alive] = 1

    palette = render.build_palette([lifeform])
    cell_size = max(1, 800 // args.size)
    geometric = (
        render.GeometricRenderer(args.shape, cell_size, 0, 0) if args.shape != "square" else None
    )

    start = time.perf_counter()
    for _ in range(args.frames):
        grid.step(rng)
        if geometric is None:
            render.render_square(screen, grid, palette, cell_size, 0, 0)
        else:
            geometric.draw(screen, grid, palette)
        pygame.display.flip()
    elapsed = time.perf_counter() - start

    fps = args.frames / elapsed if elapsed > 0 else float("inf")
    print(
        f"{fps:.1f} fps ({args.frames} frames in {elapsed:.3f}s, "
        f"shape={args.shape}, size={args.size})"
    )


if __name__ == "__main__":
    main()
