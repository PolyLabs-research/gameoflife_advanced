"""Render the README's images into docs/images without a display (SDL dummy driver): the app at
its 1200 x 800 window on each lattice, the 3D torus view (the `[torus]` extra and an OpenGL 3.3
driver; EGL works headless), the help overlay, an alert, and a banner of the three lattices
drawn grid-only. Every run is seeded, so running it again reproduces the same images.

    uv run python scripts/screenshots.py            # all of them
    uv run python scripts/screenshots.py banner ui-torus
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from latticelife.config import DEFAULT_CONFIG_PATH, Config, load_config
from latticelife.core.geometries import validate_dimensions
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.session.session import SimulationSession
from latticelife.ui import render
from latticelife.ui.app import BACKGROUND_COLOR, App

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "docs" / "images"

# The banner: three square tiles, one lattice each, on the app's background.
TILE = 392
GAP = 12
# Grid sizes whose drawn footprint is about square (render._footprint_factors), all even so
# they wrap.
BANNER_TILES = (
    ("square", 64, 64, ("B3/S23", "B36/S23", "B3/S238")),
    ("hexagon", 40, 34, ("B2/S34", "B24/S34", "B2/S345")),
    ("triangle", 80, 46, ("B34/S345", "B45/S3456", "B4/S345")),
)
SEED = 11


def _lifeforms(rules: Sequence[str]) -> list[Lifeform]:
    return [Lifeform(id=i, rule=Rule.from_bs_string(r)) for i, r in enumerate(rules, start=1)]


def _configure(
    config: Config,
    shape: str,
    width: int,
    height: int,
    n_species: int,
    *,
    seed: int = SEED,
    density: float | None = None,
) -> Config:
    grid = replace(config.grid, shape=shape, width=width, height=height, wrap=True)
    validate_dimensions(shape, width, height, True, grid.triangle_mode)
    simulation = replace(config.simulation, n_lifeforms=n_species, seed=seed)
    if density is not None:
        simulation = replace(simulation, initial_alive_percentage=density)
    return replace(config, grid=grid, simulation=simulation)


def _app_shot(
    config: Config,
    lifeforms: list[Lifeform],
    generations: int,
    path: Path,
    *,
    torus: tuple[float, float, float] | None = None,
    hints: bool = False,
) -> None:
    """The whole window after `generations` steps (fewer if an alert pauses the run). `torus`
    switches to the 3D view with the camera at (azimuth, elevation, distance)."""
    app = App(config)
    try:
        app._replace_session(config, lifeforms)
        app.settings_panel.set_lifeforms(lifeforms)
        app.session.run(generations)
        if torus is not None:
            app._toggle_view_mode()
            if app.torus_view is None:
                raise RuntimeError(app.torus_status or "3D view unavailable")
            camera = app.torus_view.camera
            camera.auto_rotate = False
            camera.azimuth, camera.elevation, camera.distance = torus
        app.hints_visible = hints
        app._draw()
        pygame.display.flip()
        pygame.image.save(app.screen, str(path))
    finally:
        app.close()
        pygame.quit()


def banner(config: Config, path: Path) -> None:
    pygame.init()
    pygame.display.set_mode((1, 1))  # the renderers' surfaces need a display format
    try:
        surface = pygame.Surface((len(BANNER_TILES) * TILE + (len(BANNER_TILES) - 1) * GAP, TILE))
        surface.fill(BACKGROUND_COLOR)
        for i, (shape, width, height, rules) in enumerate(BANNER_TILES):
            lifeforms = _lifeforms(rules)
            session = SimulationSession(
                _configure(config, shape, width, height, len(lifeforms)),
                lifeforms,
                auto_pause=False,
            )
            session.run(120)
            palette = render.build_palette(lifeforms)
            area = pygame.Rect(i * (TILE + GAP), 0, TILE, TILE)
            fitted = render.fit_grid(shape, width, height, area)
            drawn_w, drawn_h = fitted.pixel_size
            layout = render.GridLayout(
                shape,
                width,
                height,
                fitted.cell_size,
                area.x + (TILE - drawn_w) // 2,
                area.y + (TILE - drawn_h) // 2,
            )
            if shape == "square":
                render.render_square(
                    surface, session.grid, palette, layout.cell_size, layout.left, layout.top
                )
            else:
                render.GeometricRenderer(BACKGROUND_COLOR).draw(
                    surface, session.grid, palette, layout
                )
        pygame.image.save(surface, str(path))
    finally:
        pygame.quit()


def ui_square(config: Config, path: Path) -> None:
    _app_shot(
        _configure(config, "square", 50, 50, 2),
        _lifeforms(("B3/S23", "B36/S23")),
        80,
        path,
    )


def ui_hexagon(config: Config, path: Path) -> None:
    _app_shot(
        _configure(config, "hexagon", 50, 50, 3),
        _lifeforms(("B2/S34", "B24/S34", "B2/S345")),
        100,
        path,
    )


def ui_triangle(config: Config, path: Path) -> None:
    _app_shot(
        _configure(config, "triangle", 60, 50, 3),
        _lifeforms(("B34/S345", "B45/S3456", "B4/S345")),
        100,
        path,
    )


def ui_torus(config: Config, path: Path) -> None:
    _app_shot(
        _configure(config, "hexagon", 50, 50, 3),
        _lifeforms(("B2/S34", "B24/S34", "B2/S345")),
        100,
        path,
        torus=(0.7, 0.45, 3.4),
    )


def ui_help(config: Config, path: Path) -> None:
    _app_shot(
        _configure(config, "square", 50, 50, 1),
        _lifeforms(("B3/S23",)),
        40,
        path,
        hints=True,
    )


def ui_alert(config: Config, path: Path) -> None:
    # Nothing is born and everything survives on a grid over the stasis occupancy: the stasis
    # alert fires at once and pauses the run.
    frozen = Lifeform(id=1, rule=Rule(birth=set(), survive=set(range(9))))
    _app_shot(_configure(config, "square", 50, 50, 1, density=0.85), [frozen], 5, path)


SHOTS: dict[str, Callable[[Config, Path], None]] = {
    "banner": banner,
    "ui-square": ui_square,
    "ui-hexagon": ui_hexagon,
    "ui-triangle": ui_triangle,
    "ui-torus": ui_torus,
    "ui-help": ui_help,
    "ui-alert": ui_alert,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help=f"which images (default all): {', '.join(SHOTS)}")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    if unknown := [name for name in args.names if name not in SHOTS]:
        parser.error(f"unknown image(s): {', '.join(unknown)}; choose from {', '.join(SHOTS)}")

    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name in args.names or SHOTS:
        path = args.out / f"{name}.png"
        try:
            SHOTS[name](config, path)
        except Exception as exc:  # one failing shot (no OpenGL, say) must not cost the rest
            print(f"  {name}: FAILED ({exc})")
            failed += 1
            continue
        print(f"  {name}: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
