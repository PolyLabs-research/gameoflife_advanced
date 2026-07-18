"""SDL-dummy UI smoke test: drives the App through `--frames` iterations without a real display,
asserting it doesn't crash. Run with `SDL_VIDEODRIVER=dummy`.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from latticelife.config import load_config
from latticelife.ui.app import App


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--config", default="config/default.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    app = App(config)
    for _ in range(args.frames):
        if not app.step_frame():
            break

    print(f"SMOKE-OK ({args.frames} frames)")


if __name__ == "__main__":
    main()
