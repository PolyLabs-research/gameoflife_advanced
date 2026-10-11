"""UI tests run under SDL_VIDEODRIVER=dummy (§7). This must be set before pygame is imported
anywhere in the test session, so it lives at conftest module scope.

Driving the app. `App.step_frame()` advances the simulation by the wall time since the previous
frame, so tests that count generations install `FakeClock` (the `make_app` fixture does): every
frame then lasts one simulation interval — one generation per unpaused frame after the first —
without sleeping. The `ui` fixture posts real pygame events (clicks, keys, typed text, wheel) and
runs a frame, exactly as a person's input reaches the app."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dataclasses import replace
from pathlib import Path

import pygame
import pytest

from latticelife.config import load_config
from latticelife.ui.app import App

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


class FakeClock:
    """Stands in for `pygame.time.Clock`: every tick reports `ms` milliseconds without
    sleeping, and records the frame rate it was asked to cap at."""

    def __init__(self, ms: float) -> None:
        self.ms = ms
        self.framerates: list[float] = []

    def tick(self, framerate: float = 0) -> float:
        self.framerates.append(framerate)
        return self.ms


def frame_ms(fps: int) -> float:
    """A frame length that owes exactly one generation at `fps` (the tiny excess keeps float
    rounding from ever leaving a frame one hair short)."""
    return 1000 / fps + 0.001


class UI:
    """Posts input events for an App and runs one frame per action."""

    def __init__(self, app: App) -> None:
        self.app = app

    def frame(self, n: int = 1) -> None:
        for _ in range(n):
            self.app.step_frame()

    def click(self, pos, button: int = 1) -> None:
        pos = (int(pos[0]), int(pos[1]))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button))
        self.frame()

    def key(self, key: int, unicode: str = "", mod: int = 0) -> None:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=unicode, mod=mod))
        self.frame()

    def type(self, text: str) -> None:
        """Text as the OS delivers it: one TEXTINPUT event (as from an input method)."""
        pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text=text))
        self.frame()

    def clear_field(self, n: int = 10) -> None:
        for _ in range(n):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE, unicode="\b", mod=0)
            )
        self.frame()

    def move(self, pos, buttons=(0, 0, 0)) -> None:
        pygame.event.post(
            pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=buttons)
        )
        self.frame()

    def wheel(self, y: int, over=None) -> None:
        """Wheel notches (negative = scroll down), with the pointer first moved `over` a point."""
        if over is not None:
            pygame.event.post(
                pygame.event.Event(pygame.MOUSEMOTION, pos=over, rel=(0, 0), buttons=(0, 0, 0))
            )
        pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=y, flipped=False))
        self.frame()


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture
def make_app(config):
    """`make_app(grid={...}, simulation={...}, ui={...})`: an App on the default config with
    those fields replaced, its clock a `FakeClock` of one generation per frame. Each app is
    closed at teardown (releasing a 3D view's GL context)."""
    apps: list[App] = []

    def make(**sections) -> App:
        changed = config
        for name, values in sections.items():
            changed = replace(changed, **{name: replace(getattr(changed, name), **values)})
        app = App(changed)
        app.clock = FakeClock(frame_ms(changed.ui.fps))
        apps.append(app)
        return app

    yield make
    for app in apps:
        app.close()


@pytest.fixture
def ui():
    return UI


@pytest.fixture(autouse=True)
def _pygame_teardown():
    yield
    pygame.quit()
