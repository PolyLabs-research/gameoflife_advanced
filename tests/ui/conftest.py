"""UI tests run under SDL_VIDEODRIVER=dummy and are smoke-only (§7). This must be set before
pygame is imported anywhere in the test session, so it lives at conftest module scope."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path

import pygame
import pytest

from latticelife.config import load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture(autouse=True)
def _pygame_teardown():
    yield
    pygame.quit()
