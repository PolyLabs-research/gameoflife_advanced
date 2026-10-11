"""scripts/smoke_ui.py and scripts/bench_render.py report what actually ran (F61)."""

import runpy
from pathlib import Path

import pytest

from latticelife.ui.app import App

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name):
    return runpy.run_path(str(SCRIPTS / name))


def test_smoke_runs_every_shape_for_the_requested_frames(capsys):
    assert _script("smoke_ui.py")["main"](["--frames", "3"]) == 0
    out = capsys.readouterr().out
    for shape in ("square", "hexagon", "triangle"):
        assert f"{shape}: 3 frames" in out
    assert "SMOKE-OK (3 frames each: square, hexagon, triangle)" in out


def test_smoke_fails_when_the_app_stops_early(capsys, monkeypatch):
    """It used to print SMOKE-OK with the requested count even after one frame."""
    calls = iter([True, False])
    monkeypatch.setattr(App, "step_frame", lambda self: next(calls, False))
    assert _script("smoke_ui.py")["main"](["--frames", "5", "--shape", "hexagon"]) == 1
    assert "stopped after 1 of 5 frames" in capsys.readouterr().out


@pytest.mark.parametrize("shape", ["square", "triangle"])
def test_bench_reports_draw_and_step_rates(capsys, shape):
    assert (
        _script("bench_render.py")["main"](["--shape", shape, "--size", "20", "--frames", "3"]) == 0
    )
    out = capsys.readouterr().out
    assert "fps paused" in out and "fps running" in out and "gen/s step" in out


def test_bench_refuses_a_size_the_shape_cannot_wrap(capsys):
    assert _script("bench_render.py")["main"](["--shape", "hexagon", "--size", "21"]) == 1
    assert "even width" in capsys.readouterr().out
