"""Packaging metadata and the built distributions (F16, F77, F78)."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from latticelife.logging import LOG_DIR_ENV

ROOT = Path(__file__).resolve().parent.parent

# What the sdist may contain (pyproject.toml's [tool.hatch.build.targets.sdist] include), plus
# the PKG-INFO and .gitignore hatchling always adds.
SDIST_TOP_LEVEL = {"src", "tests", "config", "docs", "scripts", "README.md", "pyproject.toml"}
SDIST_TOP_LEVEL |= {"uv.lock", "PKG-INFO", ".gitignore"}


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def _project_copy(tmp_path: Path) -> Path:
    """The project's tracked layout copied under tmp_path, so a build never writes into it."""
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "uv.lock", ".gitignore"):
        shutil.copy2(ROOT / name, project / name)
    for name in ("src", "tests", "config", "docs", "scripts"):
        shutil.copytree(
            ROOT / name,
            project / name,
            ignore=shutil.ignore_patterns("__pycache__", ".claude", ".hypothesis"),
        )
    return project


def _uv_build(project: Path, kind: str, out_dir: Path) -> Path:
    """Build `kind` ("sdist" or "wheel") offline; skips the test where that is impossible."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    result = subprocess.run(
        [uv, "build", "--offline", f"--{kind}", "-o", str(out_dir)],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"offline {kind} build unavailable: {result.stderr.strip()[-300:]}")
    (artifact,) = out_dir.glob("*.tar.gz" if kind == "sdist" else "*.whl")
    return artifact


def test_requires_python_excludes_versions_without_locked_wheels():
    """F77: pygame 2.6.1 / moderngl 5.12.0 ship cp312 and cp313 wheels only."""
    requires = _pyproject()["project"]["requires-python"].replace(" ", "")
    assert requires == ">=3.12,<3.14"
    with (ROOT / "uv.lock").open("rb") as f:
        assert tomllib.load(f)["requires-python"].replace(" ", "") == requires


def test_sdist_ships_only_the_allowlisted_project_files(tmp_path):
    """F78: the sdist used to ship .claude/settings.local.json, the .hypothesis database and
    stray root screenshots, and a root-only denylist still let nested local settings, a root
    .env and scratch files through. Builds a copy of the project with such files planted."""
    project = _project_copy(tmp_path)
    planted = {
        ".claude/settings.local.json": "{}",
        "tests/.claude/settings.local.json": "{}",
        "src/latticelife/.claude/settings.local.json": "{}",
        ".hypothesis/examples/x": "x",
        "tests/.hypothesis/examples/x": "x",
        "tests/.pytest_cache/v/x": "x",
        "out/latticelife.db": "x",
        "screenshot.png": "x",
        ".env": "SECRET=1",
        "config/.env": "SECRET=1",
        "scratch_notes.txt": "x",
        "notes/todo.md": "x",
        "docs/notes.png": "x",
        "scripts/output.txt": "x",
        "config/default.toml.bak": "x",
    }
    for relative, content in planted.items():
        (project / relative).parent.mkdir(parents=True, exist_ok=True)
        (project / relative).write_text(content)

    sdist = _uv_build(project, "sdist", tmp_path / "dist")

    with tarfile.open(sdist) as tar:
        members = {name.split("/", 1)[1] for name in tar.getnames() if "/" in name}
    assert "src/latticelife/config.py" in members
    assert "config/default.toml" in members
    assert "tests/test_packaging.py" in members
    for relative in planted:
        assert relative not in members
    assert {member.split("/", 1)[0] for member in members} <= SDIST_TOP_LEVEL
    assert not any(
        part in {".claude", ".hypothesis", ".pytest_cache", ".env"}
        for member in members
        for part in member.split("/")
    )


def test_installed_wheel_runs_from_any_directory(tmp_path):
    """F16: an installed (non-editable) copy had no default config (config/default.toml was not
    in the wheel and was looked up under the current directory), so `latticelife autorun` failed
    outside a checkout and left `out/logs/app.jsonl` in whatever directory it was run from. It
    now loads the config shipped in the package and writes under the per-user data directory."""
    wheel = _uv_build(_project_copy(tmp_path), "wheel", tmp_path / "dist")
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    assert (site / "latticelife" / "default.toml").read_bytes() == (
        ROOT / "config" / "default.toml"
    ).read_bytes()

    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    data_home = tmp_path / "data"
    env = {key: value for key, value in os.environ.items() if key != LOG_DIR_ENV}
    # The unpacked copy first; the inherited PYTHONPATH after it carries the suite's out/ guard
    # (tests/conftest.py), and the runner below checks the copy is the one imported.
    env |= {
        "PYTHONPATH": os.pathsep.join([str(site), os.environ["PYTHONPATH"]]),
        "XDG_DATA_HOME": str(data_home),
        "HOME": str(tmp_path / "home"),
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
    }
    # Run the unpacked copy, not the checkout's editable install, and say so if it is not.
    runner = (
        "import sys; import latticelife.cli as cli; "
        "assert cli.__file__.startswith(sys.argv[1]), cli.__file__; "
        "sys.exit(cli.main(sys.argv[2:]))"
    )

    def latticelife(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", runner, str(site), *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    result = latticelife("autorun", "--sessions", "1", "--generations", "3", "--shape", "square")
    assert result.returncode == 0, result.stderr
    out_dir = data_home / "latticelife" / "out"
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["db_path"] == str(out_dir / "latticelife.db")
    assert (out_dir / "latticelife.db").is_file()

    # A failing command logs to the data directory too, never into the current directory.
    result = latticelife("autorun", "--sessions", "1", "--generations", "1", "--config", "no.toml")
    assert result.returncode == 2
    assert "config file not found" in result.stderr
    assert (out_dir / "logs" / "app.jsonl").is_file()

    assert list(cwd.iterdir()) == []
    assert not any(path.name == "out" for path in site.rglob("*"))
