"""Packaging metadata (F77, F78)."""

import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def test_requires_python_excludes_versions_without_locked_wheels():
    """F77: pygame 2.6.1 / moderngl 5.12.0 ship cp312 and cp313 wheels only."""
    requires = _pyproject()["project"]["requires-python"].replace(" ", "")
    assert requires == ">=3.12,<3.14"
    with (ROOT / "uv.lock").open("rb") as f:
        assert tomllib.load(f)["requires-python"].replace(" ", "") == requires


def test_sdist_leaves_out_local_files(tmp_path):
    """F78: the sdist used to ship .claude/settings.local.json, the .hypothesis database and
    stray root screenshots. Builds a copy of the project with such files planted in it."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "uv.lock", ".gitignore"):
        shutil.copy2(ROOT / name, project / name)
    for name in ("src", "config", "docs", "scripts"):
        shutil.copytree(ROOT / name, project / name, ignore=shutil.ignore_patterns("__pycache__"))
    planted = {
        ".claude/settings.local.json": "{}",
        ".hypothesis/examples/x": "x",
        "out/latticelife.db": "x",
        "screenshot.png": "x",
    }
    for relative, content in planted.items():
        (project / relative).parent.mkdir(parents=True, exist_ok=True)
        (project / relative).write_text(content)

    result = subprocess.run(
        [uv, "build", "--offline", "--sdist", "-o", str(tmp_path / "dist")],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"offline sdist build unavailable: {result.stderr.strip()[-300:]}")

    (sdist,) = (tmp_path / "dist").glob("*.tar.gz")
    with tarfile.open(sdist) as tar:
        members = {name.split("/", 1)[1] for name in tar.getnames() if "/" in name}
    assert "src/latticelife/config.py" in members
    assert "config/default.toml" in members
    for relative in planted:
        assert relative not in members
    assert not any(m.startswith((".claude", ".hypothesis", "out/")) for m in members)
