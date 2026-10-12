"""The suite's guards around the project's real out/ (tests/conftest.py). A test run once replaced
out/discovered_rules.jsonl with test rows: a regression of F17 resolved that file against the
current directory, and pytest had been started from the checkout. The guard tests below aim
their changes at a tmp stand-in for out/ that `guard_also` guards as well, so even on a suite
without the guard they never touch the real one."""

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from latticelife.config import PROJECT_ROOT

KEPT = "a row worth keeping\n"


@pytest.fixture
def fake_out(guard_also, tmp_path):
    """A guarded tmp stand-in for the project's out/, holding a file worth keeping."""
    out = tmp_path / "project" / "out"
    (out / "discovery").mkdir(parents=True)
    (out / "discovered_rules.jsonl").write_text(KEPT)
    guard_also(out)
    return out


def test_each_test_runs_in_its_own_tmp_path(tmp_path):
    """So `Path("out/discovered_rules.jsonl")` (F17's bug) lands here, not in the checkout."""
    assert Path.cwd() == tmp_path
    assert Path("out/discovered_rules.jsonl").resolve().is_relative_to(tmp_path)


def test_the_guard_watches_the_real_out_in_this_process_and_in_subprocesses(out_guard):
    real_out = str(PROJECT_ROOT / "out")
    assert out_guard.roots == (real_out,)
    assert out_guard.active
    assert os.environ["LATTICELIFE_TEST_PROTECTED_DIRS"] == real_out
    site = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0])
    assert (site / "sitecustomize.py").is_file()
    assert (site / "out_guard.py").is_file()


def test_guarding_a_stand_in_keeps_the_real_out_guarded(fake_out, out_guard):
    """A test that re-pointed the guard at its stand-in once left the real out/ open to a
    regressed worker, which wrote test rows there for the length of that test."""
    real_rules = PROJECT_ROOT / "out" / "discovered_rules.jsonl"
    assert out_guard.covers(real_rules)  # checked, never written
    assert out_guard.covers(fake_out / "discovered_rules.jsonl")
    assert os.environ["LATTICELIFE_TEST_PROTECTED_DIRS"].split(os.pathsep) == [
        str(PROJECT_ROOT / "out"),
        str(fake_out),
    ]


def _rename_out_of(out, tmp_path):
    os.replace(out / "discovered_rules.jsonl", tmp_path / "moved.jsonl")


def _rename_into(out, tmp_path):
    (tmp_path / "new.jsonl").write_text("x")
    os.replace(tmp_path / "new.jsonl", out / "discovered_rules.jsonl")


def _by_dir_fd(change):
    """`change(fd)` with `fd` open on the stand-in out/: paths relative to a directory
    descriptor, as shutil.rmtree uses them."""

    def run(out, tmp_path):
        fd = os.open(out, os.O_RDONLY)
        try:
            change(fd)
        finally:
            os.close(fd)

    return run


CHANGES = {
    "append": lambda out, tmp: open(out / "discovered_rules.jsonl", "a"),
    "truncate": lambda out, tmp: open(out / "discovered_rules.jsonl", "w"),
    "read-write": lambda out, tmp: open(out / "discovered_rules.jsonl", "r+"),
    "new file": lambda out, tmp: (out / "new.jsonl").write_text("x"),
    "touch": lambda out, tmp: (out / "discovered_rules.jsonl").touch(),
    "os.open": lambda out, tmp: os.open(out / "new.jsonl", os.O_WRONLY | os.O_CREAT),
    "relative path": lambda out, tmp: open("project/out/discovered_rules.jsonl", "a"),
    "mkdir": lambda out, tmp: (out / "logs").mkdir(),
    "rename out of": _rename_out_of,
    "rename into": _rename_into,
    "remove": lambda out, tmp: (out / "discovered_rules.jsonl").unlink(),
    "rmdir": lambda out, tmp: (out / "discovery").rmdir(),
    "rmtree": lambda out, tmp: shutil.rmtree(out),
    "rmtree of the project": lambda out, tmp: shutil.rmtree(out.parent),
    "truncate by path": lambda out, tmp: os.truncate(out / "discovered_rules.jsonl", 0),
    "sqlite": lambda out, tmp: sqlite3.connect(out / "latticelife.db"),
    "sqlite uri": lambda out, tmp: sqlite3.connect(f"file:{out}/latticelife.db?mode=rwc", uri=True),
    "remove by dir_fd": _by_dir_fd(lambda fd: os.unlink("discovered_rules.jsonl", dir_fd=fd)),
    "rmdir by dir_fd": _by_dir_fd(lambda fd: os.rmdir("discovery", dir_fd=fd)),
    "mkdir by dir_fd": _by_dir_fd(lambda fd: os.mkdir("logs", dir_fd=fd)),
}


@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES.keys())
def test_a_change_under_out_is_refused_before_it_happens(change, fake_out, out_guard, tmp_path):
    before = out_guard.snapshot(fake_out)
    with pytest.raises(PermissionError, match="must not change the project's real out/"):
        change(fake_out, tmp_path)
    assert out_guard.changes(before, out_guard.snapshot(fake_out)) == []
    assert (fake_out / "discovered_rules.jsonl").read_text() == KEPT
    # Listed for the teardown check, which fails the test even when the code caught the error
    # (Path.touch does, and tries again another way).
    assert out_guard.refused
    out_guard.refused.clear()


def test_reading_out_is_allowed(fake_out, out_guard):
    assert (fake_out / "discovered_rules.jsonl").read_text() == KEPT
    with open(fake_out / "discovered_rules.jsonl", "rb") as f:
        assert f.read() == KEPT.encode()
    assert sorted(os.listdir(fake_out)) == ["discovered_rules.jsonl", "discovery"]
    assert (fake_out / "discovery").is_dir()
    assert out_guard.refused == []


def test_changes_beside_out_are_allowed(fake_out, out_guard, tmp_path):
    (fake_out.parent / "outside.jsonl").write_text("x")
    (fake_out.parent / "out-of-date").mkdir()
    sqlite3.connect(fake_out.parent / "other.db").close()
    assert out_guard.refused == []


def test_removing_another_tree_with_an_out_inside_is_allowed(guard_also, out_guard, tmp_path):
    """shutil.rmtree removes each entry by name relative to its directory's descriptor. Read
    against the current directory instead, `rmdir("out", dir_fd=...)` looked like removing the
    guarded `./out`, and pytest's own clean-up of old tmp directories was refused."""
    guard_also(tmp_path / "out")  # tmp_path is the current directory
    tree = tmp_path / "old-run"
    (tree / "out").mkdir(parents=True)
    (tree / "out" / "discovered_rules.jsonl").write_text("x")
    shutil.rmtree(tree)
    assert not tree.exists()
    assert out_guard.refused == []


def test_a_python_subprocess_is_guarded_too(fake_out):
    """Through subprocess_site/sitecustomize.py, which PYTHONPATH carries into the child."""
    code = "import sys; open(sys.argv[1], 'a').write('a test row')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(fake_out / "discovered_rules.jsonl")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "PermissionError" in result.stderr
    assert (fake_out / "discovered_rules.jsonl").read_text() == KEPT


def test_the_session_check_sees_every_change(out_guard, tmp_path):
    """The end-of-session comparison, for writers no hook sees (here, an unguarded directory)."""
    data = tmp_path / "data"

    def changes_since(before):
        return out_guard.changes(before, out_guard.snapshot(data))

    before = out_guard.snapshot(data)
    assert before == {} and changes_since(before) == []

    (data / "discovery").mkdir(parents=True)
    rules = data / "discovered_rules.jsonl"
    rules.write_text(KEPT)
    assert changes_since(before) == [".", "discovered_rules.jsonl", "discovery"]

    before = out_guard.snapshot(data)
    with rules.open("a") as f:
        f.write("one more row\n")
    assert changes_since(before) == ["discovered_rules.jsonl"]

    before = out_guard.snapshot(data)
    rules.write_text(KEPT.upper())  # same size
    stat = rules.stat()
    os.utime(rules, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert changes_since(before) == ["discovered_rules.jsonl"]

    # A file that came and went leaves its directory's mtime behind (pinned back first, since
    # file-system clocks are coarser than this test).
    os.utime(data / "discovery", ns=(0, 0))
    before = out_guard.snapshot(data)
    (data / "discovery" / "lineage.json").write_text("{}")
    (data / "discovery" / "lineage.json").unlink()
    assert changes_since(before) == ["discovery"]
