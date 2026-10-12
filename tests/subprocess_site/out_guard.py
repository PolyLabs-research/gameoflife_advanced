"""The suite's guard around the project's real `out/` (see tests/conftest.py): an audit hook that
refuses, with PermissionError, any change under a protected directory. The suite installs it in
its own process, so every worker forked from it has it too; `sitecustomize.py` beside this file
installs it in each Python subprocess a test starts, which finds this directory on PYTHONPATH
and the protected directories in `PROTECTED_DIRS_ENV`. Plain Python (no pytest, no latticelife),
so any subprocess can import it."""

from __future__ import annotations

import errno
import os
import sys

PROTECTED_DIRS_ENV = "LATTICELIFE_TEST_PROTECTED_DIRS"  # os.pathsep-separated

# Audit events that change the file system: the argument positions of each path they change and
# of the dir_fd that path is relative to (None: the event has none). `open` and
# `sqlite3.connect` are checked separately: `open` changes something only when its flags ask to
# write, and even reading a database can create its -wal/-shm files.
_CHANGE_EVENTS = {
    "os.mkdir": ((0, 2),),
    "os.remove": ((0, 1),),  # also os.unlink
    "os.rmdir": ((0, 1),),
    "os.rename": ((0, 2), (1, 3)),  # also os.replace
    "os.link": ((1, 3),),
    "os.symlink": ((1, 2),),
    "os.truncate": ((0, None),),
    "os.utime": ((0, 3),),
    "os.chmod": ((0, 2),),
    "shutil.rmtree": ((0, 1),),
}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

Snapshot = dict[str, tuple[int, int]]  # relative path -> (size, mtime_ns)


def _resolve(path, dir_fd=None) -> str | None:
    """`path` as an absolute, normalised string, read relative to `dir_fd` when one is given (-1
    or None: the current directory). None for a file descriptor (its own open was checked
    already) or anything that cannot be resolved."""
    if isinstance(path, int):
        return None
    try:
        path = os.fsdecode(path)
        if not os.path.isabs(path):
            if dir_fd is None or dir_fd == -1:
                base = os.getcwd()
            else:  # shutil.rmtree, for one, removes entries by name relative to a dir_fd
                base = os.readlink(f"/proc/self/fd/{dir_fd}")
            path = os.path.join(base, path)
    except (TypeError, OSError):
        return None
    return os.path.normpath(path)


def _sqlite_path(database):
    """The file a `sqlite3.connect` argument names: a `file:` URI's path, else the argument."""
    if isinstance(database, str) and database.startswith("file:"):
        return database[len("file:") :].split("?", 1)[0].removeprefix("//")
    return database


class OutGuard:
    """Keeps a process out of each directory in `roots`. While `active`, a change under one —
    opening a file for writing; creating, renaming, removing or touching anything; connecting
    SQLite to a database there; removing a directory that holds one — raises PermissionError
    before it happens, and is listed in `refused` (code under test may catch the error and
    carry on)."""

    def __init__(self, *roots) -> None:
        self.roots = tuple(os.path.normpath(os.fspath(root)) for root in roots)
        self.active = False
        self.refused: list[str] = []

    def covers(self, path, dir_fd=None) -> bool:
        resolved = _resolve(path, dir_fd)
        return resolved is not None and any(
            resolved == root or resolved.startswith(root + os.sep) for root in self.roots
        )

    def _holds_a_root(self, tree, dir_fd) -> bool:
        resolved = _resolve(tree, dir_fd)
        return resolved is not None and any(
            root.startswith(resolved.rstrip(os.sep) + os.sep) for root in self.roots
        )

    def __call__(self, event: str, args: tuple) -> None:
        if not self.active:
            return
        if event == "open":
            flags = args[2] if isinstance(args[2], int) else 0
            targets = [(args[0], None)] if flags & _WRITE_FLAGS else []
        elif event == "sqlite3.connect":
            targets = [(_sqlite_path(args[0]), None)]
        elif event in _CHANGE_EVENTS:
            targets = [
                (args[at], None if fd_at is None else args[fd_at])
                for at, fd_at in _CHANGE_EVENTS[event]
            ]
        else:
            return
        for path, dir_fd in targets:
            if self.covers(path, dir_fd) or (
                event == "shutil.rmtree" and self._holds_a_root(path, dir_fd)
            ):
                self.refused.append(f"{event} {os.fsdecode(path)}")
                raise PermissionError(
                    errno.EACCES,
                    "tests must not change the project's real out/ (see tests/conftest.py)",
                    os.fsdecode(path),
                )

    @staticmethod
    def snapshot(directory) -> Snapshot:
        """Every entry under `directory`, itself included (empty when there is no such
        directory). For writers no hook sees."""
        entries = {}
        for parent, dirnames, filenames in os.walk(directory):
            for name in [os.curdir, *dirnames, *filenames]:
                path = os.path.join(parent, name)
                stat = os.lstat(path)
                entries[os.path.relpath(path, directory)] = (stat.st_size, stat.st_mtime_ns)
        return entries

    @staticmethod
    def changes(before: Snapshot, after: Snapshot) -> list[str]:
        """The entries added, removed or modified between two snapshots."""
        names = before.keys() | after.keys()
        return sorted(name for name in names if before.get(name) != after.get(name))


def install_from_environment() -> OutGuard | None:
    """Guard the directories `PROTECTED_DIRS_ENV` names, if it names any, for the rest of this
    process (an audit hook cannot be removed)."""
    roots = [root for root in os.environ.get(PROTECTED_DIRS_ENV, "").split(os.pathsep) if root]
    if not roots:
        return None
    guard = OutGuard(*roots)
    guard.active = True
    sys.addaudithook(guard)
    return guard
