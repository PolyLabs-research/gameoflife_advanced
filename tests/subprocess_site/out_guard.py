"""The suite's guard around the project's real `out/` (see tests/conftest.py): an audit hook that
refuses, with PermissionError, any change under a protected directory. The suite installs it in
its own process, so every worker forked from it has it too; `sitecustomize.py` beside this file
installs it in each Python subprocess a test starts, which finds this directory on PYTHONPATH
and the protected directories in `PROTECTED_DIRS_ENV`. Plain Python (no pytest, no latticelife),
so any subprocess can import it."""

from __future__ import annotations

import errno
import os
import stat
import sys

PROTECTED_DIRS_ENV = "LATTICELIFE_TEST_PROTECTED_DIRS"  # os.pathsep-separated

# Audit events that change the file system: for each path one changes, its argument position,
# the position of the dir_fd it is relative to (None: the event has none), and whether the change
# follows a symlink found there to its target instead of changing the link itself. `open` and
# `sqlite3.connect` are checked separately: `open` changes something only when its flags ask to
# write, and even reading a database can create its -wal/-shm files.
_CHANGE_EVENTS = {
    "os.mkdir": ((0, 2, False),),
    "os.remove": ((0, 1, False),),  # also os.unlink
    "os.rmdir": ((0, 1, False),),
    "os.rename": ((0, 2, False), (1, 3, False)),  # also os.replace
    "os.link": ((1, 3, False),),
    "os.symlink": ((1, 2, False),),
    "os.truncate": ((0, None, True),),
    # The events do not say whether follow_symlinks was turned off: assume not.
    "os.utime": ((0, 3, True),),
    "os.chmod": ((0, 2, True),),
    "shutil.rmtree": ((0, 1, False),),
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
    except (TypeError, ValueError, OSError):
        return None
    return os.path.normpath(path)


def _places(path, dir_fd=None, follow=True) -> list[str]:
    """Every absolute path `path` may stand for: as written, with the symlinks among its folders
    resolved and, when `follow`, with a symlink in its last component resolved too. A change made
    through a symlink lands on the far side of it."""
    resolved = _resolve(path, dir_fd)
    if resolved is None:
        return []
    folder, name = os.path.split(resolved)
    places = [resolved]
    try:
        places.append(os.path.join(os.path.realpath(folder), name))
        if follow:
            places.append(os.path.realpath(resolved))
    except (ValueError, OSError):
        pass
    return places


def _open_directories() -> list[int]:
    """The file descriptors this process holds open on directories."""
    try:
        fds = [int(name) for name in os.listdir("/proc/self/fd")]
    except OSError:
        return []
    directories = []
    for fd in fds:
        try:
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directories.append(fd)
        except OSError:  # closed since, such as the one listdir read the folder through
            continue
    return directories


def _is_relative(path) -> bool:
    return isinstance(path, (str, bytes, os.PathLike)) and not os.path.isabs(os.fsdecode(path))


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
    carry on).

    Paths and roots are compared both as written and with symlinks resolved, so a change made
    through a symlink to (or inside) a root, or under a root that is itself a symlink, is
    refused too. The `open` audit event does not carry the dir_fd a relative path is opened
    against (`os.open(name, flags, dir_fd=fd)`), so a relative path opened for writing is
    refused when it would land in a root read against any directory this process holds open,
    not just the current one."""

    def __init__(self, *roots) -> None:
        self.roots = tuple(os.path.normpath(os.fspath(root)) for root in roots)
        self.active = False
        self.refused: list[str] = []

    def _all_roots(self) -> set[str]:
        """The roots as written and with their symlinks resolved."""
        return {*self.roots, *map(os.path.realpath, self.roots)}

    def covers(self, path, dir_fd=None, follow=True) -> bool:
        """Whether a change to `path` (read relative to `dir_fd`, through a symlink in its last
        component when `follow`) would land in a root."""
        roots = self._all_roots()
        return any(
            place == root or place.startswith(root + os.sep)
            for place in _places(path, dir_fd, follow)
            for root in roots
        )

    def _holds_a_root(self, tree, dir_fd) -> bool:
        roots = self._all_roots()
        return any(
            root.startswith(place.rstrip(os.sep) + os.sep)
            for place in _places(tree, dir_fd, follow=False)
            for root in roots
        )

    def __call__(self, event: str, args: tuple) -> None:
        if not self.active:
            return
        if event == "open":
            flags = args[2] if isinstance(args[2], int) else 0
            targets = [(args[0], None, True)] if flags & _WRITE_FLAGS else []
            if targets and _is_relative(args[0]):  # perhaps against a dir_fd: see the docstring
                targets += [(args[0], fd, True) for fd in _open_directories()]
        elif event == "sqlite3.connect":
            targets = [(_sqlite_path(args[0]), None, True)]
        elif event in _CHANGE_EVENTS:
            targets = [
                (args[at], None if fd_at is None else args[fd_at], follow)
                for at, fd_at, follow in _CHANGE_EVENTS[event]
            ]
        else:
            return
        for path, dir_fd, follow in targets:
            if self.covers(path, dir_fd, follow) or (
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
