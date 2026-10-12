"""Run at start-up by each Python subprocess a test starts (tests/conftest.py puts this directory
on PYTHONPATH): guards the project's real out/ in that process too. It stands in for the
system's own sitecustomize, which at most installs a distribution's crash reporter."""

import out_guard

out_guard.install_from_environment()
