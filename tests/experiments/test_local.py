"""Local transients: per-block settling times whose largest is the soup's own transient."""

from dataclasses import replace

import numpy as np
import pytest

from latticelife.config import DEFAULT_CONFIG_PATH, load_config
from latticelife.core.rules import Rule
from latticelife.experiments.harness import read_jsonl
from latticelife.experiments.local import LocalTask, local_point, run_local
from latticelife.experiments.sweeps import SweepTask, sweep_point

CONFIG = load_config(DEFAULT_CONFIG_PATH)


@pytest.mark.parametrize(
    ("rule", "shape", "mode", "grid"),
    [
        ("B3/S23", "square", None, 32),
        ("B2/S34", "hexagon", None, 32),
        ("B23/S013", "triangle", "edge", 32),
        ("B3/S23", "square", None, "32x16"),
    ],
)
def test_the_largest_local_transient_is_the_global_one(rule, shape, mode, grid):
    config = (
        CONFIG if mode is None else replace(CONFIG, grid=replace(CONFIG.grid, triangle_mode=mode))
    )
    for seed in (1, 2, 3):
        record = local_point(LocalTask(config, rule, shape, grid, 0.5, seed, 4000, (4, 8, 16)))
        sweep = sweep_point(SweepTask(config, rule, shape, grid, 0.5, seed, 4000))
        assert record.transient == sweep.transient and record.period == sweep.period
        for b, times in record.local.items():
            width, height = (32, 32) if grid == 32 else (32, 16)
            assert len(times) == (width // int(b)) * (height // int(b))
            assert max(times) == record.transient and min(times) >= 0
        # a larger block's transient is the largest of the smaller blocks inside it
        fine = np.array(record.local["4"]).reshape(-1, 8 // 4, (32 // 8), 8 // 4)
        coarse = np.array(record.local["8"])
        assert np.array_equal(fine.max(axis=(1, 3)).ravel(), coarse)


def test_the_whole_grid_as_one_block_and_bad_blocks():
    record = local_point(LocalTask(CONFIG, "B3/S23", "square", 16, 0.5, 4, 4000, (16,)))
    assert record.local == {"16": [record.transient]}
    with pytest.raises(ValueError, match="does not divide"):
        local_point(LocalTask(CONFIG, "B3/S23", "square", 16, 0.5, 4, 4000, (5,)))


def test_unsettled_and_long_period_soups_have_no_local_transients():
    capped = local_point(LocalTask(CONFIG, "B3/S23", "square", 32, 0.5, 1, 3, (8,)))
    assert capped.local is None and capped.transient is None
    found = local_point(LocalTask(CONFIG, "B3/S23", "square", 32, 0.5, 1, 4000, (8,)))
    if found.period > 1:
        long = local_point(LocalTask(CONFIG, "B3/S23", "square", 32, 0.5, 1, 4000, (8,), 1))
        assert long.local is None and "max_period" in long.note


def test_run_local_resumes(tmp_path):
    out = tmp_path / "local.jsonl"
    rule = Rule.from_bs_string("B3/S23")
    first = list(run_local(CONFIG, rule, "square", [16], [0.5], [1, 2], 4000, str(out), (8,)))
    again = list(run_local(CONFIG, rule, "square", [16], [0.5], [1, 2, 3], 4000, str(out), (8,)))
    assert len(first) == 2 and len(again) == 1 and len(read_jsonl(out)) == 3
