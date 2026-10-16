"""Damage spreading as a headless experiment (`experiments.damage`)."""

from pathlib import Path

import pytest

from latticelife.config import load_config
from latticelife.core.rules import Rule
from latticelife.experiments.damage import DamageTask, damage_point, run_damage, summarise
from latticelife.experiments.harness import read_jsonl

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


def test_a_record_is_reproducible_from_its_seed(config):
    task = DamageTask(config, "B2/S34", "hexagon", 16, 0.5, seed=3, steps=40)
    first, second = damage_point(task), damage_point(task)
    assert first == second
    assert first.differing[0] == 1 and len(first.differing) == 40
    assert first.final == first.differing[-1] / first.cells
    assert first.peak == max(first.differing) / first.cells
    assert first.cells == 256


def test_twins_that_agree_agree_for_ever(config):
    # Nothing is born and nothing survives: both twins are empty after one step.
    record = damage_point(DamageTask(config, "B/S", "square", 8, 0.5, seed=1, steps=10))
    assert record.healed_at == 1
    assert record.differing[1:] == [0] * 9 and record.final == 0.0


def test_flips_warmup_and_rectangles(config):
    task = DamageTask(config, "B3/S23", "square", "8x12", 0.4, seed=2, steps=5, flips=4, warmup=3)
    record = damage_point(task)
    assert record.differing[0] == 4
    assert (record.grid, record.cells, record.warmup) == ("8x12", 96, 3)
    with pytest.raises(ValueError):
        damage_point(DamageTask(config, "B3/S23", "square", 8, 0.4, seed=2, steps=5, flips=0))


def test_run_damage_appends_and_resumes(config, tmp_path):
    out = tmp_path / "damage.jsonl"
    rule = Rule.from_bs_string("B2/S34")
    first = list(
        run_damage(config, rule, "hexagon", [8, 10], [0.5], [1, 2], steps=20, out_path=out)
    )
    assert len(first) == 4 and len(read_jsonl(out)) == 4
    assert list(run_damage(config, rule, "hexagon", [8, 10], [0.5], [1, 2], 20, out)) == []
    summary = summarise(read_jsonl(out))
    assert set(summary) == {"8", "10"} and summary["8"]["runs"] == 2
