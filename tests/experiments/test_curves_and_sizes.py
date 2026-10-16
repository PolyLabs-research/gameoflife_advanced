"""Replicate curves and size samples (`observables.damage_curves`, `observables.object_sizes`):
the raw material for a law over time with seed-to-seed error bars, and for a distribution fit."""

from pathlib import Path

import pytest

from latticelife.config import load_config
from latticelife.core.rules import Rule
from latticelife.experiments.damage import run_damage
from latticelife.experiments.harness import read_jsonl
from latticelife.experiments.objects import run_soups
from latticelife.experiments.observables import damage_curves, damage_spread, object_sizes

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


def test_damage_curves_are_each_seeds_whole_curve(config, tmp_path):
    out = tmp_path / "damage.jsonl"
    rule = Rule.from_bs_string("B2/S34")
    list(run_damage(config, rule, "hexagon", [12], [0.5], [1, 2, 3], 40, str(out)))
    records = read_jsonl(out)
    (row,) = damage_curves(records)
    assert row.seeds == [1, 2, 3] and len(row.curves) == 3
    assert all(len(c) == 40 and all(0.0 <= v <= 1.0 for v in c) for c in row.curves)
    (summary,) = damage_spread(records)
    assert summary.mean_final == pytest.approx(sum(c[-1] for c in row.curves) / 3)
    with pytest.raises(ValueError, match="two different curves"):
        bad = dict(records[0], differing=[0] * 40)
        damage_curves([*records, bad])


def test_object_sizes_pool_every_soup(config, tmp_path):
    out = tmp_path / "soups.jsonl"
    life = Rule.from_bs_string("B3/S23")
    records = list(
        run_soups(config, life, "square", 16, [1, 2, 3], 60, 0.3, str(out), max_period=30)
    )
    (row,) = object_sizes(read_jsonl(out))
    assert row.soups == 3
    assert row.objects == sum(len(r["objects"]) for r in records) == len(row.populations)
    assert sum(row.abundances) == row.objects and row.abundances == sorted(
        row.abundances, reverse=True
    )
    assert all(p >= 1 for p in row.populations)
