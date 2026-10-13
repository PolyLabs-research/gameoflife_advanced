"""The experiments' shared plumbing (`experiments.harness`)."""

import json
import os
from pathlib import Path

import pytest

from latticelife.config import load_config
from latticelife.core.rules import Rule
from latticelife.experiments.harness import (
    append_jsonl,
    map_tasks,
    read_jsonl,
    resume_keys,
    single_species_session,
    torus_config,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


def test_torus_config_sets_one_species_on_a_wrapped_square_torus():
    config = torus_config(load_config(DEFAULT_CONFIG_PATH), "hexagon", 12, 0.3)
    assert (config.grid.shape, config.grid.width, config.grid.height) == ("hexagon", 12, 12)
    assert config.grid.wrap is True
    assert config.simulation.n_lifeforms == 1
    assert config.simulation.initial_alive_percentage == 0.3


def test_single_species_session_is_reproducible_and_refuses_seed_zero():
    config = load_config(DEFAULT_CONFIG_PATH)
    rule = Rule(birth={3}, survive={2, 3})
    a = single_species_session(config, rule, "triangle", 12, 0.5, seed=3)
    b = single_species_session(config, rule, "triangle", 12, 0.5, seed=3)
    assert a.grid.occupancy.tolist() == b.grid.occupancy.tolist()
    assert a.lifeforms[0].rule == rule
    assert a.auto_pause is False
    with pytest.raises(ValueError, match="seed 0"):
        single_species_session(config, rule, "square", 12, 0.5, seed=0)


def test_jsonl_round_trip_skips_a_partial_last_line(tmp_path):
    path = tmp_path / "out" / "records.jsonl"
    assert read_jsonl(path) == []
    append_jsonl(path, {"a": 1, "k": [1, 2]})
    append_jsonl(path, {"a": 2, "k": [3]})
    with path.open("a") as f:
        f.write('{"a": 3, "k": [')  # killed mid-write
    assert read_jsonl(path) == [{"a": 1, "k": [1, 2]}, {"a": 2, "k": [3]}]
    assert resume_keys(path, ("a", "k")) == {(1, (1, 2)), (2, (3,))}
    assert resume_keys(path, ("missing",)) == set()


def _square(x):
    return (os.getpid(), x * x)


def test_map_tasks_runs_inline_or_in_a_pool():
    inline = list(map_tasks(_square, [1, 2, 3], workers=1))
    assert [v for _, v in inline] == [1, 4, 9]
    assert {pid for pid, _ in inline} == {os.getpid()}
    pooled = list(map_tasks(_square, range(8), workers=2))
    assert sorted(v for _, v in pooled) == [x * x for x in range(8)]
    assert os.getpid() not in {pid for pid, _ in pooled}


def test_jsonl_lines_are_plain_json(tmp_path):
    path = tmp_path / "r.jsonl"
    append_jsonl(path, {"rule": "B3/S23"})
    assert json.loads(path.read_text()) == {"rule": "B3/S23"}
