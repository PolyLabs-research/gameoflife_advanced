"""The rule census (`experiments.census`)."""

from pathlib import Path

import numpy as np
import pytest

from latticelife.config import load_config
from latticelife.core.rules import Rule
from latticelife.experiments.census import (
    CensusTask,
    classify_rule,
    classify_session,
    enumerate_rules,
    langton_lambda,
    mean_field_fixed_points,
    mean_field_map,
    rule_from_masks,
    rule_masks,
    rule_space_size,
    run_census,
    summarise,
)
from latticelife.experiments.harness import read_jsonl, single_species_session

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"
LIFE = Rule(birth={3}, survive={2, 3})


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


def test_rule_space_sizes_match_the_degree_table():
    assert rule_space_size(3) == 256
    assert rule_space_size(6) == 16_384
    assert rule_space_size(8) == 262_144
    assert rule_space_size(12) == 67_108_864


def test_enumerate_rules_lists_every_rule_once_in_a_fixed_order():
    rules = list(enumerate_rules(3))
    assert len(rules) == 256
    assert len({rule_masks(r, 3) for r in rules}) == 256
    assert rules[0] == Rule(birth=set(), survive=set())
    assert rules[-1] == Rule(birth={0, 1, 2, 3}, survive={0, 1, 2, 3})
    assert rule_from_masks(*rule_masks(LIFE, 8), 8) == LIFE


def test_enumerate_rules_up_to_duality_covers_the_space_through_complements():
    n = 3
    kept = list(enumerate_rules(n, up_to_duality=True))
    keys = {rule_masks(r, n) for r in kept}
    assert len(keys) == len(kept)
    for rule in enumerate_rules(n):
        assert rule_masks(rule, n) in keys or rule_masks(rule.complement(n), n) in keys
    self_dual = sum(1 for r in kept if r.complement(n) == r)
    assert len(kept) == (256 - self_dual) / 2 + self_dual


def test_langton_lambda_of_life():
    # (C(8,3) + C(8,2) + C(8,3)) / 2^9 = 140 / 512
    assert langton_lambda(LIFE, 8) == pytest.approx(140 / 512)
    assert langton_lambda(Rule(set(), set()), 8) == 0.0
    assert langton_lambda(Rule(set(range(9)), set(range(9))), 8) == 1.0


def test_mean_field_map_is_the_binomial_density_map():
    f = mean_field_map(LIFE, 8)
    for rho in (0.0, 0.1, 0.37, 0.8, 1.0):
        from math import comb

        expected = (1 - rho) * comb(8, 3) * rho**3 * (1 - rho) ** 5 + rho * sum(
            comb(8, k) * rho**k * (1 - rho) ** (8 - k) for k in (2, 3)
        )
        assert float(f(rho)) == pytest.approx(expected)


def test_mean_field_fixed_points_of_life_and_the_identity_rule():
    fixed, stable = mean_field_fixed_points(LIFE, 8)
    assert 0.0 in fixed and 0.0 in stable
    assert any(0.3 < p < 0.4 for p in stable)  # the well-known mean-field density of Life
    f = mean_field_map(LIFE, 8)
    for point in fixed:
        assert float(f(point)) == pytest.approx(point, abs=1e-5)
    assert mean_field_fixed_points(Rule(set(), set(range(9))), 8) == ([], [])
    assert mean_field_fixed_points(Rule(set(), set()), 8) == ([0.0], [0.0])


def test_classify_session_outcomes(config):
    def session_with(cells, size=10):
        session = single_species_session(config, LIFE, "square", size, 0.0, seed=1)
        for r, c in cells:
            session.set_cell(r * size + c, 1)
        return session

    empty = classify_session(session_with([]), steps=20, tail=5)
    assert empty["outcome"] == "extinct"
    assert (empty["period"], empty["transient"]) == (1, 0)
    assert empty["cluster_count"] == 0

    block = classify_session(session_with([(4, 4), (4, 5), (5, 4), (5, 5)]), steps=20, tail=5)
    assert block["outcome"] == "fixed"
    assert block["changed_final"] == 0.0
    assert (block["period"], block["transient"]) == (1, 0)
    assert (block["rho_final"], block["cluster_count"], block["max_cluster_size"]) == (0.04, 1, 4)

    blinker = classify_session(session_with([(4, 4), (4, 5), (4, 6)]), steps=20, tail=4)
    assert blinker["outcome"] == "periodic"
    assert (blinker["period"], blinker["transient"]) == (2, 0)
    assert blinker["activity_final"] == 0.0  # same population in both phases
    assert blinker["changed_final"] == 0.04  # yet 4 of 100 cells change every step

    # Three cells in an L: one step of transient into a block.
    corner = classify_session(session_with([(4, 4), (4, 5), (5, 4)]), steps=20, tail=5)
    assert (corner["outcome"], corner["period"], corner["transient"]) == ("fixed", 1, 1)

    short = classify_session(session_with([(4, 4), (4, 5), (4, 6)]), steps=2, tail=5)
    assert short["outcome"] == "unsettled"
    with pytest.raises(ValueError, match="tail"):
        classify_session(session_with([]), steps=5, tail=0)


def test_classify_rule_builds_a_complete_record(config):
    task = CensusTask(config, "B3/S23", "hexagon", 12, 40, 1, 0.5, 10)
    record = classify_rule(task)
    assert (record.rule, record.shape, record.triangle_mode) == ("B3/S23", "hexagon", None)
    assert (record.grid, record.steps, record.seed, record.density) == (12, 40, 1, 0.5)
    assert record.outcome in ("extinct", "fixed", "periodic", "unsettled")
    assert record.langton_lambda == pytest.approx(langton_lambda(LIFE, 6))
    assert record.dual == LIFE.complement(6).to_bs_string(6)
    assert record.mean_field == mean_field_fixed_points(LIFE, 6)[1]
    assert set(record.to_json()) >= {"outcome", "period", "rho_final", "mean_field", "dual"}

    triangle = classify_rule(CensusTask(config, "B2/S3", "triangle", 12, 40, 1, 0.5, 10))
    assert triangle.triangle_mode == config.grid.triangle_mode


def test_run_census_writes_records_and_resumes_without_repeating(config, tmp_path):
    out = tmp_path / "census.jsonl"
    rules = [LIFE, Rule(birth={2}, survive={3, 4})]
    first = list(run_census(config, "square", rules, [8, 10], [1, 2], 20, 0.5, out, tail=5))
    assert len(first) == 8
    assert len(read_jsonl(out)) == 8

    again = list(run_census(config, "square", rules, [8, 10], [1, 2], 20, 0.5, out, tail=5))
    assert again == []
    assert len(read_jsonl(out)) == 8

    more = list(run_census(config, "square", rules, [8, 10], [1, 2, 3], 20, 0.5, out, tail=5))
    assert len(more) == 4 and all(r.seed == 3 for r in more)
    # A record from a different experiment (other steps) does not count as done.
    other = list(run_census(config, "square", rules[:1], [8], [1], 30, 0.5, out, tail=5))
    assert len(other) == 1
    assert len(read_jsonl(out)) == 13


def test_run_census_is_the_same_in_parallel(config, tmp_path):
    rules = list(enumerate_rules(3))[40:46]
    serial = sorted(
        r.to_json().items() and (r.rule, r.grid, r.seed, r.outcome, r.period, r.rho_final)
        for r in run_census(config, "triangle", rules, [8], [1], 30, 0.5, tmp_path / "a.jsonl")
    )
    parallel = sorted(
        (r.rule, r.grid, r.seed, r.outcome, r.period, r.rho_final)
        for r in run_census(
            config, "triangle", rules, [8], [1], 30, 0.5, tmp_path / "b.jsonl", workers=2
        )
    )
    assert serial == parallel


def test_summarise_counts_outcomes_and_periods(config, tmp_path):
    records = list(
        run_census(config, "square", [LIFE], [8], [1, 2, 3], 60, 0.4, tmp_path / "c.jsonl")
    )
    summary = summarise(records)
    assert summary["records"] == 3
    assert sum(summary["outcomes"].values()) == 3
    assert all(int(p) > 1 for p in summary["periods"])


def test_rule_strings_round_trip_through_records_on_twelve_neighbour_triangles(config, tmp_path):
    rule = Rule(birth={2, 10}, survive={0, 12})
    record = classify_rule(CensusTask(config, rule.to_bs_string(12), "triangle", 8, 10, 1, 0.5, 5))
    assert Rule.from_bs_string(record.rule) == rule
    assert Rule.from_bs_string(record.dual) == rule.complement(12)
    assert np.isfinite(record.langton_lambda)
