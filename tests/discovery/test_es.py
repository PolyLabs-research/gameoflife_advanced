from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latticelife.config import load_config
from latticelife.core.geometries import max_degree
from latticelife.core.rules import Rule
from latticelife.discovery import es
from latticelife.discovery.es import Candidate, ESState

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.toml"


def _config():
    return load_config(DEFAULT_CONFIG_PATH)


def _fast_config():
    """A config trimmed for evaluate_candidate() calls: same population shape as the real
    default, but a tiny grid/step count so simulation cost stays sub-second per test."""
    base = _config()
    return replace(base, discovery=replace(base.discovery, grid_size=10, steps=20))


def test_initial_population_size_and_rule_validity():
    config = _config()
    rng = np.random.default_rng(0)
    population = es.initial_population("square", "edge+vertex", config.discovery, rng)

    assert len(population) == config.discovery.mu
    max_n = max_degree("square")
    for candidate in population:
        assert candidate.rule.birth <= set(range(max_n + 1))
        assert candidate.rule.survive <= set(range(max_n + 1))
        assert candidate.eval_count == 0


def test_mutate_population_size_and_freshness():
    config = _config()
    rng = np.random.default_rng(0)
    parents = es.initial_population("square", config.grid.triangle_mode, config.discovery, rng)

    offspring = es.mutate_population(parents, "square", config, rng)

    assert len(offspring) == config.discovery.n_offspring
    assert all(c.eval_count == 0 for c in offspring)


def test_select_elites_picks_best_mu_by_mean_score():
    candidates = [
        Candidate(Rule(birth={i}, survive={2}), total_score=float(i), eval_count=1)
        for i in range(5)
    ]
    elites = es.select_elites(candidates, mu=2)
    assert [c.mean_score for c in elites] == [4.0, 3.0]


def test_candidate_running_mean_across_evaluations():
    candidate = Candidate(Rule(birth={3}, survive={2, 3}))
    candidate.add_evaluation(0.5)
    candidate.add_evaluation(0.9)
    assert candidate.eval_count == 2
    assert candidate.mean_score == pytest.approx(0.7)


def test_evaluate_candidate_invalid_rule_short_circuits_to_zero():
    """A rule with no birth or no survive condition can never produce/sustain life — legacy
    short-circuited this without spending a simulation on it."""
    config = _fast_config()
    rng = np.random.default_rng(0)
    assert es.evaluate_candidate(Rule(birth=set(), survive={2, 3}), "square", config, rng) == 0.0
    assert es.evaluate_candidate(Rule(birth={3}, survive=set()), "square", config, rng) == 0.0


def test_evaluate_candidate_deterministic_given_same_rng_seed():
    config = _fast_config()
    rule = Rule(birth={3}, survive={2, 3})
    score_a = es.evaluate_candidate(rule, "square", config, np.random.default_rng(7))
    score_b = es.evaluate_candidate(rule, "square", config, np.random.default_rng(7))
    assert score_a == score_b


def test_evaluate_candidate_score_in_unit_interval():
    config = _fast_config()
    rule = Rule(birth={3}, survive={2, 3})
    score = es.evaluate_candidate(rule, "square", config, np.random.default_rng(1))
    assert 0.0 <= score <= 1.0


def test_checkpoint_round_trip_preserves_state():
    rule = Rule(birth={3}, survive={2, 3})
    candidate = Candidate(rule=rule, total_score=1.5, eval_count=3)
    state = ESState(generation=5, population=[candidate], shape="square", entropy=12345)

    restored = es.from_checkpoint_dict(es.to_checkpoint_dict(state))

    assert restored.generation == 5
    assert restored.shape == "square"
    assert restored.entropy == 12345
    assert restored.population[0].rule == rule
    assert restored.population[0].eval_count == 3
    assert restored.population[0].mean_score == pytest.approx(0.5)
