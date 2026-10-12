import re
import signal
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latticelife.config import (
    ConfigError,
    load_config,
    min_discovery_steps,
    validate_discovery_config,
)
from latticelife.core.geometries import max_degree
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.discovery import es
from latticelife.discovery.es import Candidate, ESState
from latticelife.experiments.criticality import BURN_IN_FRAC
from latticelife.experiments.runner import run_timeseries
from latticelife.session.session import SimulationSession

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


def test_evaluate_candidate_ranks_a_living_rule_above_dead_and_frozen_ones():
    """F64: the old `0 <= score <= 1` assertion could not fail (scores are clamped). Conway
    keeps a 10x10 soup active; B8/S8 dies at once; B8/S012345678 freezes at once."""
    config = _fast_config()
    scores = {
        text: [
            es.evaluate_candidate(Rule.from_bs_string(text), "square", config, rng)
            for rng in np.random.default_rng(1).spawn(3)
        ]
        for text in ("B3/S23", "B8/S8", "B8/S012345678")
    }
    assert min(scores["B3/S23"]) > 0.5
    assert max(scores["B8/S8"]) < 0.2
    assert max(scores["B8/S012345678"]) < min(scores["B3/S23"])


def _settings(shape="square", **discovery_overrides):
    base = _config()
    config = replace(base, discovery=replace(base.discovery, **discovery_overrides))
    return es.search_settings(config, shape)


def _state(population, shape="square", generation=5):
    return ESState(
        generation=generation,
        population=population,
        shape=shape,
        entropy=12345,
        settings=_settings(shape),
    )


def test_checkpoint_round_trip_preserves_state():
    rule = Rule(birth={3}, survive={2, 3})
    candidate = Candidate(rule=rule, total_score=1.5, eval_count=3)
    state = _state([candidate])

    data = es.to_checkpoint_dict(state)
    restored = es.from_checkpoint_dict(data)

    assert data["version"] == es.CHECKPOINT_VERSION
    assert data["settings"] == state.settings
    assert restored.generation == 5
    assert restored.shape == "square"
    assert restored.entropy == 12345
    assert restored.settings == state.settings
    assert restored.population[0].rule == rule
    assert restored.population[0].eval_count == 3
    assert restored.population[0].mean_score == pytest.approx(0.5)


def test_checkpoint_without_settings_is_rejected():
    """F20: a pre-settings checkpoint says nothing about the landscape it was evolved in."""
    data = es.to_checkpoint_dict(_state([Candidate(Rule({3}, {2, 3}), 1.0, 1)]))
    legacy = {k: v for k, v in data.items() if k not in ("version", "settings")}
    with pytest.raises(ValueError, match="version"):
        es.from_checkpoint_dict(legacy)


def test_checkpoint_with_counts_the_shape_cannot_have_is_rejected():
    """F20: a square population (counts 7, 8) stored under hexagon settings (max degree 6)."""
    data = es.to_checkpoint_dict(_state([Candidate(Rule({0, 4, 5, 7, 8}, {8}), 1.0, 1)]))
    data["shape"] = "hexagon"
    data["settings"] = _settings("hexagon")
    with pytest.raises(ValueError, match="outside 0..6"):
        es.from_checkpoint_dict(data)


def test_checkpoint_population_copies_are_merged():
    """F21: old checkpoints held one rule several times (e.g. B12568/S1 twice)."""
    rule = Rule({1, 2, 5, 6, 8}, {1})
    data = es.to_checkpoint_dict(
        _state(
            [
                Candidate(rule, 0.9 * 3, 3),
                Candidate(Rule({3}, {2}), 0.5, 1),
                Candidate(rule, 0.6 * 3, 3),
            ]
        )
    )
    restored = es.from_checkpoint_dict(data)
    assert len(restored.population) == 2
    merged = restored.population[0]
    assert merged.rule == rule
    assert merged.eval_count == 6
    assert merged.mean_score == pytest.approx(0.75)


def test_search_settings_capture_the_fitness_landscape():
    """F20: everything that changes what a score means changes the lineage; ES
    hyperparameters do not."""
    base = _settings()
    assert _settings("hexagon") != base
    assert _settings(grid_size=32) != base
    assert _settings(steps=300) != base
    assert _settings(freeze_window=50) != base
    assert _settings(mu=9, n_offspring=3, mutation_rate=0.2, evals_per_candidate=5) == base

    config = _config()
    criticality = replace(config.criticality, freeze_penalty_weight=0.25)
    assert es.search_settings(replace(config, criticality=criticality), "square") != base

    def with_mode(shape, mode):
        return es.search_settings(
            replace(config, grid=replace(config.grid, triangle_mode=mode)), shape
        )

    assert with_mode("triangle", "edge") != with_mode("triangle", "edge+vertex")
    assert with_mode("square", "edge") == with_mode("square", "edge+vertex")


def test_checkpoint_names_identify_shape_mode_and_settings():
    square = es.checkpoint_name(_settings())
    assert square.startswith("checkpoint-square-") and square.endswith(".json")
    assert es.checkpoint_name(_settings(steps=300)) != square
    config = _config()
    triangle = es.search_settings(
        replace(config, grid=replace(config.grid, triangle_mode="edge")), "triangle"
    )
    assert es.checkpoint_name(triangle).startswith("checkpoint-triangle-edge-")
    assert es.settings_id(_settings()) == es.settings_id(_settings())


def test_a_lineage_does_not_depend_on_how_its_numbers_are_spelt(tmp_path):
    """`low = 0` and `low = 0.0` load as equal configs, but the integer was fingerprinted as
    written: the run silently left its lineage (population and evidence) for a fresh one."""
    floats = DEFAULT_CONFIG_PATH.read_text()
    integers, spelt = re.subn(r"(?m)^(\w+ = -?\d+)\.0$", r"\1", floats)
    assert spelt >= 3  # the bands' `low = 0.0`, at least
    (tmp_path / "floats.toml").write_text(floats)
    (tmp_path / "integers.toml").write_text(integers)
    a = load_config(tmp_path / "floats.toml")
    b = load_config(tmp_path / "integers.toml")
    for shape in ("square", "hexagon", "triangle"):
        assert es.checkpoint_name(es.search_settings(b, shape)) == es.checkpoint_name(
            es.search_settings(a, shape)
        )


@pytest.mark.parametrize(
    ("shape", "mode"),
    [
        ("square", "edge+vertex"),
        ("hexagon", "edge+vertex"),
        ("triangle", "edge"),
        ("triangle", "edge+vertex"),
    ],
)
@pytest.mark.parametrize("rate", [0.0, 0.01, 0.08])
def test_offspring_never_clone_a_parent_or_each_other(shape, mode, rate):
    """F21: Rule.mutate returns its parent unchanged 22% (square) to 51% (triangle edge) of the
    time at the default rate, and those clones used to enter the pool as extra candidates."""
    base = _config()
    config = replace(
        base,
        grid=replace(base.grid, triangle_mode=mode),
        discovery=replace(base.discovery, mutation_rate=rate, n_offspring=15),
    )
    rng = np.random.default_rng(4)
    parents = es.initial_population(shape, mode, config.discovery, rng)
    for _ in range(20):
        offspring = es.mutate_population(parents, shape, config, rng)
        keys = [es.rule_key(c.rule) for c in parents + offspring]
        assert len(offspring) == 15
        assert len(set(keys)) == len(keys)
        n = max_degree(shape, mode)
        for child in offspring:
            assert child.rule.birth | child.rule.survive <= set(range(n + 1))


def test_merge_duplicates_pools_the_evidence_of_one_rule():
    rule = Rule({3}, {2, 3})
    pool = [Candidate(rule, 2.4, 3), Candidate(Rule({2}, {2}), 0.1, 1), Candidate(rule, 0.6, 1)]
    merged = es.merge_duplicates(pool)
    assert [c.rule for c in merged] == [rule, Rule({2}, {2})]
    assert merged[0].eval_count == 4
    assert merged[0].mean_score == pytest.approx(0.75)


def test_select_elites_returns_distinct_rules():
    rule = Rule({3}, {2, 3})
    pool = [Candidate(rule, 0.9, 1), Candidate(rule, 0.8, 1), Candidate(Rule({2}, {2}), 0.1, 1)]
    elites = es.select_elites(pool, mu=2)
    assert [c.rule for c in elites] == [rule, Rule({2}, {2})]


def test_elite_sets_stay_distinct_over_a_synthetic_run():
    """F21's reproduction: 40 generations with a noisy synthetic fitness. The elite set used to
    hold a duplicated rule in 37 of 40 generations."""
    config = _config()
    rng = np.random.default_rng(0)
    population = es.initial_population("square", "edge+vertex", config.discovery, rng)
    for _ in range(40):
        offspring = es.mutate_population(population, "square", config, rng)
        pool = es.merge_duplicates([*population, *offspring])
        for candidate in pool:
            for _ in range(config.discovery.evals_per_candidate):
                value = (len(candidate.rule.birth) * 7 + len(candidate.rule.survive) * 3) % 10
                candidate.add_evaluation(value / 10 + rng.normal(0, 0.05))
        population = es.select_elites(pool, config.discovery.mu)
        keys = [es.rule_key(c.rule) for c in population]
        assert len(set(keys)) == len(keys) == config.discovery.mu


# --- F24: freeze detection must fit inside discovery.steps --------------------------------------

FREEZES_AT_ONCE = Rule.from_bs_string("B8/S012345678")


def _write_config(tmp_path, **discovery):
    text = DEFAULT_CONFIG_PATH.read_text()
    for key, value in discovery.items():
        text = re.sub(rf"(?m)^{key} = \S+", f"{key} = {value}", text)
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_steps_too_short_for_freeze_detection_are_rejected_for_discovery(tmp_path):
    def load(**discovery):
        return load_config(_write_config(tmp_path, **discovery), for_discovery=True)

    with pytest.raises(ConfigError, match=r"discovery\.steps \(100\).*steps >= 166"):
        load(steps=100)
    with pytest.raises(ConfigError, match="steps >= 166"):
        load(steps=165)
    assert load(steps=166).discovery.steps == 166
    assert load(steps=100, freeze_window=20).discovery.freeze_window == 20


def test_steps_too_short_for_freeze_detection_still_load_for_other_commands(tmp_path):
    """The freeze rule only matters to discovery scoring; `ui` and `autorun` (plain
    `load_config`) never read discovery.steps, so a quick-run steps = 100 must not stop them."""
    config = load_config(_write_config(tmp_path, steps=100))
    assert (config.discovery.steps, config.discovery.freeze_window) == (100, 100)
    with pytest.raises(ConfigError, match=r"discovery\.steps \(100\)"):
        validate_discovery_config(config)


def test_min_discovery_steps_is_the_smallest_steps_that_detects_a_freeze():
    for freeze_window in range(1, 400):
        expected = 1
        while int(expected * BURN_IN_FRAC) + freeze_window > expected:
            expected += 1
        assert min_discovery_steps(freeze_window) == expected, freeze_window
    assert min_discovery_steps(30_000_000) == 49_999_999


def _raise_timeout(signum, frame):
    raise TimeoutError("took too long")


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX interval timers")
def test_a_huge_freeze_window_is_rejected_at_once(tmp_path):
    """The suggested minimum steps was found by counting up one step at a time, so a slip like
    freeze_window = 30000000 took seconds to report and 10**12 would have taken days."""
    path = _write_config(tmp_path, freeze_window=10**12)
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        with pytest.raises(ConfigError, match=r"steps >= 1666666666666 or"):
            load_config(path, for_discovery=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_default_steps_keep_the_legacy_freeze_detection():
    config = _config()
    assert (config.discovery.steps, config.discovery.freeze_window) == (200, 100)
    session_config = es._session_config_for_discovery(
        replace(config, discovery=replace(config.discovery, grid_size=16)), "square"
    )
    session = SimulationSession(session_config, [Lifeform(1, FREEZES_AT_ONCE)], seed=1)
    assert run_timeseries(session, 1, steps=200).frozen_at == 179


def test_evaluate_candidate_uses_the_configured_freeze_window():
    """With steps=100 the legacy window of 100 can never detect a freeze, so a rule that
    freezes immediately escaped the penalty; a window of 20 applies it."""
    base = _config()

    def score(freeze_window):
        config = replace(
            base,
            discovery=replace(base.discovery, grid_size=16, steps=100, freeze_window=freeze_window),
        )
        return es.evaluate_candidate(FREEZES_AT_ONCE, "square", config, np.random.default_rng(1))

    assert score(20) < score(100)
