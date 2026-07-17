import numpy as np

from latticelife.core.rules import Rule


def test_to_bs_string_formats_sorted_counts():
    rule = Rule(birth={3}, survive={2, 3})
    assert rule.to_bs_string(8) == "B3/S23"


def test_to_bs_string_clamps_to_n():
    rule = Rule(birth={3, 9}, survive={2, 3})
    assert rule.to_bs_string(8) == "B3/S23"


def test_random_is_reproducible_with_seeded_generator():
    rule_a = Rule.random(8, p_birth=0.3, p_survive=0.3, rng=np.random.default_rng(42))
    rule_b = Rule.random(8, p_birth=0.3, p_survive=0.3, rng=np.random.default_rng(42))
    assert rule_a == rule_b


def test_mutate_is_pure():
    rule = Rule(birth={3}, survive={2, 3})
    mutated = rule.mutate(8, rate=0.5, rng=np.random.default_rng(0))
    assert rule.birth == {3}
    assert rule.survive == {2, 3}
    assert isinstance(mutated, Rule)
