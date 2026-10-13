from itertools import combinations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from latticelife.core.rules import (
    Rule,
    format_count_list,
    format_counts,
    parse_count_list,
    parse_counts,
)


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


# --- persisted notation (F3) -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("counts", "text"),
    [
        (set(), ""),
        ({3}, "3"),
        ({2, 3}, "23"),
        ({0, 1, 2, 9}, "0129"),
        ({2, 5, 8, 10}, "2,5,8,10"),
        ({0, 8, 12}, "0,8,12"),
        ({12}, "12,"),
        ({10}, "10,"),
    ],
)
def test_format_counts_compact_below_ten_and_comma_separated_from_ten(counts, text):
    assert format_counts(counts) == text
    assert parse_counts(text) == counts


def test_triangle_rules_that_used_to_collide_get_distinct_strings():
    """F3: {12} and {1, 2} both used to persist as '12', {0, 12} and {0, 1, 2} as '012'."""
    a = Rule(birth={12}, survive={0, 12})
    b = Rule(birth={1, 2}, survive={0, 1, 2})
    assert a.to_bs_string(12) != b.to_bs_string(12)
    assert Rule.from_bs_string(a.to_bs_string(12)) == a
    assert Rule.from_bs_string(b.to_bs_string(12)) == b


def test_format_parse_round_trips_every_subset_of_twelve_neighbours():
    seen: dict[str, frozenset[int]] = {}
    for size in range(13 + 1):
        for subset in combinations(range(13), size):
            counts = set(subset)
            text = format_counts(counts)
            assert parse_counts(text) == counts
            assert seen.setdefault(text, frozenset(counts)) == frozenset(counts), text


@given(st.sets(st.integers(0, 12)), st.sets(st.integers(0, 12)))
def test_bs_string_round_trips(birth, survive):
    rule = Rule(birth=birth, survive=survive)
    assert Rule.from_bs_string(rule.to_bs_string(12)) == rule


def test_to_bs_string_uses_commas_for_edge_vertex_triangle_rules():
    rule = Rule(birth={2, 5, 8, 10}, survive={0, 8, 12})
    assert rule.to_bs_string(12) == "B2,5,8,10/S0,8,12"


@pytest.mark.parametrize(
    ("birth", "survive", "text"),
    [
        ({1, 3, 4, 5, 9}, {7, 9, 12}, "B1,3,4,5,9/S7,9,12"),
        ({7, 9, 12}, {1, 3, 4, 5, 9}, "B7,9,12/S1,3,4,5,9"),
        ({12}, {2}, "B12,/S2"),
        ({2}, {12}, "B2/S12,"),
        ({2, 3}, {10}, "B2,3/S10,"),
        ({12}, {2, 3}, "B12,/S2,3"),
        (set(), {0, 12}, "B/S0,12"),
        ({3}, {2, 3}, "B3/S23"),
    ],
)
def test_to_bs_string_picks_one_form_for_the_whole_rule(birth, survive, text):
    """The form was chosen per half, so a 12-neighbour rule could read 'B13459/S7,9,12': compact
    digits beside commas, and '13459' read by a person as holding 13."""
    rule = Rule(birth=birth, survive=survive)
    assert rule.to_bs_string(12) == text
    assert Rule.from_bs_string(text) == rule


def test_to_bs_string_picks_the_form_from_the_counts_it_keeps():
    assert Rule(birth={3}, survive={2, 3, 10}).to_bs_string(8) == "B3/S23"


@given(st.sets(st.integers(0, 12)), st.sets(st.integers(0, 12)))
def test_bs_string_halves_share_one_form(birth, survive):
    text = Rule(birth=birth, survive=survive).to_bs_string(12)
    b_text, s_text = text[1:].split("/S")
    if all(v <= 9 for v in birth | survive):
        assert "," not in text
    else:
        for half, counts in ((b_text, birth), (s_text, survive)):
            assert len(counts) < 2 or "," in half, text  # no run of digits holding two counts


def test_from_bs_string_still_reads_strings_with_mixed_halves():
    """Rows written while the form was chosen per half stay loadable."""
    assert Rule.from_bs_string("B13459/S7,9,12") == Rule({1, 3, 4, 5, 9}, {7, 9, 12})


def test_format_counts_comma_form_can_be_forced():
    assert format_counts({2, 3}, comma_form=True) == "2,3"
    assert format_counts({2}, comma_form=True) == "2"
    assert format_counts({12}, comma_form=True) == "12,"
    assert format_counts(set(), comma_form=True) == ""
    assert format_counts({2, 3}, comma_form=False) == "23"
    for counts in ({2, 3}, {2}, {12}, set()):
        assert parse_counts(format_counts(counts, comma_form=True)) == counts
    with pytest.raises(ValueError, match="comma form"):
        format_counts({2, 12}, comma_form=False)


@pytest.mark.parametrize("text", ["B3/S23", "b3/s23", " B3/S23 ", "B3/S2,3", "B3/S2 3"])
def test_from_bs_string_accepts_both_forms(text):
    assert Rule.from_bs_string(text) == Rule(birth={3}, survive={2, 3})


def test_from_bs_string_accepts_empty_parts():
    assert Rule.from_bs_string("B/S") == Rule(birth=set(), survive=set())


@pytest.mark.parametrize("text", ["", "3/23", "B3S23", "B3/S2x", "B3/S23/S4", "B²/S2"])
def test_from_bs_string_rejects_garbage(text):
    with pytest.raises(ValueError):
        Rule.from_bs_string(text)


def test_format_counts_rejects_negative_counts():
    with pytest.raises(ValueError):
        format_counts({-1, 2})


# --- display notation for rule fields (F2) -----------------------------------------------------


def test_format_count_list_is_always_comma_separated():
    assert format_count_list({3, 2}) == "2,3"
    assert format_count_list({12}) == "12"
    assert format_count_list(set()) == ""


@pytest.mark.parametrize(
    ("text", "counts"),
    [
        ("2,3", {2, 3}),
        ("2, 3", {2, 3}),
        ("2 3", {2, 3}),
        ("3;2;3", {2, 3}),
        ("2,5,8,10", {2, 5, 8, 10}),
        ("0,1,12", {0, 1, 12}),
        ("  ", set()),
        ("", set()),
    ],
)
def test_parse_count_list_reads_integer_tokens(text, counts):
    assert parse_count_list(text, max_count=12) == counts


@given(st.sets(st.integers(0, 12)))
def test_count_list_round_trips(counts):
    assert parse_count_list(format_count_list(counts), max_count=12) == counts


@pytest.mark.parametrize("text", ["²", "2,３", "٣", "½"])
def test_parse_count_list_rejects_non_ascii_digits(text):
    with pytest.raises(ValueError, match="not a digit 0-9"):
        parse_count_list(text, max_count=8)


def test_parse_count_list_rejects_text_without_digits():
    with pytest.raises(ValueError, match="no neighbour counts"):
        parse_count_list("abc", max_count=8)


def test_parse_count_list_rejects_counts_above_the_shape_degree():
    with pytest.raises(ValueError, match="9 exceeds the 8 neighbours"):
        parse_count_list("3,9", max_count=8)
    assert parse_count_list("3,9") == {3, 9}  # no limit given


def test_parse_count_list_echoes_a_long_count_shortened():
    """A long digit run typed into a rule field was echoed whole, one unbroken word wider than
    the settings panel."""
    with pytest.raises(ValueError) as caught:
        parse_count_list("3," + "1" * 60, max_count=8)
    assert str(caught.value) == "11111…11111 exceeds the 8 neighbours this shape has"


def test_parse_count_list_refuses_a_run_past_the_int_conversion_limit_as_too_big():
    """int() of more than 4300 digits raises its own 'Exceeds the limit ...' ValueError."""
    with pytest.raises(ValueError, match="exceeds the 12 neighbours"):
        parse_count_list("9" * 5000, max_count=12)


def test_parse_count_list_reads_leading_zeros_and_shortens_echoed_text():
    assert parse_count_list("007, 0", max_count=8) == {0, 7}
    with pytest.raises(ValueError, match="^9 exceeds"):
        parse_count_list("0009", max_count=8)
    with pytest.raises(ValueError, match=r"^no neighbour counts in ',,,,,…,,,,,'$"):
        parse_count_list("," * 40, max_count=8)
