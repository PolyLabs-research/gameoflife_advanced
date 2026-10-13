"""Lattice-agnostic rule representation.

Salvaged from the legacy `gameoflife_advanced` repo's `rules.py`. `Rule` is the
*only* rule representation in this codebase — the gene-string codec that used
to live in `rule_discovery.py` is discarded; discovery operates on `Rule`
objects directly.

Neighbour-count notation. A set of counts has two text forms, and every module that writes or
reads one goes through the helpers here rather than joining digits itself:

- **Persisted form** (`format_counts` / `parse_counts`, used by `Rule.to_bs_string`, the
  recorder's `lifeform_meta` columns and `discovered_rules.jsonl`): compact digits while every
  count is <= 9, so the spec's examples stay byte-identical (`'3'`, `'23'`, `B3/S23`), and
  comma-separated as soon as any count is >= 10 (`'2,5,8,10'`). Plain digit concatenation is
  not injective once counts reach 10 — {12} and {1, 2} would both read `'12'` — and triangle
  edge+vertex has 12 neighbours (§6.1). A lone count >= 10 keeps a trailing comma (`'12,'`) so
  the text still carries the separator that tells the parser it is not two digits. A B/S
  string picks one form for the whole rule (`B1,3,4,5,9/S7,9,12`, never `B13459/S7,9,12`);
  the recorder's separate birth and survive columns each pick their own. The parser reads each
  half on its own, so strings in either style load.
- **Display form** (`format_count_list` / `parse_count_list`, used by the settings panel's rule
  fields): always comma-separated (`'2,3'`), parsed leniently from whatever separators a person
  types, validated against the shape's maximum degree.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

_ASCII_DIGITS = "0123456789"
_LIST_SEPARATORS = re.compile(r"[,\s]+")
_DIGIT_RUNS = re.compile(r"[0-9]+")
_BS_STRING = re.compile(r"^\s*B([^/]*)/S(.*?)\s*$", re.IGNORECASE)


def _sorted_counts(counts: Iterable[int]) -> list[int]:
    values = sorted({int(v) for v in counts})
    if values and values[0] < 0:
        raise ValueError(f"neighbour counts cannot be negative: {values}")
    return values


def format_counts(counts: Iterable[int], *, comma_form: bool | None = None) -> str:
    """Persisted form of a count set: `'23'` while every count is <= 9, else `'2,10,12'`
    (with a trailing comma for a single count >= 10, e.g. `'12,'`). Inverse of
    `parse_counts`.

    `comma_form=True` writes commas even when every count is <= 9 (`'2,3'`; a single count
    <= 9 needs no comma, `'2'` reads back the same either way), so the two halves of one rule
    can share a form (`Rule.to_bs_string`). `None` picks the form from these counts alone.
    `False` with a count >= 10 raises ValueError: compact digits cannot hold it."""
    values = _sorted_counts(counts)
    needs_commas = any(v > 9 for v in values)
    if comma_form is None:
        comma_form = needs_commas
    elif not comma_form and needs_commas:
        raise ValueError(f"counts >= 10 need the comma form: {values}")
    if not comma_form:
        return "".join(str(v) for v in values)
    text = ",".join(str(v) for v in values)
    return f"{text}," if len(values) == 1 and values[0] > 9 else text


def parse_counts(text: str) -> set[int]:
    """Parse the persisted form back into a set. Text containing a comma or whitespace is read
    as integer tokens (`'2,10,12'`, `'12,'`, `'2 3'`); otherwise each digit is one count
    (`'23'` -> {2, 3}). Raises ValueError for anything that is not ASCII digits and separators."""
    stripped = text.strip()
    if not stripped:
        return set()
    if "," in stripped or any(ch.isspace() for ch in stripped):
        tokens = [token for token in _LIST_SEPARATORS.split(stripped) if token]
    else:
        tokens = list(stripped)
    counts: set[int] = set()
    for token in tokens:
        if not all(ch in _ASCII_DIGITS for ch in token):
            raise ValueError(f"invalid neighbour count {token!r} in {text!r}")
        counts.add(int(token))
    return counts


def format_count_list(counts: Iterable[int]) -> str:
    """Display form for editable rule fields: always comma-separated, e.g. `'2,3'`."""
    return ",".join(str(v) for v in _sorted_counts(counts))


def _excerpt(text: str, limit: int = 12) -> str:
    """`text` as an error message echoes it: whole when short, else its two ends around an
    ellipsis, so one long run typed into a field cannot swamp the message."""
    return text if len(text) <= limit else f"{text[:5]}…{text[-5:]}"


def parse_count_list(text: str, max_count: int | None = None) -> set[int]:
    """Parse a typed rule field. Every run of ASCII digits 0-9 is one count and anything else
    separates them (`'2,3'`, `'2 3'`, `'2;10'`). Raises ValueError with a message fit to show
    the user when the text holds a non-ASCII digit (e.g. `'²'` or `'３'`, which would otherwise
    vanish silently as a separator), has no digits at all, or names a count above `max_count`.
    Long text the message echoes is shortened (`_excerpt`). Empty text is the empty set."""
    stripped = text.strip()
    if not stripped:
        return set()
    for ch in stripped:
        if ch not in _ASCII_DIGITS and (ch.isdigit() or ch.isnumeric()):
            raise ValueError(f"'{ch}' is not a digit 0-9")
    # Without leading zeros a count's length bounds its size, so a long run is refused as too
    # big without int(), which raises its own error past 4300 digits.
    tokens = {token.lstrip("0") or "0" for token in _DIGIT_RUNS.findall(stripped)}
    if not tokens:
        raise ValueError(f"no neighbour counts in {_excerpt(stripped)!r}")
    if max_count is not None:
        too_big = sorted(
            (t for t in tokens if len(t) > len(str(max_count)) or int(t) > max_count),
            key=lambda t: (len(t), t),
        )
        if too_big:
            listed = ", ".join(_excerpt(t) for t in too_big)
            raise ValueError(f"{listed} exceeds the {max_count} neighbours this shape has")
    return {int(token) for token in tokens}


@dataclass(frozen=True)
class Rule:
    birth: set[int]
    survive: set[int]

    def to_bs_string(self, n: int) -> str:
        """Format the rule as B*/S* in the persisted notation (`format_counts`), only including
        counts up to n: `B3/S23`, or `B2,5,8,10/S0,8,12` once any count of the rule reaches 10.
        The form is chosen for the whole rule, not per half, so a 12-neighbour rule never reads
        `B13459/S7,9,12` (is that 13 or 1 and 3?) but `B1,3,4,5,9/S7,9,12`."""
        birth = [x for x in self.birth if x <= n]
        survive = [x for x in self.survive if x <= n]
        comma_form = any(x > 9 for x in birth + survive)
        b = format_counts(birth, comma_form=comma_form)
        s = format_counts(survive, comma_form=comma_form)
        return f"B{b}/S{s}"

    @staticmethod
    def from_bs_string(text: str) -> Rule:
        """Inverse of `to_bs_string`: accepts `B3/S23`, `B2,10/S0,12`, `B12,/S2` and the empty
        `B/S` (case-insensitive). Each half is parsed on its own (`parse_counts`), so a string
        whose halves use different forms (`B13459/S7,9,12`) also loads. Raises ValueError for
        anything else."""
        match = _BS_STRING.match(text)
        if match is None:
            raise ValueError(f"not a B/S rule string: {text!r}")
        return Rule(birth=parse_counts(match.group(1)), survive=parse_counts(match.group(2)))

    @staticmethod
    def random(n: int, p_birth: float, p_survive: float, rng: np.random.Generator) -> Rule:
        """Sample a random rule with independent inclusion probabilities."""
        birth = {i for i in range(n + 1) if rng.random() < p_birth}
        survive = {i for i in range(n + 1) if rng.random() < p_survive}
        return Rule(birth=birth, survive=survive)

    def complement(self, n: int) -> Rule:
        """The dual rule under swapping alive and dead, for a lattice of degree `n`: running
        this rule on a grid and the complement rule on the complemented grid keeps the two grids
        complementary for ever (checked by the test suite on every shape). A dead cell of the
        complemented grid is a live cell here with `n - k` live neighbours, so it is born there
        exactly when it dies here: `B' = {n - k : k not in S}`, and likewise
        `S' = {n - k : k not in B}`. An involution; Conway's B3/S23 maps to AntiLife
        B0123478/S01234678. Counts above `n` are dropped, as `to_bs_string` drops them."""
        return Rule(
            birth={n - k for k in range(n + 1) if k not in self.survive},
            survive={n - k for k in range(n + 1) if k not in self.birth},
        )

    def mutate(self, n: int, rate: float, rng: np.random.Generator) -> Rule:
        """Flip bits with probability `rate` for each neighbor count up to n."""
        birth = set(self.birth)
        survive = set(self.survive)

        for i in range(n + 1):
            if rng.random() < rate:
                birth.symmetric_difference_update({i})
            if rng.random() < rate:
                survive.symmetric_difference_update({i})

        return Rule(birth=birth, survive=survive)
