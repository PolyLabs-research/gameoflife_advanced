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
  the text still carries the separator that tells the parser it is not two digits.
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


def format_counts(counts: Iterable[int]) -> str:
    """Persisted form of a count set: `'23'` while every count is <= 9, else `'2,10,12'`
    (with a trailing comma for a single count >= 10, e.g. `'12,'`). Inverse of
    `parse_counts`."""
    values = _sorted_counts(counts)
    if all(v <= 9 for v in values):
        return "".join(str(v) for v in values)
    text = ",".join(str(v) for v in values)
    return text if len(values) > 1 else f"{text},"


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


def parse_count_list(text: str, max_count: int | None = None) -> set[int]:
    """Parse a typed rule field. Every run of ASCII digits 0-9 is one count and anything else
    separates them (`'2,3'`, `'2 3'`, `'2;10'`). Raises ValueError with a message fit to show
    the user when the text holds a non-ASCII digit (e.g. `'²'` or `'３'`, which would otherwise
    vanish silently as a separator), has no digits at all, or names a count above `max_count`.
    Empty text is the empty set."""
    stripped = text.strip()
    if not stripped:
        return set()
    for ch in stripped:
        if ch not in _ASCII_DIGITS and (ch.isdigit() or ch.isnumeric()):
            raise ValueError(f"'{ch}' is not a digit 0-9")
    tokens = _DIGIT_RUNS.findall(stripped)
    if not tokens:
        raise ValueError(f"no neighbour counts in {stripped!r}")
    counts = {int(token) for token in tokens}
    if max_count is not None:
        too_big = sorted(v for v in counts if v > max_count)
        if too_big:
            listed = ", ".join(str(v) for v in too_big)
            raise ValueError(f"{listed} exceeds the {max_count} neighbours this shape has")
    return counts


@dataclass(frozen=True)
class Rule:
    birth: set[int]
    survive: set[int]

    def to_bs_string(self, n: int) -> str:
        """Format the rule as B*/S* in the persisted notation (`format_counts`), only including
        counts up to n: `B3/S23`, or `B2,5,8,10/S0,8,12` once a count reaches 10."""
        b = format_counts(x for x in self.birth if x <= n)
        s = format_counts(x for x in self.survive if x <= n)
        return f"B{b}/S{s}"

    @staticmethod
    def from_bs_string(text: str) -> Rule:
        """Inverse of `to_bs_string`: accepts `B3/S23`, `B2,10/S0,12,` and the empty `B/S`
        (case-insensitive). Raises ValueError for anything else."""
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
