"""Lattice-agnostic rule representation.

Salvaged from the legacy `gameoflife_advanced` repo's `rules.py`. `Rule` is the
*only* rule representation in this codebase — the gene-string codec that used
to live in `rule_discovery.py` is discarded; discovery operates on `Rule`
objects directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rule:
    birth: set[int]
    survive: set[int]

    def to_bs_string(self, n: int) -> str:
        """Format the rule as B*/S*, only including counts up to n."""
        b = "".join(str(v) for v in sorted(x for x in self.birth if x <= n))
        s = "".join(str(v) for v in sorted(x for x in self.survive if x <= n))
        return f"B{b}/S{s}"

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
