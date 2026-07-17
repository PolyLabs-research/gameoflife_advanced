"""Lifeform = (id, Rule). No colors, no UI — those live in ui/render.py, keyed by id."""

from __future__ import annotations

from dataclasses import dataclass

from latticelife.core.rules import Rule


@dataclass(frozen=True)
class Lifeform:
    id: int
    rule: Rule
