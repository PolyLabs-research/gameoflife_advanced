"""Sudden-death / stasis alerts, evaluated per generation (§6.7).

Closes D-4: in the legacy `game.py:update_simulation`, the stasis check's `else:
self.crash_alert_message = None` branch ran unconditionally right after the sudden-death check,
silently clobbering a `"Sudden death: ..."` message set two lines earlier in the same tick —
there was no precedence between the two checks, just overwrite-in-sequence. Here precedence is
explicit and structural: `evaluate()` is a single if/elif/else chain, so a sudden-death alert is
returned immediately and the stasis condition is never even consulted that tick.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from latticelife.config import AlertsConfig


class AlertKind(Enum):
    SUDDEN_DEATH = "sudden_death"
    STASIS = "stasis"


@dataclass(frozen=True)
class Alert:
    kind: AlertKind
    message: str


def _sudden_death(
    alive_history: Sequence[int], current_alive: int, config: AlertsConfig, generation: int
) -> Alert | None:
    window = config.sudden_death_window
    if len(alive_history) < window:
        return None
    prev_alive = alive_history[-window]
    if prev_alive <= 0:
        return None
    drop = prev_alive - current_alive
    if drop > config.sudden_death_drop * prev_alive:
        return Alert(
            AlertKind.SUDDEN_DEATH,
            f"Sudden death: gen {generation} drop {prev_alive}->{current_alive}",
        )
    return None


def _stasis(occupancy: float, volatility: float, config: AlertsConfig) -> Alert | None:
    if occupancy > config.stasis_occupancy and volatility < config.stasis_volatility:
        return Alert(AlertKind.STASIS, "CRITICAL STASIS DETECTED")
    return None


def evaluate(
    alive_history: Sequence[int],
    current_alive: int,
    occupancy: float,
    volatility: float,
    config: AlertsConfig,
    generation: int,
) -> Alert | None:
    """Precedence: sudden-death > stasis > none, evaluated fresh each tick.

    `alive_history` holds alive counts from prior generations only (this tick's count is not
    yet appended). `occupancy` is alive_count / total_cells as a fraction in [0, 1]; `volatility`
    is the §6.6 grid-level metric.
    """
    sudden_death = _sudden_death(alive_history, current_alive, config, generation)
    if sudden_death is not None:
        return sudden_death
    return _stasis(occupancy, volatility, config)
