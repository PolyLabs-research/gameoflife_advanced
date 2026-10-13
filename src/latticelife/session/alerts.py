"""Sudden-death / stasis alerts, evaluated per generation (§6.7).

Closes D-4: in the legacy `game.py:update_simulation`, the stasis check's `else:
self.crash_alert_message = None` branch ran unconditionally right after the sudden-death check,
silently clobbering a `"Sudden death: ..."` message set two lines earlier in the same tick —
there was no precedence between the two checks, just overwrite-in-sequence. Here precedence is
explicit and structural: `evaluate()` is a single if/elif/else chain, so a sudden-death alert is
returned immediately and the stasis condition is never even consulted that tick.

Sudden death fires when the drop is at least `sudden_death_drop` of the population
`sudden_death_window` generations back (§6.7 says "≥"; legacy `game.py` used a strict ">", so an
exact 80% drop, or total extinction with `sudden_death_drop = 1.0`, never fired). The test
compares the dropped fraction itself, `drop / prev_alive`, so an exact fraction such as 0.07
equals the configured literal rather than missing it by a rounding error in `0.07 * 100`.
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


def is_sudden_drop(prev_alive: int, current_alive: int, config: AlertsConfig) -> bool:
    """Whether a fall from `prev_alive` to `current_alive` is a sudden-death drop: at least
    `sudden_death_drop` of a population above zero."""
    return prev_alive > 0 and (prev_alive - current_alive) / prev_alive >= config.sudden_death_drop


def _sudden_death(
    alive_history: Sequence[int], current_alive: int, config: AlertsConfig, generation: int
) -> Alert | None:
    window = config.sudden_death_window
    if len(alive_history) < window:
        return None
    prev_alive = alive_history[-window]
    if is_sudden_drop(prev_alive, current_alive, config):
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

    `alive_history` holds alive counts from prior generations only, oldest first (this tick's
    count is not yet appended); sudden death compares against the entry `sudden_death_window`
    generations back and needs that many entries. `occupancy` is alive_count / total_cells as a
    fraction in [0, 1]; `volatility` is the §6.6 grid-level metric.
    """
    sudden_death = _sudden_death(alive_history, current_alive, config, generation)
    if sudden_death is not None:
        return sudden_death
    return _stasis(occupancy, volatility, config)
