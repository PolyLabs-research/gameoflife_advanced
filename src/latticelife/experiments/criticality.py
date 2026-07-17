"""Heuristic criticality scoring based on rho/activity shape.

Salvaged from the legacy `analysis/criticality_score.py`. Scoring semantics are
unchanged; band thresholds and weights that used to be hardcoded now come from
`config/default.toml`'s `[criticality]` table (see `latticelife.config`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Band:
    low: float
    high: float
    target_low: float
    target_high: float


@dataclass(frozen=True)
class CriticalityWeights:
    alive: float
    activity: float
    rho_var: float
    act_var: float
    damage: float


@dataclass(frozen=True)
class DamageSplit:
    band: float
    growth: float


@dataclass(frozen=True)
class CriticalityConfig:
    alive_band: Band
    activity_band: Band
    damage_band: Band
    damage_growth_band: Band
    rho_std_scale: float
    act_std_scale: float
    weights: CriticalityWeights
    damage_split: DamageSplit
    freeze_penalty_weight: float


def _band_score(value: float, band: Band) -> float:
    """Piecewise reward: linear up into the target band, flat in band, linear down."""
    if value < band.low or value > band.high:
        return 0.0
    if value < band.target_low:
        return (value - band.low) / (band.target_low - band.low)
    if value > band.target_high:
        return (band.high - value) / (band.high - band.target_high)
    return 1.0


def score_timeseries(
    rho: np.ndarray,
    activity: np.ndarray,
    frozen_at: int | None,
    steps: int,
    config: CriticalityConfig,
    h: np.ndarray | None = None,
) -> float:
    """Compute a heuristic criticality score from timeseries.

    Heuristics:
    - Alive-band: prefer rho_mean in the configured band.
    - Activity-band: prefer moderate activity (not zero, not maximal).
    - Variance: reward some variance but clamp so pure noise doesn't dominate.
    - Freeze penalty: early freeze is penalized.
    """
    if len(rho) == 0 or len(activity) == 0:
        return 0.0

    burn_in = int(steps * 0.4)
    rho_post = rho[burn_in:]
    activity_post = activity[burn_in:]

    rho_mean = float(np.mean(rho_post))
    rho_std = float(np.std(rho_post))
    act_mean = float(np.mean(activity_post))
    act_std = float(np.std(activity_post))

    alive_score = _band_score(rho_mean, config.alive_band)
    activity_score = _band_score(act_mean, config.activity_band)

    rho_var_score = min(rho_std / config.rho_std_scale, 1.0)
    act_var_score = min(act_std / config.act_std_scale, 1.0)

    damage_score = 0.0
    if h is not None and len(h) >= steps:
        h_post = h[burn_in:]
        h_mean = float(np.mean(h_post))
        h_growth = float(h[-1] - h[burn_in])
        h_band = _band_score(h_mean, config.damage_band)
        h_growth_band = _band_score(h_growth, config.damage_growth_band)
        damage_score = (
            config.damage_split.band * h_band + config.damage_split.growth * h_growth_band
        )

    freeze_penalty = 0.0
    if frozen_at is not None:
        freeze_penalty = max(0.0, 1.0 - frozen_at / max(1, steps))

    w = config.weights
    score = (
        w.alive * alive_score
        + w.activity * activity_score
        + w.rho_var * rho_var_score
        + w.act_var * act_var_score
        + w.damage * damage_score
    )

    score *= 1.0 - config.freeze_penalty_weight * freeze_penalty

    return max(0.0, min(1.0, score))
