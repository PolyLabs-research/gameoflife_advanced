"""TOML config load + typed access. All tunables live here; no module-level constants."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from latticelife.experiments.criticality import (
    Band,
    CriticalityConfig,
    CriticalityWeights,
    DamageSplit,
)


@dataclass(frozen=True)
class GridConfig:
    shape: str
    width: int
    height: int
    wrap: bool
    triangle_mode: str


@dataclass(frozen=True)
class SimulationConfig:
    initial_alive_percentage: float
    n_lifeforms: int
    static_threshold: int
    history_limit: int
    seed: int


@dataclass(frozen=True)
class MetricsConfig:
    cluster_stride: int


@dataclass(frozen=True)
class AlertsConfig:
    sudden_death_window: int
    sudden_death_drop: float
    stasis_occupancy: float
    stasis_volatility: float


@dataclass(frozen=True)
class RecordingConfig:
    db_path: str
    batch_size: int


@dataclass(frozen=True)
class DiscoveryConfig:
    mu: int
    n_offspring: int
    evals_per_candidate: int
    grid_size: int
    steps: int
    p_birth: float
    p_survive: float
    mutation_rate: float
    save_threshold: float
    checkpoint_path: str


@dataclass(frozen=True)
class UIConfig:
    fps: int


@dataclass(frozen=True)
class Config:
    grid: GridConfig
    simulation: SimulationConfig
    metrics: MetricsConfig
    alerts: AlertsConfig
    recording: RecordingConfig
    discovery: DiscoveryConfig
    criticality: CriticalityConfig
    ui: UIConfig


def _criticality_config(data: dict) -> CriticalityConfig:
    return CriticalityConfig(
        alive_band=Band(**data["alive_band"]),
        activity_band=Band(**data["activity_band"]),
        damage_band=Band(**data["damage_band"]),
        damage_growth_band=Band(**data["damage_growth_band"]),
        rho_std_scale=data["rho_std_scale"],
        act_std_scale=data["act_std_scale"],
        weights=CriticalityWeights(**data["weights"]),
        damage_split=DamageSplit(**data["damage_split"]),
        freeze_penalty_weight=data["freeze_penalty_weight"],
    )


def load_config(path: str | Path) -> Config:
    """Load and type a TOML config file, e.g. `config/default.toml`."""
    with Path(path).open("rb") as f:
        data = tomllib.load(f)

    return Config(
        grid=GridConfig(**data["grid"]),
        simulation=SimulationConfig(**data["simulation"]),
        metrics=MetricsConfig(**data["metrics"]),
        alerts=AlertsConfig(**data["alerts"]),
        recording=RecordingConfig(**data["recording"]),
        discovery=DiscoveryConfig(**data["discovery"]),
        criticality=_criticality_config(data["criticality"]),
        ui=UIConfig(**data["ui"]),
    )
