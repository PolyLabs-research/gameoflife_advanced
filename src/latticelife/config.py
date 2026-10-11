"""TOML config load + typed access. All tunables live here; no module-level constants.

`load_config()` validates everything it loads (`validate_config()`): each table must have exactly
its known keys, each value the right type and range, and the grid (plus discovery's always-wrapped
`grid_size`, for every shape) must pass `core.geometries.validate_dimensions`. Any problem raises
`ConfigError` naming the `section.key` — a bad value fails at load time, before a session, a
recording or a discovery run has started on it.

Paths. `DEFAULT_CONFIG_PATH` is the checkout's `config/default.toml`, found from this file's
location rather than the current directory. Relative output paths in a config (`recording.db_path`,
`discovery.checkpoint_path`, `discovery.discovered_path`, `logging.dir`) are resolved against the
config file's project root — the nearest directory at or above the file that holds a
`pyproject.toml`, else the file's own directory — and stored as absolute paths, so the same config
writes to the same `out/` whichever directory a command is started from.
"""

from __future__ import annotations

import tomllib
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

from latticelife.core.geometries import SHAPES, TRIANGLE_MODES, validate_dimensions
from latticelife.experiments.criticality import (
    BURN_IN_FRAC,
    Band,
    CriticalityConfig,
    CriticalityWeights,
    DamageSplit,
)


def _source_project_root() -> Path:
    """The checkout this package runs from (src layout: `<root>/src/latticelife/config.py`).
    Falls back to the current directory for an installed, non-editable copy, which has no
    project directory of its own."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd()


PROJECT_ROOT = _source_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"

# Bounds that are not tunables: the int8 occupancy array and the panel's species and FPS
# controls are built around these.
MAX_LIFEFORMS = 10
MIN_FPS = 1
MAX_FPS = 60


class ConfigError(ValueError):
    """A config value is missing, unknown, of the wrong type or out of range. The message
    names the offending `section.key` (and the file, when raised by `load_config`)."""


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
    # Optional in TOML (default `out/discovered_rules.jsonl` under the project root).
    discovered_path: str = str(PROJECT_ROOT / "out" / "discovered_rules.jsonl")
    # Optional in TOML: quiet steps after burn-in that count as a freeze (legacy: 100).
    freeze_window: int = 100


@dataclass(frozen=True)
class LoggingConfig:
    """`[logging]` is optional in TOML; `dir` defaults to `out/logs` under the project root."""

    dir: str = str(PROJECT_ROOT / "out" / "logs")


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
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Relative defaults for optional keys, resolved against the config's project root like any
# value written in the file.
_OPTIONAL_DEFAULTS: dict[str, dict[str, str]] = {
    "discovery": {"discovered_path": "out/discovered_rules.jsonl"},
    "logging": {"dir": "out/logs"},
}
_OPTIONAL_TABLES = frozenset({"logging"})
_PATH_KEYS: dict[str, tuple[str, ...]] = {
    "recording": ("db_path",),
    "discovery": ("checkpoint_path", "discovered_path"),
    "logging": ("dir",),
}
_SECTIONS: dict[str, type] = {
    "grid": GridConfig,
    "simulation": SimulationConfig,
    "metrics": MetricsConfig,
    "alerts": AlertsConfig,
    "recording": RecordingConfig,
    "discovery": DiscoveryConfig,
    "ui": UIConfig,
    "logging": LoggingConfig,
}
_CRITICALITY_TABLES: dict[str, type] = {
    "alive_band": Band,
    "activity_band": Band,
    "damage_band": Band,
    "damage_growth_band": Band,
    "weights": CriticalityWeights,
    "damage_split": DamageSplit,
}


def project_root_for(config_path: str | Path) -> Path:
    """The directory a config's relative output paths resolve against (see module docstring)."""
    config_dir = Path(config_path).resolve().parent
    for candidate in (config_dir, *config_dir.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return config_dir


# --- typed construction -----------------------------------------------------------------------


def _type_ok(value: Any, annotation: str) -> bool:
    if annotation == "bool":
        return type(value) is bool
    if annotation == "int":
        return type(value) is int
    if annotation == "float":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if annotation == "str":
        return isinstance(value, str)
    return True


_TYPE_NAMES = {"bool": "true or false", "int": "an integer", "float": "a number", "str": "a string"}


def _build(cls: type, table: Any, name: str) -> Any:
    """Construct dataclass `cls` from TOML table `name`, rejecting missing/unknown keys and
    values of the wrong type."""
    if not isinstance(table, dict):
        raise ConfigError(f"[{name}] must be a table")
    values = dict(table)
    known = {f.name: f for f in fields(cls)}
    for key in values:
        if key not in known:
            raise ConfigError(f"unknown key {name}.{key}")
    for key, f in known.items():
        if key not in values:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ConfigError(f"missing key {name}.{key}")
            continue
        annotation = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        if not _type_ok(values[key], annotation):
            raise ConfigError(
                f"{name}.{key} must be {_TYPE_NAMES.get(annotation, annotation)} "
                f"(got {values[key]!r})"
            )
    return cls(**values)


def _criticality_config(data: Any) -> CriticalityConfig:
    if not isinstance(data, dict):
        raise ConfigError("[criticality] must be a table")
    nested = {}
    for key, cls in _CRITICALITY_TABLES.items():
        if key not in data:
            raise ConfigError(f"missing table [criticality.{key}]")
        nested[key] = _build(cls, data[key], f"criticality.{key}")
    scalars = {key: value for key, value in data.items() if key not in _CRITICALITY_TABLES}
    return _build(CriticalityConfig, {**scalars, **nested}, "criticality")


def _resolve_paths(table: dict[str, Any], section: str, root: Path) -> dict[str, Any]:
    resolved = dict(table)
    for key in _PATH_KEYS.get(section, ()):
        value = resolved.get(key)
        if isinstance(value, str) and value.strip():
            resolved[key] = str(root / Path(value).expanduser())
    return resolved


def _config_from_data(data: dict[str, Any], root: Path) -> Config:
    for section in data:
        if section not in _SECTIONS and section != "criticality":
            raise ConfigError(f"unknown table [{section}]")
    built: dict[str, Any] = {}
    for section, cls in _SECTIONS.items():
        if section in data:
            table = data[section]
        elif section in _OPTIONAL_TABLES:
            table = {}
        else:
            raise ConfigError(f"missing table [{section}]")
        if isinstance(table, dict):
            table = _resolve_paths({**_OPTIONAL_DEFAULTS.get(section, {}), **table}, section, root)
        built[section] = _build(cls, table, section)
    if "criticality" not in data:
        raise ConfigError("missing table [criticality]")
    built["criticality"] = _criticality_config(data["criticality"])
    return Config(**built)


# --- range validation -------------------------------------------------------------------------


def _check(ok: bool, key: str, requirement: str, value: Any) -> None:
    if not ok:
        raise ConfigError(f"{key} must be {requirement} (got {value!r})")


def _check_range(
    value: float,
    key: str,
    low: float | None = None,
    high: float | None = None,
    low_inclusive: bool = True,
) -> None:
    above = low is None or (value >= low if low_inclusive else value > low)
    below = high is None or value <= high
    if above and below:
        return
    if low is not None and high is not None:
        opener = "[" if low_inclusive else "("
        requirement = f"in {opener}{low}, {high}]"
    elif low is not None:
        requirement = f">= {low}" if low_inclusive else f"> {low}"
    else:
        requirement = f"<= {high}"
    raise ConfigError(f"{key} must be {requirement} (got {value!r})")


def _check_path(value: str, key: str) -> None:
    _check(bool(value.strip()), key, "a non-empty path", value)


def _validate_band(band: Band, key: str) -> None:
    ordered = band.low <= band.target_low <= band.target_high <= band.high
    _check(
        ordered,
        key,
        "ordered low <= target_low <= target_high <= high",
        (band.low, band.target_low, band.target_high, band.high),
    )


def validate_config(config: Config) -> None:
    """Range and cross-field checks for an already-typed Config. Raises ConfigError naming the
    first offending `section.key`. `load_config` calls this; code that builds a Config with
    `dataclasses.replace` can call it too."""
    grid = config.grid
    _check(grid.shape in SHAPES, "grid.shape", f"one of {', '.join(SHAPES)}", grid.shape)
    _check(
        grid.triangle_mode in TRIANGLE_MODES,
        "grid.triangle_mode",
        f"one of {', '.join(TRIANGLE_MODES)}",
        grid.triangle_mode,
    )
    try:
        validate_dimensions(grid.shape, grid.width, grid.height, grid.wrap, grid.triangle_mode)
    except ValueError as exc:
        raise ConfigError(f"grid.width/grid.height: {exc}") from None

    sim = config.simulation
    _check_range(sim.initial_alive_percentage, "simulation.initial_alive_percentage", 0, 1)
    _check_range(sim.n_lifeforms, "simulation.n_lifeforms", 1, MAX_LIFEFORMS)
    _check_range(sim.static_threshold, "simulation.static_threshold", 0)
    _check_range(sim.history_limit, "simulation.history_limit", 1)
    _check_range(sim.seed, "simulation.seed", 0)

    _check_range(config.metrics.cluster_stride, "metrics.cluster_stride", 1)

    alerts = config.alerts
    _check_range(alerts.sudden_death_window, "alerts.sudden_death_window", 1)
    _check_range(alerts.sudden_death_drop, "alerts.sudden_death_drop", 0, 1, low_inclusive=False)
    _check_range(alerts.stasis_occupancy, "alerts.stasis_occupancy", 0, 1)
    _check_range(alerts.stasis_volatility, "alerts.stasis_volatility", 0)
    if sim.history_limit < alerts.sudden_death_window:
        raise ConfigError(
            f"simulation.history_limit ({sim.history_limit}) must be >= "
            f"alerts.sudden_death_window ({alerts.sudden_death_window}), or sudden-death "
            "alerts can never fire"
        )

    _check_path(config.recording.db_path, "recording.db_path")
    _check_range(config.recording.batch_size, "recording.batch_size", 1)

    disc = config.discovery
    _check_range(disc.mu, "discovery.mu", 1)
    _check_range(disc.n_offspring, "discovery.n_offspring", 1)
    _check_range(disc.evals_per_candidate, "discovery.evals_per_candidate", 1)
    _check_range(disc.steps, "discovery.steps", 2)
    _check_range(disc.freeze_window, "discovery.freeze_window", 1)
    # Freeze detection counts quiet steps only after burn-in; if the run ends first, a rule
    # that freezes at once is never penalised for it.
    if int(disc.steps * BURN_IN_FRAC) + disc.freeze_window > disc.steps:
        min_steps = disc.steps
        while int(min_steps * BURN_IN_FRAC) + disc.freeze_window > min_steps:
            min_steps += 1
        raise ConfigError(
            f"discovery.steps ({disc.steps}) is too short to detect a freeze: it needs "
            f"burn-in ({BURN_IN_FRAC:g} x steps) + discovery.freeze_window "
            f"({disc.freeze_window}) <= steps. Use steps >= {min_steps} or a smaller "
            "freeze_window"
        )
    _check_range(disc.p_birth, "discovery.p_birth", 0, 1)
    _check_range(disc.p_survive, "discovery.p_survive", 0, 1)
    _check_range(disc.mutation_rate, "discovery.mutation_rate", 0, 1)
    _check_range(disc.save_threshold, "discovery.save_threshold", 0, 1, low_inclusive=False)
    _check_path(disc.checkpoint_path, "discovery.checkpoint_path")
    _check_path(disc.discovered_path, "discovery.discovered_path")
    # Discovery always wraps a grid_size x grid_size grid, and `discover --shape` picks the
    # shape only at run time, so grid_size has to suit every shape.
    for shape in SHAPES:
        try:
            validate_dimensions(shape, disc.grid_size, disc.grid_size, True, grid.triangle_mode)
        except ValueError as exc:
            raise ConfigError(
                f"discovery.grid_size ({disc.grid_size}) is invalid for {shape} discovery: {exc}"
            ) from None

    crit = config.criticality
    for key in ("alive_band", "activity_band", "damage_band", "damage_growth_band"):
        _validate_band(getattr(crit, key), f"criticality.{key}")
    _check_range(crit.rho_std_scale, "criticality.rho_std_scale", 0, low_inclusive=False)
    _check_range(crit.act_std_scale, "criticality.act_std_scale", 0, low_inclusive=False)
    for f in fields(crit.weights):
        _check_range(getattr(crit.weights, f.name), f"criticality.weights.{f.name}", 0)
    for f in fields(crit.damage_split):
        _check_range(getattr(crit.damage_split, f.name), f"criticality.damage_split.{f.name}", 0)
    _check_range(crit.freeze_penalty_weight, "criticality.freeze_penalty_weight", 0, 1)

    _check_range(config.ui.fps, "ui.fps", MIN_FPS, MAX_FPS)
    _check_path(config.logging.dir, "logging.dir")


def load_config(path: str | Path) -> Config:
    """Load, type and validate a TOML config file, e.g. `config/default.toml`. Raises
    ConfigError (a ValueError) naming the file and the offending key on any problem."""
    path = Path(path)
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"{path}: config file not found") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from None

    try:
        config = _config_from_data(data, project_root_for(path))
        validate_config(config)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    return config
