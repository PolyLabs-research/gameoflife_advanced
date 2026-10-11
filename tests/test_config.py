import copy
import tomllib
from pathlib import Path

import pytest

from latticelife.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    ConfigError,
    load_config,
    project_root_for,
)

REPO_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.toml"


def test_load_default_config():
    config = load_config(REPO_DEFAULT_CONFIG)

    assert config.grid.shape == "square"
    assert config.grid.wrap is True
    assert config.simulation.seed == 0
    assert config.discovery.mu == 5
    assert config.criticality.weights.alive == 0.3
    assert config.criticality.alive_band.target_high == 0.40
    assert config.ui.fps == 30


def _default_data() -> dict:
    with REPO_DEFAULT_CONFIG.open("rb") as f:
        return tomllib.load(f)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(value)


def _dump_toml(data: dict, prefix: str = "") -> str:
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    lines = [f"[{prefix}]"] if prefix and scalars else []
    lines += [f"{key} = {_toml_value(value)}" for key, value in scalars.items()]
    text = "\n".join(lines) + "\n" if lines else ""
    for key, table in tables.items():
        text += "\n" + _dump_toml(table, f"{prefix}.{key}" if prefix else key)
    return text


def _write_config(directory: Path, data: dict, name: str = "config.toml") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(_dump_toml(data))
    return path


def _with(data: dict, dotted_key: str, value) -> dict:
    changed = copy.deepcopy(data)
    *sections, key = dotted_key.split(".")
    table = changed
    for section in sections:
        table = table[section]
    table[key] = value
    return changed


def test_dumped_default_round_trips(tmp_path):
    assert load_config(_write_config(tmp_path, _default_data())).grid.width == 50


# (key, bad value, text the error must contain). F10: none of these used to be rejected.
BAD_VALUES = [
    ("grid.shape", "pentagon", "grid.shape"),
    ("grid.triangle_mode", "vertex", "grid.triangle_mode"),
    ("grid.width", 0, "at least 1"),
    ("grid.width", 50.5, "grid.width must be an integer"),
    ("grid.height", True, "grid.height must be an integer"),
    ("grid.wrap", "false", "grid.wrap must be true or false"),
    ("simulation.initial_alive_percentage", 50, "simulation.initial_alive_percentage"),
    ("simulation.initial_alive_percentage", -0.1, "simulation.initial_alive_percentage"),
    ("simulation.n_lifeforms", 0, "simulation.n_lifeforms"),
    ("simulation.n_lifeforms", 11, "simulation.n_lifeforms"),
    ("simulation.n_lifeforms", 200, "simulation.n_lifeforms"),
    ("simulation.static_threshold", -1, "simulation.static_threshold"),
    ("simulation.history_limit", 0, "simulation.history_limit"),
    ("simulation.history_limit", 5, "alerts.sudden_death_window"),
    ("simulation.seed", -1, "simulation.seed"),
    ("metrics.cluster_stride", 0, "metrics.cluster_stride"),
    ("alerts.sudden_death_window", 0, "alerts.sudden_death_window"),
    ("alerts.sudden_death_drop", 0, "alerts.sudden_death_drop"),
    ("alerts.sudden_death_drop", 1.5, "alerts.sudden_death_drop"),
    ("alerts.stasis_occupancy", 2, "alerts.stasis_occupancy"),
    ("alerts.stasis_volatility", -1.0, "alerts.stasis_volatility"),
    ("recording.db_path", "", "recording.db_path"),
    ("recording.batch_size", 0, "recording.batch_size"),
    ("discovery.mu", 0, "discovery.mu"),
    ("discovery.n_offspring", 0, "discovery.n_offspring"),
    ("discovery.evals_per_candidate", 0, "discovery.evals_per_candidate"),
    ("discovery.steps", 0, "discovery.steps"),
    ("discovery.grid_size", 63, "discovery.grid_size"),
    ("discovery.grid_size", 4, "discovery.grid_size"),
    ("discovery.p_birth", 1.5, "discovery.p_birth"),
    ("discovery.p_survive", -0.5, "discovery.p_survive"),
    ("discovery.mutation_rate", 2, "discovery.mutation_rate"),
    ("discovery.save_threshold", 0, "discovery.save_threshold"),
    ("discovery.checkpoint_path", " ", "discovery.checkpoint_path"),
    ("criticality.rho_std_scale", 0, "criticality.rho_std_scale"),
    ("criticality.freeze_penalty_weight", 1.5, "criticality.freeze_penalty_weight"),
    ("criticality.alive_band.target_low", 0.9, "criticality.alive_band"),
    ("criticality.weights.alive", -0.3, "criticality.weights.alive"),
    ("criticality.damage_split.band", "half", "criticality.damage_split.band"),
    ("ui.fps", 0, "ui.fps"),
    ("ui.fps", -5, "ui.fps"),
    ("ui.fps", 120, "ui.fps"),
    ("ui.fps", 1000, "ui.fps"),
    ("logging.dir", "", "logging.dir"),
]


@pytest.mark.parametrize(("key", "value", "expected"), BAD_VALUES)
def test_invalid_values_are_rejected_naming_the_key(tmp_path, key, value, expected):
    path = _write_config(tmp_path, _with(_default_data(), key, value))
    with pytest.raises(ConfigError, match=expected.replace(".", r"\.")) as excinfo:
        load_config(path)
    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize(
    ("shape", "width", "height", "message"),
    [
        ("hexagon", 51, 50, "even width"),
        ("triangle", 50, 51, "even width and height"),
        ("triangle", 51, 50, "even width and height"),
    ],
)
def test_wrapped_sizes_that_do_not_tile_are_rejected(tmp_path, shape, width, height, message):
    """F1: odd hexagon/triangle sizes are refused with a readable error, never snapped."""
    data = _with(_with(_with(_default_data(), "grid.shape", shape), "grid.width", width),
                 "grid.height", height)
    with pytest.raises(ConfigError, match=message):
        load_config(_write_config(tmp_path, data))


def test_odd_sizes_are_fine_without_wrap(tmp_path):
    data = _with(_with(_default_data(), "grid.shape", "hexagon"), "grid.width", 51)
    data = _with(data, "grid.wrap", False)
    assert load_config(_write_config(tmp_path, data)).grid.width == 51


def test_unknown_key_is_rejected(tmp_path):
    data = _with(_default_data(), "grid.widht", 40)
    with pytest.raises(ConfigError, match=r"unknown key grid\.widht"):
        load_config(_write_config(tmp_path, data))


def test_missing_key_is_rejected(tmp_path):
    data = _default_data()
    del data["alerts"]["sudden_death_drop"]
    with pytest.raises(ConfigError, match=r"missing key alerts\.sudden_death_drop"):
        load_config(_write_config(tmp_path, data))


def test_unknown_and_missing_tables_are_rejected(tmp_path):
    data = _default_data()
    data["simulaton"] = data.pop("simulation")
    with pytest.raises(ConfigError, match=r"unknown table \[simulaton\]"):
        load_config(_write_config(tmp_path, data))

    data = _default_data()
    del data["criticality"]["weights"]
    with pytest.raises(ConfigError, match=r"missing table \[criticality\.weights\]"):
        load_config(_write_config(tmp_path, data))


def test_missing_file_and_bad_toml_raise_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("[grid\nwidth = ")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(broken)
    assert issubclass(ConfigError, ValueError)


# --- paths (F16) -------------------------------------------------------------------------------


def test_default_config_path_does_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert DEFAULT_CONFIG_PATH.is_absolute()
    assert DEFAULT_CONFIG_PATH == REPO_DEFAULT_CONFIG
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.recording.db_path == str(PROJECT_ROOT / "out" / "latticelife.db")
    assert config.discovery.checkpoint_path == str(PROJECT_ROOT / "out" / "discovery")
    assert config.discovery.discovered_path == str(PROJECT_ROOT / "out" / "discovered_rules.jsonl")
    assert config.logging.dir == str(PROJECT_ROOT / "out" / "logs")


def test_relative_output_paths_resolve_against_the_config_project_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    path = _write_config(project / "config", _default_data(), "mine.toml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    config = load_config(path)

    assert project_root_for(path) == project.resolve()
    assert config.recording.db_path == str(project.resolve() / "out" / "latticelife.db")
    assert config.discovery.checkpoint_path == str(project.resolve() / "out" / "discovery")
    assert config.logging.dir == str(project.resolve() / "out" / "logs")


def test_without_a_project_paths_resolve_next_to_the_config(tmp_path):
    data = _with(_default_data(), "recording.db_path", "runs/a.db")
    path = _write_config(tmp_path / "experiment", data)
    config = load_config(path)
    assert config.recording.db_path == str((tmp_path / "experiment").resolve() / "runs" / "a.db")


def test_absolute_output_paths_are_kept(tmp_path):
    target = tmp_path / "abs" / "x.db"
    path = _write_config(tmp_path, _with(_default_data(), "recording.db_path", str(target)))
    assert load_config(path).recording.db_path == str(target)


def test_logging_table_and_discovered_path_are_optional(tmp_path):
    data = _default_data()
    del data["logging"]
    del data["discovery"]["discovered_path"]
    config = load_config(_write_config(tmp_path, data))
    root = tmp_path.resolve()
    assert config.logging.dir == str(root / "out" / "logs")
    assert config.discovery.discovered_path == str(root / "out" / "discovered_rules.jsonl")


def test_tmp_output_config_keeps_every_output_out_of_the_project(tmp_output_config, tmp_path):
    config = tmp_output_config
    outputs = [
        config.recording.db_path,
        config.discovery.checkpoint_path,
        config.discovery.discovered_path,
        config.logging.dir,
    ]
    for output in outputs:
        assert Path(output).is_relative_to(tmp_path)
