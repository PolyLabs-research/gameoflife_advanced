"""(μ+λ) evolution strategy over `Rule` objects (§6.11). Named honestly: elitist ES, not a GA —
the legacy `rule_discovery.py` called itself a "genetic algorithm" but never did crossover, only
mutation-and-select, which is exactly (μ+λ)-ES. Operates directly on `Rule` via
`Rule.random()`/`Rule.mutate()` with an injected Generator; the legacy gene-string codec
(`rule_to_gene`/`gene_to_rule`) is gone — `Rule` is the only rule representation.

Pure mechanics only: no file I/O, no multiprocessing awareness. `discovery/worker.py` wraps this
with checkpointing, the discovered-rules JSONL sink, and stop-event responsiveness.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from latticelife.config import Config, DiscoveryConfig
from latticelife.core.geometries import max_degree
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.experiments.criticality import score_timeseries
from latticelife.experiments.runner import run_damage_spreading, run_timeseries
from latticelife.session.session import SimulationSession


@dataclass
class Candidate:
    rule: Rule
    total_score: float = 0.0
    eval_count: int = 0

    @property
    def mean_score(self) -> float:
        return self.total_score / self.eval_count if self.eval_count else 0.0

    def add_evaluation(self, score: float) -> None:
        self.total_score += score
        self.eval_count += 1


@dataclass(frozen=True)
class ESState:
    generation: int
    population: list[Candidate]
    shape: str
    entropy: int  # SeedSequence entropy this run's per-generation RNGs are derived from


def initial_population(
    shape: str, triangle_mode: str, discovery_cfg: DiscoveryConfig, rng: np.random.Generator
) -> list[Candidate]:
    n = max_degree(shape, triangle_mode)
    return [
        Candidate(Rule.random(n, discovery_cfg.p_birth, discovery_cfg.p_survive, rng))
        for _ in range(discovery_cfg.mu)
    ]


def mutate_population(
    parents: list[Candidate], shape: str, config: Config, rng: np.random.Generator
) -> list[Candidate]:
    n = max_degree(shape, config.grid.triangle_mode)
    offspring = []
    for _ in range(config.discovery.n_offspring):
        parent = parents[int(rng.integers(0, len(parents)))]
        child_rule = parent.rule.mutate(n, config.discovery.mutation_rate, rng)
        offspring.append(Candidate(child_rule))
    return offspring


def select_elites(pool: list[Candidate], mu: int) -> list[Candidate]:
    return sorted(pool, key=lambda c: c.mean_score, reverse=True)[:mu]


def _session_config_for_discovery(config: Config, shape: str) -> Config:
    """A fresh single-species session config for evaluation: grid_size/p0 from `[discovery]`,
    and cluster metrics disabled (discovery only ever reads rho/activity/occupancy — computing
    cluster labels every `cluster_stride` generations here would be pure waste)."""
    discovery_cfg = config.discovery
    grid_size = discovery_cfg.grid_size
    return replace(
        config,
        grid=replace(config.grid, shape=shape, width=grid_size, height=grid_size, wrap=True),
        simulation=replace(config.simulation, initial_alive_percentage=0.5, n_lifeforms=1),
        metrics=replace(config.metrics, cluster_stride=discovery_cfg.steps + 1),
    )


def evaluate_candidate(rule: Rule, shape: str, config: Config, rng: np.random.Generator) -> float:
    """Legacy pipeline (`rule_discovery.py:run_simulation_for_gene`): timeseries run + damage
    spreading, each on its own fresh SimulationSession, scored by `experiments.criticality`.
    A rule with no birth or no survive condition can never produce or sustain life — scored 0
    without spending a simulation on it, matching the legacy short-circuit."""
    if not rule.birth or not rule.survive:
        return 0.0

    session_config = _session_config_for_discovery(config, shape)
    lifeform = Lifeform(id=1, rule=rule)
    steps = config.discovery.steps

    ts_seed_rng, damage_rng = rng.spawn(2)
    ts_seed = int(ts_seed_rng.integers(0, 2**32))
    ts_session = SimulationSession(session_config, [lifeform], seed=ts_seed)
    series = run_timeseries(ts_session, lifeform_id=1, steps=steps)

    damage_seed_rng, damage_step_rng = damage_rng.spawn(2)
    damage_seed = int(damage_seed_rng.integers(0, 2**32))
    damage_session = SimulationSession(session_config, [lifeform], seed=damage_seed)
    h = run_damage_spreading(damage_session, steps, damage_step_rng, flip_cells=1)

    return score_timeseries(
        series.rho, series.activity, series.frozen_at, steps, config.criticality, h=h
    )


def to_checkpoint_dict(state: ESState) -> dict:
    return {
        "generation": state.generation,
        "shape": state.shape,
        "entropy_seed": state.entropy,
        "population": [
            {"birth": sorted(c.rule.birth), "survive": sorted(c.rule.survive)}
            for c in state.population
        ],
        "scores": [c.mean_score for c in state.population],
        "eval_counts": [c.eval_count for c in state.population],
    }


def from_checkpoint_dict(data: dict) -> ESState:
    population = []
    for rule_data, score, eval_count in zip(
        data["population"], data["scores"], data["eval_counts"], strict=True
    ):
        rule = Rule(birth=set(rule_data["birth"]), survive=set(rule_data["survive"]))
        candidate = Candidate(rule=rule, total_score=score * eval_count, eval_count=eval_count)
        population.append(candidate)
    return ESState(
        generation=data["generation"],
        population=population,
        shape=data["shape"],
        entropy=data["entropy_seed"],
    )
