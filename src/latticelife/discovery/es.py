"""(μ+λ) evolution strategy over `Rule` objects (§6.11). Named honestly: elitist ES, not a GA —
the legacy `rule_discovery.py` called itself a "genetic algorithm" but never did crossover, only
mutation-and-select, which is exactly (μ+λ)-ES. Operates directly on `Rule` via
`Rule.random()`/`Rule.mutate()` with an injected Generator; the legacy gene-string codec
(`rule_to_gene`/`gene_to_rule`) is gone — `Rule` is the only rule representation.

One candidate per rule. Offspring always differ from their parent and from every rule already
in the pool, and `merge_duplicates` folds any remaining copies of a rule into one candidate, so a
rule's evaluations accumulate in a single running mean (§6.11) instead of being split across
copies that crowd the elite set.

Search settings. A lineage (a population and its accumulated scores) is only meaningful for the
fitness landscape it was evolved in: the shape, the triangle neighbourhood, the evaluation grid
and step count, the freeze window and every criticality weight. `search_settings` collects those;
the checkpoint stores them and is named after their fingerprint (`checkpoint_name`), so a run
with different settings resumes its own lineage, never another landscape's.

Pure mechanics only: no file I/O, no multiprocessing awareness. `discovery/worker.py` wraps this
with checkpointing, the discovered-rules JSONL sink, and stop-event responsiveness.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from latticelife.config import Config, DiscoveryConfig
from latticelife.core.geometries import max_degree
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.experiments.criticality import score_timeseries
from latticelife.experiments.runner import run_damage_spreading, run_timeseries
from latticelife.session.session import SimulationSession

CHECKPOINT_VERSION = 2

# Rule.mutate draws before a child falls back to a single-bit flip (see mutate_population).
_MUTATION_DRAWS = 16

RuleKey = tuple[frozenset[int], frozenset[int]]


def rule_key(rule: Rule) -> RuleKey:
    """Hashable identity of a rule (Rule holds plain sets, so it is not hashable itself)."""
    return frozenset(rule.birth), frozenset(rule.survive)


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
    settings: dict[str, Any]  # search_settings() of the run that evolved this population


def search_settings(config: Config, shape: str) -> dict[str, Any]:
    """Everything that defines the fitness landscape a discovery lineage lives in, as plain
    JSON data. ES hyperparameters (mu, lambda, mutation rate, evals per candidate, initial
    probabilities) are left out: changing them changes how a lineage searches, not what its
    accumulated scores mean."""
    triangle_mode = config.grid.triangle_mode if shape == "triangle" else None
    max_degree(shape, config.grid.triangle_mode)  # raises ValueError for an unknown shape
    settings = {
        "shape": shape,
        "triangle_mode": triangle_mode,
        "grid_size": config.discovery.grid_size,
        "steps": config.discovery.steps,
        "freeze_window": config.discovery.freeze_window,
        "criticality": dataclasses.asdict(config.criticality),
    }
    return json.loads(json.dumps(settings))  # normalised exactly as a checkpoint reads it back


def settings_id(settings: dict[str, Any]) -> str:
    """Short, stable fingerprint of `search_settings` output."""
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def checkpoint_name(settings: dict[str, Any]) -> str:
    """`checkpoint-square-<id>.json`, `checkpoint-triangle-edge+vertex-<id>.json`, ..."""
    parts = [settings["shape"]]
    if settings["triangle_mode"] is not None:
        parts.append(settings["triangle_mode"])
    return f"checkpoint-{'-'.join(parts)}-{settings_id(settings)}.json"


def settings_max_degree(settings: dict[str, Any]) -> int:
    return max_degree(settings["shape"], settings["triangle_mode"] or "edge+vertex")


def initial_population(
    shape: str, triangle_mode: str, discovery_cfg: DiscoveryConfig, rng: np.random.Generator
) -> list[Candidate]:
    """`mu` distinct random rules (a repeated draw is redrawn, up to `_MUTATION_DRAWS` times
    per member)."""
    n = max_degree(shape, triangle_mode)
    population: list[Candidate] = []
    known: set[RuleKey] = set()
    for _ in range(discovery_cfg.mu):
        for _ in range(_MUTATION_DRAWS):
            rule = Rule.random(n, discovery_cfg.p_birth, discovery_cfg.p_survive, rng)
            if rule_key(rule) not in known:
                break
        known.add(rule_key(rule))
        population.append(Candidate(rule))
    return merge_duplicates(population)


def _single_flips(rule: Rule, n: int, rng: np.random.Generator) -> list[Rule]:
    """Every rule exactly one of the 2(n+1) birth/survive bits away (the legacy mutation), in
    random order."""
    flips = []
    for bit in rng.permutation(2 * (n + 1)):
        birth, survive = set(rule.birth), set(rule.survive)
        target = birth if bit <= n else survive
        target.symmetric_difference_update({int(bit) % (n + 1)})
        flips.append(Rule(birth=birth, survive=survive))
    return flips


def _fallback_child(
    parent: Rule, others: list[Rule], n: int, known: set[RuleKey], rng: np.random.Generator
) -> Rule:
    """A single-bit flip of `parent` that is not a known rule; failing that, of another parent;
    failing that (every such rule is already in the pool), a flip of `parent` anyway."""
    first = _single_flips(parent, n, rng)
    for candidate_parent in (None, *others):
        flips = first if candidate_parent is None else _single_flips(candidate_parent, n, rng)
        for child in flips:
            if rule_key(child) not in known:
                return child
    return first[0]


def mutate_population(
    parents: list[Candidate], shape: str, config: Config, rng: np.random.Generator
) -> list[Candidate]:
    """λ offspring, each a mutation of a uniformly drawn parent. `Rule.mutate` flips every bit
    independently, so it returns the parent unchanged with probability (1-rate)^(2(n+1)) — 22%
    on square at the default rate — and two mutations can coincide. A draw that repeats a rule
    already in the pool (the parents or an earlier child) is redrawn; after `_MUTATION_DRAWS`
    such draws (e.g. at a very low rate) the child is a single-bit flip instead, which always
    differs from its parent and, unless every single flip of every parent is already in the
    pool, from every other rule too (`merge_duplicates` pools the evidence if not)."""
    n = max_degree(shape, config.grid.triangle_mode)
    rate = config.discovery.mutation_rate
    known = {rule_key(c.rule) for c in parents}
    offspring = []
    for _ in range(config.discovery.n_offspring):
        parent = parents[int(rng.integers(0, len(parents)))]
        for _ in range(_MUTATION_DRAWS):
            child_rule = parent.rule.mutate(n, rate, rng)
            if rule_key(child_rule) not in known:
                break
        else:
            others = [c.rule for c in parents if c is not parent]
            child_rule = _fallback_child(parent.rule, others, n, known, rng)
        known.add(rule_key(child_rule))
        offspring.append(Candidate(child_rule))
    return offspring


def merge_duplicates(pool: list[Candidate]) -> list[Candidate]:
    """One candidate per rule, in first-seen order: later copies' scores and evaluation counts
    are added to the first copy, so every evaluation of a rule counts toward one running mean."""
    merged: dict[RuleKey, Candidate] = {}
    for candidate in pool:
        key = rule_key(candidate.rule)
        first = merged.get(key)
        if first is None:
            merged[key] = candidate
        else:
            first.total_score += candidate.total_score
            first.eval_count += candidate.eval_count
    return list(merged.values())


def select_elites(pool: list[Candidate], mu: int) -> list[Candidate]:
    """The `mu` best distinct rules by mean score (stable for ties)."""
    return sorted(merge_duplicates(pool), key=lambda c: c.mean_score, reverse=True)[:mu]


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
    without spending a simulation on it, matching the legacy short-circuit. Freeze detection
    uses `discovery.freeze_window` (legacy: 100), which config validation keeps short enough
    for a freeze to be detectable within `discovery.steps`."""
    if not rule.birth or not rule.survive:
        return 0.0

    session_config = _session_config_for_discovery(config, shape)
    lifeform = Lifeform(id=1, rule=rule)
    steps = config.discovery.steps

    ts_seed_rng, damage_rng = rng.spawn(2)
    ts_seed = int(ts_seed_rng.integers(0, 2**32))
    ts_session = SimulationSession(session_config, [lifeform], seed=ts_seed)
    series = run_timeseries(
        ts_session, lifeform_id=1, steps=steps, freeze_window=config.discovery.freeze_window
    )

    damage_seed_rng, damage_step_rng = damage_rng.spawn(2)
    damage_seed = int(damage_seed_rng.integers(0, 2**32))
    damage_session = SimulationSession(session_config, [lifeform], seed=damage_seed)
    h = run_damage_spreading(damage_session, steps, damage_step_rng, flip_cells=1)

    return score_timeseries(
        series.rho, series.activity, series.frozen_at, steps, config.criticality, h=h
    )


def to_checkpoint_dict(state: ESState) -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "generation": state.generation,
        "shape": state.shape,
        "entropy_seed": state.entropy,
        "settings": state.settings,
        "population": [
            {"birth": sorted(c.rule.birth), "survive": sorted(c.rule.survive)}
            for c in state.population
        ],
        "scores": [c.mean_score for c in state.population],
        "eval_counts": [c.eval_count for c in state.population],
    }


def _counts(values: Any, max_count: int, what: str) -> set[int]:
    if not isinstance(values, list) or not all(type(v) is int for v in values):
        raise ValueError(f"{what} must be a list of integers (got {values!r})")
    if any(v < 0 or v > max_count for v in values):
        raise ValueError(f"{what} {values} has counts outside 0..{max_count}")
    return set(values)


def from_checkpoint_dict(data: dict) -> ESState:
    """Inverse of `to_checkpoint_dict`. Raises ValueError (or KeyError/TypeError for a
    malformed file) for anything that is not a complete, current-version checkpoint, including
    rules with neighbour counts the stored shape cannot have. Copies of one rule are merged."""
    if not isinstance(data, dict) or data.get("version") != CHECKPOINT_VERSION:
        version = data.get("version") if isinstance(data, dict) else None
        raise ValueError(f"unsupported checkpoint version {version!r}")
    settings = data["settings"]
    if not isinstance(settings, dict) or settings.get("shape") != data["shape"]:
        raise ValueError("checkpoint settings do not match its shape")
    max_count = settings_max_degree(settings)
    generation, entropy = data["generation"], data["entropy_seed"]
    if type(generation) is not int or generation < 0 or type(entropy) is not int:
        raise ValueError("checkpoint generation/entropy_seed must be non-negative integers")

    population = []
    for rule_data, score, eval_count in zip(
        data["population"], data["scores"], data["eval_counts"], strict=True
    ):
        rule = Rule(
            birth=_counts(rule_data["birth"], max_count, "birth"),
            survive=_counts(rule_data["survive"], max_count, "survive"),
        )
        if type(eval_count) is not int or eval_count < 0:
            raise ValueError(f"eval_count must be a non-negative integer (got {eval_count!r})")
        if not np.isfinite(score):
            raise ValueError(f"score must be finite (got {score!r})")
        population.append(
            Candidate(rule=rule, total_score=float(score) * eval_count, eval_count=eval_count)
        )
    if not population:
        raise ValueError("checkpoint population is empty")
    return ESState(
        generation=generation,
        population=merge_duplicates(population),
        shape=data["shape"],
        entropy=entropy,
        settings=settings,
    )
