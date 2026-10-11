"""D-3 / §6.4 as an automated gate (F64): no code under src/ may seed or draw from the global
numpy or stdlib RNGs. Every random number comes from an injected `np.random.Generator` (or a
`random.Random` instance), so nothing one component does can shift another's stream — the legacy
discovery thread reseeded the shared global RNGs and corrupted the main simulation."""

import re
from pathlib import Path

import latticelife

SRC = Path(latticelife.__file__).resolve().parent

# Module-level functions of numpy.random and random that use (or seed) hidden global state.
# Constructors of explicit generators (np.random.default_rng, SeedSequence, Generator, PCG64,
# random.Random) are what the policy asks for and stay allowed.
_GLOBAL_NUMPY = re.compile(
    r"\bnp\.random\.(?!default_rng\b|SeedSequence\b|Generator\b|PCG64\b|BitGenerator\b)\w+"
)
_GLOBAL_STDLIB = re.compile(
    r"(?<![\w.])random\.(seed|random|randint|randrange|choice|choices|sample|shuffle|uniform|"
    r"gauss|normalvariate|getrandbits|betavariate|expovariate|triangular)\b"
)
_STAR_OR_FUNCTION_IMPORT = re.compile(
    r"^\s*from\s+(numpy\.random|random)\s+import\s+(?!Random\b|default_rng\b|SeedSequence\b|"
    r"Generator\b)",
    re.MULTILINE,
)


def _violations(text: str) -> list[str]:
    found = []
    for pattern in (_GLOBAL_NUMPY, _GLOBAL_STDLIB, _STAR_OR_FUNCTION_IMPORT):
        found += [match.group(0).strip() for match in pattern.finditer(text)]
    return found


def test_patterns_catch_global_rng_use():
    assert _violations("np.random.seed(0)") == ["np.random.seed"]
    assert _violations("x = np.random.rand(3)") == ["np.random.rand"]
    assert _violations("random.seed(1); random.shuffle(xs)") == ["random.seed", "random.shuffle"]
    assert _violations("from numpy.random import randint") == ["from numpy.random import"]
    assert _violations("rng = np.random.default_rng(np.random.SeedSequence(1))") == []
    assert _violations("rng = random.Random(3); rng.sample(xs, 2); self.random.choice") == []


def test_src_never_uses_global_rng_state():
    offenders = {
        str(path.relative_to(SRC)): found
        for path in sorted(SRC.rglob("*.py"))
        if (found := _violations(path.read_text()))
    }
    assert offenders == {}
