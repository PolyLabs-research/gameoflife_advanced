"""Rule presets and example patterns for the settings panel.

A rule preset is a named rule the panel can load into a species' rule fields (staged, like
anything typed there, until Apply). `rule_presets(shape, triangle_mode)` lists the ones that
suit a shape: those written for it, plus any shape-free ones whose counts fit its neighbour
count.

An example is a ready-made demonstration: a shape, a rule and a pattern. Loading one (the
panel's Drawing section) applies the shape and the rule, clears the grid, places the pattern in
the middle and pauses, so pressing Play shows it at once. Spaceships in particular are worth
having this way: a census of the hexagon rules found that random soups almost never make them.
Patterns are pattern keys (`core.translations`): (row, column) cells relative to an origin on an
even column, in the odd-q offset coordinates the hexagon lattice uses. Each one's behaviour
under its rule is checked by `tests/ui/test_presets.py` with `experiments.objects.identify`.
"""

from __future__ import annotations

from dataclasses import dataclass

from latticelife.core.geometries import max_degree, validate_dimensions
from latticelife.core.rules import Rule

Cell = tuple[int, int]


@dataclass(frozen=True)
class RulePreset:
    name: str
    rule: str  # B/S notation
    # The shapes it is meant for; None means any shape whose neighbour count fits its counts.
    shapes: tuple[str, ...] | None = None

    @property
    def label(self) -> str:
        return f"{self.name} · {self.rule}"

    def as_rule(self) -> Rule:
        return Rule.from_bs_string(self.rule)


@dataclass(frozen=True)
class Example:
    name: str
    shape: str
    rule: str
    cells: tuple[Cell, ...]
    # What the example shows, for the panel's hint line.
    note: str
    # The grid it is loaded on when the current one is smaller or does not suit its shape.
    size: tuple[int, int] = (64, 64)

    def as_rule(self) -> Rule:
        return Rule.from_bs_string(self.rule)


RULE_PRESETS: tuple[RulePreset, ...] = (
    RulePreset("Conway's Life", "B3/S23", ("square",)),
    RulePreset("Conway's numbers", "B3/S23", ("hexagon", "triangle")),
    RulePreset("Seeds", "B2/S"),
    # Square: the classic Life-like rules.
    RulePreset("HighLife", "B36/S23", ("square",)),
    RulePreset("Day & Night", "B3678/S34678", ("square",)),
    RulePreset("Life without death", "B3/S012345678", ("square",)),
    RulePreset("Maze", "B3/S12345", ("square",)),
    RulePreset("Replicator", "B1357/S1357", ("square",)),
    RulePreset("2x2", "B36/S125", ("square",)),
    RulePreset("Morley", "B368/S245", ("square",)),
    RulePreset("Diamoeba", "B35678/S5678", ("square",)),
    RulePreset("Anneal", "B4678/S35678", ("square",)),
    # Hexagon: rules from the hexagon census and its follow-up runs.
    RulePreset("Bays' Life, c/5 ship", "B2/S35", ("hexagon",)),
    RulePreset("Bays' glider", "B245/S3", ("hexagon",)),
    RulePreset("c/3 ship", "B24/S245", ("hexagon",)),
    RulePreset("Diagonal ship", "B2/S2", ("hexagon",)),
    RulePreset("Period-48 oscillator", "B2/S34", ("hexagon",)),
    RulePreset("Slow chaos", "B136/S1345", ("hexagon",)),
    RulePreset("Honeycomb with walls", "B134/S12346", ("hexagon",)),
    RulePreset("Order invades chaos", "B2/S012456", ("hexagon",)),
)


EXAMPLES: tuple[Example, ...] = (
    Example(
        "Glider",
        "square",
        "B3/S23",
        ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
        "Conway's glider: 5 cells, one diagonal step every 4 generations.",
    ),
    Example(
        "Lightweight spaceship",
        "square",
        "B3/S23",
        ((0, 1), (0, 4), (1, 0), (2, 0), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3)),
        "Conway's LWSS: moves two cells every 4 generations.",
    ),
    Example(
        "Gosper glider gun",
        "square",
        "B3/S23",
        (
            (0, 24),
            (1, 22),
            (1, 24),
            (2, 12),
            (2, 13),
            (2, 20),
            (2, 21),
            (2, 34),
            (2, 35),
            (3, 11),
            (3, 15),
            (3, 20),
            (3, 21),
            (3, 34),
            (3, 35),
            (4, 0),
            (4, 1),
            (4, 10),
            (4, 16),
            (4, 20),
            (4, 21),
            (5, 0),
            (5, 1),
            (5, 10),
            (5, 14),
            (5, 16),
            (5, 17),
            (5, 22),
            (5, 24),
            (6, 10),
            (6, 16),
            (6, 24),
            (7, 11),
            (7, 15),
            (8, 12),
            (8, 13),
        ),
        "Gosper's gun: a new glider every 30 generations, until they wrap round.",
        size=(80, 80),
    ),
    Example(
        "Pulsar",
        "square",
        "B3/S23",
        tuple(
            (r, c)
            for r in range(13)
            for c in range(13)
            if (r in (0, 5, 7, 12) and c in (2, 3, 4, 8, 9, 10))
            or (c in (0, 5, 7, 12) and r in (2, 3, 4, 8, 9, 10))
        ),
        "A period-3 oscillator of 48 cells.",
    ),
    Example(
        "R-pentomino",
        "square",
        "B3/S23",
        ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),
        "Five cells that churn for over a thousand generations on an open plane.",
    ),
    Example(
        "Bays' c/5 ship",
        "hexagon",
        "B2/S35",
        (
            (-5, -2),
            (-5, 0),
            (-3, -4),
            (-3, -2),
            (-3, 0),
            (-3, 2),
            (-2, -3),
            (-2, 1),
            (0, -2),
            (0, 0),
        ),
        "10-18 cells, one cell every 5 generations; runs under B2/S3 to B2/S356.",
    ),
    Example(
        "Bays' glider",
        "hexagon",
        "B245/S3",
        ((-5, -3), (-4, -1), (-3, -5), (-3, -3), (-2, 0), (-1, -4), (-1, -2), (0, -3), (0, 0)),
        "Period 10, two cells per period; runs under B245/S3 and B245/S36.",
    ),
    Example(
        "c/3 ship",
        "hexagon",
        "B24/S245",
        ((-3, -2), (-3, -1), (-3, 0), (-2, 1), (-1, 0), (0, 0)),
        "6-7 cells, one cell every 3 generations; runs under 16 rules.",
    ),
    Example(
        "Diagonal ship",
        "hexagon",
        "B2/S2",
        ((-4, 0), (-3, -3), (-2, -4), (-2, -3), (-2, -2), (-2, 0), (0, 0)),
        "7-10 cells, one diagonal step every 3 generations; runs under B2/S2 to B256/S256.",
    ),
    Example(
        "Period-48 oscillator",
        "hexagon",
        "B2/S34",
        ((-4, -3), (-4, -2), (-3, -1), (-2, -2), (-2, -1), (0, 0)),
        "6 cells that cycle through 48 phases, growing to 16.",
    ),
)


def preset_fits(preset: RulePreset, shape: str, triangle_mode: str) -> bool:
    """Whether `preset` is offered for `shape`: written for it (or for any shape) and every
    count within its neighbour count."""
    if preset.shapes is not None and shape not in preset.shapes:
        return False
    rule = preset.as_rule()
    return max(rule.birth | rule.survive, default=0) <= max_degree(shape, triangle_mode)


def rule_presets(shape: str, triangle_mode: str = "edge+vertex") -> list[RulePreset]:
    return [p for p in RULE_PRESETS if preset_fits(p, shape, triangle_mode)]


def examples_for(shape: str | None = None) -> list[Example]:
    """Every example, or those of one shape."""
    return [e for e in EXAMPLES if shape is None or e.shape == shape]


def example_size(
    example: Example, width: int, height: int, wrap: bool, triangle_mode: str
) -> tuple[int, int]:
    """The grid an example is loaded on: the current one when it suits the example's shape and
    is at least the example's size, else the example's own size."""
    min_width, min_height = example.size
    if width >= min_width and height >= min_height:
        try:
            validate_dimensions(example.shape, width, height, wrap, triangle_mode)
        except ValueError:
            pass
        else:
            return width, height
    return example.size


def place_cells(cells: tuple[Cell, ...], shape: str, width: int, height: int) -> list[int]:
    """Flat cell indices of a pattern key centred on a `width` x `height` grid. The key's origin
    keeps its column parity (even on hexagons, so the pattern keeps its shape). Cells wrap
    round the edges; examples are only loaded on grids that hold them."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    row = height // 2 - (min(rows) + max(rows)) // 2
    column = width // 2 - (min(cols) + max(cols)) // 2
    if shape == "hexagon" and column % 2:
        column -= 1
    return sorted({((r + row) % height) * width + (c + column) % width for r, c in cells})
