"""The settings panel, rebuilt against `Config`/`Lifeform` rather than the legacy Game/grid
objects. Widget wiring only — the widgets themselves live in `ui/widgets.py`.

Sections, top to bottom:

- Grid (staged): shape, size, wrap, the triangle neighbourhood (triangles only), the starting
  density and the seed (0 = a fresh random grid each time).
- Speed (live): generations per second, and whether a sudden-death alert pauses the run.
- Species (staged): how many, a preset-rule menu for the chosen species, and each species'
  birth and survival counts with an R (random rule) button.
- View (live): 2D/3D (as the V key), torus auto-rotation, a camera reset (3D), and the
  keyboard-shortcut help (as '?').
- Drawing and examples (live): load an example (a shape, a rule and a pattern, placed on a clear
  grid and paused, ready for Play), clear the grid (as the C key), and how painting works.

Staged settings take effect only when Apply is clicked, through the caller's `on_apply` (the
app owns rebuilding the SimulationSession from a new Config). Apply reads "Apply changes" and is
highlighted while a staged setting differs from the one running. Live controls act at once
through their own callbacks; the app tells the panel what they show (`view_mode`,
`auto_rotate`, `alert_pause`) before each draw.

Apply validates before it calls back: grid sizes that `core.geometries.validate_dimensions`
rejects (e.g. an odd width for a wrapped hexagon grid) and rule fields that do not parse, or name
a count above the selected shape's neighbour count, set `error` — drawn in red above the Apply
button — outline the field(s) at fault in red (opening their section and scrolling them into
view), and nothing is applied. Rule fields use the comma-separated display form of
`core.rules` (`'2,3'`), so the 10-12 neighbour counts of edge+vertex triangles round-trip.

Layout and events. Given a height, the panel is a viewport: the sections scroll (mouse wheel)
and are clipped to it, while the error text and Apply stay pinned below them, so Apply is always
reachable however many species there are. A click goes to exactly one widget — the visible one
under the pointer — after any focused field the click is not on has committed and blurred (a
species-count change made that way adds or removes rows before the click is routed, so it never
reaches a removed row). An open drop-down list takes the click for itself. Widgets in a
collapsed section have no on-screen rect and get no events. Only what lies inside the viewport
can be clicked — a click on a field's scrolled-away part is a click elsewhere — and a focused
field with no part left in view (scrolled away, or squeezed out by a smaller window) commits and
blurs, so keys go back to the app's shortcuts. The error text is wrapped (breaking an over-long
word) to the panel's width and at most `MAX_ERROR_LINES` lines, so it never draws past the panel
or pushes Apply off the window.

Each species row starts with the species' colour; clicking it (or pressing its number key, see
`ui/app.py`) picks the species a grid click places — `selected_species`, highlighted here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pygame

from latticelife.config import MAX_FPS, MAX_LIFEFORMS, MIN_FPS, Config
from latticelife.core.geometries import SHAPES, TRIANGLE_MODES, max_degree, validate_dimensions
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule, format_count_list, parse_count_list
from latticelife.ui.presets import EXAMPLES, Example, RulePreset, rule_presets
from latticelife.ui.widgets import (
    ACCENT_PRIMARY,
    ERROR_COLOR,
    PLACEHOLDER_COLOR,
    SCROLLBAR_TRACK,
    Button,
    CollapsibleSection,
    Dropdown,
    NumericInput,
    Slider,
    TextInput,
    blit_clipped,
)

SECTION_GAP = 12
CONTROL_GAP = 8
ROW_HEIGHT = 32
CAPTION_HEIGHT = 18
LABEL_WIDTH = 90
LABEL_COLOR = (235, 235, 245)
SWATCH_SIZE = 14
SCROLLBAR_WIDTH = 4
WHEEL_STEP = ROW_HEIGHT
# Most lines of error text drawn above Apply (more is cut short with an ellipsis), so a long
# message cannot push Apply off the window. Every message Apply makes fits at the narrowest panel.
MAX_ERROR_LINES = 6
# Characters a rule field accepts from the keyboard: counts and the separators people type.
RULE_CHARS = "0123456789, "
# Largest width/height the fields take (a bigger loaded config raises it to its own size).
# There is no fixed smallest beyond 1: Apply reports each shape's real minimum.
MAX_GRID_SIZE = 500
# Largest seed the Seed field takes (a bigger loaded config raises it to its own seed).
MAX_SEED = 2**31 - 1
# Mixed into the simulation seed for the UI's own random stream (see `ui_rng`).
UI_RNG_KEY = 0x5549
# The most neighbours any shape has: rule fields are compared at this bound for "pending".
_MAX_COUNT = max(max_degree(shape, mode) for shape in SHAPES for mode in TRIANGLE_MODES)


def ui_rng(seed: int) -> np.random.Generator:
    """The generator the interactive UI draws random rules from (R buttons, New). A nonzero
    `simulation.seed` makes the sequence reproducible (§6.4); mixing in `UI_RNG_KEY` keeps it
    independent of the session's own stream. Seed 0 means fresh entropy, as everywhere else."""
    return np.random.default_rng([seed, UI_RNG_KEY]) if seed else np.random.default_rng()


def _fitting_length(word: str, font: pygame.font.Font, width: int) -> int:
    """The longest prefix of `word` (at least one character) no wider than `width` pixels."""
    low, high = 1, len(word)
    while low < high:
        mid = (low + high + 1) // 2
        if font.size(word[:mid])[0] <= width:
            low = mid
        else:
            high = mid - 1
    return low


def _wrap_text(
    text: str, font: pygame.font.Font, width: int, max_lines: int | None = None
) -> list[str]:
    """Greedy word wrap of `text` to lines no wider than `width` pixels. A word too wide for a
    line of its own (a long run typed into a field and echoed by an error) starts a new line
    and is broken between characters. With `max_lines`, text past that many lines is dropped
    and the last line kept ends in an ellipsis."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        if max_lines is not None and len(lines) > max_lines:
            break
        candidate = f"{line} {word}" if line else word
        if font.size(candidate)[0] <= width:
            line = candidate
            continue
        if line:
            lines.append(line)
        while font.size(word)[0] > width and (max_lines is None or len(lines) <= max_lines):
            cut = _fitting_length(word, font, width)
            lines.append(word[:cut])
            word = word[cut:]
        line = word
    if line:
        lines.append(line)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and font.size(last + "…")[0] > width:
            last = last[:-1]
        lines[-1] = last + "…"
    return lines


def _example_label(example: Example) -> str:
    return f"{example.name} · {example.shape}"


def _typed_int(widget: NumericInput) -> int:
    """What a number field will hold once committed: its typed text, clamped, or its value."""
    try:
        return widget.clamp(int(widget.text))
    except ValueError:
        return widget.value


Row = tuple[int, Callable[[int], object]]


@dataclass
class SettingsPanel:
    config: Config
    lifeforms: list[Lifeform]
    font: pygame.font.Font
    on_apply: Callable[[Config, list[Lifeform]], None]
    # Called with the new FPS while the slider moves (live; no Apply needed).
    on_fps_change: Callable[[int], None] | None = None
    # Called with a species id when its row's colour/label is clicked.
    on_select_species: Callable[[int], None] | None = None
    # Random rules for the R buttons; default `ui_rng(config.simulation.seed)`.
    rng: np.random.Generator | None = None
    # Live controls: each is called when its button is clicked.
    on_alert_pause_toggle: Callable[[], None] | None = None
    on_view_toggle: Callable[[], None] | None = None
    on_auto_rotate_toggle: Callable[[], None] | None = None
    on_reset_camera: Callable[[], None] | None = None
    on_help: Callable[[], None] | None = None
    on_clear: Callable[[], None] | None = None
    # Called with the example chosen from the examples menu.
    on_example: Callable[[Example], None] | None = None

    grid_section: CollapsibleSection = field(init=False)
    speed_section: CollapsibleSection = field(init=False)
    species_section: CollapsibleSection = field(init=False)
    view_section: CollapsibleSection = field(init=False)
    drawing_section: CollapsibleSection = field(init=False)
    shape_dropdown: Dropdown = field(init=False)
    triangle_dropdown: Dropdown = field(init=False)
    width_input: NumericInput = field(init=False)
    height_input: NumericInput = field(init=False)
    wrap_button: Button = field(init=False)
    density_slider: Slider = field(init=False)
    seed_input: NumericInput = field(init=False)
    fps_slider: Slider = field(init=False)
    alert_button: Button = field(init=False)
    species_count_input: NumericInput = field(init=False)
    preset_dropdown: Dropdown = field(init=False)
    rule_inputs: list[tuple[TextInput, TextInput]] = field(init=False)
    randomize_buttons: list[Button] = field(init=False)
    view_button: Button = field(init=False)
    rotate_button: Button = field(init=False)
    camera_button: Button = field(init=False)
    help_button: Button = field(init=False)
    example_dropdown: Dropdown = field(init=False)
    clear_button: Button = field(init=False)
    apply_button: Button = field(init=False)
    error: str | None = field(default=None, init=False)
    # Colour of each applied species, by row (set by the app); rows beyond it are staged only.
    species_colors: list[tuple[int, int, int]] = field(default_factory=list, init=False)
    # The species id a grid click places (set by the app), highlighted in its row.
    selected_species: int = field(default=1, init=False)
    # What the live controls show (set by the app before each draw).
    view_mode: str = field(default="2D", init=False)
    auto_rotate: bool = field(default=True, init=False)
    alert_pause: bool = field(default=True, init=False)
    scroll_y: int = field(default=0, init=False)
    max_scroll: int = field(default=0, init=False)
    _content_clip: pygame.Rect | None = field(default=None, init=False)
    _scroll_area: pygame.Rect | None = field(default=None, init=False)
    _species_label_rects: list[pygame.Rect] = field(default_factory=list, init=False)
    _pointer: tuple[int, int] | None = field(default=None, init=False)
    # A field to scroll into view at the next draw (one a failed Apply flagged).
    _reveal: object | None = field(default=None, init=False)
    # The settings and rules running now, which the staged ones are compared with.
    _applied: Config = field(init=False)
    _applied_rules: list[tuple[frozenset[int], frozenset[int]]] = field(init=False)
    _presets: dict[str, RulePreset] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = ui_rng(self.config.simulation.seed)
        font = self.font
        none = pygame.Rect(0, 0, 0, 0)
        self.grid_section = CollapsibleSection("Grid", font)
        self.speed_section = CollapsibleSection("Speed", font)
        self.species_section = CollapsibleSection("Species", font)
        self.view_section = CollapsibleSection("View", font)
        self.drawing_section = CollapsibleSection("Drawing and examples", font)

        grid, simulation = self.config.grid, self.config.simulation
        self.shape_dropdown = Dropdown(list(SHAPES), font, selected=grid.shape)
        self.triangle_dropdown = Dropdown(list(TRIANGLE_MODES), font, selected=grid.triangle_mode)
        width, height = grid.width, grid.height
        self.width_input = NumericInput(width, 1, max(MAX_GRID_SIZE, width), font)
        self.height_input = NumericInput(height, 1, max(MAX_GRID_SIZE, height), font)
        self.wrap_button = Button(none.copy(), self._wrap_label(), font, on_click=self._toggle_wrap)
        self.density_slider = Slider(
            "Density",
            0,
            100,
            round(simulation.initial_alive_percentage * 100),
            font,
            format_fn=lambda v: f"{round(v)}%",
            step=1,
            show_value=True,
        )
        seed = simulation.seed
        self.seed_input = NumericInput(seed, 0, max(MAX_SEED, seed), font, zero_text="random")
        self.fps_slider = Slider(
            "FPS",
            MIN_FPS,
            MAX_FPS,
            self.config.ui.fps,
            font,
            format_fn=lambda v: f"{round(v)} fps",
            live_callback=self._fps_moved,
            step=1,
            show_value=True,
        )
        self.alert_button = Button(
            none.copy(), "", font, on_click=lambda: self._call(self.on_alert_pause_toggle)
        )
        self.species_count_input = NumericInput(
            len(self.lifeforms), 1, MAX_LIFEFORMS, font, on_commit=self._resize_species
        )
        self.preset_dropdown = Dropdown([], font, on_commit=self._load_preset)
        self.rule_inputs = []
        self.randomize_buttons = []
        # One row per species, up to the count the input shows (it clamps to MAX_LIFEFORMS),
        # so what is listed is exactly what Apply will apply.
        self._resize_species(self.species_count_input.value)
        self._show_rules(self.lifeforms)

        self.view_button = Button(
            none.copy(), "", font, on_click=lambda: self._call(self.on_view_toggle)
        )
        self.rotate_button = Button(
            none.copy(), "", font, on_click=lambda: self._call(self.on_auto_rotate_toggle)
        )
        self.camera_button = Button(
            none.copy(), "Reset camera", font, on_click=lambda: self._call(self.on_reset_camera)
        )
        self.help_button = Button(
            none.copy(), "Shortcuts (?)", font, on_click=lambda: self._call(self.on_help)
        )
        self.example_dropdown = Dropdown(
            [_example_label(e) for e in EXAMPLES],
            font,
            on_commit=self._choose_example,
            placeholder="Load an example...",
        )
        self.clear_button = Button(
            none.copy(), "Clear grid (C)", font, on_click=lambda: self._call(self.on_clear)
        )
        self.apply_button = Button(none.copy(), "Apply", font, primary=True, on_click=self._apply)
        self._remember_applied(self.config, self.lifeforms)
        self._refresh()

    # --- actions -------------------------------------------------------------

    @staticmethod
    def _call(callback: Callable[[], None] | None) -> None:
        if callback is not None:
            callback()

    def _wrap_label(self) -> str:
        return "Wrap: On" if self.config.grid.wrap else "Wrap: Off"

    def _toggle_wrap(self) -> None:
        new_grid = replace(self.config.grid, wrap=not self.config.grid.wrap)
        self.config = replace(self.config, grid=new_grid)
        self.wrap_button.text = self._wrap_label()

    def _fps_moved(self, value: float) -> None:
        fps = int(round(value))
        if fps != self.config.ui.fps:
            self.config = replace(self.config, ui=replace(self.config.ui, fps=fps))
            if self.on_fps_change is not None:
                self.on_fps_change(fps)

    def _staged_shape(self) -> str:
        return self.shape_dropdown.selected or self.config.grid.shape

    def _staged_mode(self) -> str:
        return self.triangle_dropdown.selected or self.config.grid.triangle_mode

    def _make_randomizer(self, index: int) -> Callable[[], None]:
        return lambda: self._randomize_rule(index)

    def _randomize_rule(self, index: int) -> None:
        """Populates B/S with a random rule sized to the selected shape's neighbor count,
        reusing the same `Rule.random` the ES discovery process samples candidates from."""
        if not 0 <= index < len(self.rule_inputs):
            return
        n = max_degree(self._staged_shape(), self._staged_mode())
        discovery = self.config.discovery
        assert self.rng is not None
        rule = Rule.random(n, discovery.p_birth, discovery.p_survive, self.rng)
        self._set_rule_row(index, rule)

    def _set_rule_row(self, index: int, rule: Rule) -> None:
        birth_input, survive_input = self.rule_inputs[index]
        birth_input.text = format_count_list(rule.birth)
        survive_input.text = format_count_list(rule.survive)

    def _preset_row(self) -> int:
        """The species row a preset goes into: the chosen species', or the first."""
        index = self.selected_species - 1
        return index if 0 <= index < len(self.rule_inputs) else 0

    def _load_preset(self, label: str) -> None:
        """A preset rule into the chosen species' fields, staged until Apply."""
        self.preset_dropdown.selected = None  # a menu, not a setting
        preset = self._presets.get(label)
        if preset is not None and self.rule_inputs:
            self._set_rule_row(self._preset_row(), preset.as_rule())

    def _choose_example(self, label: str) -> None:
        self.example_dropdown.selected = None  # a menu, not a setting
        example = next((e for e in EXAMPLES if _example_label(e) == label), None)
        if example is not None and self.on_example is not None:
            self.on_example(example)

    def _rule_input(self, text: str, placeholder: str) -> TextInput:
        return TextInput(text, self.font, placeholder=placeholder, allowed_chars=RULE_CHARS)

    def _resize_species(self, count: int) -> None:
        current = len(self.rule_inputs)
        if count > current:
            for _ in range(count - current):
                self.rule_inputs.append(
                    (self._rule_input("3", "birth"), self._rule_input("2,3", "survive"))
                )
                self.randomize_buttons.append(
                    Button(
                        pygame.Rect(0, 0, 0, 0),
                        "R",
                        self.font,
                        on_click=self._make_randomizer(len(self.rule_inputs) - 1),
                    )
                )
        elif count < current:
            del self.rule_inputs[count:]
            del self.randomize_buttons[count:]
            del self._species_label_rects[count:]

    def _show_rules(self, lifeforms: list[Lifeform]) -> None:
        for (birth_input, survive_input), lf in zip(self.rule_inputs, lifeforms, strict=False):
            birth_input.text = format_count_list(lf.rule.birth)
            survive_input.text = format_count_list(lf.rule.survive)

    def _remember_applied(self, config: Config, lifeforms: list[Lifeform]) -> None:
        self._applied = config
        self._applied_rules = [
            (frozenset(lf.rule.birth), frozenset(lf.rule.survive)) for lf in lifeforms
        ]

    def set_lifeforms(self, lifeforms: list[Lifeform]) -> None:
        """Show `lifeforms`' rules in the species rows (one row each, adding or removing rows to
        match) and clear any error — for when the app replaces the species itself, e.g. New."""
        self.lifeforms = lifeforms
        self.species_count_input.set_value(len(lifeforms))
        self._resize_species(len(lifeforms))
        self._show_rules(lifeforms)
        self._applied_rules = [
            (frozenset(lf.rule.birth), frozenset(lf.rule.survive)) for lf in lifeforms
        ]
        self._fail(None)

    def show_settings(self, config: Config, lifeforms: list[Lifeform]) -> None:
        """Show a config and species the app applied itself (an example) in every staged field,
        so nothing reads as pending. The live speed stays as the slider has it."""
        grid, simulation = config.grid, config.simulation
        self.config = replace(config, ui=self.config.ui)
        self.shape_dropdown.selected = grid.shape
        self.triangle_dropdown.selected = grid.triangle_mode
        for size_field, size in ((self.width_input, grid.width), (self.height_input, grid.height)):
            size_field.max_val = max(size_field.max_val, size)
            size_field.set_value(size)
        self.wrap_button.text = self._wrap_label()
        self.density_slider.value = self.density_slider._clamp(
            round(simulation.initial_alive_percentage * 100)
        )
        self.seed_input.max_val = max(self.seed_input.max_val, simulation.seed)
        self.seed_input.set_value(simulation.seed)
        self._remember_applied(self.config, lifeforms)
        self.set_lifeforms(lifeforms)

    def _fail(self, message: str | None, *fields) -> None:
        """Show `message` above Apply (None clears it) and outline `fields` in red, clearing
        every other field's outline. The first field's section is opened and the field
        scrolled into view on the next draw, so the problem is always on screen."""
        self.error = message
        for widget in self._inputs():
            widget.invalid = any(widget is f for f in fields)
        if fields:
            grid_fields = (self.width_input, self.height_input)
            in_grid = any(fields[0] is f for f in grid_fields)
            (self.grid_section if in_grid else self.species_section).expanded = True
            self._reveal = fields[0]

    def _bad_size_fields(self, grid) -> list[NumericInput]:
        """The size field(s) at fault for a grid `validate_dimensions` rejects. Each dimension's
        rule (parity, minimum) is separate, so each is checked against a size the other
        dimension can always have."""
        roomy = 2 * MAX_GRID_SIZE  # even and above every shape's minimum
        bad = []
        for size_field, width, height in (
            (self.width_input, grid.width, roomy),
            (self.height_input, roomy, grid.height),
        ):
            try:
                validate_dimensions(grid.shape, width, height, grid.wrap, grid.triangle_mode)
            except ValueError:
                bad.append(size_field)
        return bad or [self.width_input, self.height_input]

    def _staged_density(self) -> float:
        """The starting density Apply will use: the slider's percentage, or the running value
        unchanged while the slider still shows it (a loaded 0.333 stays 0.333)."""
        applied = self._applied.simulation.initial_alive_percentage
        percent = int(round(self.density_slider.value))
        return applied if percent == round(applied * 100) else percent / 100

    def pending(self) -> bool:
        """True while any staged setting differs from what is running, so Apply would change
        something beyond re-seeding the grid."""
        applied = self._applied
        grid = applied.grid
        staged_grid = (
            self._staged_shape(),
            self._staged_mode(),
            _typed_int(self.width_input),
            _typed_int(self.height_input),
            self.config.grid.wrap,
        )
        if staged_grid != (grid.shape, grid.triangle_mode, grid.width, grid.height, grid.wrap):
            return True
        if self._staged_density() != applied.simulation.initial_alive_percentage:
            return True
        if _typed_int(self.seed_input) != applied.simulation.seed:
            return True
        if len(self.rule_inputs) != len(self._applied_rules):
            return True
        if _typed_int(self.species_count_input) != len(self.rule_inputs):
            return True
        for (birth_input, survive_input), applied_rule in zip(
            self.rule_inputs, self._applied_rules, strict=True
        ):
            try:
                staged = (
                    frozenset(parse_count_list(birth_input.text, _MAX_COUNT)),
                    frozenset(parse_count_list(survive_input.text, _MAX_COUNT)),
                )
            except ValueError:
                return True
            if staged != applied_rule:
                return True
        return False

    def _apply(self) -> None:
        """Validate the staged settings and hand them to `on_apply`; on any invalid value set
        `error`, outline the field(s) at fault, and apply nothing."""
        self.width_input.commit()
        self.height_input.commit()
        self.seed_input.commit()
        self.species_count_input.commit()

        grid = replace(
            self.config.grid,
            shape=self._staged_shape(),
            triangle_mode=self._staged_mode(),
            width=self.width_input.value,
            height=self.height_input.value,
        )
        try:
            validate_dimensions(grid.shape, grid.width, grid.height, grid.wrap, grid.triangle_mode)
        except ValueError as exc:
            self._fail(str(exc), *self._bad_size_fields(grid))
            return

        n = max_degree(grid.shape, grid.triangle_mode)
        lifeforms = []
        for i, (birth_input, survive_input) in enumerate(self.rule_inputs):
            counts: dict[str, set[int]] = {}
            for part, text_input in (("birth", birth_input), ("survive", survive_input)):
                try:
                    counts[part] = parse_count_list(text_input.text, n)
                except ValueError as exc:
                    self._fail(f"Species {i + 1} {part}: {exc}.", text_input)
                    return
            lifeforms.append(Lifeform(id=i + 1, rule=Rule(**counts)))

        self._show_rules(lifeforms)
        self._fail(None)

        ui = replace(self.config.ui, fps=int(round(self.fps_slider.value)))
        simulation = replace(
            self.config.simulation,
            n_lifeforms=len(lifeforms),
            initial_alive_percentage=self._staged_density(),
            seed=self.seed_input.value,
        )
        new_config = replace(self.config, grid=grid, ui=ui, simulation=simulation)
        self.config = new_config
        self.lifeforms = lifeforms
        self._remember_applied(new_config, lifeforms)
        self.on_apply(new_config, lifeforms)

    # --- events ----------------------------------------------------------------

    def _rule_widgets(self) -> list:
        return [w for pair in self.rule_inputs for w in pair]

    def _inputs(self) -> list:
        return [
            self.width_input,
            self.height_input,
            self.seed_input,
            self.species_count_input,
            *self._rule_widgets(),
        ]

    def _dropdowns(self) -> list[Dropdown]:
        return [self.shape_dropdown, self.triangle_dropdown, self.preset_dropdown,
                self.example_dropdown]  # fmt: skip

    def _open_dropdown(self) -> Dropdown | None:
        return next((d for d in self._dropdowns() if d.active), None)

    def _sliders(self) -> list[Slider]:
        return [self.density_slider, self.fps_slider]

    def _focused_inputs(self) -> list:
        return [w for w in self._inputs() if w.active]

    def has_focus(self) -> bool:
        """True while a text or number field is being typed in."""
        return any(w.active for w in self._inputs())

    def focused_rect(self) -> pygame.Rect | None:
        """The visible part of the field being typed in, if any (for the IME candidate box)."""
        visible = (self._visible_part(w.rect) for w in self._focused_inputs())
        return next((rect for rect in visible if rect is not None), None)

    def _visible_part(self, rect: pygame.Rect | None) -> pygame.Rect | None:
        """The part of a content widget's `rect` inside the scrolling viewport, or None if no
        part of it is (or it has no rect)."""
        if rect is None or self._content_clip is None:
            return rect
        visible = rect.clip(self._content_clip)
        return visible if visible.width and visible.height else None

    def _blur_out_of_view(self) -> None:
        """Commit and blur a focused field with no part left in the viewport — scrolled away or
        squeezed out by a smaller window — as a click elsewhere would: typing must never edit a
        field nobody can see, and the app's single-key shortcuts must work again."""
        for widget in self._focused_inputs():
            if widget.rect is not None and self._visible_part(widget.rect) is None:
                widget.blur()

    def _sections(self) -> list[CollapsibleSection]:
        return [
            self.grid_section,
            self.speed_section,
            self.species_section,
            self.view_section,
            self.drawing_section,
        ]

    def _section_widgets(self, section: CollapsibleSection) -> list:
        """Every widget of `section`, shown or not."""
        if section is self.grid_section:
            return [
                self.shape_dropdown,
                self.width_input,
                self.height_input,
                self.wrap_button,
                self.triangle_dropdown,
                self.density_slider,
                self.seed_input,
            ]
        if section is self.speed_section:
            return [self.fps_slider, self.alert_button]
        if section is self.view_section:
            return [self.view_button, self.rotate_button, self.camera_button, self.help_button]
        if section is self.drawing_section:
            return [self.example_dropdown, self.clear_button]
        return [
            self.species_count_input,
            self.preset_dropdown,
            *self._rule_widgets(),
            *self.randomize_buttons,
        ]

    def _shown_widgets(self, section: CollapsibleSection) -> list:
        """The widgets of an expanded `section` that are on screen: the triangle neighbourhood
        only while triangles are the staged shape."""
        widgets = self._section_widgets(section)
        if section is self.grid_section and self._staged_shape() != "triangle":
            widgets.remove(self.triangle_dropdown)
        return widgets

    def _hide_section(self, section: CollapsibleSection) -> None:
        """A collapsed section's widgets commit any edit, close, and lose their rects."""
        for widget in self._section_widgets(section):
            widget.hide()
        if section is self.species_section:
            self._species_label_rects = []

    def _content_widgets(self) -> list:
        """The widgets currently on screen in the scrolling area, in layout order."""
        widgets: list = []
        for section in self._sections():
            widgets.append(section)
            if section.expanded:
                widgets += self._shown_widgets(section)
        return widgets

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Route one event; True if the panel used it (see the module docstring)."""
        if hasattr(event, "pos") and event.type in (
            pygame.MOUSEMOTION,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
        ):
            self._pointer = event.pos
        if event.type == pygame.MOUSEBUTTONDOWN:
            return event.button == 1 and self._handle_click(event)
        if event.type == pygame.MOUSEWHEEL:
            return self._handle_wheel(event)
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP, pygame.WINDOWFOCUSLOST):
            return any([slider.handle_event(event) for slider in self._sliders()])
        if event.type in (pygame.KEYDOWN, pygame.TEXTINPUT, pygame.TEXTEDITING):
            dropdown = self._open_dropdown()
            if dropdown is not None and dropdown.handle_event(event):
                return True
            focused = self._focused_inputs()
            return any([w.handle_event(event) for w in focused])
        return False

    def _handle_click(self, event: pygame.event.Event) -> bool:
        dropdown = self._open_dropdown()
        if dropdown is not None:
            # An open list takes the click whatever it lands on: an option is chosen, anything
            # else only closes the list. Nothing under the list sees it.
            return dropdown.handle_event(event)

        pos = event.pos
        for widget in self._focused_inputs():
            # Only the field's visible part counts as on it: a click on a stat card or Apply
            # that its scrolled-away part lies under is elsewhere.
            visible = self._visible_part(widget.rect)
            if visible is None or not visible.collidepoint(pos):
                widget.blur()  # may resize the species rows: routed below with the new list

        if self.apply_button.handle_event(event):
            return True
        if self._content_clip is not None and not self._content_clip.collidepoint(pos):
            return False
        for widget in self._content_widgets():
            if widget.handle_event(event):
                if isinstance(widget, CollapsibleSection) and not widget.expanded:
                    self._hide_section(widget)
                return True
        for i, rect in enumerate(self._species_label_rects):
            if i < len(self.species_colors) and rect.collidepoint(pos):
                self.selected_species = i + 1
                if self.on_select_species is not None:
                    self.on_select_species(i + 1)
                return True
        return False

    def _handle_wheel(self, event: pygame.event.Event) -> bool:
        dropdown = self._open_dropdown()
        if dropdown is not None:
            return dropdown.handle_event(event)
        pointer = self._pointer if self._pointer is not None else pygame.mouse.get_pos()
        if self._scroll_area is None or not self._scroll_area.collidepoint(pointer):
            return False
        if self.max_scroll > 0:
            previous = self.scroll_y
            self.scroll_y = max(0, min(self.max_scroll, self.scroll_y - event.y * WHEEL_STEP))
            # Move the drawn rects with the content now, not at the next draw, so the events
            # after this one in the same frame (a click, typed text) meet the widgets where the
            # scroll put them — and a field it took out of view no longer takes the keys.
            self._move_content(previous - self.scroll_y)
            self._blur_out_of_view()
        return True

    def _move_content(self, dy: int) -> None:
        """Shift every on-screen rect in the scrolling area `dy` pixels down."""
        for widget in self._content_widgets():
            name = "header_rect" if isinstance(widget, CollapsibleSection) else "rect"
            rect = getattr(widget, name)
            if rect is not None:
                setattr(widget, name, rect.move(0, dy))
        self._species_label_rects = [rect.move(0, dy) for rect in self._species_label_rects]

    # --- drawing -----------------------------------------------------------------

    def _refresh(self) -> None:
        """Labels and menus that follow the app's state and the staged settings."""
        self.alert_button.text = f"Auto-pause: {'On' if self.alert_pause else 'Off'}"
        self.view_button.text = "2D view (V)" if self.view_mode == "3D" else "3D view (V)"
        self.rotate_button.text = f"Rotate: {'On' if self.auto_rotate else 'Off'}"
        self.camera_button.enabled = self.view_mode == "3D"
        species = self._preset_row() + 1
        self.preset_dropdown.placeholder = f"Preset rule for species {species}..."
        if not self.preset_dropdown.active:  # never change a list under the pointer
            presets = rule_presets(self._staged_shape(), self._staged_mode())
            self._presets = {preset.label: preset for preset in presets}
            self.preset_dropdown.options = list(self._presets)
        pending = self.pending()
        self.apply_button.text = "Apply changes" if pending else "Apply"
        self.apply_button.primary = pending

    def _hint(self) -> str:
        species = self.selected_species
        return (
            f"Click or drag on the grid to paint species {species}; a drag that starts on a "
            f"species-{species} cell erases. Examples load paused: press Play."
        )

    def _rows(self, surface: pygame.Surface, x: int, width: int) -> list[Row]:
        """The scrolling content as (height, draw-at-y) rows; hidden widgets are hidden here."""
        rows: list[Row] = []

        def header(section: CollapsibleSection) -> bool:
            rows.append(
                (
                    section.HEADER_HEIGHT + CONTROL_GAP,
                    lambda y: section.draw_header(surface, x, y, width),
                )
            )
            if not section.expanded:
                self._hide_section(section)
            return section.expanded

        def labelled(label: str, widget) -> None:
            rows.append((ROW_HEIGHT, lambda y: self._draw_row(surface, x, y, width, label, widget)))

        def full(widget) -> None:
            rows.append((ROW_HEIGHT, lambda y: self._draw_full(surface, x, y, width, widget)))

        def pair(left: Button, right: Button) -> None:
            rows.append((ROW_HEIGHT, lambda y: self._draw_pair(surface, x, y, width, left, right)))

        def gap() -> None:
            rows.append((SECTION_GAP, lambda y: None))

        if header(self.grid_section):
            labelled("Shape", self.shape_dropdown)
            rows.append((ROW_HEIGHT, lambda y: self._draw_size(surface, x, y, width)))
            rows.append((ROW_HEIGHT, lambda y: self._draw_wrap(surface, x, y, width)))
            if self._staged_shape() == "triangle":
                labelled("Neighbours", self.triangle_dropdown)
            else:
                self.triangle_dropdown.hide()
            labelled("Density", self.density_slider)
            labelled("Seed", self.seed_input)
        gap()

        if header(self.speed_section):
            labelled("FPS", self.fps_slider)
            labelled("Alerts", self.alert_button)
        gap()

        if header(self.species_section):
            labelled("Count", self.species_count_input)
            full(self.preset_dropdown)
            rows.append((CAPTION_HEIGHT, lambda y: self._draw_captions(surface, x, y, width)))
            for i in range(len(self.rule_inputs)):
                rows.append(
                    (ROW_HEIGHT, lambda y, i=i: self._draw_species_row(surface, x, y, width, i))
                )
        gap()

        if header(self.view_section):
            pair(self.view_button, self.rotate_button)
            pair(self.camera_button, self.help_button)
        gap()

        if header(self.drawing_section):
            full(self.example_dropdown)
            full(self.clear_button)
            lines = _wrap_text(self._hint(), self.font, width)
            line_height = self.font.get_linesize()
            rows.append(
                (
                    len(lines) * line_height + CONTROL_GAP,
                    lambda y: self._draw_lines(surface, x, y, width, lines, line_height),
                )
            )
        gap()
        return rows

    def draw(
        self, surface: pygame.Surface, x: int, y: int, width: int, height: int | None = None
    ) -> int:
        """Draw at (x, y), `width` wide. With `height`, the sections scroll inside it and the
        error text and Apply sit at the bottom; without, everything is laid out in full.
        Returns the height used."""
        self._refresh()
        rows = self._rows(surface, x, width)
        content_height = sum(row_height for row_height, _ in rows)
        error_lines = []
        if self.error:
            error_lines = _wrap_text(self.error, self.font, width, MAX_ERROR_LINES)
        line_height = self.font.get_linesize()
        footer_height = ROW_HEIGHT
        if error_lines:
            footer_height += len(error_lines) * line_height + CONTROL_GAP

        visible = content_height if height is None else max(0, height - footer_height)
        visible = min(visible, content_height)
        self.max_scroll = content_height - visible
        self.scroll_y = max(0, min(self.max_scroll, self.scroll_y))
        content_rect = pygame.Rect(x, y, width, visible)
        self._content_clip = content_rect
        # The wheel scrolls the panel anywhere over its column, margins included.
        self._scroll_area = content_rect.inflate(2 * CONTROL_GAP + 2 * SCROLLBAR_WIDTH + 16, 0)

        previous_clip = surface.get_clip()
        surface.set_clip(content_rect.clip(previous_clip))
        cursor = y - self.scroll_y
        for row_height, draw_row in rows:
            draw_row(cursor)
            cursor += row_height
        surface.set_clip(previous_clip)
        self._blur_out_of_view()
        self._scroll_to_reveal(content_rect)
        if self.max_scroll > 0:
            self._draw_scrollbar(surface, content_rect, content_height)

        cursor = content_rect.bottom
        for line in error_lines:
            line_rect = pygame.Rect(x, cursor, width, line_height)
            label = self.font.render(line, True, ERROR_COLOR)
            blit_clipped(surface, label, line_rect.topleft, line_rect)
            cursor += line_height
        if error_lines:
            cursor += CONTROL_GAP
        self.apply_button.rect = pygame.Rect(x, cursor, width, ROW_HEIGHT)
        self.apply_button.draw(surface)
        cursor += ROW_HEIGHT

        # Drawn last so an open option list paints on top of the rows below it instead of
        # being immediately overdrawn by them.
        for dropdown in self._dropdowns():
            dropdown.draw_options(surface)
        return cursor - y

    def _scroll_to_reveal(self, content_rect: pygame.Rect) -> None:
        """Scroll (from the next frame) just enough to bring `_reveal` fully into view."""
        rect = getattr(self._reveal, "rect", None)
        self._reveal = None
        if rect is None:
            return
        if rect.top < content_rect.top:
            self.scroll_y -= content_rect.top - rect.top
        elif rect.bottom > content_rect.bottom:
            self.scroll_y += rect.bottom - content_rect.bottom
        self.scroll_y = max(0, min(self.max_scroll, self.scroll_y))

    def _draw_scrollbar(
        self, surface: pygame.Surface, content_rect: pygame.Rect, content_height: int
    ) -> None:
        track = pygame.Rect(
            content_rect.right + 6, content_rect.y, SCROLLBAR_WIDTH, content_rect.height
        )
        pygame.draw.rect(surface, SCROLLBAR_TRACK, track, border_radius=2)
        thumb_height = max(16, track.height * track.height // max(content_height, 1))
        thumb_y = track.y + (track.height - thumb_height) * self.scroll_y // self.max_scroll
        thumb = pygame.Rect(track.x, thumb_y, SCROLLBAR_WIDTH, thumb_height)
        pygame.draw.rect(surface, PLACEHOLDER_COLOR, thumb, border_radius=2)

    def _draw_label(self, surface: pygame.Surface, x: int, y: int, text: str) -> None:
        label = self.font.render(text, True, LABEL_COLOR)
        blit_clipped(surface, label, (x, y + 6), pygame.Rect(x, y, LABEL_WIDTH - 4, ROW_HEIGHT))

    def _draw_row(
        self, surface: pygame.Surface, x: int, y: int, width: int, label_text: str, widget
    ) -> None:
        self._draw_label(surface, x, y, label_text)
        rect = pygame.Rect(x + LABEL_WIDTH, y, width - LABEL_WIDTH, ROW_HEIGHT - 4)
        if isinstance(widget, Button):
            widget.rect = rect
            widget.draw(surface)
        else:
            widget.draw(surface, rect)

    def _draw_full(self, surface: pygame.Surface, x: int, y: int, width: int, widget) -> None:
        """A widget across the whole row, with no label."""
        rect = pygame.Rect(x, y, width, ROW_HEIGHT - 4)
        if isinstance(widget, Button):
            widget.rect = rect
            widget.draw(surface)
        else:
            widget.draw(surface, rect)

    def _draw_pair(
        self, surface: pygame.Surface, x: int, y: int, width: int, left: Button, right: Button
    ) -> None:
        half = (width - CONTROL_GAP) // 2
        left.rect = pygame.Rect(x, y, half, ROW_HEIGHT - 4)
        right.rect = pygame.Rect(x + width - half, y, half, ROW_HEIGHT - 4)
        left.draw(surface)
        right.draw(surface)

    def _draw_size(self, surface: pygame.Surface, x: int, y: int, width: int) -> None:
        """Width and height side by side: "Size  [w] x [h]"."""
        self._draw_label(surface, x, y, "Size")
        times = self.font.render("×", True, PLACEHOLDER_COLOR)
        gap = times.get_width() + 2 * CONTROL_GAP
        field_width = (width - LABEL_WIDTH - gap) // 2
        left = pygame.Rect(x + LABEL_WIDTH, y, field_width, ROW_HEIGHT - 4)
        right = pygame.Rect(x + width - field_width, y, field_width, ROW_HEIGHT - 4)
        self.width_input.draw(surface, left)
        self.height_input.draw(surface, right)
        surface.blit(times, times.get_rect(center=((left.right + right.left) // 2, left.centery)))

    def _draw_wrap(self, surface: pygame.Surface, x: int, y: int, width: int) -> None:
        self.wrap_button.rect = pygame.Rect(x + LABEL_WIDTH, y, width - LABEL_WIDTH, ROW_HEIGHT - 4)
        self.wrap_button.draw(surface)

    def _draw_lines(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        width: int,
        lines: list[str],
        line_height: int,
    ) -> None:
        for i, line in enumerate(lines):
            label = self.font.render(line, True, PLACEHOLDER_COLOR)
            line_rect = pygame.Rect(x, y + i * line_height, width, line_height)
            blit_clipped(surface, label, line_rect.topleft, line_rect)

    def _rule_columns(self, x: int, width: int) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        """(birth, survive, R button) rects of a species row at y=0."""
        button_size = ROW_HEIGHT - 4
        half = (width - LABEL_WIDTH - button_size - CONTROL_GAP) // 2
        birth = pygame.Rect(x + LABEL_WIDTH, 0, half - 4, ROW_HEIGHT - 4)
        survive = pygame.Rect(x + LABEL_WIDTH + half, 0, half - 4, ROW_HEIGHT - 4)
        button = pygame.Rect(x + width - button_size, 0, button_size, button_size)
        return birth, survive, button

    def _draw_captions(self, surface: pygame.Surface, x: int, y: int, width: int) -> None:
        birth, survive, _ = self._rule_columns(x, width)
        for text, column in (("Birth", birth), ("Survive", survive)):
            caption = self.font.render(text, True, PLACEHOLDER_COLOR)
            surface.blit(caption, (column.x + 2, y))

    def _draw_species_row(
        self, surface: pygame.Surface, x: int, y: int, width: int, i: int
    ) -> None:
        applied = i < len(self.species_colors)
        label_rect = pygame.Rect(x, y, LABEL_WIDTH - 6, ROW_HEIGHT - 4)
        if applied and i + 1 == self.selected_species:
            pygame.draw.rect(surface, ACCENT_PRIMARY, label_rect, 2, border_radius=4)
        swatch = pygame.Rect(0, 0, SWATCH_SIZE, SWATCH_SIZE)
        swatch.midleft = (label_rect.x + 6, label_rect.centery)
        if applied:
            pygame.draw.rect(surface, self.species_colors[i], swatch, border_radius=3)
        else:
            pygame.draw.rect(surface, PLACEHOLDER_COLOR, swatch, 1, border_radius=3)
        name_box = pygame.Rect(swatch.right + 6, label_rect.y, 0, label_rect.height)
        name_box.width = label_rect.right - 4 - name_box.x
        # One format for every row: the long one only if the widest label fits.
        long_fits = self.font.size(f"Species {len(self.rule_inputs)}")[0] <= name_box.width
        name = self.font.render(f"{'Species' if long_fits else 'Sp.'} {i + 1}", True, LABEL_COLOR)
        blit_clipped(surface, name, name.get_rect(midleft=name_box.midleft), name_box)
        while len(self._species_label_rects) <= i:
            self._species_label_rects.append(pygame.Rect(0, 0, 0, 0))
        self._species_label_rects[i] = label_rect

        birth, survive, button = self._rule_columns(x, width)
        birth_input, survive_input = self.rule_inputs[i]
        birth_input.draw(surface, birth.move(0, y))
        survive_input.draw(surface, survive.move(0, y))
        randomize_button = self.randomize_buttons[i]
        randomize_button.rect = button.move(0, y)
        randomize_button.draw(surface)
