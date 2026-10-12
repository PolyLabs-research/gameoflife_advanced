"""Settings panel covering grid dims/shape/species count/rules/speed (§9 acceptance), rebuilt
against `Config`/`Lifeform` rather than the legacy Game/grid objects. Widget wiring only — the
widgets themselves live in `ui/widgets.py`. Grid and species changes are staged in the panel's
own widget state and only take effect when "Apply" is clicked, via the `on_apply` callback the
caller supplies (the app owns rebuilding the SimulationSession from a new Config). The FPS
slider is the exception: it changes the running simulation's speed live, through
`on_fps_change`, without restarting anything.

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
reaches a removed row). An open Shape list takes the click for itself. Widgets in a collapsed
section have no on-screen rect and get no events. Only what lies inside the viewport can be
clicked — a click on a field's scrolled-away part is a click elsewhere — and a focused field
with no part left in view (scrolled away, or squeezed out by a smaller window) commits and
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
from latticelife.core.geometries import SHAPES, max_degree, validate_dimensions
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule, format_count_list, parse_count_list
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
# Mixed into the simulation seed for the UI's own random stream (see `ui_rng`).
UI_RNG_KEY = 0x5549


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

    grid_section: CollapsibleSection = field(init=False)
    speed_section: CollapsibleSection = field(init=False)
    species_section: CollapsibleSection = field(init=False)
    shape_dropdown: Dropdown = field(init=False)
    width_input: NumericInput = field(init=False)
    height_input: NumericInput = field(init=False)
    wrap_button: Button = field(init=False)
    fps_slider: Slider = field(init=False)
    species_count_input: NumericInput = field(init=False)
    rule_inputs: list[tuple[TextInput, TextInput]] = field(init=False)
    randomize_buttons: list[Button] = field(init=False)
    apply_button: Button = field(init=False)
    error: str | None = field(default=None, init=False)
    # Colour of each applied species, by row (set by the app); rows beyond it are staged only.
    species_colors: list[tuple[int, int, int]] = field(default_factory=list, init=False)
    # The species id a grid click places (set by the app), highlighted in its row.
    selected_species: int = field(default=1, init=False)
    scroll_y: int = field(default=0, init=False)
    max_scroll: int = field(default=0, init=False)
    _content_clip: pygame.Rect | None = field(default=None, init=False)
    _scroll_area: pygame.Rect | None = field(default=None, init=False)
    _species_label_rects: list[pygame.Rect] = field(default_factory=list, init=False)
    _pointer: tuple[int, int] | None = field(default=None, init=False)
    # A field to scroll into view at the next draw (one a failed Apply flagged).
    _reveal: object | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = ui_rng(self.config.simulation.seed)
        self.grid_section = CollapsibleSection("Grid", self.font)
        self.speed_section = CollapsibleSection("Speed", self.font)
        self.species_section = CollapsibleSection("Species", self.font)

        self.shape_dropdown = Dropdown(list(SHAPES), self.font, selected=self.config.grid.shape)
        width, height = self.config.grid.width, self.config.grid.height
        self.width_input = NumericInput(width, 1, max(MAX_GRID_SIZE, width), self.font)
        self.height_input = NumericInput(height, 1, max(MAX_GRID_SIZE, height), self.font)
        self.wrap_button = Button(
            pygame.Rect(0, 0, 0, 0), self._wrap_label(), self.font, on_click=self._toggle_wrap
        )
        self.fps_slider = Slider(
            "FPS",
            MIN_FPS,
            MAX_FPS,
            self.config.ui.fps,
            self.font,
            format_fn=lambda v: f"{round(v)} fps",
            live_callback=self._fps_moved,
            step=1,
            show_value=True,
        )
        self.species_count_input = NumericInput(
            len(self.lifeforms), 1, MAX_LIFEFORMS, self.font, on_commit=self._resize_species
        )
        self.rule_inputs = []
        self.randomize_buttons = []
        # One row per species, up to the count the input shows (it clamps to MAX_LIFEFORMS),
        # so what is listed is exactly what Apply will apply.
        self._resize_species(self.species_count_input.value)
        self._show_rules(self.lifeforms)
        self.apply_button = Button(
            pygame.Rect(0, 0, 0, 0), "Apply", self.font, primary=True, on_click=self._apply
        )

    # --- actions -------------------------------------------------------------

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

    def _make_randomizer(self, index: int) -> Callable[[], None]:
        return lambda: self._randomize_rule(index)

    def _randomize_rule(self, index: int) -> None:
        """Populates B/S with a random rule sized to the selected shape's neighbor count,
        reusing the same `Rule.random` the ES discovery process samples candidates from."""
        if not 0 <= index < len(self.rule_inputs):
            return
        shape = self.shape_dropdown.selected or self.config.grid.shape
        n = max_degree(shape, self.config.grid.triangle_mode)
        discovery = self.config.discovery
        assert self.rng is not None
        rule = Rule.random(n, discovery.p_birth, discovery.p_survive, self.rng)
        birth_input, survive_input = self.rule_inputs[index]
        birth_input.text = format_count_list(rule.birth)
        survive_input.text = format_count_list(rule.survive)

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

    def set_lifeforms(self, lifeforms: list[Lifeform]) -> None:
        """Show `lifeforms`' rules in the species rows (one row each, adding or removing rows to
        match) and clear any error — for when the app replaces the species itself, e.g. New."""
        self.lifeforms = lifeforms
        self.species_count_input.set_value(len(lifeforms))
        self._resize_species(len(lifeforms))
        self._show_rules(lifeforms)
        self._fail(None)

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

    def _apply(self) -> None:
        """Validate the staged settings and hand them to `on_apply`; on any invalid value set
        `error`, outline the field(s) at fault, and apply nothing."""
        self.width_input.commit()
        self.height_input.commit()
        self.species_count_input.commit()

        grid = replace(
            self.config.grid,
            shape=self.shape_dropdown.selected or self.config.grid.shape,
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
        simulation = replace(self.config.simulation, n_lifeforms=len(lifeforms))
        new_config = replace(self.config, grid=grid, ui=ui, simulation=simulation)
        self.config = new_config
        self.lifeforms = lifeforms
        self.on_apply(new_config, lifeforms)

    # --- events ----------------------------------------------------------------

    def _rule_widgets(self) -> list:
        return [w for pair in self.rule_inputs for w in pair]

    def _inputs(self) -> list:
        return [
            self.width_input,
            self.height_input,
            self.species_count_input,
            *self._rule_widgets(),
        ]

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

    def _section_widgets(self, section: CollapsibleSection) -> list:
        if section is self.grid_section:
            return [self.shape_dropdown, self.width_input, self.height_input, self.wrap_button]
        if section is self.speed_section:
            return [self.fps_slider]
        return [self.species_count_input, *self._rule_widgets(), *self.randomize_buttons]

    def _hide_section(self, section: CollapsibleSection) -> None:
        """A collapsed section's widgets commit any edit, close, and lose their rects."""
        for widget in self._section_widgets(section):
            widget.hide()
        if section is self.species_section:
            self._species_label_rects = []

    def _content_widgets(self) -> list:
        """The widgets currently on screen in the scrolling area, in layout order."""
        widgets: list = [self.grid_section]
        if self.grid_section.expanded:
            widgets += self._section_widgets(self.grid_section)
        widgets.append(self.speed_section)
        if self.speed_section.expanded:
            widgets.append(self.fps_slider)
        widgets.append(self.species_section)
        if self.species_section.expanded:
            widgets += self._section_widgets(self.species_section)
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
            return self.fps_slider.handle_event(event)
        if event.type in (pygame.KEYDOWN, pygame.TEXTINPUT, pygame.TEXTEDITING):
            if self.shape_dropdown.active and self.shape_dropdown.handle_event(event):
                return True
            focused = self._focused_inputs()
            return any([w.handle_event(event) for w in focused])
        return False

    def _handle_click(self, event: pygame.event.Event) -> bool:
        if self.shape_dropdown.active:
            # An open list takes the click whatever it lands on: an option is chosen, anything
            # else only closes the list. Nothing under the list sees it.
            return self.shape_dropdown.handle_event(event)

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
        if self.shape_dropdown.active:
            return self.shape_dropdown.handle_event(event)
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

    def _rows(self, surface: pygame.Surface, x: int, width: int) -> list[Row]:
        """The scrolling content as (height, draw-at-y) rows; hidden widgets are hidden here."""
        rows: list[Row] = []

        def header(section: CollapsibleSection) -> None:
            rows.append(
                (
                    section.HEADER_HEIGHT + CONTROL_GAP,
                    lambda y: section.draw_header(surface, x, y, width),
                )
            )

        def gap() -> None:
            rows.append((SECTION_GAP, lambda y: None))

        header(self.grid_section)
        if self.grid_section.expanded:
            for label, widget in (
                ("Shape", self.shape_dropdown),
                ("Width", self.width_input),
                ("Height", self.height_input),
            ):
                rows.append(
                    (
                        ROW_HEIGHT,
                        lambda y, label=label, widget=widget: self._draw_row(
                            surface, x, y, width, label, widget
                        ),
                    )
                )
            rows.append((ROW_HEIGHT, lambda y: self._draw_wrap(surface, x, y, width)))
        else:
            self._hide_section(self.grid_section)
        gap()

        header(self.speed_section)
        if self.speed_section.expanded:
            rows.append(
                (ROW_HEIGHT, lambda y: self._draw_row(surface, x, y, width, "FPS", self.fps_slider))
            )
        else:
            self._hide_section(self.speed_section)
        gap()

        header(self.species_section)
        if self.species_section.expanded:
            rows.append(
                (
                    ROW_HEIGHT,
                    lambda y: self._draw_row(
                        surface, x, y, width, "Count", self.species_count_input
                    ),
                )
            )
            rows.append((CAPTION_HEIGHT, lambda y: self._draw_captions(surface, x, y, width)))
            for i in range(len(self.rule_inputs)):
                rows.append(
                    (ROW_HEIGHT, lambda y, i=i: self._draw_species_row(surface, x, y, width, i))
                )
        else:
            self._hide_section(self.species_section)
        gap()
        return rows

    def draw(
        self, surface: pygame.Surface, x: int, y: int, width: int, height: int | None = None
    ) -> int:
        """Draw at (x, y), `width` wide. With `height`, the sections scroll inside it and the
        error text and Apply sit at the bottom; without, everything is laid out in full.
        Returns the height used."""
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
        self.shape_dropdown.draw_options(surface)
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

    def _draw_row(
        self, surface: pygame.Surface, x: int, y: int, width: int, label_text: str, widget
    ) -> None:
        label = self.font.render(label_text, True, LABEL_COLOR)
        surface.blit(label, (x, y + 6))
        widget.draw(surface, pygame.Rect(x + LABEL_WIDTH, y, width - LABEL_WIDTH, ROW_HEIGHT - 4))

    def _draw_wrap(self, surface: pygame.Surface, x: int, y: int, width: int) -> None:
        self.wrap_button.rect = pygame.Rect(x + LABEL_WIDTH, y, width - LABEL_WIDTH, ROW_HEIGHT - 4)
        self.wrap_button.draw(surface)

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
