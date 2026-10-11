"""Settings panel covering grid dims/shape/species count/rules/speed (§9 acceptance), rebuilt
against `Config`/`Lifeform` rather than the legacy Game/grid objects. Widget wiring only — the
widgets themselves live in `ui/widgets.py`. Changes are staged in the panel's own widget state
and only take effect when "Apply" is clicked, via the `on_apply` callback the caller supplies
(the app owns rebuilding the SimulationSession from a new Config).

Apply validates before it calls back: grid sizes that `core.geometries.validate_dimensions`
rejects (e.g. an odd width for a wrapped hexagon grid) and rule fields that do not parse, or name
a count above the selected shape's neighbour count, set `error` — drawn in red above the Apply
button — and nothing is applied. Rule fields use the comma-separated display form of
`core.rules` (`'2,3'`), so the 10-12 neighbour counts of edge+vertex triangles round-trip.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pygame

from latticelife.config import MAX_FPS, MAX_LIFEFORMS, MIN_FPS, Config
from latticelife.core.geometries import max_degree, validate_dimensions
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule, format_count_list, parse_count_list
from latticelife.ui.widgets import (
    Button,
    CollapsibleSection,
    Dropdown,
    NumericInput,
    Slider,
    TextInput,
)

SHAPES = ["square", "hexagon", "triangle"]
SECTION_GAP = 12
CONTROL_GAP = 8
ROW_HEIGHT = 32
LABEL_WIDTH = 90
LABEL_COLOR = (235, 235, 245)
ERROR_COLOR = (248, 113, 113)


def _wrap_text(text: str, font: pygame.font.Font, width: int) -> list[str]:
    """Greedy word wrap of `text` to lines no wider than `width` pixels (a single word wider
    than that gets a line of its own)."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}" if line else word
        if line and font.size(candidate)[0] > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


@dataclass
class SettingsPanel:
    config: Config
    lifeforms: list[Lifeform]
    font: pygame.font.Font
    on_apply: Callable[[Config, list[Lifeform]], None]

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

    def __post_init__(self) -> None:
        self.grid_section = CollapsibleSection("Grid", self.font)
        self.speed_section = CollapsibleSection("Speed", self.font)
        self.species_section = CollapsibleSection("Species", self.font)

        self.shape_dropdown = Dropdown(list(SHAPES), self.font, selected=self.config.grid.shape)
        self.width_input = NumericInput(self.config.grid.width, 5, 500, self.font)
        self.height_input = NumericInput(self.config.grid.height, 5, 500, self.font)
        self.wrap_button = Button(
            pygame.Rect(0, 0, 0, 0), self._wrap_label(), self.font, on_click=self._toggle_wrap
        )
        self.fps_slider = Slider(
            "FPS",
            MIN_FPS,
            MAX_FPS,
            self.config.ui.fps,
            self.font,
            format_fn=lambda v: f"{int(v)} fps",
        )
        self.species_count_input = NumericInput(
            len(self.lifeforms), 1, MAX_LIFEFORMS, self.font, on_commit=self._resize_species
        )
        self.rule_inputs = [
            (
                TextInput(format_count_list(lf.rule.birth), self.font, placeholder="birth"),
                TextInput(format_count_list(lf.rule.survive), self.font, placeholder="survive"),
            )
            for lf in self.lifeforms
        ]
        self.randomize_buttons = [
            Button(pygame.Rect(0, 0, 0, 0), "R", self.font, on_click=self._make_randomizer(i))
            for i in range(len(self.lifeforms))
        ]
        self.apply_button = Button(
            pygame.Rect(0, 0, 0, 0), "Apply", self.font, primary=True, on_click=self._apply
        )

    def _wrap_label(self) -> str:
        return "Wrap: On" if self.config.grid.wrap else "Wrap: Off"

    def _toggle_wrap(self) -> None:
        new_grid = replace(self.config.grid, wrap=not self.config.grid.wrap)
        self.config = replace(self.config, grid=new_grid)
        self.wrap_button.text = self._wrap_label()

    def _make_randomizer(self, index: int) -> Callable[[], None]:
        return lambda: self._randomize_rule(index)

    def _randomize_rule(self, index: int) -> None:
        """Populates B/S with a random rule sized to the selected shape's neighbor count,
        reusing the same `Rule.random` the ES discovery process samples candidates from."""
        shape = self.shape_dropdown.selected or self.config.grid.shape
        n = max_degree(shape, self.config.grid.triangle_mode)
        discovery = self.config.discovery
        rule = Rule.random(n, discovery.p_birth, discovery.p_survive, np.random.default_rng())
        birth_input, survive_input = self.rule_inputs[index]
        birth_input.text = format_count_list(rule.birth)
        survive_input.text = format_count_list(rule.survive)

    def _resize_species(self, count: int) -> None:
        current = len(self.rule_inputs)
        if count > current:
            for _ in range(count - current):
                self.rule_inputs.append((TextInput("3", self.font), TextInput("2,3", self.font)))
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

    def set_lifeforms(self, lifeforms: list[Lifeform]) -> None:
        """Show `lifeforms`' rules in the species rows (one row each, adding or removing rows to
        match) and clear any error — for when the app replaces the species itself, e.g. New."""
        self.lifeforms = lifeforms
        self.species_count_input.set_value(len(lifeforms))
        self._resize_species(len(lifeforms))
        for (birth_input, survive_input), lf in zip(self.rule_inputs, lifeforms, strict=True):
            birth_input.text = format_count_list(lf.rule.birth)
            survive_input.text = format_count_list(lf.rule.survive)
        self.error = None

    def _apply(self) -> None:
        """Validate the staged settings and hand them to `on_apply`; on any invalid value set
        `error` instead and apply nothing."""
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
            self.error = str(exc)
            return

        n = max_degree(grid.shape, grid.triangle_mode)
        lifeforms = []
        for i, (birth_input, survive_input) in enumerate(self.rule_inputs):
            counts: dict[str, set[int]] = {}
            for part, text_input in (("birth", birth_input), ("survive", survive_input)):
                try:
                    counts[part] = parse_count_list(text_input.text, n)
                except ValueError as exc:
                    self.error = f"Species {i + 1} {part}: {exc}."
                    return
            lifeforms.append(Lifeform(id=i + 1, rule=Rule(**counts)))

        for (birth_input, survive_input), lf in zip(self.rule_inputs, lifeforms, strict=True):
            birth_input.text = format_count_list(lf.rule.birth)
            survive_input.text = format_count_list(lf.rule.survive)
        self.error = None

        ui = replace(self.config.ui, fps=int(self.fps_slider.value))
        new_config = replace(self.config, grid=grid, ui=ui)
        self.on_apply(new_config, lifeforms)

    def handle_event(self, event: pygame.event.Event) -> bool:
        widgets: list = [
            self.grid_section,
            self.speed_section,
            self.species_section,
            self.shape_dropdown,
            self.width_input,
            self.height_input,
            self.wrap_button,
            self.fps_slider,
            self.species_count_input,
            self.apply_button,
        ]
        for birth_input, survive_input in self.rule_inputs:
            widgets.append(birth_input)
            widgets.append(survive_input)
        widgets.extend(self.randomize_buttons)
        # Not a generator: a blurring TextInput/NumericInput reports "handled" for the same
        # click that lands on another widget (e.g. Apply), and `any` must not short-circuit
        # before that other widget also sees the event.
        handled = [widget.handle_event(event) for widget in widgets]
        return any(handled)

    def draw(self, surface: pygame.Surface, x: int, y: int, width: int) -> int:
        cursor = y
        cursor += self.grid_section.draw_header(surface, x, cursor, width) + CONTROL_GAP
        if self.grid_section.expanded:
            cursor = self._draw_row(surface, x, cursor, width, "Shape", self.shape_dropdown)
            cursor = self._draw_row(surface, x, cursor, width, "Width", self.width_input)
            cursor = self._draw_row(surface, x, cursor, width, "Height", self.height_input)
            wrap_width = width - LABEL_WIDTH
            self.wrap_button.rect = pygame.Rect(x + LABEL_WIDTH, cursor, wrap_width, ROW_HEIGHT - 4)
            self.wrap_button.draw(surface)
            cursor += ROW_HEIGHT
        cursor += SECTION_GAP

        cursor += self.speed_section.draw_header(surface, x, cursor, width) + CONTROL_GAP
        if self.speed_section.expanded:
            self.fps_slider.draw(
                surface, pygame.Rect(x + LABEL_WIDTH, cursor, width - LABEL_WIDTH, ROW_HEIGHT - 4)
            )
            cursor += ROW_HEIGHT
        cursor += SECTION_GAP

        cursor += self.species_section.draw_header(surface, x, cursor, width) + CONTROL_GAP
        if self.species_section.expanded:
            cursor = self._draw_row(surface, x, cursor, width, "Count", self.species_count_input)
            button_size = ROW_HEIGHT - 4
            half = (width - LABEL_WIDTH - button_size - CONTROL_GAP) // 2
            for i, (birth_input, survive_input) in enumerate(self.rule_inputs):
                label = self.font.render(f"B{i + 1}", True, LABEL_COLOR)
                surface.blit(label, (x, cursor + 6))
                birth_rect = pygame.Rect(x + LABEL_WIDTH, cursor, half - 4, ROW_HEIGHT - 4)
                birth_input.draw(surface, birth_rect)
                survive_input.draw(
                    surface, pygame.Rect(x + LABEL_WIDTH + half, cursor, half - 4, ROW_HEIGHT - 4)
                )
                randomize_button = self.randomize_buttons[i]
                randomize_button.rect = pygame.Rect(
                    x + width - button_size, cursor, button_size, button_size
                )
                randomize_button.draw(surface)
                cursor += ROW_HEIGHT
        cursor += SECTION_GAP

        if self.error:
            for line in _wrap_text(self.error, self.font, width):
                surface.blit(self.font.render(line, True, ERROR_COLOR), (x, cursor))
                cursor += self.font.get_linesize()
            cursor += CONTROL_GAP

        self.apply_button.rect = pygame.Rect(x, cursor, width, ROW_HEIGHT)
        self.apply_button.draw(surface)
        cursor += ROW_HEIGHT

        # Drawn last so an open option list paints on top of the rows below it instead of
        # being immediately overdrawn by them.
        self.shape_dropdown.draw_options(surface)

        return cursor - y

    def _draw_row(
        self, surface: pygame.Surface, x: int, y: int, width: int, label_text: str, widget
    ) -> int:
        label = self.font.render(label_text, True, LABEL_COLOR)
        surface.blit(label, (x, y + 6))
        widget.draw(surface, pygame.Rect(x + LABEL_WIDTH, y, width - LABEL_WIDTH, ROW_HEIGHT - 4))
        return y + ROW_HEIGHT
