"""Settings panel covering grid dims/shape/species count/rules/speed (§9 acceptance), rebuilt
against `Config`/`Lifeform` rather than the legacy Game/grid objects. Widget wiring only — the
widgets themselves live in `ui/widgets.py`. Changes are staged in the panel's own widget state
and only take effect when "Apply" is clicked, via the `on_apply` callback the caller supplies
(the app owns rebuilding the SimulationSession from a new Config).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pygame

from latticelife.config import Config
from latticelife.core.geometries import max_degree
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
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


def _rule_string(counts: set[int]) -> str:
    return "".join(str(v) for v in sorted(counts))


def _parse_rule(text: str) -> set[int]:
    return {int(ch) for ch in text if ch.isdigit()}


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
            "FPS", 1, 60, self.config.ui.fps, self.font, format_fn=lambda v: f"{int(v)} fps"
        )
        self.species_count_input = NumericInput(
            len(self.lifeforms), 1, 10, self.font, on_commit=self._resize_species
        )
        self.rule_inputs = [
            (
                TextInput(_rule_string(lf.rule.birth), self.font, placeholder="birth"),
                TextInput(_rule_string(lf.rule.survive), self.font, placeholder="survive"),
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
        rule = Rule.random(
            n, self.config.discovery.p_birth, self.config.discovery.p_survive, np.random.default_rng()
        )
        birth_input, survive_input = self.rule_inputs[index]
        birth_input.text = _rule_string(rule.birth)
        survive_input.text = _rule_string(rule.survive)

    def _resize_species(self, count: int) -> None:
        current = len(self.rule_inputs)
        if count > current:
            for _ in range(count - current):
                self.rule_inputs.append((TextInput("3", self.font), TextInput("23", self.font)))
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

    def _apply(self) -> None:
        self.width_input.commit()
        self.height_input.commit()
        self.species_count_input.commit()

        grid = replace(
            self.config.grid,
            shape=self.shape_dropdown.selected or self.config.grid.shape,
            width=self.width_input.value,
            height=self.height_input.value,
        )
        ui = replace(self.config.ui, fps=int(self.fps_slider.value))
        new_config = replace(self.config, grid=grid, ui=ui)

        lifeforms = [
            Lifeform(id=i + 1, rule=Rule(birth=_parse_rule(b.text), survive=_parse_rule(s.text)))
            for i, (b, s) in enumerate(self.rule_inputs)
        ]

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
