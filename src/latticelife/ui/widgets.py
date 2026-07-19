"""Widgets salvaged from the legacy `settings_panel.py`, trimmed: panel wiring is rebuilt
against `Config` in `ui/panel.py`, but the draw-yourself / handle-your-own-events shape of each
widget survives. `CollapsibleSection` drops the legacy `content_height`/`animation_progress`
fields — they were set at construction and never read or animated anywhere in the legacy file.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import pygame

TEXT_COLOR = (235, 235, 245)
PANEL_BACKGROUND_COLOR = (24, 26, 32)
BUTTON_COLOR = (55, 58, 72)
BUTTON_HOVER_COLOR = (75, 78, 96)
BUTTON_TEXT_COLOR = (255, 255, 255)
BUTTON_BORDER_RADIUS = 6
ACCENT_PRIMARY = (99, 102, 241)
SCROLLBAR_TRACK = (35, 38, 48)


@dataclass
class Button:
    rect: pygame.Rect
    text: str
    font: pygame.font.Font
    primary: bool = False
    on_click: Callable[[], None] | None = None

    def is_hovered(self, mouse_pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)

    def draw(self, surface: pygame.Surface) -> None:
        hovered = self.is_hovered(pygame.mouse.get_pos())
        fill = ACCENT_PRIMARY if self.primary else (BUTTON_HOVER_COLOR if hovered else BUTTON_COLOR)
        pygame.draw.rect(surface, fill, self.rect, border_radius=BUTTON_BORDER_RADIUS)
        pygame.draw.rect(surface, TEXT_COLOR, self.rect, 1, border_radius=BUTTON_BORDER_RADIUS)
        label = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        surface.blit(label, label.get_rect(center=self.rect.center))

    def handle_event(self, event: pygame.event.Event) -> bool:
        is_click = event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
        if is_click and self.is_hovered(event.pos):
            if self.on_click is not None:
                self.on_click()
            return True
        return False


@dataclass
class TextInput:
    text: str
    font: pygame.font.Font
    placeholder: str = ""
    on_commit: Callable[[str], None] | None = None
    active: bool = field(default=False, init=False)
    rect: pygame.Rect | None = field(default=None, init=False)

    def _commit(self) -> None:
        if self.on_commit is not None:
            self.on_commit(self.text)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            was_active = self.active
            self.active = self.rect is not None and self.rect.collidepoint(event.pos)
            if was_active and not self.active:
                self._commit()
            return was_active or self.active
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self._commit()
                self.active = False
            else:
                self.text += event.unicode
            return True
        return False

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self.rect = rect
        border = BUTTON_HOVER_COLOR if self.active else TEXT_COLOR
        pygame.draw.rect(surface, BUTTON_COLOR, rect, border_radius=4)
        pygame.draw.rect(surface, border, rect, 2, border_radius=4)
        if self.text:
            label = self.font.render(self.text, True, TEXT_COLOR)
        else:
            label = self.font.render(self.placeholder, True, BUTTON_HOVER_COLOR)
        surface.blit(label, label.get_rect(midleft=(rect.x + 8, rect.centery)))


@dataclass
class NumericInput:
    value: int
    min_val: int
    max_val: int
    font: pygame.font.Font
    on_commit: Callable[[int], None] | None = None
    text: str = field(init=False)
    active: bool = field(default=False, init=False)
    rect: pygame.Rect | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.text = str(self.value)

    def clamp(self, val: int) -> int:
        return max(self.min_val, min(self.max_val, val))

    def set_value(self, val: int) -> None:
        self.value = self.clamp(val)
        self.text = str(self.value)

    def commit(self) -> bool:
        try:
            parsed = int(self.text)
        except ValueError:
            parsed = self.value
        clamped = self.clamp(parsed)
        changed = clamped != self.value
        self.value = clamped
        self.text = str(self.value)
        if changed and self.on_commit is not None:
            self.on_commit(self.value)
        return changed

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            was_active = self.active
            self.active = self.rect is not None and self.rect.collidepoint(event.pos)
            if was_active and not self.active:
                self.commit()
            return was_active or self.active
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.commit()
                self.active = False
            elif event.unicode.isdigit():
                self.text += event.unicode
            return True
        return False

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self.rect = rect
        border = BUTTON_HOVER_COLOR if self.active else TEXT_COLOR
        pygame.draw.rect(surface, BUTTON_COLOR, rect, border_radius=3)
        pygame.draw.rect(surface, border, rect, 2, border_radius=3)
        label = self.font.render(self.text or "0", True, TEXT_COLOR)
        surface.blit(label, label.get_rect(center=rect.center))


@dataclass
class Slider:
    label: str
    min_val: float
    max_val: float
    value: float
    font: pygame.font.Font
    format_fn: Callable[[float], str] = str
    live_callback: Callable[[float], None] | None = None
    release_callback: Callable[[float], None] | None = None
    rect: pygame.Rect | None = field(default=None, init=False)
    dragging: bool = field(default=False, init=False)
    hovered: bool = field(default=False, init=False)

    TRACK_HEIGHT = 6
    HANDLE_RADIUS = 10
    ACTIVE_HANDLE_RADIUS = 12

    def _handle_x(self) -> float:
        if self.rect is None or self.max_val <= self.min_val:
            return 0.0 if self.rect is None else float(self.rect.x)
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + ratio * self.rect.width

    def _value_at(self, mouse_x: int) -> float:
        if self.rect is None or self.rect.width == 0:
            return self.value
        ratio = max(0.0, min(1.0, (mouse_x - self.rect.x) / self.rect.width))
        return self.min_val + ratio * (self.max_val - self.min_val)

    def _near_handle(self, pos: tuple[int, int]) -> bool:
        if self.rect is None:
            return False
        distance = math.hypot(pos[0] - self._handle_x(), pos[1] - self.rect.centery)
        return distance <= self.ACTIVE_HANDLE_RADIUS

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.rect is None:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self._near_handle(event.pos)
            if self.dragging:
                self.value = self._value_at(event.pos[0])
                if self.live_callback is not None:
                    self.live_callback(self.value)
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._near_handle(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
                self.value = self._value_at(event.pos[0])
                if self.live_callback is not None:
                    self.live_callback(self.value)
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            if self.release_callback is not None:
                self.release_callback(self.value)
            return True
        return False

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self.rect = rect
        track_y = rect.centery
        pygame.draw.rect(
            surface,
            SCROLLBAR_TRACK,
            (rect.x, track_y - self.TRACK_HEIGHT // 2, rect.width, self.TRACK_HEIGHT),
            border_radius=3,
        )
        handle_x = self._handle_x()
        pygame.draw.rect(
            surface,
            ACCENT_PRIMARY,
            (rect.x, track_y - self.TRACK_HEIGHT // 2, handle_x - rect.x, self.TRACK_HEIGHT),
            border_radius=3,
        )
        active = self.dragging or self.hovered
        radius = self.ACTIVE_HANDLE_RADIUS if active else self.HANDLE_RADIUS
        handle_color = (130, 133, 255) if self.dragging else ACCENT_PRIMARY
        pygame.draw.circle(surface, handle_color, (int(handle_x), track_y), radius)
        pygame.draw.circle(surface, TEXT_COLOR, (int(handle_x), track_y), radius, 2)
        if self.dragging:
            tooltip = self.font.render(self.format_fn(self.value), True, TEXT_COLOR)
            tooltip_rect = tooltip.get_rect(midbottom=(int(handle_x), track_y - radius - 6))
            pygame.draw.rect(surface, (50, 52, 65), tooltip_rect.inflate(12, 8), border_radius=4)
            surface.blit(tooltip, tooltip_rect)


@dataclass
class Dropdown:
    options: list[str]
    font: pygame.font.Font
    selected: str | None = None
    on_commit: Callable[[str], None] | None = None
    active: bool = field(default=False, init=False)
    scroll_y: int = field(default=0, init=False)
    rect: pygame.Rect | None = field(default=None, init=False)
    _option_rects: list[pygame.Rect] = field(default_factory=list, init=False)

    ITEM_HEIGHT = 28
    MAX_HEIGHT = 200
    SCROLL_SPEED = 20

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.rect is None:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self.active:
                if self.rect.collidepoint(event.pos):
                    self.active = True
                    return True
                return False
            for option, option_rect in zip(self.options, self._option_rects, strict=False):
                if option_rect.collidepoint(event.pos):
                    self.selected = option
                    self.active = False
                    if self.on_commit is not None:
                        self.on_commit(option)
                    return True
            self.active = False
            return True
        if self.active and event.type == pygame.MOUSEWHEEL:
            max_scroll = max(0, len(self.options) * self.ITEM_HEIGHT - self.MAX_HEIGHT)
            self.scroll_y = max(0, min(max_scroll, self.scroll_y - event.y * self.SCROLL_SPEED))
            return True
        return False

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """Draws only the closed box. Callers must also call `draw_options` — after
        everything else in the panel — so an open list paints on top instead of being
        overdrawn by whatever rows come after this one in layout order."""
        self.rect = rect
        pygame.draw.rect(surface, BUTTON_COLOR, rect, border_radius=4)
        border = TEXT_COLOR if self.active else BUTTON_HOVER_COLOR
        pygame.draw.rect(surface, border, rect, 2, border_radius=4)
        label_text = self.selected if self.selected is not None else "Select..."
        label = self.font.render(label_text, True, TEXT_COLOR)
        surface.blit(label, label.get_rect(midleft=(rect.x + 8, rect.centery)))

    def draw_options(self, surface: pygame.Surface) -> None:
        self._option_rects = []
        if not self.active or self.rect is None:
            return
        rect = self.rect

        list_height = min(self.MAX_HEIGHT, len(self.options) * self.ITEM_HEIGHT)
        list_rect = pygame.Rect(rect.x, rect.bottom, rect.width, list_height)
        pygame.draw.rect(surface, PANEL_BACKGROUND_COLOR, list_rect)
        pygame.draw.rect(surface, BUTTON_COLOR, list_rect, 1)

        clip = surface.get_clip()
        surface.set_clip(list_rect)
        mouse_pos = pygame.mouse.get_pos()
        for i, option in enumerate(self.options):
            row_y = rect.bottom + i * self.ITEM_HEIGHT - self.scroll_y
            option_rect = pygame.Rect(rect.x, row_y, rect.width, self.ITEM_HEIGHT)
            if option_rect.bottom < list_rect.top or option_rect.top > list_rect.bottom:
                continue
            if option_rect.collidepoint(mouse_pos):
                pygame.draw.rect(surface, BUTTON_HOVER_COLOR, option_rect)
            option_label = self.font.render(option, True, TEXT_COLOR)
            label_pos = (option_rect.x + 8, option_rect.centery)
            surface.blit(option_label, option_label.get_rect(midleft=label_pos))
            self._option_rects.append(option_rect)
        surface.set_clip(clip)


@dataclass
class CollapsibleSection:
    title: str
    font: pygame.font.Font
    expanded: bool = True
    header_rect: pygame.Rect | None = field(default=None, init=False)

    HEADER_HEIGHT = 32

    def toggle(self) -> None:
        self.expanded = not self.expanded

    def handle_event(self, event: pygame.event.Event) -> bool:
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.header_rect is not None
            and self.header_rect.collidepoint(event.pos)
        ):
            self.toggle()
            return True
        return False

    def draw_header(self, surface: pygame.Surface, x: int, y: int, width: int) -> int:
        self.header_rect = pygame.Rect(x, y, width, self.HEADER_HEIGHT)
        hovered = self.header_rect.collidepoint(pygame.mouse.get_pos())
        fill = BUTTON_HOVER_COLOR if hovered else BUTTON_COLOR
        pygame.draw.rect(surface, fill, self.header_rect, border_radius=6)
        indicator = "▼" if self.expanded else "▶"
        label = self.font.render(f"{indicator} {self.title}", True, TEXT_COLOR)
        surface.blit(label, label.get_rect(midleft=(x + 10, self.header_rect.centery)))
        return self.HEADER_HEIGHT
