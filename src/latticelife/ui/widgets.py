"""Widgets salvaged from the legacy `settings_panel.py`, trimmed: panel wiring is rebuilt
against `Config` in `ui/panel.py`, but the draw-yourself / handle-your-own-events shape of each
widget survives. `CollapsibleSection` drops the legacy `content_height`/`animation_progress`
fields — they were set at construction and never read or animated anywhere in the legacy file.

Event contract. `handle_event` returns True only when the event was meant for this widget: a click
inside it, or a key or text event while it has focus. A click outside a focused text field still
commits and blurs it, but returns False, so the same click goes on to whatever it landed on (a
button, a grid cell). Typed characters come from `pygame.TEXTINPUT`, never from `KEYDOWN.unicode`:
that is the only event an input method or on-screen keyboard delivers, and reading both would
type every character twice. `KEYDOWN` handles the editing keys only (Backspace, Enter, Esc).
While an input method composes, `pygame.TEXTEDITING` carries its uncommitted text: the focused
field shows it underlined after its own text, and types nothing until the TEXTINPUT that commits
it. (The app turns SDL text entry on only while a field has focus — see `App._sync_text_entry`.)

Every widget draws inside the rect it is given: text that does not fit is clipped, and a focused
field scrolls so its end (where typing happens) stays visible.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import pygame

TEXT_COLOR = (235, 235, 245)
PLACEHOLDER_COLOR = (128, 130, 145)
PANEL_BACKGROUND_COLOR = (24, 26, 32)
BUTTON_COLOR = (55, 58, 72)
BUTTON_HOVER_COLOR = (75, 78, 96)
BUTTON_TEXT_COLOR = (255, 255, 255)
BUTTON_BORDER_RADIUS = 6
ACCENT_PRIMARY = (99, 102, 241)
ERROR_COLOR = (248, 113, 113)
SCROLLBAR_TRACK = (35, 38, 48)

ASCII_DIGITS = "0123456789"
TEXT_PADDING = 6

_ENTER_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER)


def blit_clipped(surface: pygame.Surface, image: pygame.Surface, pos, clip: pygame.Rect) -> None:
    """Blit `image` at `pos`, drawing nothing outside `clip` (intersected with any clip the
    surface already has)."""
    previous = surface.get_clip()
    surface.set_clip(clip.clip(previous))
    surface.blit(image, pos)
    surface.set_clip(previous)


@dataclass
class Button:
    rect: pygame.Rect
    text: str
    font: pygame.font.Font
    primary: bool = False
    on_click: Callable[[], None] | None = None
    # A disabled button is drawn dimmed and takes a click without acting on it.
    enabled: bool = True

    def is_hovered(self, mouse_pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)

    def hide(self) -> None:
        """Forget the drawn position, so a button that is not on screen cannot be clicked."""
        self.rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, surface: pygame.Surface) -> None:
        if not self.enabled:
            pygame.draw.rect(surface, PANEL_BACKGROUND_COLOR, self.rect, border_radius=6)
            pygame.draw.rect(surface, BUTTON_COLOR, self.rect, 1, border_radius=6)
            label = self.font.render(self.text, True, PLACEHOLDER_COLOR)
            blit_clipped(surface, label, label.get_rect(center=self.rect.center), self.rect)
            return
        hovered = self.is_hovered(pygame.mouse.get_pos())
        fill = ACCENT_PRIMARY if self.primary else (BUTTON_HOVER_COLOR if hovered else BUTTON_COLOR)
        pygame.draw.rect(surface, fill, self.rect, border_radius=BUTTON_BORDER_RADIUS)
        pygame.draw.rect(surface, TEXT_COLOR, self.rect, 1, border_radius=BUTTON_BORDER_RADIUS)
        label = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        blit_clipped(surface, label, label.get_rect(center=self.rect.center), self.rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        is_click = event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
        if is_click and self.is_hovered(event.pos):
            if self.enabled and self.on_click is not None:
                self.on_click()
            return True
        return False


class _FocusableInput:
    """Shared focus/blur/edit handling for `TextInput` and `NumericInput`: click to focus, click
    elsewhere or Enter to commit and blur, Esc to blur and restore the text from before focus.
    `invalid` (set by the owner, e.g. a failed Apply) outlines the field in red. `composition`
    is an input method's uncommitted text, shown but not part of `text`. Subclasses provide
    `text`, `font`, `active`, `invalid`, `rect`, `composition`, `_accepts(ch)` and
    `_commit_text()`."""

    text: str
    font: pygame.font.Font
    active: bool
    invalid: bool
    rect: pygame.Rect | None
    composition: str
    _text_at_focus: str

    def _accepts(self, ch: str) -> bool:
        raise NotImplementedError

    def _commit_text(self) -> None:
        raise NotImplementedError

    def focus(self) -> None:
        if not self.active:
            self.active = True
            self._text_at_focus = self.text
            self.composition = ""

    def blur(self, commit: bool = True) -> None:
        """Lose focus. `commit=False` (Esc) puts back the text the field had when focused. An
        uncommitted composition is dropped (stopping text entry cancels it too)."""
        if not self.active:
            return
        self.active = False
        self.composition = ""
        if commit:
            self._commit_text()
        else:
            self.text = self._text_at_focus

    def hide(self) -> None:
        """Commit and blur if focused, and forget the drawn position (the field is off screen)."""
        self.blur()
        self.rect = None

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect is not None and self.rect.collidepoint(event.pos):
                self.focus()
                return True
            self.blur()
            return False
        if not self.active:
            return False
        if event.type == pygame.TEXTEDITING:
            # The whole composition so far ("" once it is committed or cancelled).
            self.composition = event.text
            return True
        if event.type == pygame.TEXTINPUT:
            self.composition = ""
            self.text += "".join(ch for ch in event.text if ch.isprintable() and self._accepts(ch))
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key in _ENTER_KEYS:
                self.blur()
            elif event.key == pygame.K_ESCAPE:
                self.blur(commit=False)
            # Any other key is swallowed: its character (if any) arrives as TEXTINPUT, and the
            # app's single-key shortcuts must not fire while a field is being typed in.
            return True
        return False

    def _draw_box(self, surface: pygame.Surface, rect: pygame.Rect, radius: int) -> None:
        if self.invalid:
            border = ERROR_COLOR
        else:
            border = ACCENT_PRIMARY if self.active else BUTTON_HOVER_COLOR
        pygame.draw.rect(surface, BUTTON_COLOR, rect, border_radius=radius)
        pygame.draw.rect(surface, border, rect, 2, border_radius=radius)

    def _draw_text(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: pygame.Surface,
        centered: bool,
    ) -> None:
        """Blit `label` inside `rect`, clipped to it, and any input-method composition after it,
        underlined. A focused field that overflows shows its end, with a caret after the text
        (and composition)."""
        inner = rect.inflate(-2 * TEXT_PADDING, -4)
        text_width = label.get_width() if self.text else 0
        preedit = self.font.render(self.composition, True, TEXT_COLOR) if self.composition else None
        shown_width = text_width + (preedit.get_width() if preedit is not None else 0)
        if shown_width > inner.width - 2:
            x = inner.right - 2 - shown_width if self.active else inner.x
        elif centered and shown_width:
            x = rect.centerx - shown_width // 2
        else:
            x = inner.x
        y = rect.centery - label.get_height() // 2
        half = label.get_height() // 2
        blit_clipped(surface, label, (x, y), inner)
        if preedit is not None:
            left = x + text_width
            blit_clipped(surface, preedit, (left, y), inner)
            underline = pygame.Rect(left, rect.centery + half, preedit.get_width(), 1)
            pygame.draw.rect(surface, TEXT_COLOR, underline.clip(inner))
        if self.active:
            caret_x = min(x + shown_width + 1, inner.right - 1)
            pygame.draw.line(
                surface, TEXT_COLOR, (caret_x, rect.centery - half), (caret_x, rect.centery + half)
            )


@dataclass
class TextInput(_FocusableInput):
    text: str
    font: pygame.font.Font
    placeholder: str = ""
    on_commit: Callable[[str], None] | None = None
    # Characters a person may type into the field (None: any printable character).
    allowed_chars: str | None = None
    active: bool = field(default=False, init=False)
    invalid: bool = field(default=False, init=False)
    rect: pygame.Rect | None = field(default=None, init=False)
    composition: str = field(default="", init=False)
    _text_at_focus: str = field(default="", init=False, repr=False)

    def _accepts(self, ch: str) -> bool:
        return self.allowed_chars is None or ch in self.allowed_chars

    def _commit_text(self) -> None:
        if self.on_commit is not None:
            self.on_commit(self.text)

    def fitted_text(self, rect: pygame.Rect) -> str:
        """The text an unfocused field shows in `rect`: all of it if it fits, else a prefix cut
        back to whole comma-separated items and marked "...". Clipping alone could end on a whole
        digit, so "0,1,2,5,7,12" read as the valid, different rule "0,1,2,5,7,1"."""
        room = rect.width - 2 * TEXT_PADDING - 2
        if self.active or self.font.size(self.text)[0] <= room:
            return self.text
        shown = self.text
        while shown and self.font.size(shown + "...")[0] > room:
            shown = shown[:-1]
        if "," in self.text:
            shown = shown[: shown.rfind(",") + 1]
        return shown + "..."

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self.rect = rect
        self._draw_box(surface, rect, radius=4)
        if self.text or self.composition:
            label = self.font.render(self.fitted_text(rect), True, TEXT_COLOR)
        else:
            label = self.font.render(self.placeholder, True, PLACEHOLDER_COLOR)
        self._draw_text(surface, rect, label, centered=False)


@dataclass
class NumericInput(_FocusableInput):
    value: int
    min_val: int
    max_val: int
    font: pygame.font.Font
    on_commit: Callable[[int], None] | None = None
    # Shown, dimmed, in place of a 0 while the field is not being typed in (e.g. "random").
    zero_text: str | None = None
    text: str = field(init=False)
    active: bool = field(default=False, init=False)
    invalid: bool = field(default=False, init=False)
    rect: pygame.Rect | None = field(default=None, init=False)
    composition: str = field(default="", init=False)
    _text_at_focus: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        # The shown value must be the one Apply will use, so clamp an out-of-range start value.
        self.value = self.clamp(self.value)
        self.text = str(self.value)

    def clamp(self, val: int) -> int:
        return max(self.min_val, min(self.max_val, val))

    def set_value(self, val: int) -> None:
        self.value = self.clamp(val)
        self.text = str(self.value)

    def _accepts(self, ch: str) -> bool:
        # ASCII only: str.isdigit() also accepts '²' and full-width digits that int() rejects
        # or silently reads.
        return ch in ASCII_DIGITS

    def _commit_text(self) -> None:
        self.commit()

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

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self.rect = rect
        self._draw_box(surface, rect, radius=3)
        if self.zero_text is not None and not self.active and self.text in ("", "0"):
            label = self.font.render(self.zero_text, True, PLACEHOLDER_COLOR)
            self._draw_text(surface, rect, label, centered=True)
            return
        label = self.font.render(self.text or ("" if self.active else "0"), True, TEXT_COLOR)
        self._draw_text(surface, rect, label, centered=True)


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
    # Snap values to multiples of `step` above min_val (None: continuous).
    step: float | None = None
    # Draw `format_fn(value)` at the right of the control at all times, not only while dragging.
    show_value: bool = False
    rect: pygame.Rect | None = field(default=None, init=False)
    dragging: bool = field(default=False, init=False)
    hovered: bool = field(default=False, init=False)

    TRACK_HEIGHT = 6
    HANDLE_RADIUS = 10
    ACTIVE_HANDLE_RADIUS = 12
    VALUE_GAP = 14

    def __post_init__(self) -> None:
        self.value = self._clamp(self.value)

    def _clamp(self, value: float) -> float:
        value = max(self.min_val, min(self.max_val, value))
        if self.step:
            value = self.min_val + round((value - self.min_val) / self.step) * self.step
            value = max(self.min_val, min(self.max_val, value))
        return value

    def _handle_x(self) -> float:
        if self.rect is None or self.max_val <= self.min_val:
            return 0.0 if self.rect is None else float(self.rect.x)
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + max(0.0, min(1.0, ratio)) * self.rect.width

    def _value_at(self, mouse_x: int) -> float:
        if self.rect is None or self.rect.width == 0:
            return self.value
        ratio = max(0.0, min(1.0, (mouse_x - self.rect.x) / self.rect.width))
        return self._clamp(self.min_val + ratio * (self.max_val - self.min_val))

    def _near_handle(self, pos: tuple[int, int]) -> bool:
        if self.rect is None:
            return False
        distance = math.hypot(pos[0] - self._handle_x(), pos[1] - self.rect.centery)
        return distance <= self.ACTIVE_HANDLE_RADIUS

    def _set_value(self, value: float) -> None:
        if value != self.value:
            self.value = value
            if self.live_callback is not None:
                self.live_callback(self.value)

    def end_drag(self) -> None:
        """Stop dragging at the current value (also used when the button release never
        arrives, e.g. the window lost focus mid-drag)."""
        if self.dragging:
            self.dragging = False
            if self.release_callback is not None:
                self.release_callback(self.value)

    def hide(self) -> None:
        self.end_drag()
        self.hovered = False
        self.rect = None

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.WINDOWFOCUSLOST:
            self.end_drag()
            return False
        if self.rect is None:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self._near_handle(event.pos)
            if self.dragging:
                if not getattr(event, "buttons", (1,))[0]:
                    # The release was lost: never let the handle follow a bare mouse move.
                    self.end_drag()
                    return False
                self._set_value(self._value_at(event.pos[0]))
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.end_drag()  # a press while still "dragging" means the last release was lost
            if self._near_handle(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
                self._set_value(self._value_at(event.pos[0]))
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.end_drag()
            return True
        return False

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw in `rect`. With `show_value`, the formatted value takes the right end of `rect`
        and the track the rest; `self.rect` is the track."""
        track = rect
        if self.show_value:
            widest = max(self.font.size(self.format_fn(v))[0] for v in (self.min_val, self.max_val))
            value_label = self.font.render(self.format_fn(self.value), True, TEXT_COLOR)
            track = pygame.Rect(
                rect.x, rect.y, max(0, rect.width - widest - self.VALUE_GAP), rect.h
            )
            label_pos = value_label.get_rect(midright=(rect.right, rect.centery))
            blit_clipped(surface, value_label, label_pos, rect)
        self.rect = track
        track_y = rect.centery
        pygame.draw.rect(
            surface,
            SCROLLBAR_TRACK,
            (self.rect.x, track_y - self.TRACK_HEIGHT // 2, self.rect.width, self.TRACK_HEIGHT),
            border_radius=3,
        )
        handle_x = self._handle_x()
        pygame.draw.rect(
            surface,
            ACCENT_PRIMARY,
            (
                self.rect.x,
                track_y - self.TRACK_HEIGHT // 2,
                handle_x - self.rect.x,
                self.TRACK_HEIGHT,
            ),
            border_radius=3,
        )
        active = self.dragging or self.hovered
        radius = self.ACTIVE_HANDLE_RADIUS if active else self.HANDLE_RADIUS
        handle_color = (130, 133, 255) if self.dragging else ACCENT_PRIMARY
        pygame.draw.circle(surface, handle_color, (int(handle_x), track_y), radius)
        pygame.draw.circle(surface, TEXT_COLOR, (int(handle_x), track_y), radius, 2)
        if self.dragging and not self.show_value:
            tooltip = self.font.render(self.format_fn(self.value), True, TEXT_COLOR)
            tooltip_rect = tooltip.get_rect(midbottom=(int(handle_x), track_y - radius - 6))
            pygame.draw.rect(surface, (50, 52, 65), tooltip_rect.inflate(12, 8), border_radius=4)
            surface.blit(tooltip, tooltip_rect)


@dataclass
class Dropdown:
    """A closed box that opens a list of options. While open it takes every click for itself:
    a click on an option selects it, any other click just closes the list — nothing underneath
    the list, or elsewhere on screen, receives that click."""

    options: list[str]
    font: pygame.font.Font
    selected: str | None = None
    on_commit: Callable[[str], None] | None = None
    # The box's text while nothing is selected (a menu that acts on choice keeps it None).
    placeholder: str = "Select..."
    active: bool = field(default=False, init=False)
    scroll_y: int = field(default=0, init=False)
    rect: pygame.Rect | None = field(default=None, init=False)
    # The options drawn last frame, each with its on-screen rect (clipped to the list).
    _option_rects: list[tuple[str, pygame.Rect]] = field(default_factory=list, init=False)
    _list_rect: pygame.Rect | None = field(default=None, init=False)

    ITEM_HEIGHT = 28
    MAX_HEIGHT = 200
    SCROLL_SPEED = 20

    def open(self) -> None:
        self.active = True
        self.scroll_y = 0

    def close(self) -> None:
        self.active = False
        self.scroll_y = 0
        self._option_rects = []
        self._list_rect = None

    def hide(self) -> None:
        self.close()
        self.rect = None

    def _max_scroll(self) -> int:
        return max(0, len(self.options) * self.ITEM_HEIGHT - self.MAX_HEIGHT)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.rect is None:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self.active:
                if self.rect.collidepoint(event.pos):
                    self.open()
                    return True
                return False
            chosen = None
            if self._list_rect is not None and self._list_rect.collidepoint(event.pos):
                chosen = next(
                    (
                        opt
                        for opt, opt_rect in self._option_rects
                        if opt_rect.collidepoint(event.pos)
                    ),
                    None,
                )
            self.close()
            if chosen is not None:
                self.selected = chosen
                if self.on_commit is not None:
                    self.on_commit(chosen)
            return True
        if not self.active:
            return False
        if event.type == pygame.MOUSEWHEEL:
            self.scroll_y = max(
                0, min(self._max_scroll(), self.scroll_y - event.y * self.SCROLL_SPEED)
            )
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.close()
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
        if self.selected is not None:
            label = self.font.render(self.selected, True, TEXT_COLOR)
        else:
            label = self.font.render(self.placeholder, True, PLACEHOLDER_COLOR)
        # The text stops short of the chevron.
        text_box = pygame.Rect(rect.x, rect.y, max(0, rect.width - 26), rect.height)
        blit_clipped(surface, label, label.get_rect(midleft=(rect.x + 8, rect.centery)), text_box)
        # Chevron at the right end, drawn rather than typed (font-independent).
        cx, cy = rect.right - 14, rect.centery
        pygame.draw.polygon(surface, TEXT_COLOR, [(cx - 4, cy - 2), (cx + 4, cy - 2), (cx, cy + 3)])

    def draw_options(self, surface: pygame.Surface, bounds: pygame.Rect | None = None) -> None:
        """Draw the open list below the box, or above it when it would run past the bottom of
        `bounds` (default: the surface) and there is room above."""
        self._option_rects = []
        self._list_rect = None
        if not self.active or self.rect is None:
            return
        rect = self.rect
        bounds = bounds if bounds is not None else surface.get_rect()

        list_height = min(self.MAX_HEIGHT, len(self.options) * self.ITEM_HEIGHT)
        top = rect.bottom
        if top + list_height > bounds.bottom and rect.top - list_height >= bounds.top:
            top = rect.top - list_height
        list_rect = pygame.Rect(rect.x, top, rect.width, list_height)
        self._list_rect = list_rect
        pygame.draw.rect(surface, PANEL_BACKGROUND_COLOR, list_rect)
        pygame.draw.rect(surface, BUTTON_COLOR, list_rect, 1)

        clip = surface.get_clip()
        surface.set_clip(list_rect)
        mouse_pos = pygame.mouse.get_pos()
        for i, option in enumerate(self.options):
            row_y = list_rect.top + i * self.ITEM_HEIGHT - self.scroll_y
            option_rect = pygame.Rect(rect.x, row_y, rect.width, self.ITEM_HEIGHT)
            visible = option_rect.clip(list_rect)
            if visible.height <= 0:
                continue
            if visible.collidepoint(mouse_pos):
                pygame.draw.rect(surface, BUTTON_HOVER_COLOR, visible)
            option_label = self.font.render(option, True, TEXT_COLOR)
            label_pos = (option_rect.x + 8, option_rect.centery)
            surface.blit(option_label, option_label.get_rect(midleft=label_pos))
            self._option_rects.append((option, visible))
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
        # The expanded/collapsed indicator is a drawn triangle: the '▶' glyph is missing from
        # common fallback fonts (Arial) and rendered as a box.
        cx, cy = x + 14, self.header_rect.centery
        if self.expanded:
            arrow = [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 4)]
        else:
            arrow = [(cx - 3, cy - 5), (cx - 3, cy + 5), (cx + 4, cy)]
        pygame.draw.polygon(surface, TEXT_COLOR, arrow)
        label = self.font.render(self.title, True, TEXT_COLOR)
        surface.blit(label, label.get_rect(midleft=(x + 26, self.header_rect.centery)))
        return self.HEADER_HEIGHT
