"""Widget event and drawing contracts (ui/widgets.py), headless under the SDL dummy driver."""

import numpy as np
import pygame
import pytest

from latticelife.ui import widgets
from latticelife.ui.widgets import (
    CollapsibleSection,
    Dropdown,
    NumericInput,
    Slider,
    TextInput,
)


@pytest.fixture
def font():
    pygame.init()
    return pygame.font.Font(None, 16)


def _down(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


def _key(key, unicode=""):
    return pygame.event.Event(pygame.KEYDOWN, key=key, unicode=unicode, mod=0)


def _text(text):
    return pygame.event.Event(pygame.TEXTINPUT, text=text)


def _focused_text_input(font, text="3"):
    field = TextInput(text, font)
    field.draw(pygame.Surface((300, 100)), pygame.Rect(10, 10, 80, 28))
    assert field.handle_event(_down((20, 20))) is True
    return field


def test_a_blur_click_commits_but_is_not_reported_as_handled(font):
    """F72: a click outside a focused field reported "handled", so the grid never saw it."""
    committed = []
    field = TextInput("3", font, on_commit=committed.append)
    field.draw(pygame.Surface((300, 100)), pygame.Rect(10, 10, 80, 28))
    field.handle_event(_down((20, 20)))
    assert field.active

    assert field.handle_event(_down((200, 50))) is False
    assert not field.active and committed == ["3"]

    number = NumericInput(50, 5, 500, font)
    number.draw(pygame.Surface((300, 100)), pygame.Rect(10, 10, 80, 28))
    number.handle_event(_down((20, 20)))
    number.text = "70"
    assert number.handle_event(_down((200, 50))) is False
    assert number.value == 70


def test_control_characters_are_never_typed(font):
    """F53: Tab, Delete and Esc appended '\\t', '\\x7f' and '\\x1b'."""
    field = _focused_text_input(font)
    for key, char in ((pygame.K_TAB, "\t"), (pygame.K_DELETE, "\x7f")):
        assert field.handle_event(_key(key, char)) is True
        field.handle_event(_text(char))
    assert field.text == "3"


def test_keypad_enter_commits_and_escape_cancels(font):
    field = _focused_text_input(font)
    field.handle_event(_text("4"))
    field.handle_event(_key(pygame.K_KP_ENTER, "\r"))
    assert (field.text, field.active) == ("34", False)

    field.handle_event(_down((20, 20)))
    field.handle_event(_text(",5"))
    field.handle_event(_key(pygame.K_ESCAPE, "\x1b"))
    assert (field.text, field.active) == ("34", False)

    number = NumericInput(50, 5, 500, font)
    number.draw(pygame.Surface((300, 100)), pygame.Rect(10, 10, 80, 28))
    number.handle_event(_down((20, 20)))
    number.handle_event(_key(pygame.K_BACKSPACE, "\b"))
    number.handle_event(_key(pygame.K_ESCAPE, "\x1b"))
    assert (number.text, number.value, number.active) == ("50", 50, False)


def test_allowed_chars_filter_typed_text(font):
    field = TextInput("", font, allowed_chars="0123456789, ")
    field.draw(pygame.Surface((300, 100)), pygame.Rect(10, 10, 80, 28))
    field.handle_event(_down((20, 20)))
    field.handle_event(_text("2;3²,a 4"))
    assert field.text == "23, 4"


def test_long_text_is_clipped_to_its_box(font):
    """F53: a long rule drew under the R button and past the panel edge."""
    surface = pygame.Surface((300, 60))
    surface.fill((1, 2, 3))
    field = TextInput("0,1,2,3,4,5,6,7,8,9,10,11,12", font)
    for active in (False, True):
        field.active = active
        field.draw(surface, pygame.Rect(10, 10, 60, 28))
        right = pygame.surfarray.array3d(surface)[70:, :]
        assert np.all(right == (1, 2, 3))


def test_numeric_input_clamps_its_starting_value(font):
    """F49: a count of 12 showed as 12 while Apply would make it 10."""
    assert NumericInput(12, 1, 10, font).text == "10"
    assert NumericInput(0, 1, 10, font).value == 1


@pytest.mark.parametrize(("start", "expected"), [(120, 60), (1000, 60), (0, 1), (-5, 1)])
def test_slider_clamps_its_starting_value_and_stays_on_its_track(font, start, expected):
    """F50: fps 120 drew the handle across the grid; fps 0 drew it left of the track."""
    slider = Slider("FPS", 1, 60, start, font)
    assert slider.value == expected
    slider.draw(pygame.Surface((400, 100)), pygame.Rect(100, 10, 180, 28))
    assert slider.rect.left <= slider._handle_x() <= slider.rect.right


def test_slider_snaps_to_its_step_and_reaches_both_ends(font):
    slider = Slider("FPS", 1, 60, 30, font, step=1)
    slider.draw(pygame.Surface((400, 100)), pygame.Rect(100, 10, 180, 28))
    slider.handle_event(_down((slider.rect.right - 1, slider.rect.centery)))
    assert slider.value == 60
    slider.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(0, 0), button=1))
    slider.handle_event(_down((slider.rect.left, slider.rect.centery)))
    assert slider.value == 1


def test_a_lost_button_release_ends_the_drag(font):
    """F71: if the release never arrived (focus stolen mid-drag), the handle followed plain
    mouse movement and the next Apply saved that value."""
    changes = []
    slider = Slider("FPS", 1, 60, 30, font, step=1, live_callback=changes.append)
    slider.draw(pygame.Surface((400, 100)), pygame.Rect(100, 10, 180, 28))
    centre = (int(slider._handle_x()), slider.rect.centery)

    slider.handle_event(_down(centre))
    slider.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    assert not slider.dragging
    slider.handle_event(
        pygame.event.Event(pygame.MOUSEMOTION, pos=(260, 20), rel=(0, 0), buttons=(0, 0, 0))
    )
    assert slider.value == 30

    slider.handle_event(_down(centre))
    slider.handle_event(
        pygame.event.Event(pygame.MOUSEMOTION, pos=(260, 20), rel=(0, 0), buttons=(0, 0, 0))
    )
    assert not slider.dragging and slider.value == 30

    slider.handle_event(_down(centre))
    slider.handle_event(
        pygame.event.Event(pygame.MOUSEMOTION, pos=(260, 20), rel=(0, 0), buttons=(1, 0, 0))
    )
    assert slider.dragging and slider.value > 30
    assert changes and changes[-1] == slider.value


def test_scrolled_dropdown_selects_the_option_drawn_under_the_click(font):
    """F52: rects were paired with options from index 0, so once scrolled a click picked an
    option further up; a click just below the list picked a hidden option."""
    options = [f"opt{i}" for i in range(12)]
    dropdown = Dropdown(options, font, selected="opt0")
    surface = pygame.Surface((300, 600))
    dropdown.draw(surface, pygame.Rect(10, 10, 150, 24))
    dropdown.handle_event(_down((20, 20)))
    dropdown.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-3, flipped=False))
    assert dropdown.scroll_y == 60
    dropdown.draw_options(surface)

    shown = dict(dropdown._option_rects)
    assert "opt0" not in shown and "opt3" in shown
    for rect in shown.values():
        assert dropdown._list_rect.contains(rect)
    dropdown.handle_event(_down(shown["opt4"].center))
    assert dropdown.selected == "opt4" and not dropdown.active

    dropdown.handle_event(_down((20, 20)))
    dropdown.draw_options(surface)
    assert dropdown.scroll_y == 0  # reopens at the top
    below = (20, dropdown._list_rect.bottom + 5)
    assert dropdown.handle_event(_down(below)) is True  # the open list takes the click
    assert dropdown.selected == "opt4" and not dropdown.active


def test_dropdown_opens_upward_when_there_is_no_room_below(font):
    dropdown = Dropdown(["a", "b", "c"], font)
    surface = pygame.Surface((300, 200))
    dropdown.draw(surface, pygame.Rect(10, 170, 150, 24))
    dropdown.handle_event(_down((20, 180)))
    dropdown.draw_options(surface)
    assert dropdown._list_rect.bottom == 170


class _RecordingFont:
    """A font that remembers every string it renders."""

    def __init__(self, font):
        self.font, self.rendered = font, []

    def render(self, text, *args):
        self.rendered.append(text)
        return self.font.render(text, *args)


def test_collapsed_indicator_is_drawn_not_a_missing_glyph(font):
    """F54: '▶' is missing from Arial, the usual fallback for 'Segoe UI', and showed as a box.
    The indicator is now a filled triangle drawn without any font."""
    surface = pygame.Surface((300, 40))
    for expanded, centroid in ((True, (14, 15)), (False, (13, 16))):
        surface.fill((0, 0, 0))
        recording = _RecordingFont(font)
        CollapsibleSection("Grid", recording, expanded=expanded).draw_header(surface, 0, 0, 300)
        assert recording.rendered == ["Grid"]
        assert tuple(surface.get_at(centroid))[:3] == widgets.TEXT_COLOR
