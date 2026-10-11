"""Interactive pygame shell (§6.14). Drives a `SimulationSession` — zero simulation logic lives
here; only presentation. `LayoutManager`/`FontManager`/action-bar/stat-card/hints concepts are
mined from the legacy `game.py`, rebuilt against the new session/config instead of the discarded
Game/grid classes. Single exit path: `run()`'s loop just returns when `step_frame()` reports the
app should stop — the interpreter-exit builtin never appears on this path, and the 3D torus
renderer is imported lazily, only inside the V-toggle handler.

Frame loop. The window handles input and redraws at a steady `UI_FPS`, whatever the simulation
speed. The simulation runs on its own clock — `config.ui.fps` generations per second, which the
panel's FPS slider changes live — fed by a time accumulator, so a slow simulation never makes
clicks and typing lag. A frame that resets or replaces the session draws generation 0 before
any step.

Editing (2D). A grid click is mapped to a cell by the same `render.GridLayout` that drew it and
applied through `SimulationSession.set_cell`, so growth baselines and the sudden-death window
see the edit. The click places the chosen species — number keys 1-9 and 0 (species 10), or a
click on a species' colour in the settings panel, choose it — and clears a cell that already
holds it. An edit made while running holds the simulation for `EDIT_HOLD_SECONDS`, so the edit is
seen before the rule acts on it (legacy click parity).

New (N) gives every current species a fresh random rule sized to the shape's neighbour count and
re-seeds the grid; Reset (R) keeps the rules and re-seeds. Reset, New and Apply keep a pause the
user chose, so generation 0 can be inspected and single-stepped, and lift one a sudden-death
alert imposed. The help overlay ('?' on any keyboard
layout, or F1) is modal: while it is open, input does nothing else, and a click closes it.

3D view (V, `[torus]` extra). The grid area shows the grid on a torus (`ui/torus/view.py`),
rendered offscreen and blitted, so the window stays a normal pygame surface and the panel, chart
and overlays work unchanged. It follows the live session every frame — Apply, New, Reset and
window resizes need no rebuild. A drag on the torus rotates it and the wheel over it zooms;
grid clicks edit cells in 2D only. Leaving 3D (V again, or exiting) releases every GL object.
If 3D cannot start, or fails while drawing, the app stays in (or returns to) 2D and a short
status line between the chart and the grid says why for `STATUS_SECONDS`.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pygame

from latticelife.config import DEFAULT_CONFIG_PATH, MIN_FPS, Config, load_config
from latticelife.core.geometries import max_degree
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.session.alerts import Alert, AlertKind
from latticelife.session.session import SimulationSession
from latticelife.ui import render
from latticelife.ui.panel import SettingsPanel, ui_rng
from latticelife.ui.widgets import Button, blit_clipped

if TYPE_CHECKING:  # never imported at run time: the [torus] extra is optional
    from latticelife.ui.torus.view import TorusView

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# The input/redraw rate, independent of the simulation speed (config.ui.fps).
UI_FPS = 60
# At most this many generations per frame; a backlog beyond it is dropped, not replayed.
MAX_STEPS_PER_FRAME = 2
# A longer frame (a stall, a dragged window) counts as this long.
MAX_FRAME_SECONDS = 0.25
# How long the simulation waits after a grid edit made while it runs.
EDIT_HOLD_SECONDS = 0.25
# Key repeat (delay, interval in ms) while a settings field has focus.
TEXT_KEY_REPEAT = (400, 35)
# How long the 3D view's status message stays up.
STATUS_SECONDS = 6.0
# Half the space between the alert message and the 3D status when both share the status strip.
STATUS_GAP = 8

BACKGROUND_COLOR = (18, 18, 24)
PANEL_BACKGROUND_COLOR = (24, 26, 32)
TEXT_COLOR = (235, 235, 245)
CHART_BACKGROUND = (28, 30, 38)
CHART_GRID_LINES = (45, 48, 58)
CHART_AXIS_COLOR = (128, 130, 145)
ACCENT_PRIMARY = (99, 102, 241)
ACCENT_DANGER = (248, 113, 113)

FONT_NAME = "Segoe UI"
FONT_SIZES = {"xs": 10, "sm": 12, "base": 14, "lg": 16, "xl": 20, "2xl": 24}

SHORTCUTS = [
    ("Space", "Play / pause"),
    ("R", "Reset: same rules, new random grid"),
    ("N", "New: random rules for every species"),
    ("S", "Step forward (when paused)"),
    ("Click", "Place / remove the chosen species (2D)"),
    ("1-9, 0", "Choose the species clicks place"),
    ("V", "Toggle 2D / 3D torus view"),
    ("Drag, wheel", "Rotate / zoom the torus (3D)"),
    ("Wheel", "Scroll the settings panel"),
    ("? or F1", "Toggle this help"),
]

# Number keys by keycode, for when no digit character came with the key (US-style Shift+1,
# the keypad with Num Lock off); `App._species_key` reads the typed digit first.
_SPECIES_KEYS = {
    **{getattr(pygame, f"K_{d}"): d or 10 for d in range(10)},
    **{getattr(pygame, f"K_KP{d}"): d or 10 for d in range(10)},
}
_SHORTCUT_BLOCKING_MODS = pygame.KMOD_CTRL | pygame.KMOD_ALT | pygame.KMOD_GUI


class LayoutManager:
    """Ported from the legacy `game.py:LayoutManager` — same regions, same ratios."""

    MIN_PANEL_WIDTH = 240
    MAX_PANEL_WIDTH = 400
    PANEL_WIDTH_RATIO = 0.25

    def __init__(self, window_width: int, window_height: int) -> None:
        self.update(window_width, window_height)

    def update(self, window_width: int, window_height: int) -> None:
        self.window_width = window_width
        self.window_height = window_height

        self.panel_width = max(
            self.MIN_PANEL_WIDTH,
            min(self.MAX_PANEL_WIDTH, int(window_width * self.PANEL_WIDTH_RATIO)),
        )

        margin = 16
        self.chart_x = self.panel_width + margin
        self.chart_y = margin
        self.chart_width = window_width - self.chart_x - margin
        self.chart_height = min(180, int(window_height * 0.25))

        self.grid_x = self.panel_width + margin
        self.grid_y = self.chart_y + self.chart_height + margin + 20
        self.grid_width = window_width - self.grid_x - margin
        self.grid_height = window_height - self.grid_y - margin

    def panel_rect(self) -> pygame.Rect:
        return pygame.Rect(0, 0, self.panel_width, self.window_height)

    def chart_rect(self) -> pygame.Rect:
        return pygame.Rect(self.chart_x, self.chart_y, self.chart_width, self.chart_height)

    def grid_rect(self) -> pygame.Rect:
        return pygame.Rect(self.grid_x, self.grid_y, self.grid_width, self.grid_height)


class FontManager:
    """Ported from the legacy `game.py:FontManager` — same named tiers."""

    def __init__(self) -> None:
        self.fonts = {
            key: pygame.font.SysFont(FONT_NAME, size, bold=key in ("xl", "2xl"))
            for key, size in FONT_SIZES.items()
        }
        self.fonts["mono"] = pygame.font.SysFont("Consolas", FONT_SIZES["sm"])

    def get(self, size: str = "base") -> pygame.font.Font:
        return self.fonts.get(size, self.fonts["base"])


def _default_lifeforms(config: Config) -> list[Lifeform]:
    conway = Rule(birth={3}, survive={2, 3})
    return [Lifeform(id=i, rule=conway) for i in range(1, config.simulation.n_lifeforms + 1)]


def _fit_text(font: pygame.font.Font, text: str, max_width: int) -> str:
    """`text`, cut short with an ellipsis if it would be wider than `max_width` pixels."""
    if font.size(text)[0] <= max_width:
        return text
    while text and font.size(text + "…")[0] > max_width:
        text = text[:-1]
    return text + "…"


def _short_error(exc: BaseException) -> str:
    """The first line of an exception's message (its type name if it has none)."""
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__


def _alert_label(alert: Alert | None) -> str:
    """Short card label for an alert; the full message is drawn in the status strip, between
    the chart and the grid."""
    if alert is None:
        return "None"
    return "Sudden death" if alert.kind is AlertKind.SUDDEN_DEATH else "Stasis"


def _stat_values(session: SimulationSession) -> list[tuple[str, str]]:
    """(label, value) of each stat card."""
    return [
        ("Generation", str(session.generation)),
        # The live grid, so a cell edited while paused shows at once.
        ("Population", str(int(np.count_nonzero(session.grid.occupancy)))),
        ("Status", "Paused" if session.paused else "Running"),
        ("Alert", _alert_label(session.alert)),
    ]


def _draw_stat_cards(
    surface: pygame.Surface,
    fonts: FontManager,
    x: int,
    y: int,
    width: int,
    session: SimulationSession,
) -> int:
    stats = _stat_values(session)
    card_width = (width - 8) // 2
    card_height = 56
    for i, (label, value) in enumerate(stats):
        col, row = i % 2, i // 2
        rect = pygame.Rect(
            x + col * (card_width + 8), y + row * (card_height + 8), card_width, card_height
        )
        pygame.draw.rect(surface, PANEL_BACKGROUND_COLOR, rect, border_radius=6)
        pygame.draw.rect(surface, CHART_GRID_LINES, rect, 1, border_radius=6)
        color = ACCENT_DANGER if label == "Alert" and session.alert is not None else TEXT_COLOR
        text_width = rect.width - 16
        label_surf = fonts.get("xs").render(
            _fit_text(fonts.get("xs"), label, text_width), True, CHART_AXIS_COLOR
        )
        value_surf = fonts.get("lg").render(
            _fit_text(fonts.get("lg"), value, text_width), True, color
        )
        inner = rect.inflate(-2, -2)
        blit_clipped(surface, label_surf, (rect.x + 8, rect.y + 6), inner)
        blit_clipped(surface, value_surf, (rect.x + 8, rect.y + 22), inner)
    rows = (len(stats) + 1) // 2
    return rows * (card_height + 8)


def _draw_population_chart(
    surface: pygame.Surface, rect: pygame.Rect, session: SimulationSession, palette
) -> None:
    """One line per lifeform, colored to match its grid palette entry, sharing a y-axis
    scaled to total population (every species's count is <= the total at any generation)."""
    pygame.draw.rect(surface, CHART_BACKGROUND, rect, border_radius=6)
    pygame.draw.rect(surface, CHART_GRID_LINES, rect, 1, border_radius=6)
    records = session.history_records
    if len(records) < 2:
        return
    n = len(records)
    # The plotted generations' own totals: `history_alive` also holds the unplotted
    # generation-0 total, which would squash every line under a crowded start.
    max_v = max(record.grid_metrics.total_alive for record in records) or 1

    def to_points(values: list[int]) -> list[tuple[float, float]]:
        return [
            (rect.x + i / (n - 1) * rect.width, rect.bottom - (v / max_v) * (rect.height - 8) - 4)
            for i, v in enumerate(values)
        ]

    for lf in session.lifeforms:
        per_species = [record.lifeform_metrics[lf.id].alive for record in records]
        color = tuple(int(c) for c in palette[lf.id])
        pygame.draw.lines(surface, color, False, to_points(per_species), 2)


def _draw_alert_banner(
    surface: pygame.Surface,
    fonts: FontManager,
    strip: pygame.Rect,
    alert: Alert | None,
    max_width: int,
) -> None:
    """The full alert message (the stat card only names it), at the left of `strip` — the
    status strip between the chart and the grid — in a box at most `max_width` px wide. Drawn
    over the chart instead, it hid the lines' early peaks for as long as the alert lasted."""
    if alert is None:
        return
    font = fonts.get("sm")
    text = font.render(_fit_text(font, alert.message, max_width - 12), True, ACCENT_DANGER)
    box = text.get_rect().inflate(12, 6)
    box.midleft = strip.midleft
    pygame.draw.rect(surface, CHART_BACKGROUND, box, border_radius=4)
    surface.blit(text, text.get_rect(center=box.center))


def _draw_keyboard_hints(
    surface: pygame.Surface,
    fonts: FontManager,
    visible: bool,
    window_width: int,
    window_height: int,
    footer: str = "Press ? for shortcuts",
) -> None:
    if not visible:
        hint = fonts.get("xs").render(footer, True, CHART_AXIS_COLOR)
        hint_pos = (window_width - hint.get_width() - 12, window_height - hint.get_height() - 8)
        surface.blit(hint, hint_pos)
        return

    overlay = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    row_height = 28
    panel_width, panel_height = 420, len(SHORTCUTS) * row_height + 84
    panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
    panel_rect.center = surface.get_rect().center
    pygame.draw.rect(surface, PANEL_BACKGROUND_COLOR, panel_rect, border_radius=10)
    pygame.draw.rect(surface, CHART_GRID_LINES, panel_rect, 1, border_radius=10)

    title = fonts.get("lg").render("Keyboard Shortcuts", True, TEXT_COLOR)
    surface.blit(title, (panel_rect.x + 20, panel_rect.y + 16))
    for i, (key, description) in enumerate(SHORTCUTS):
        row_y = panel_rect.y + 52 + i * row_height
        key_surf = fonts.get("sm").render(key, True, ACCENT_PRIMARY)
        desc_surf = fonts.get("sm").render(description, True, TEXT_COLOR)
        surface.blit(key_surf, (panel_rect.x + 20, row_y))
        surface.blit(desc_surf, (panel_rect.x + 110, row_y))
    close = fonts.get("xs").render("Click, Esc or ? to close", True, CHART_AXIS_COLOR)
    surface.blit(close, close.get_rect(bottomright=(panel_rect.right - 16, panel_rect.bottom - 12)))


class App:
    def __init__(self, config: Config) -> None:
        # The 3D view renders with an OpenGL context of its own. On some Linux GPUs SDL would
        # present this (non-GL) window through a hidden GL context, and the two would compete
        # for the thread; this keeps SDL on its plain framebuffer (unless the user set it).
        os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")
        pygame.init()
        pygame.display.set_caption("LatticeLife")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        # Typed characters reach the settings fields as TEXTINPUT (keyboard layouts and input
        # methods included), and '?' and the species digits are recognised from their translated
        # characters on any layout.
        pygame.key.start_text_input()

        self.config = config
        # Random rules for New and the panel's R buttons (reproducible from a nonzero seed).
        self.ui_rng = ui_rng(config.simulation.seed)
        self.lifeforms = _default_lifeforms(config)
        self.session = SimulationSession(config, self.lifeforms)

        self.layout = LayoutManager(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.fonts = FontManager()

        self.hints_visible = False
        self.view_mode = "2D"
        # The 3D view (a `ui.torus.view.TorusView`) while view_mode is "3D", else None, and the
        # status message about it with the seconds it has left on screen.
        self.torus_view: TorusView | None = None
        self.torus_status: str | None = None
        self._torus_status_left = 0.0

        self.palette = render.build_palette(self.lifeforms)
        self.geometric_renderer = render.GeometricRenderer(BACKGROUND_COLOR)
        # The species id a grid click places.
        self.selected_species = 1

        # Simulation time owed (s), the last frame's length (s), and the edit hold left (s).
        self._sim_time = 0.0
        self._frame_seconds = 0.0
        self._edit_hold = 0.0
        self._key_repeat_on = False
        self._text_input_rect: pygame.Rect | None = None

        self.settings_panel = SettingsPanel(
            self.config,
            self.lifeforms,
            self.fonts.get("sm"),
            self._apply_settings,
            on_fps_change=self._set_fps,
            on_select_species=self._select_species,
            rng=self.ui_rng,
        )
        self._sync_panel_species()

        action_font = self.fonts.get("sm")
        empty_rect = pygame.Rect(0, 0, 0, 0)
        self.play_button = Button(empty_rect, "Pause", action_font, on_click=self._toggle_pause)
        self.reset_button = Button(empty_rect.copy(), "Reset", action_font, on_click=self._reset)
        self.step_button = Button(empty_rect.copy(), "Step", action_font, on_click=self._step_once)
        self.new_button = Button(empty_rect.copy(), "New", action_font, on_click=self._new_random)

    # --- actions ---------------------------------------------------------

    def _toggle_pause(self) -> None:
        self.session.paused = not self.session.paused
        self._sim_time = 0.0

    def _session_changed(self) -> None:
        """After a reset or a new session: draw generation 0 before stepping, repaint fully."""
        self._sim_time = 0.0
        self._frame_seconds = 0.0  # this frame owes no time to the new session
        self._edit_hold = 0.0
        self.geometric_renderer.reset()
        self._set_torus_status(None)  # the user has moved on

    def _reset(self) -> None:
        self.session.reset()
        self._session_changed()

    def _step_once(self) -> None:
        if self.session.paused:
            self.session.step()

    def _replace_session(
        self, config: Config, lifeforms: list[Lifeform], seed: int | None = None
    ) -> None:
        # The run state carries over as in `SimulationSession.reset()` (F48): a pause the user
        # chose is kept, so the new generation 0 can be inspected and single-stepped; one a
        # sudden-death alert imposed belonged to the old run and is lifted.
        old = self.session
        paused = old.paused and not old.paused_by_alert
        self.config = config
        self.lifeforms = lifeforms
        self.session = SimulationSession(config, lifeforms, seed=seed, paused=paused)
        self.palette = render.build_palette(lifeforms)
        if not 1 <= self.selected_species <= len(lifeforms):
            self.selected_species = 1
        self._sync_panel_species()
        self._session_changed()

    def _new_random(self) -> None:
        """New: a fresh random rule for every current species, sized to the shape's neighbour
        count, on a newly seeded grid; the panel's rule fields show the new rules."""
        grid, discovery = self.config.grid, self.config.discovery
        n = max_degree(grid.shape, grid.triangle_mode)
        lifeforms = [
            Lifeform(lf.id, Rule.random(n, discovery.p_birth, discovery.p_survive, self.ui_rng))
            for lf in self.lifeforms
        ]
        # A fixed config seed would put every New on the same grid; derive a fresh, still
        # reproducible, seed from the UI stream instead. Seed 0 stays "fresh entropy".
        seed = int(self.ui_rng.integers(1, 2**63)) if self.config.simulation.seed else None
        self._replace_session(self.config, lifeforms, seed)
        self.settings_panel.set_lifeforms(lifeforms)

    def _apply_settings(self, new_config: Config, lifeforms: list[Lifeform]) -> None:
        self._replace_session(new_config, lifeforms)

    def _set_fps(self, fps: int) -> None:
        """Live speed change from the FPS slider: the running session is kept."""
        self.config = replace(self.config, ui=replace(self.config.ui, fps=fps))

    def _select_species(self, species_id: int) -> None:
        if 1 <= species_id <= len(self.lifeforms):
            self.selected_species = species_id
            self.settings_panel.selected_species = species_id

    def _sync_panel_species(self) -> None:
        panel = self.settings_panel
        panel.species_colors = [tuple(int(c) for c in self.palette[lf.id]) for lf in self.lifeforms]
        panel.selected_species = self.selected_species

    def _set_torus_status(self, message: str | None) -> None:
        self.torus_status = message
        self._torus_status_left = STATUS_SECONDS if message is not None else 0.0

    def _toggle_view_mode(self) -> None:
        if self.view_mode == "3D":
            self._leave_3d()
            return

        try:
            # Lazy: the [torus] extra (moderngl) is optional and only needed from here on.
            from latticelife.ui.torus.view import TorusView
        except ImportError:
            self._set_torus_status(
                "3D view unavailable: install the [torus] extra (uv sync --extra torus)."
            )
            return

        try:
            view = TorusView(BACKGROUND_COLOR)
        except Exception as exc:  # no OpenGL 3.3, a driver that rejects the shaders, ...
            self._set_torus_status(f"3D view unavailable: {_short_error(exc)}")
            return

        self.torus_view = view
        self.view_mode = "3D"
        self._set_torus_status(None)

    def _leave_3d(self, status: str | None = None) -> None:
        """Back to 2D, releasing the 3D view's GL resources."""
        view, self.torus_view = self.torus_view, None
        self.view_mode = "2D"
        self._set_torus_status(status)
        if view is not None:
            view.release()

    def close(self) -> None:
        """Release what the app holds outside pygame (the 3D view's GL context)."""
        if self.torus_view is not None:
            self._leave_3d()

    # --- events ------------------------------------------------------------

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.layout.update(event.w, event.h)
                self.geometric_renderer.reset()
                continue
            if self.hints_visible and self._handle_help_event(event):
                continue

            # One click, one target. A focused settings field that a click is not on commits
            # and blurs but does not report the click as handled, so the same click still
            # reaches an action button or the grid; an open Shape list does report it, so
            # nothing underneath acts on it.
            if self.settings_panel.handle_event(event):
                continue
            buttons = (self.play_button, self.reset_button, self.step_button, self.new_button)
            if any(b.handle_event(event) for b in buttons):
                continue

            if event.type == pygame.KEYDOWN:
                if self._handle_keydown(event):
                    continue

            if self.torus_view is not None:
                # 3D: a drag on the torus rotates it, the wheel over it zooms.
                self.torus_view.handle_event(event, self.layout.grid_rect())
                continue

            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.view_mode == "2D"
            ):
                grid_rect = self.layout.grid_rect()
                if grid_rect.collidepoint(event.pos):
                    self._toggle_cell(event.pos)

        self._sync_text_entry()
        return True

    def _handle_help_event(self, event: pygame.event.Event) -> bool:
        """While the help overlay is open: '?', F1, Esc or a click closes it, and no other
        key, click, wheel or typed text acts on anything behind it. Returns True if used."""
        if event.type == pygame.KEYDOWN:
            if self._is_help_key(event) or event.key == pygame.K_ESCAPE:
                self.hints_visible = False
            return True
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.hints_visible = False
            return True
        return event.type in (pygame.MOUSEMOTION, pygame.MOUSEWHEEL, pygame.TEXTINPUT)

    @staticmethod
    def _is_help_key(event: pygame.event.Event) -> bool:
        """'?' by its typed character (any keyboard layout), US-style Shift+/ when no character
        came with the key, or F1."""
        if getattr(event, "unicode", "") == "?" or event.key == pygame.K_F1:
            return True
        return event.key == pygame.K_SLASH and bool(getattr(event, "mod", 0) & pygame.KMOD_SHIFT)

    @staticmethod
    def _species_key(event: pygame.event.Event) -> int | None:
        """The species a number key chooses ('0' is species 10), or None for any other key. The
        typed digit comes first, so layouts that type the top-row digits with Shift (AZERTY) work
        too; the keycode covers keys that came with no digit (US-style Shift+1, the keypad with
        Num Lock off)."""
        char = getattr(event, "unicode", "")
        if len(char) == 1 and "0" <= char <= "9":
            return int(char) or 10
        return _SPECIES_KEYS.get(event.key)

    def _handle_keydown(self, event: pygame.event.Event) -> bool:
        if self._is_help_key(event):
            self.hints_visible = not self.hints_visible
            return True
        if getattr(event, "mod", 0) & _SHORTCUT_BLOCKING_MODS:
            return False  # Ctrl+N, Alt+R, ... are not shortcuts
        if event.key == pygame.K_SPACE:
            self._toggle_pause()
        elif event.key == pygame.K_r:
            self._reset()
        elif event.key == pygame.K_n:
            self._new_random()
        elif event.key == pygame.K_s and self.session.paused:
            self._step_once()
        elif event.key == pygame.K_v:
            self._toggle_view_mode()
        elif (species := self._species_key(event)) is not None:
            self._select_species(species)
        else:
            return False
        return True

    def _sync_text_entry(self) -> None:
        """Key repeat only while a settings field has focus (held Backspace deletes, but held
        Space/N/R never repeat a shortcut), and the input-method box beside that field."""
        panel = self.settings_panel
        focused = panel.has_focus()
        if focused != self._key_repeat_on:
            if focused:
                pygame.key.set_repeat(*TEXT_KEY_REPEAT)
            else:
                pygame.key.set_repeat()
            self._key_repeat_on = focused
        rect = panel.focused_rect()
        if rect is not None and rect != self._text_input_rect:
            pygame.key.set_text_input_rect(rect)
        self._text_input_rect = rect

    def _grid_layout(self) -> render.GridLayout:
        grid = self.session.grid
        return render.fit_grid(grid.shape, grid.width, grid.height, self.layout.grid_rect())

    def _toggle_cell(self, pos: tuple[int, int]) -> None:
        """Place the chosen species on the clicked cell, or clear it if it already holds it."""
        index = self._grid_layout().cell_at(*pos)
        if index is None or not self.lifeforms:
            return
        species = self.lifeforms[self.selected_species - 1].id
        target = 0 if int(self.session.grid.occupancy[index]) == species else species
        if self.session.set_cell(index, target) and not self.session.paused:
            self._edit_hold = EDIT_HOLD_SECONDS

    # --- drawing -------------------------------------------------------------

    def _footer_hint(self) -> str:
        if self.torus_view is not None:
            return "Drag to rotate · wheel to zoom · V for 2D   ·   Press ? for shortcuts"
        if len(self.lifeforms) > 1:
            return f"Clicks place species {self.selected_species}   ·   Press ? for shortcuts"
        return "Press ? for shortcuts"

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)

        panel_rect = self.layout.panel_rect()
        pygame.draw.rect(self.screen, PANEL_BACKGROUND_COLOR, panel_rect)
        cursor_y = 16
        self.play_button.text = "Play" if self.session.paused else "Pause"
        buttons = (self.play_button, self.reset_button, self.step_button, self.new_button)
        button_width = (panel_rect.width - 3 * 8 - 32) // 4
        for i, button in enumerate(buttons):
            button.rect = pygame.Rect(16 + i * (button_width + 8), cursor_y, button_width, 32)
            button.draw(self.screen)
        cursor_y += 32 + 16

        cursor_y += _draw_stat_cards(
            self.screen, self.fonts, 16, cursor_y, panel_rect.width - 32, self.session
        )
        cursor_y += 16
        self.settings_panel.draw(
            self.screen,
            16,
            cursor_y,
            panel_rect.width - 32,
            max(0, self.layout.window_height - cursor_y - 16),
        )

        chart_rect = self.layout.chart_rect()
        _draw_population_chart(self.screen, chart_rect, self.session, self.palette)

        if self.torus_view is not None:
            self._draw_torus()
        else:
            self._draw_grid()

        strip = self._status_strip()
        status = self._torus_status_text()
        alert_width = strip.width if status is None else strip.width // 2 - STATUS_GAP
        _draw_alert_banner(self.screen, self.fonts, strip, self.session.alert, alert_width)
        if status is not None:
            self.screen.blit(*status)

        _draw_keyboard_hints(
            self.screen,
            self.fonts,
            self.hints_visible,
            self.layout.window_width,
            self.layout.window_height,
            self._footer_hint(),
        )

    def _draw_grid(self) -> None:
        grid_rect = self.layout.grid_rect()
        grid = self.session.grid
        layout = self._grid_layout()
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(grid_rect)  # nothing may spill onto the panel or chart
        if grid.shape == "square":
            render.render_square(
                self.screen, grid, self.palette, layout.cell_size, layout.left, layout.top
            )
        else:
            self.geometric_renderer.draw(self.screen, grid, self.palette, layout)
        self.screen.set_clip(previous_clip)

    def _draw_torus(self) -> None:
        """The torus over the grid area; if the GL side fails, back to 2D with the reason."""
        assert self.torus_view is not None
        try:
            self.torus_view.draw(
                self.screen, self.layout.grid_rect(), self.session.grid, self.palette
            )
        except Exception as exc:  # a lost GPU, a grid larger than a GL texture, ...
            self._leave_3d(f"3D view stopped: {_short_error(exc)}")
            self._draw_grid()

    def _status_strip(self) -> pygame.Rect:
        """The strip between the chart and the grid area, as wide as the chart. The alert
        message and the 3D view's status are written here, so neither covers the chart's lines
        or the grid's cells."""
        chart = self.layout.chart_rect()
        return pygame.Rect(chart.x, chart.bottom, chart.width, self.layout.grid_y - chart.bottom)

    def _torus_status_text(self) -> tuple[pygame.Surface, pygame.Rect] | None:
        """The 3D view's status message and where it goes in the status strip: at its left, or
        at the right of its right half while the alert message takes the left half."""
        if self.torus_status is None:
            return None
        strip = self._status_strip()
        font = self.fonts.get("sm")
        if self.session.alert is None:
            text = font.render(_fit_text(font, self.torus_status, strip.width), True, ACCENT_DANGER)
            return text, text.get_rect(midleft=strip.midleft)
        half = strip.width // 2 - STATUS_GAP
        text = font.render(_fit_text(font, self.torus_status, half), True, ACCENT_DANGER)
        return text, text.get_rect(midright=strip.midright)

    # --- main loop -------------------------------------------------------------

    def _advance_simulation(self, seconds: float) -> None:
        """Run the generations owed after `seconds` of wall time at `config.ui.fps`."""
        if self.session.paused:
            self._sim_time = 0.0
            return
        if self._edit_hold > 0:
            self._edit_hold = max(0.0, self._edit_hold - seconds)
            self._sim_time = 0.0
            return
        interval = 1.0 / max(MIN_FPS, self.config.ui.fps)
        self._sim_time += seconds
        steps = 0
        while self._sim_time >= interval and steps < MAX_STEPS_PER_FRAME:
            self._sim_time -= interval
            self.session.step()
            steps += 1
            if self.session.paused:  # an alert auto-paused the session
                self._sim_time = 0.0
                return
        self._sim_time = min(self._sim_time, interval)

    def step_frame(self) -> bool:
        """Runs one frame: events, the simulation generations owed since the last frame (if
        unpaused), draw, flip, clock tick at UI_FPS. Returns False when the app should exit
        (QUIT event)."""
        running = self._handle_events()
        if running:
            self._advance_simulation(self._frame_seconds)
        self._draw()
        pygame.display.flip()
        self._frame_seconds = min(self.clock.tick(UI_FPS) / 1000.0, MAX_FRAME_SECONDS)
        if self.torus_status is not None:
            self._torus_status_left -= self._frame_seconds
            if self._torus_status_left <= 0:
                self._set_torus_status(None)
        return running

    def run(self) -> None:
        try:
            while self.step_frame():
                pass
        finally:
            self.close()


def main(config_path: str = str(DEFAULT_CONFIG_PATH)) -> None:
    config = load_config(config_path)
    App(config).run()


if __name__ == "__main__":
    main()
