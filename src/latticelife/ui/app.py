"""Interactive pygame shell (§6.14). Drives a `SimulationSession` — zero simulation logic lives
here; only presentation. `LayoutManager`/`FontManager`/action-bar/stat-card/hints concepts are
mined from the legacy `game.py`, rebuilt against the new session/config instead of the discarded
Game/grid classes. Single exit path: `run()`'s loop just returns when `step_frame()` reports the
app should stop — the interpreter-exit builtin never appears on this path, and the 3D torus
renderer is imported lazily, only inside the V-toggle handler.
"""

from __future__ import annotations

import pygame

from latticelife.config import Config, load_config
from latticelife.core.lifeform import Lifeform
from latticelife.core.rules import Rule
from latticelife.session.session import SimulationSession
from latticelife.ui import render
from latticelife.ui.panel import SettingsPanel
from latticelife.ui.widgets import Button

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

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
    ("R", "Reset"),
    ("N", "New random lifeforms"),
    ("S", "Step forward (when paused)"),
    ("Click", "Toggle a cell (2D only)"),
    ("V", "Toggle 2D / 3D view"),
    ("?", "Toggle this help"),
]


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


def _draw_stat_cards(
    surface: pygame.Surface,
    fonts: FontManager,
    x: int,
    y: int,
    width: int,
    session: SimulationSession,
) -> int:
    stats = [
        ("Generation", str(session.generation)),
        ("Population", str(session.history_alive[-1]) if session.history_alive else "0"),
        ("Status", "Paused" if session.paused else "Running"),
        ("Alert", session.alert.message if session.alert is not None else "None"),
    ]
    card_width = (width - 8) // 2
    card_height = 56
    for i, (label, value) in enumerate(stats):
        col, row = i % 2, i // 2
        rect = pygame.Rect(
            x + col * (card_width + 8), y + row * (card_height + 8), card_width, card_height
        )
        pygame.draw.rect(surface, PANEL_BACKGROUND_COLOR, rect, border_radius=6)
        pygame.draw.rect(surface, CHART_GRID_LINES, rect, 1, border_radius=6)
        label_surf = fonts.get("xs").render(label, True, CHART_AXIS_COLOR)
        value_surf = fonts.get("lg").render(value, True, TEXT_COLOR)
        surface.blit(label_surf, (rect.x + 8, rect.y + 6))
        surface.blit(value_surf, (rect.x + 8, rect.y + 22))
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
    max_v = max(session.history_alive) or 1

    def to_points(values: list[int]) -> list[tuple[float, float]]:
        return [
            (rect.x + i / (n - 1) * rect.width, rect.bottom - (v / max_v) * (rect.height - 8) - 4)
            for i, v in enumerate(values)
        ]

    for lf in session.lifeforms:
        per_species = [record.lifeform_metrics[lf.id].alive for record in records]
        color = tuple(int(c) for c in palette[lf.id])
        pygame.draw.lines(surface, color, False, to_points(per_species), 2)


def _draw_keyboard_hints(
    surface: pygame.Surface,
    fonts: FontManager,
    visible: bool,
    window_width: int,
    window_height: int,
) -> None:
    if not visible:
        hint = fonts.get("xs").render("Press ? for shortcuts", True, CHART_AXIS_COLOR)
        hint_pos = (window_width - hint.get_width() - 12, window_height - hint.get_height() - 8)
        surface.blit(hint, hint_pos)
        return

    overlay = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    panel_width, panel_height = 360, len(SHORTCUTS) * 32 + 60
    panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
    panel_rect.center = surface.get_rect().center
    pygame.draw.rect(surface, PANEL_BACKGROUND_COLOR, panel_rect, border_radius=10)
    pygame.draw.rect(surface, CHART_GRID_LINES, panel_rect, 1, border_radius=10)

    title = fonts.get("lg").render("Keyboard Shortcuts", True, TEXT_COLOR)
    surface.blit(title, (panel_rect.x + 20, panel_rect.y + 16))
    for i, (key, description) in enumerate(SHORTCUTS):
        row_y = panel_rect.y + 56 + i * 32
        key_surf = fonts.get("sm").render(key, True, ACCENT_PRIMARY)
        desc_surf = fonts.get("sm").render(description, True, TEXT_COLOR)
        surface.blit(key_surf, (panel_rect.x + 20, row_y))
        surface.blit(desc_surf, (panel_rect.x + 100, row_y))


class App:
    def __init__(self, config: Config) -> None:
        pygame.init()
        pygame.display.set_caption("LatticeLife")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.config = config
        self.lifeforms = _default_lifeforms(config)
        self.session = SimulationSession(config, self.lifeforms)

        self.layout = LayoutManager(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.fonts = FontManager()

        self.hints_visible = False
        self.view_mode = "2D"
        self.torus_status: str | None = None
        self.torus_renderer = None

        self.palette = render.build_palette(self.lifeforms)
        self.geometric_renderer: render.GeometricRenderer | None = None

        self.settings_panel = SettingsPanel(
            self.config, self.lifeforms, self.fonts.get("sm"), self._apply_settings
        )

        action_font = self.fonts.get("sm")
        empty_rect = pygame.Rect(0, 0, 0, 0)
        self.play_button = Button(
            empty_rect, "Play/Pause", action_font, on_click=self._toggle_pause
        )
        self.reset_button = Button(empty_rect.copy(), "Reset", action_font, on_click=self._reset)
        self.step_button = Button(empty_rect.copy(), "Step", action_font, on_click=self._step_once)
        self.new_button = Button(empty_rect.copy(), "New", action_font, on_click=self._new_random)

    # --- actions ---------------------------------------------------------

    def _toggle_pause(self) -> None:
        self.session.paused = not self.session.paused

    def _reset(self) -> None:
        self.session.reset()
        if self.geometric_renderer is not None:
            self.geometric_renderer.reset()

    def _step_once(self) -> None:
        if self.session.paused:
            self.session.step()

    def _new_random(self) -> None:
        self.lifeforms = _default_lifeforms(self.config)
        self.session = SimulationSession(self.config, self.lifeforms)
        self.palette = render.build_palette(self.lifeforms)
        if self.geometric_renderer is not None:
            self.geometric_renderer.reset()

    def _apply_settings(self, new_config: Config, lifeforms: list[Lifeform]) -> None:
        self.config = new_config
        self.lifeforms = lifeforms
        self.session = SimulationSession(new_config, lifeforms)
        self.palette = render.build_palette(lifeforms)
        self.geometric_renderer = None

    def _toggle_view_mode(self) -> None:
        if self.view_mode == "3D":
            self.view_mode = "2D"
            self.torus_renderer = None
            return

        try:
            from latticelife.ui.torus.torus_renderer import TorusRenderer
        except ImportError:
            self.torus_status = "3D view unavailable: install the [torus] extra (moderngl, pyrr)."
            return

        try:
            self.torus_renderer = TorusRenderer(
                self.screen, (self.config.grid.width, self.config.grid.height), self.palette
            )
        except Exception as exc:  # pragma: no cover - requires a real GL context
            self.torus_status = f"3D view unavailable: {exc}"
            return

        self.view_mode = "3D"
        self.torus_status = None

    # --- events ------------------------------------------------------------

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.layout.update(event.w, event.h)
                if self.geometric_renderer is not None:
                    self.geometric_renderer.reset()
                continue

            # Both handlers must run regardless of what the other returns: a TextInput/
            # NumericInput in the settings panel reports "handled" when it blurs, even if
            # the same click actually lands on one of the action buttons below.
            panel_handled = self.settings_panel.handle_event(event)
            buttons = (self.play_button, self.reset_button, self.step_button, self.new_button)
            buttons_handled = any([b.handle_event(event) for b in buttons])
            if panel_handled or buttons_handled:
                continue

            if event.type == pygame.KEYDOWN:
                if self._handle_keydown(event):
                    continue

            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.view_mode == "2D"
            ):
                grid_rect = self.layout.grid_rect()
                if grid_rect.collidepoint(event.pos):
                    self._toggle_cell(event.pos, grid_rect)

        return True

    def _handle_keydown(self, event: pygame.event.Event) -> bool:
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
        elif event.key == pygame.K_SLASH and event.mod & pygame.KMOD_SHIFT:
            self.hints_visible = not self.hints_visible
        elif event.key == pygame.K_ESCAPE and self.hints_visible:
            self.hints_visible = False
        else:
            return False
        return True

    def _toggle_cell(self, pos: tuple[int, int], grid_rect: pygame.Rect) -> None:
        grid = self.session.grid
        cell_size = render.fit_cell_size(
            grid.shape, grid.width, grid.height, grid_rect.width, grid_rect.height
        )
        col = (pos[0] - grid_rect.x) // cell_size
        row = (pos[1] - grid_rect.y) // cell_size
        if 0 <= row < grid.height and 0 <= col < grid.width:
            idx = row * grid.width + col
            if grid.occupancy[idx] > 0:
                grid.occupancy[idx] = 0
                grid.lifespans[idx] = 0
            elif self.lifeforms:
                grid.occupancy[idx] = self.lifeforms[0].id
                grid.lifespans[idx] = 1

    # --- drawing -------------------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)

        panel_rect = self.layout.panel_rect()
        pygame.draw.rect(self.screen, PANEL_BACKGROUND_COLOR, panel_rect)
        cursor_y = 16
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
        self.settings_panel.draw(self.screen, 16, cursor_y, panel_rect.width - 32)

        _draw_population_chart(self.screen, self.layout.chart_rect(), self.session, self.palette)

        if self.view_mode == "3D" and self.torus_renderer is not None:
            grid = self.session.grid
            self.torus_renderer.update_texture(grid.occupancy, grid.lifespans)
            self.torus_renderer.render()
        else:
            self._draw_grid()

        if self.torus_status is not None:
            status_surf = self.fonts.get("sm").render(self.torus_status, True, ACCENT_DANGER)
            self.screen.blit(status_surf, (self.layout.grid_x, self.layout.window_height - 28))

        _draw_keyboard_hints(
            self.screen,
            self.fonts,
            self.hints_visible,
            self.layout.window_width,
            self.layout.window_height,
        )

    def _draw_grid(self) -> None:
        grid_rect = self.layout.grid_rect()
        grid = self.session.grid
        cell_size = render.fit_cell_size(
            grid.shape, grid.width, grid.height, grid_rect.width, grid_rect.height
        )

        if grid.shape == "square":
            render.render_square(
                self.screen, grid, self.palette, cell_size, grid_rect.x, grid_rect.y
            )
            return

        if self.geometric_renderer is None or self.geometric_renderer.shape != grid.shape:
            self.geometric_renderer = render.GeometricRenderer(
                grid.shape, cell_size, grid_rect.x, grid_rect.y
            )
        self.geometric_renderer.cell_size = cell_size
        self.geometric_renderer.offset_x = grid_rect.x
        self.geometric_renderer.offset_y = grid_rect.y
        self.geometric_renderer.draw(self.screen, grid, self.palette)

    # --- main loop -------------------------------------------------------------

    def step_frame(self) -> bool:
        """Runs one frame: events, a simulation step if unpaused, draw, flip, clock tick.
        Returns False when the app should exit (QUIT event)."""
        running = self._handle_events()
        if running and not self.session.paused:
            self.session.step()
        self._draw()
        pygame.display.flip()
        self.clock.tick(self.config.ui.fps)
        return running

    def run(self) -> None:
        while self.step_frame():
            pass


def main(config_path: str = "config/default.toml") -> None:
    config = load_config(config_path)
    App(config).run()


if __name__ == "__main__":
    main()
