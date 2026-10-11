"""Orbital camera for the 3D torus view, and the two view/projection matrices it needs.

The matrices are plain numpy (no `pyrr`: its 2019 release sets `ndarray.shape`, which NumPy 2.5
deprecates). They use the column-vector convention, `clip = projection @ view @ point`;
`gl_bytes` turns one into the column-major float32 bytes a GLSL `mat4` uniform reads. For the
same arguments each equals the transpose of the row-major matrix `pyrr.matrix44.create_look_at`
/ `create_perspective_projection` gave, i.e. the same uniform bytes.

The camera orbits the origin with +z up: `azimuth` turns it around the torus axis and
`elevation` lifts it above (or below) the torus plane. `handle_event` maps pygame input onto
it: a left-button drag that starts on the torus rotates it (and stops the slow auto-rotation),
and the wheel over the torus zooms.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

# Field of view across the narrower side of the view (vertical when it is wider than tall).
FIELD_OF_VIEW_DEGREES = 45.0
# The torus lies within 1.4 of the origin (major radius 1 + minor 0.4); from this far away that
# sphere spans asin(1.4 / 4) = 20.5 degrees each way, inside the 22.5-degree half field of view,
# so the whole torus is in view from every angle, in a view of any shape.
DEFAULT_DISTANCE = 4.0
NEAR_PLANE = 0.1
FAR_PLANE = 100.0

MIN_DISTANCE = 1.5
MAX_DISTANCE = 10.0
# Elevation stays this far short of straight up/down, where the +z up vector degenerates.
ELEVATION_MARGIN = 0.1
# Radians of rotation per pixel dragged, and the zoom factor per wheel notch.
DRAG_RADIANS_PER_PIXEL = 0.01
ZOOM_PER_NOTCH = 0.9


def look_at(eye, target, up) -> np.ndarray:
    """4x4 view matrix: world -> camera space for a camera at `eye` looking at `target`, with
    `up` as the screen's upward direction (camera looks down its -z axis)."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, np.asarray(up, dtype=np.float64))
    side /= np.linalg.norm(side)
    upward = np.cross(side, forward)
    view = np.identity(4)
    view[0, :3], view[1, :3], view[2, :3] = side, upward, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fovy_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    """4x4 OpenGL perspective projection (depth -near..-far maps to NDC -1..1)."""
    f = 1.0 / math.tan(math.radians(fovy_degrees) / 2.0)
    projection = np.zeros((4, 4))
    projection[0, 0] = f / aspect
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = 2.0 * far * near / (near - far)
    projection[3, 2] = -1.0
    return projection


def gl_bytes(matrix: np.ndarray) -> bytes:
    """A column-vector matrix as the column-major float32 bytes of a GLSL `mat4` uniform."""
    return np.ascontiguousarray(np.asarray(matrix).T, dtype=np.float32).tobytes()


class OrbitalCamera:
    """Camera that orbits the torus. Supports mouse drag rotation and wheel zoom."""

    def __init__(self, distance: float = DEFAULT_DISTANCE, aspect_ratio: float = 16 / 9) -> None:
        self.distance = distance
        self.azimuth = 0.0  # radians around the torus axis; 0 looks at the torus from +x
        self.elevation = 0.3  # radians above the torus plane
        self.target = np.zeros(3)
        self.aspect_ratio = aspect_ratio

        # Auto-rotation, until the first drag.
        self.auto_rotate = True
        self.auto_rotate_speed = 0.005  # radians per frame

        self._last_mouse_pos: tuple[int, int] | None = None  # set while dragging
        self._pointer: tuple[int, int] | None = None  # last known pointer position

    @property
    def dragging(self) -> bool:
        return self._last_mouse_pos is not None

    def handle_event(self, event: pygame.event.Event, area: pygame.Rect) -> bool:
        """Camera input for a view drawn over `area`. A left-button press on the area starts a
        drag that rotates the camera until the button is released, wherever the pointer goes;
        the wheel over the area zooms. True if the event was used."""
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            self._pointer = event.pos
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and area.collidepoint(event.pos):
                self.handle_mouse_down(event.pos)
                return True
            return False
        if event.type == pygame.MOUSEMOTION:
            if not self.dragging:
                return False
            if not getattr(event, "buttons", (1,))[0]:
                self.handle_mouse_up()  # the release happened outside the window
                return False
            self.handle_mouse_move(event.pos)
            return True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.dragging:
                self.handle_mouse_up()
                return True
            return False
        if event.type == pygame.WINDOWFOCUSLOST:
            self.handle_mouse_up()
            return False
        if event.type == pygame.MOUSEWHEEL:
            # One notch per wheel step; the legacy button 4/5 events pygame also sends for the
            # same step are not used, so a step zooms once.
            pointer = self._pointer if self._pointer is not None else pygame.mouse.get_pos()
            if area.collidepoint(pointer):
                self.handle_scroll(event.y)
                return True
        return False

    @property
    def position(self) -> np.ndarray:
        """Camera position from its spherical coordinates around the target."""
        horizontal = self.distance * math.cos(self.elevation)
        return self.target + np.array(
            [
                horizontal * math.cos(self.azimuth),
                horizontal * math.sin(self.azimuth),
                self.distance * math.sin(self.elevation),
            ]
        )

    @property
    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.target, (0.0, 0.0, 1.0))

    @property
    def field_of_view(self) -> float:
        """Vertical field of view in degrees, chosen so the narrower side of the view spans
        FIELD_OF_VIEW_DEGREES (a view taller than wide gets a wider vertical angle)."""
        if self.aspect_ratio >= 1:
            return FIELD_OF_VIEW_DEGREES
        across = math.tan(math.radians(FIELD_OF_VIEW_DEGREES / 2)) / self.aspect_ratio
        return 2 * math.degrees(math.atan(across))

    @property
    def projection_matrix(self) -> np.ndarray:
        return perspective(self.field_of_view, self.aspect_ratio, NEAR_PLANE, FAR_PLANE)

    def update(self) -> None:
        """Advance one frame (auto-rotation)."""
        if self.auto_rotate:
            self.azimuth += self.auto_rotate_speed

    def handle_mouse_down(self, pos: tuple[int, int]) -> None:
        self._last_mouse_pos = pos
        self.auto_rotate = False  # the user has taken over

    def handle_mouse_up(self) -> None:
        self._last_mouse_pos = None

    def handle_mouse_move(self, pos: tuple[int, int]) -> None:
        """While dragging: the surface follows the pointer (drag right turns the torus right,
        drag down tips its top towards the viewer)."""
        if self._last_mouse_pos is None:
            return
        dx = pos[0] - self._last_mouse_pos[0]
        dy = pos[1] - self._last_mouse_pos[1]
        self.azimuth -= dx * DRAG_RADIANS_PER_PIXEL
        limit = math.pi / 2 - ELEVATION_MARGIN
        self.elevation = min(limit, max(-limit, self.elevation + dy * DRAG_RADIANS_PER_PIXEL))
        self._last_mouse_pos = pos

    def handle_scroll(self, notches: float) -> None:
        """Zoom: positive notches (wheel up / away from the user) move closer."""
        distance = self.distance * ZOOM_PER_NOTCH**notches
        self.distance = min(MAX_DISTANCE, max(MIN_DISTANCE, distance))

    def on_resize(self, width: int, height: int) -> None:
        """Match the projection to a `width` x `height` viewport (a zero size is treated as 1)."""
        self.aspect_ratio = max(width, 1) / max(height, 1)
