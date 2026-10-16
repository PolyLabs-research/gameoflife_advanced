"""3D torus camera (§6.14): numpy view/projection matrices (no `pyrr`, F79) and pygame input
(F33, F34). No GL context needed; these run without the [torus] extra."""

import math

import numpy as np
import pygame
import pytest

from latticelife.ui.torus.camera import (
    MAX_DISTANCE,
    MIN_DISTANCE,
    OrbitalCamera,
    gl_bytes,
    look_at,
    perspective,
)

# Row-major matrices from pyrr 0.10.3 (`matrix44.create_look_at` and
# `create_perspective_projection(45, aspect, 0.1, 100)`), pinned when pyrr was dropped. The
# GL uniforms were their bytes, so the new matrices must equal their transposes.
PYRR_LOOK_AT = {
    (3.0, 0.0, 0.3): [
        [0.0, -0.295520207, 0.955336489, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.955336489, 0.295520207, 0.0],
        [0.0, 0.0, -3.0, 1.0],
    ],
    (4.5, 1.2, -0.7): [
        [-0.932039086, 0.233437275, 0.277146498, 0.0],
        [0.362357754, 0.600436064, 0.712862813, 0.0],
        [0.0, 0.764842187, -0.644217687, 0.0],
        [0.0, 0.0, -4.5, 1.0],
    ],
}
PYRR_PERSPECTIVE = {
    1.5: [
        [1.609475708, 0.0, 0.0, 0.0],
        [0.0, 2.414213562, 0.0, 0.0],
        [0.0, 0.0, -1.002002002, -1.0],
        [0.0, 0.0, -0.2002002, 0.0],
    ],
    0.75: [
        [3.218951416, 0.0, 0.0, 0.0],
        [0.0, 2.414213562, 0.0, 0.0],
        [0.0, 0.0, -1.002002002, -1.0],
        [0.0, 0.0, -0.2002002, 0.0],
    ],
}


@pytest.mark.parametrize("key", sorted(PYRR_LOOK_AT))
def test_view_matrix_matches_the_pyrr_original(key):
    distance, azimuth, elevation = key
    camera = OrbitalCamera(distance=distance)
    camera.azimuth, camera.elevation = azimuth, elevation
    np.testing.assert_allclose(camera.view_matrix.T, PYRR_LOOK_AT[key], atol=1e-8)


@pytest.mark.parametrize("aspect", sorted(PYRR_PERSPECTIVE))
def test_perspective_matches_the_pyrr_original(aspect):
    projection = perspective(45.0, aspect, 0.1, 100.0)
    np.testing.assert_allclose(projection.T, PYRR_PERSPECTIVE[aspect], atol=1e-8)


def test_a_wide_view_keeps_the_original_projection():
    camera = OrbitalCamera(aspect_ratio=1.5)
    np.testing.assert_allclose(camera.projection_matrix.T, PYRR_PERSPECTIVE[1.5], atol=1e-8)


@pytest.mark.parametrize("aspect", [2.0, 1.5, 1.0, 0.75, 0.3])
def test_the_whole_torus_fits_the_narrower_side_at_the_default_distance(aspect):
    """A tall grid area cut the ring off: the field of view was always 45 degrees vertically."""
    camera = OrbitalCamera(aspect_ratio=aspect)
    projection = camera.projection_matrix
    half_across = math.atan(1 / projection[0, 0])
    half_down = math.atan(1 / projection[1, 1])
    assert math.degrees(min(half_across, half_down)) == pytest.approx(22.5)
    # The torus lies within 1.4 of the origin, so its outline stays within asin(1.4 / distance).
    assert min(half_across, half_down) > math.asin(1.4 / camera.distance)


def test_look_at_puts_the_eye_at_the_origin_and_the_target_straight_ahead():
    eye, target = np.array([2.0, -1.0, 3.0]), np.array([0.5, 0.5, 0.0])
    view = look_at(eye, target, (0, 0, 1))
    np.testing.assert_allclose(view @ np.append(eye, 1.0), [0, 0, 0, 1], atol=1e-12)
    ahead = view @ np.append(target, 1.0)
    distance = np.linalg.norm(target - eye)
    np.testing.assert_allclose(ahead, [0, 0, -distance, 1], atol=1e-12)


def test_perspective_maps_the_near_and_far_planes_to_the_depth_range():
    projection = perspective(45.0, 2.0, 0.1, 100.0)
    for depth, ndc in ((-0.1, -1.0), (-100.0, 1.0)):
        clip = projection @ np.array([0.0, 0.0, depth, 1.0])
        assert clip[2] / clip[3] == pytest.approx(ndc)


def test_gl_bytes_are_column_major_float32():
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    values = np.frombuffer(gl_bytes(matrix), dtype=np.float32)
    np.testing.assert_array_equal(values[:4], matrix[:, 0])  # the first column comes first


@pytest.mark.filterwarnings("error")
def test_a_camera_frame_raises_no_warnings():
    """F79: pyrr assigned ndarray.shape, deprecated in NumPy 2.5 (5 warnings per frame)."""
    camera = OrbitalCamera()
    camera.update()
    camera.view_matrix, camera.projection_matrix, camera.position  # noqa: B018


def test_drag_rotates_and_stops_auto_rotation():
    camera = OrbitalCamera()
    azimuth, elevation = camera.azimuth, camera.elevation
    camera.handle_mouse_down((100, 100))
    camera.handle_mouse_move((150, 80))  # right and up
    camera.handle_mouse_up()
    assert camera.auto_rotate is False
    assert camera.azimuth == pytest.approx(azimuth - 0.5)
    assert camera.elevation == pytest.approx(elevation - 0.2)
    camera.update()
    assert camera.azimuth == pytest.approx(azimuth - 0.5)  # no more auto-rotation


def test_elevation_is_clamped_short_of_the_poles():
    camera = OrbitalCamera()
    camera.handle_mouse_down((0, 0))
    camera.handle_mouse_move((0, 100_000))
    assert camera.elevation < math.pi / 2
    camera.handle_mouse_move((0, -100_000))
    assert camera.elevation > -math.pi / 2
    assert np.all(np.isfinite(camera.view_matrix))


def test_wheel_up_zooms_in_and_distance_is_clamped():
    camera = OrbitalCamera(distance=3.0)
    camera.handle_scroll(1)
    assert camera.distance < 3.0
    camera.handle_scroll(-1000)
    assert camera.distance == MAX_DISTANCE
    camera.handle_scroll(1000)
    assert camera.distance == MIN_DISTANCE


def test_on_resize_sets_the_aspect_and_survives_a_zero_height():
    """F34: the aspect must follow the viewport; a minimised window reports height 0."""
    camera = OrbitalCamera()
    camera.on_resize(800, 400)
    assert camera.aspect_ratio == 2.0
    camera.on_resize(800, 0)
    assert math.isfinite(camera.aspect_ratio)


# --- pygame input --------------------------------------------------------------------------

AREA = pygame.Rect(300, 200, 400, 300)


def _down(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


def _up(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button)


def _move(pos, pressed=True):
    return pygame.event.Event(
        pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(int(pressed), 0, 0)
    )


def _wheel(y):
    return pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=y, flipped=False)


def test_a_drag_that_starts_on_the_area_rotates_even_off_it():
    camera = OrbitalCamera()
    azimuth = camera.azimuth
    assert camera.handle_event(_down(AREA.center), AREA)
    assert camera.handle_event(_move((AREA.right + 50, AREA.centery)), AREA)  # left the area
    assert camera.handle_event(_up((AREA.right + 50, AREA.centery)), AREA)
    assert camera.azimuth < azimuth
    assert not camera.dragging
    assert not camera.handle_event(_move(AREA.center), AREA)  # a bare move does nothing


def test_a_press_off_the_area_or_with_another_button_is_not_a_drag():
    camera = OrbitalCamera()
    assert not camera.handle_event(_down((10, 10)), AREA)
    assert not camera.handle_event(_down(AREA.center, button=3), AREA)
    assert not camera.dragging
    assert camera.auto_rotate


def test_a_release_lost_outside_the_window_ends_the_drag():
    camera = OrbitalCamera()
    camera.handle_event(_down(AREA.center), AREA)
    azimuth = camera.azimuth
    assert not camera.handle_event(_move((AREA.centerx + 40, AREA.centery), False), AREA)
    assert not camera.dragging
    assert camera.azimuth == azimuth


def test_wheel_zooms_only_over_the_area_and_once_per_step():
    camera = OrbitalCamera(distance=3.0)
    camera.handle_event(_move((10, 10), False), AREA)
    assert not camera.handle_event(_wheel(1), AREA)
    assert camera.distance == 3.0
    camera.handle_event(_move(AREA.center, False), AREA)
    assert camera.handle_event(_wheel(1), AREA)
    # pygame also posts the legacy button-4 press for the same wheel step: it must not zoom.
    assert not camera.handle_event(_down(AREA.center, button=4), AREA)
    assert camera.distance == pytest.approx(2.7)


def test_reset_puts_the_camera_back_and_leaves_rotation_alone():
    camera = OrbitalCamera()
    camera.handle_mouse_down((0, 0))
    camera.handle_mouse_move((40, 25))
    camera.handle_scroll(4)
    assert camera.auto_rotate is False
    camera.reset()
    start = OrbitalCamera()
    assert (camera.distance, camera.azimuth, camera.elevation) == (
        start.distance,
        start.azimuth,
        start.elevation,
    )
    assert not camera.dragging and camera.auto_rotate is False
