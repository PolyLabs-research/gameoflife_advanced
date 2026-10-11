"""Torus mesh for the 3D view (§6.14).

The torus lies in the xy-plane around the z axis. A surface point at major angle theta (around
the ring) and minor angle phi (around the tube) is
    ((R + r cos phi) cos theta, (R + r cos phi) sin theta, r sin phi),
so phi = 0 is the outer equator and phi rises up the outer wall. Each vertex carries
uv = (theta / 2pi, phi / 2pi); the fragment shader turns uv into a grid cell (see `shaders.py`).
"""

from __future__ import annotations

import numpy as np

VERTEX_DTYPE = np.dtype([("in_position", "3f4"), ("in_normal", "3f4"), ("in_uv", "2f4")])


def generate_torus_mesh(
    R: float, r: float, major_segments: int, minor_segments: int
) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved vertices (`VERTEX_DTYPE`: position, unit normal, uv) and uint32 triangle
    indices. The seam rows (theta = 2pi, phi = 2pi) are separate vertices with u = 1 or v = 1, so
    uv never jumps back to 0 inside a triangle."""
    theta = 2 * np.pi * np.arange(major_segments + 1) / major_segments
    phi = 2 * np.pi * np.arange(minor_segments + 1) / minor_segments
    t, p = np.meshgrid(theta, phi, indexing="ij")  # vertex (i, j) at index i*(minor+1) + j
    ring = R + r * np.cos(p)

    vertices = np.zeros(t.size, dtype=VERTEX_DTYPE)
    vertices["in_position"] = np.stack(
        [ring * np.cos(t), ring * np.sin(t), r * np.sin(p)], axis=-1
    ).reshape(-1, 3)
    vertices["in_normal"] = np.stack(
        [np.cos(p) * np.cos(t), np.cos(p) * np.sin(t), np.sin(p)], axis=-1
    ).reshape(-1, 3)
    vertices["in_uv"] = np.stack(
        [t / (2 * np.pi), p / (2 * np.pi)], axis=-1
    ).reshape(-1, 2)

    i, j = np.meshgrid(np.arange(major_segments), np.arange(minor_segments), indexing="ij")
    i0 = (i * (minor_segments + 1) + j).ravel()
    i1 = i0 + minor_segments + 1  # next theta
    i2 = i0 + 1  # next phi
    i3 = i1 + 1
    indices = np.stack([i0, i1, i2, i1, i3, i2], axis=-1).astype(np.uint32).ravel()
    return vertices, indices
