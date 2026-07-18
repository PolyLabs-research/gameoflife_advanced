
import numpy as np


def torus_point(theta: float, phi: float, R: float, r: float) -> tuple[float, float, float]:
    """
    Generate a point on the torus surface.
    
    Args:
        theta: Major angle (around the donut)
        phi: Minor angle (around the tube)
        R: Major radius (center to tube)
        r: Minor radius (tube thickness)
    
    Returns:
        (x, y, z) coordinates
    """
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return (x, y, z)

def generate_torus_mesh(
    R: float, 
    r: float, 
    major_segments: int, 
    minor_segments: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate torus mesh with vertices, normals, and UV coordinates.
    
    Returns:
        vertices: (N, 3) float32 array of positions
        normals: (N, 3) float32 array of surface normals
        uvs: (N, 2) float32 array of texture coordinates
    """
    vertices = []
    normals = []
    uvs = []
    
    for i in range(major_segments + 1):
        theta = 2 * np.pi * i / major_segments
        
        for j in range(minor_segments + 1):
            phi = 2 * np.pi * j / minor_segments
            
            # Vertex position
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            vertices.append((x, y, z))
            
            # Normal
            nx = np.cos(phi) * np.cos(theta)
            ny = np.cos(phi) * np.sin(theta)
            nz = np.sin(phi)
            normals.append((nx, ny, nz))
            
            # UV coordinates
            u = i / major_segments
            v = j / minor_segments
            uvs.append((u, v))

    indices = []
    for i in range(major_segments):
        for j in range(minor_segments):
            i0 = i * (minor_segments + 1) + j
            i1 = (i + 1) * (minor_segments + 1) + j
            i2 = i * (minor_segments + 1) + (j + 1)
            i3 = (i + 1) * (minor_segments + 1) + (j + 1)
            
            indices.extend([i0, i1, i2])
            indices.extend([i1, i3, i2])

    # Unpack vertices, normals, and uvs for indexed drawing
    final_vertices = np.array(vertices, dtype=np.float32)
    final_normals = np.array(normals, dtype=np.float32)
    final_uvs = np.array(uvs, dtype=np.float32)
    
    # Create an interleaved VBO data array
    vbo_data = np.zeros(len(vertices), dtype=[
        ('in_position', '3f4'),
        ('in_normal', '3f4'),
        ('in_uv', '2f4'),
    ])
    
    vbo_data['in_position'] = final_vertices
    vbo_data['in_normal'] = final_normals
    vbo_data['in_uv'] = final_uvs

    return vbo_data, np.array(indices, dtype=np.uint32)
