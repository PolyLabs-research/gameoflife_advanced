
import numpy as np
from pyrr import Matrix44, Vector3, matrix44


class OrbitalCamera:
    """
    Camera that orbits around the torus.
    Supports mouse drag rotation and scroll zoom.
    """
    
    def __init__(self, distance: float = 3.0, aspect_ratio: float = 16/9):
        self.distance = distance
        self.azimuth = 0.0      # Horizontal angle (radians)
        self.elevation = 0.3    # Vertical angle (radians)
        self.target = Vector3([0.0, 0.0, 0.0])
        self.aspect_ratio = aspect_ratio
        
        # Auto-rotation
        self.auto_rotate = True
        self.auto_rotate_speed = 0.005  # radians per frame
        
        # Mouse drag state
        self._dragging = False
        self._last_mouse_pos = None
    
    @property
    def position(self) -> Vector3:
        """Calculate camera position from spherical coordinates."""
        x = self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        y = self.distance * np.cos(self.elevation) * np.sin(self.azimuth)
        z = self.distance * np.sin(self.elevation)
        return Vector3([x, y, z]) + self.target
    
    @property
    def view_matrix(self) -> Matrix44:
        """Generate view matrix (look-at)."""
        return matrix44.create_look_at(self.position, self.target, Vector3([0.0, 0.0, 1.0]))
    
    @property
    def projection_matrix(self) -> Matrix44:
        """Generate perspective projection matrix."""
        return matrix44.create_perspective_projection(45.0, self.aspect_ratio, 0.1, 100.0)
    
    def update(self):
        """Update camera (auto-rotation)."""
        if self.auto_rotate:
            self.azimuth += self.auto_rotate_speed
    
    def handle_mouse_down(self, pos: tuple[int, int]):
        self._dragging = True
        self._last_mouse_pos = pos
        self.auto_rotate = False  # Stop auto-rotation when user interacts
    
    def handle_mouse_up(self):
        self._dragging = False
        self._last_mouse_pos = None

    def handle_mouse_move(self, pos: tuple[int, int]):
        if self._dragging and self._last_mouse_pos:
            dx = pos[0] - self._last_mouse_pos[0]
            dy = pos[1] - self._last_mouse_pos[1]
            
            self.azimuth -= dx * 0.01
            self.elevation += dy * 0.01
            self.elevation = np.clip(self.elevation, -np.pi/2 + 0.1, np.pi/2 - 0.1)
            
            self._last_mouse_pos = pos
    
    def handle_scroll(self, direction: int):
        """Zoom in/out."""
        self.distance *= 1.1 if direction < 0 else 0.9
        self.distance = np.clip(self.distance, 1.5, 10.0)

    def on_resize(self, width: int, height: int):
        self.aspect_ratio = width / height
