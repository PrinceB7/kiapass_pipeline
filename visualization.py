import cv2
import numpy as np
from typing import List, Tuple

class DisplayManager:
    def __init__(self, num_cameras: int, grid_size: Tuple[int, int] = None):
        self.num_cameras = num_cameras
        if grid_size is None:
            # Calculate optimal grid size
            grid_width = int(np.ceil(np.sqrt(num_cameras)))
            grid_height = int(np.ceil(num_cameras / grid_width))
            self.grid_size = (grid_width, grid_height)
        else:
            self.grid_size = grid_size
            
    def create_display_grid(self, frames: List[np.ndarray], target_width: int = 1920) -> np.ndarray:
        if not frames:
            return None
            
        grid_w, grid_h = self.grid_size
        cell_width = target_width // grid_w
        cell_height = int(cell_width * (frames[0].shape[0] / frames[0].shape[1]))
        
        # Create empty grid
        grid = np.zeros((cell_height * grid_h, cell_width * grid_w, 3), dtype=np.uint8)
        
        # Place frames in grid
        for idx, frame in enumerate(frames):
            if idx >= self.num_cameras:
                break
                
            i, j = idx // grid_w, idx % grid_w
            resized_frame = cv2.resize(frame, (cell_width, cell_height))
            grid[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width] = resized_frame
            
        return grid