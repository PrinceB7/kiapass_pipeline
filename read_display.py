import cv2
import numpy as np
import json
import time

# Configuration parameters
CONFIG_FILE = 'config/video_data.json'
GRID_ROWS = 5
GRID_COLUMNS = 5
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# Load video paths from JSON
with open(CONFIG_FILE, 'r') as f:
    video_paths = [video['path'] for video in json.load(f)]

# Initialize video captures
video_captures = [cv2.VideoCapture(path) for path in video_paths]

# Calculate grid cell size
cell_width = VIDEO_WIDTH // GRID_COLUMNS
cell_height = VIDEO_HEIGHT // GRID_ROWS

# Main loop for reading and displaying video frames
previous_time = time.time()
grid = np.zeros((GRID_ROWS * cell_height, GRID_COLUMNS * cell_width, 3), dtype=np.uint8)
while True:
    # Read frames from all videos
    frames = [cap.read()[1] for cap in video_captures]

    # Fill the grid with resized video frames
    for i in range(GRID_ROWS):
        for j in range(GRID_COLUMNS):
            idx = i * GRID_COLUMNS + j
            if idx < len(frames) and frames[idx] is not None:
                resized_frame = cv2.resize(frames[idx], (cell_width, cell_height))
                grid[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width] = resized_frame

    # Display the grid
    cv2.imwrite('img.png', grid)
    # cv2.imshow('Video Grid', grid)

    # Check for user input to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Calculate and print FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    print(f"FPS: {fps:.2f}")

# Release resources
for cap in video_captures:
    cap.release()

cv2.destroyAllWindows()
