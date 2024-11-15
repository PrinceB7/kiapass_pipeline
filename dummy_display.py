import cv2
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Load video paths from JSON file
json_path = 'config/video_data.json'
with open(json_path, 'r') as file:
    video_data = json.load(file)

# Extract paths of the 25 videos
video_paths = [item['path'] for item in video_data]

# Number of videos
num_videos = len(video_paths)

# Initialize video capture objects
video_captures = [cv2.VideoCapture(path) for path in video_paths]

# Dimensions for grid display (5x5 grid)
grid_rows, grid_cols = 2, 2
display_width, display_height = 1920, 1080  # Each video window size
# frame_width, frame_height = 640, 360  # Each small frame size in the grid
frame_width, frame_height = display_width//grid_cols, display_height//grid_rows  # Each small frame size in the grid

# Function to read a single frame from a video capture
def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
        ret, frame = cap.read()
    return frame

# Function to get all frames from all video captures
def get_all_frames(video_captures):
    with ThreadPoolExecutor(max_workers=num_videos) as executor:
        frames = list(executor.map(read_frame, video_captures))
    return frames

# Function to combine frames into a 5x5 grid
def combine_frames(frames):
    # Resize frames to fit the grid
    resized_frames = [cv2.resize(frame, (frame_width, frame_height)) for frame in frames]
    
    # Combine rows
    rows = []
    for i in range(grid_rows):
        row_frames = resized_frames[i * grid_cols:(i + 1) * grid_cols]
        row = np.hstack(row_frames)
        rows.append(row)
    
    # Stack rows to form the final grid
    grid_frame = np.vstack(rows)
    
    # Resize to display window size
    grid_frame = cv2.resize(grid_frame, (display_width, display_height))
    
    return grid_frame

# Main loop to read and display frames
try:
    previous_time = time.time()
    while True:
        # Get frames from all videos
        frames = get_all_frames(video_captures)
        
        # Combine frames into a grid
        grid_frame = combine_frames(frames)
        
        # Display the grid
        # cv2.imshow('Video Grid (5x5)', grid_frame)
        cv2.imwrite('img.png', grid_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Calculate the time taken for this iteration and update FPS
        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time
        
        # Calculate FPS as frames per second
        print(f"FPS: {fps:.2f}")  # Print the current FPS

finally:
    # Release video captures and close the display window
    for cap in video_captures:
        cap.release()
    cv2.destroyAllWindows()
