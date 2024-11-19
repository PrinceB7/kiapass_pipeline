import av
import numpy as np
import cv2
import json
import time
import threading
from queue import Queue

# Load video data from the JSON file
with open('config/video_data.json', 'r') as file:
    video_data = json.load(file)

# List of video sources from the JSON file
video_sources = [entry['path'] for entry in video_data]

# Ensure we have exactly 25 sources
assert len(video_sources) == 25, "There should be exactly 25 camera/video sources."

# Initialize PyAV containers
video_containers = [av.open(src) for src in video_sources]
streams = [container.streams.video[0] for container in video_containers]

# Grid dimensions (5x5 grid)
grid_rows, grid_cols = 5, 5
frame_width, frame_height = 384, 216  # Size of each resized frame

# Queue to hold frames read by threads
frame_queue = Queue(maxsize=25)

# Function to read a single frame from a video container using PyAV
def read_frame(container, stream, index):
    while True:
        try:
            frame = next(container.decode(stream))
            frame = frame.to_ndarray(format='bgr24')
            resized_frame = cv2.resize(frame, (frame_width, frame_height))
        except av.AVError:
            container.seek(0)
            frame = next(container.decode(stream))
            frame = frame.to_ndarray(format='bgr24')
            resized_frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Put the frame in the queue with its index
        frame_queue.put((index, resized_frame))

# Start separate threads for each video container
threads = []
for i, (container, stream) in enumerate(zip(video_containers, streams)):
    thread = threading.Thread(target=read_frame, args=(container, stream, i))
    thread.daemon = True  # Set as a daemon so threads exit when the main program exits
    thread.start()
    threads.append(thread)

# Function to combine frames into a 5x5 grid
def combine_frames(frames):
    # Combine rows
    rows = []
    for i in range(grid_rows):
        row_frames = frames[i * grid_cols:(i + 1) * grid_cols]
        row = np.hstack(row_frames)
        rows.append(row)
    
    # Stack rows to form the final grid
    grid_frame = np.vstack(rows)
    
    return grid_frame

# Main loop to read and display frames
try:
    previous_time = time.time()
    frame_count = 0
    added_FPS = 0
    # Placeholder for frames to be displayed
    current_frames = [np.zeros((frame_height, frame_width, 3), dtype=np.uint8) for _ in range(25)]

    while True:
        # Update frames from the queue
        while not frame_queue.empty():
            index, frame = frame_queue.get()
            current_frames[index] = frame
        
        # Combine frames into a grid
        grid_frame = combine_frames(current_frames)
        
        # Display the grid
        cv2.imshow('25 Camera Grid (1920x1080)', grid_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Calculate and print FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        print(f"FPS: {fps:.2f}")
        frame_count += 1
        added_FPS += fps

finally:
    # Close video containers and destroy the display window
    for container in video_containers:
        container.close()
    cv2.destroyAllWindows()
    print(f"Final FPS: {added_FPS/frame_count:.2f}")
