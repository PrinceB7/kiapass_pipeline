import cv2
import numpy as np
import json
import time
from threading import Thread, Event
from ultralytics import YOLO

# Configuration parameters
CONFIG_FILE = 'config/video_data.json'
GRID_ROWS = 5
GRID_COLUMNS = 5
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
MODEL_PATH = 'yolov8n-seg.pt'
# MODEL_PATH = 'yolov8n-seg.pt'

# Load video paths from JSON
with open(CONFIG_FILE, 'r') as f:
    video_paths = [video['path'] for video in json.load(f)]

# Initialize video captures
video_captures = [cv2.VideoCapture(path) for path in video_paths]

# Calculate grid cell size
cell_width = VIDEO_WIDTH // GRID_COLUMNS
cell_height = VIDEO_HEIGHT // GRID_ROWS

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Shared state between threads
frames = [None] * 25
inference_results = [None] * 25
frame_ready_event = Event()
inference_ready_event = Event()
stop_event = Event()

def video_reader_loop():
    global frames
    while not stop_event.is_set():
        # Read frames from all videos
        for i, cap in enumerate(video_captures):
            frames[i] = cap.read()[1]
        frame_ready_event.set()
        frame_ready_event.clear()

def inference_loop():
    global inference_results
    while not stop_event.is_set():
        # Wait for new frames to be available
        frame_ready_event.wait()

        # Run inference on all frames in batch
        results = model.predict(frames, imgsz=640, device=0, verbose=False, classes=[0])
        inference_results = results

        inference_ready_event.set()
        inference_ready_event.clear()

def main_loop():
    global previous_time
    previous_time = time.time()

    video_reader_thread = Thread(target=video_reader_loop)
    inference_thread = Thread(target=inference_loop)

    video_reader_thread.start()
    inference_thread.start()
    grid = np.zeros((GRID_ROWS * cell_height, GRID_COLUMNS * cell_width, 3), dtype=np.uint8)

    while True:
        # Wait for inference results to be available
        inference_ready_event.wait()

        # Process the results and create the grid
        for i in range(GRID_ROWS):
            for j in range(GRID_COLUMNS):
                idx = i * GRID_COLUMNS + j
                if frames[idx] is not None:
                    resized_frame = cv2.resize(inference_results[idx].plot(boxes=False, labels=False), (cell_width, cell_height))
                    grid[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width] = resized_frame

        # Display the grid
        cv2.imshow('Video Grid', grid)

        # Calculate and print FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        print(f"FPS: {fps:.2f}")

        # Check for user input to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Signal the threads to stop
    stop_event.set()

    # Wait for the threads to finish
    video_reader_thread.join()
    inference_thread.join()

    # Release resources
    for cap in video_captures:
        cap.release()

    cv2.destroyAllWindows()

main_loop()