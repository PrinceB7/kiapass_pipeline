import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from typing import List
import cv2
import time
import numpy as np
from camera_reader import VideoManager
from segmentation import SegmentationProcessor
from visualization import DisplayManager

def process_batch(frame_batch: List[np.ndarray], segmentation_processor: SegmentationProcessor):
    return segmentation_processor.process_batch(frame_batch)

def main():
    # Configuration
    config_path = 'config/video_data.json'
    model_path = 'weights/yolov8n-seg.pt'
    batch_size = 5  # Number of frames to process in parallel
    
    # Initialize video manager and components
    video_manager = VideoManager(config_path, buffer_size=10)
    video_manager.initialize_readers()
    num_cameras = video_manager.get_reader_count()
    
    segmentation_processor = SegmentationProcessor(model_path)
    display_manager = DisplayManager(num_cameras)
    
    # Start video readers
    video_manager.start_all()
    
    try:
        previous_time = time.time()
        while True:
            # Collect frames from all videos
            frames = []
            frame_info = []
            for reader in video_manager.video_readers:
                if reader.running():
                    frame_count, frame = reader.read_with_info()
                    frames.append(frame)
                    frame_info.append(frame_count)
                    
            if not frames:
                break
                
            # Process frames in batches using multiprocessing
            batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
            processed_batches = []
            
            for batch in batches:
                result = process_batch(batch, segmentation_processor)
                processed_batches.extend(result)
            
            # Extract processed frames
            processed_frames = [frame for frame, _ in processed_batches]
            
            # Create and display grid
            display_grid = display_manager.create_display_grid(processed_frames)
            if display_grid is not None:
                # Add frame count overlay
                for idx, count in enumerate(frame_info):
                    i, j = idx // display_manager.grid_size[1], idx % display_manager.grid_size[0]
                    x = j * (display_grid.shape[1] // display_manager.grid_size[0]) + 10
                    y = i * (display_grid.shape[0] // display_manager.grid_size[1]) + 30
                    cv2.putText(display_grid, f"Frame: {count}", (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # cv2.imwrite('img.png', display_grid)
                cv2.imshow('Multi-Camera Feed', display_grid)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Calculate the time taken for this iteration and update FPS
            current_time = time.time()
            fps = 1/(current_time - previous_time)
            previous_time = current_time
            
            # Calculate FPS as frames per second
            print(f"FPS: {fps:.2f}")  # Print the current FPS
                
            # time.sleep(0.001)  # Prevent CPU overload
            
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        video_manager.stop_all()

if __name__ == '__main__':
    main()