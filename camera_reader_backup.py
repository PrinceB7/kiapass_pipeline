import cv2
from threading import Thread
import queue
import time
import json
import os

class VideoReader:
    def __init__(self, video_path: str, buffer_size=30):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.video_path = video_path
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.frame_count = 0
        
    def start(self):
        thread = Thread(target=self.update, args=(), daemon=True)
        thread.start()
        return self
        
    def update(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.stopped = True
            return
            
        while not self.stopped:
            if not self.buffer.full():
                ret, frame = cap.read()
                if not ret:
                    # Loop back to the beginning of the video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
                self.frame_count += 1
                if not self.buffer.full():
                    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                    self.buffer.put((self.frame_count, frame))
            else:
                time.sleep(0.0001)  # Prevent CPU overload
                
        cap.release()
        
    def read(self):
        return self.buffer.get()[1]  # Return just the frame
        
    def read_with_info(self):
        return self.buffer.get()  # Return both frame count and frame
        
    def running(self):
        return not self.stopped
        
    def stop(self):
        self.stopped = True
    
    def __del__(self):
        """Ensure proper cleanup of resources."""
        self.stop()
        
        
class VideoManager:
    def __init__(self, config_path: str, buffer_size: int = 30):
        self.config_path = config_path
        self.buffer_size = buffer_size
        self.video_readers = []
        self.load_config()
        
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                self.video_configs = json.load(f)
        except Exception as e:
            raise Exception(f"Error loading video configuration: {str(e)}")
            
    def initialize_readers(self):
        for config in self.video_configs:
            reader = VideoReader(config['path'], self.buffer_size)
            self.video_readers.append(reader)
            
    def start_all(self):
        for reader in self.video_readers:
            reader.start()
            
    def stop_all(self):
        for reader in self.video_readers:
            reader.stop()
            
    def get_reader_count(self):
        return len(self.video_readers)