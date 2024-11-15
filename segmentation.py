from ultralytics import YOLO
import numpy as np
from typing import List, Tuple

class SegmentationProcessor:
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        self.device = device
        self.model = YOLO(model_path)
        
    def process_batch(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, dict]]:
        
        results = self.model.predict(frames, imgsz=640, device=self.device, 
                                    verbose=False, classes=[0])
        processed = []
        
        for result in results:
            # Draw segmentation masks
            annotated_frame = result.plot(boxes=False, labels=False)
            
            # Extract metadata
            metadata = {
                'boxes': result.boxes.data.cpu().numpy() if result.boxes else None,
                'masks': result.masks.data.cpu().numpy() if result.masks else None,
                'names': result.names
            }
            
            processed.append((annotated_frame, metadata))
            
        return processed
    
    # def process_batch(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, dict]]:
        
    #     processed = []
        
    #     for frame in frames:
    #         # Get the original frame
            
    #         metadata = {
    #             'boxes': "boxes",
    #             'masks': "masks",
    #             'names': "names"
    #         }
            
    #         processed.append((frame, metadata))
            
    #     return processed