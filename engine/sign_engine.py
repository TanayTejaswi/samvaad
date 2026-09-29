import asyncio
import time
import numpy as np
import cv2
import logging
from collections import deque
from inference.hands import HandPipeline

logger = logging.getLogger("samvaad.engine")

class SignEngine:
    def __init__(self, vocab_map=None, model=None):
        self.hand_pipeline = HandPipeline(device="cpu") # We use CPU as default if NPU model missing
        self.hand_pipeline.load()
        self.model = model
        self.vocab_map = vocab_map
        
        # Ring buffers
        self.max_frames = 150
        self.frame_buffer = deque(maxlen=self.max_frames)
        self.landmark_buffer = deque(maxlen=self.max_frames)
        
        # State machine
        self.is_signing = False
        self.sign_start_time = 0
        self.silence_frames = 0
        self.silence_threshold = 15 # 15 frames of no hands = end of sign
        
    def process_frame(self, frame: np.ndarray):
        """Processes a single BGR frame, tracks hands, and emits signs if completed."""
        result = self.hand_pipeline.infer({"image": frame})
        
        has_hands = result["left"] is not None or result["right"] is not None
        
        self.landmark_buffer.append(result)
        
        if not self.is_signing and has_hands:
            self.is_signing = True
            self.silence_frames = 0
            self.sign_start_time = time.time()
            return {"type": "STATUS", "status": "signing", "landmarks": result}
            
        if self.is_signing:
            if not has_hands:
                self.silence_frames += 1
            else:
                self.silence_frames = 0
                
            if self.silence_frames >= self.silence_threshold:
                # Sign ended!
                self.is_signing = False
                sign_event = self._recognize_segment()
                self.landmark_buffer.clear() # Reset for next sign
                sign_event["landmarks"] = result
                return sign_event
                
        return {"type": "STATUS", "status": "idle" if not self.is_signing else "signing", "landmarks": result}
        
    def _recognize_segment(self):
        """Runs the SignNet model on the buffered sequence."""
        if not self.model or not self.vocab_map:
            # Mock mode if no model is loaded
            return {
                "type": "SIGN_RECOGNIZED",
                "gloss": "HELLO",
                "confidence": 0.95,
                "alternatives": ["THANK YOU", "PLEASE"],
                "latency_ms": 150
            }
            
        # TODO: Format landmarks from self.landmark_buffer into (T, 2, 21, 3) 
        # and run through self.model.
        
        return {
            "type": "SIGN_RECOGNIZED",
            "gloss": "UNIMPLEMENTED",
            "confidence": 0.0,
            "alternatives": [],
            "latency_ms": 0
        }
