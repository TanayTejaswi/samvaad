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
        if len(self.landmark_buffer) < 5:
            # Too short to be a sign
            return {"type": "STATUS", "status": "idle"}
            
        start_time = time.time()
        
        # 1. Format into (T, 2, 21, 3)
        T = len(self.landmark_buffer)
        sequence = np.zeros((T, 2, 21, 3), dtype=np.float32)
        
        for t, res in enumerate(self.landmark_buffer):
            if res["left"] is not None:
                sequence[t, 0] = res["left"]
            if res["right"] is not None:
                sequence[t, 1] = res["right"]
                
        # 2. Extract features using Phase 3 pipeline
        from inference.features import extract_features
        features = extract_features(sequence, target_length=48)
        
        # 3. Run Inference
        if not self.model or not self.vocab_map:
            # Mock mode if no model is loaded yet
            return {
                "type": "SIGN_RECOGNIZED",
                "gloss": "HELLO (Mock)",
                "confidence": 0.99,
                "alternatives": ["THANK YOU", "PLEASE"],
                "latency_ms": int((time.time() - start_time) * 1000)
            }
            
        # NPU / ONNX Inference
        # Model expects (1, 48, 254)
        input_tensor = np.expand_dims(features, axis=0)
        
        try:
            # Assuming self.model is an ONNX InferenceSession wrapper
            out = self.model.infer({"input": input_tensor})
            logits = out["logits"][0] # (num_classes,)
            live_embedding = out["embedding"][0] # (128,)
            
            # --- PHASE 9: PERSONAL SIGN OVERRIDE ---
            from engine.enrollment import EnrollmentEngine, get_cosine_similarity
            enroll_engine = EnrollmentEngine()
            
            best_custom_word = None
            best_custom_score = 0.0
            
            for word, prototype in enroll_engine.personal_vocab.items():
                sim = get_cosine_similarity(live_embedding, prototype)
                if sim > best_custom_score:
                    best_custom_score = sim
                    best_custom_word = word
                    
            if best_custom_score > 0.85:
                # Override!
                return {
                    "type": "SIGN_RECOGNIZED",
                    "gloss": best_custom_word,
                    "confidence": float(best_custom_score),
                    "alternatives": ["(Custom Sign)"],
                    "latency_ms": int((time.time() - start_time) * 1000)
                }
            # ---------------------------------------
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            top3_idx = np.argsort(probs)[-3:][::-1]
            top_prob = probs[top3_idx[0]]
            
            gloss = self.vocab_map.get(str(top3_idx[0]), f"Unknown_{top3_idx[0]}")
            alts = [self.vocab_map.get(str(i), f"Unknown_{i}") for i in top3_idx[1:]]
            
            if top_prob < 0.4:
                gloss = "UNSURE"
            
            return {
                "type": "SIGN_RECOGNIZED",
                "gloss": gloss,
                "confidence": float(top_prob),
                "alternatives": alts,
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        except Exception as e:
            logger.error("SignNet inference failed: %s", e)
            return {"type": "STATUS", "status": "idle"}
