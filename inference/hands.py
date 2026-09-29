import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional
import time

from inference.base import InferenceBackend
from inference.qnn_session import create_session

logger = logging.getLogger("samvaad.inference.hands")

# Standard MediaPipe SSD Anchor generation
def generate_anchors(input_size_width: int, input_size_height: int) -> np.ndarray:
    """Generates anchor boxes for SSD detection."""
    # Based on MediaPipe's SsdAnchorsCalculator options
    min_scale = 0.1484375
    max_scale = 0.75
    num_layers = 4
    feature_map_width = [0, 0, 0, 0]
    feature_map_height = [0, 0, 0, 0]
    strides = [8, 16, 16, 16]
    aspect_ratios = [1.0]
    reduce_boxes_in_lowest_layer = False
    interpolated_scale_aspect_ratio = 1.0
    fixed_anchor_size = True

    anchors = []
    layer_id = 0
    while layer_id < num_layers:
        anchor_height = []
        anchor_width = []
        aspect_ratios_list = []
        scales = []
        
        last_same_stride_layer = layer_id
        while last_same_stride_layer < num_layers and strides[last_same_stride_layer] == strides[layer_id]:
            scale = min_scale + (max_scale - min_scale) * 1.0 * last_same_stride_layer / (num_layers - 1.0)
            if last_same_stride_layer == 0 and reduce_boxes_in_lowest_layer:
                aspect_ratios_list.append(1.0)
                aspect_ratios_list.append(2.0)
                aspect_ratios_list.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)
            else:
                aspect_ratios_list.append(1.0)
                scales.append(scale)
                if interpolated_scale_aspect_ratio > 0.0:
                    scale_next = 1.0 if last_same_stride_layer == num_layers - 1 else min_scale + (max_scale - min_scale) * 1.0 * (last_same_stride_layer + 1) / (num_layers - 1.0)
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios_list.append(interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
            
        for i in range(len(aspect_ratios_list)):
            ratio_sqrts = np.sqrt(aspect_ratios_list[i])
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)
            
        stride = strides[layer_id]
        feature_map_height = int(np.ceil(1.0 * input_size_height / stride))
        feature_map_width = int(np.ceil(1.0 * input_size_width / stride))
        
        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + 0.5) / feature_map_width
                    y_center = (y + 0.5) / feature_map_height
                    if fixed_anchor_size:
                        w = 1.0
                        h = 1.0
                    else:
                        w = anchor_width[anchor_id]
                        h = anchor_height[anchor_id]
                    anchors.append([x_center, y_center, w, h])
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)

def decode_bboxes(raw_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Decodes SSD raw boxes into [x_center, y_center, w, h, keypoint_x, keypoint_y...] relative coordinates."""
    # raw_boxes shape: (num_anchors, 18) [dx, dy, w, h, 7 * (kx, ky)]
    x_center = raw_boxes[:, 0] / 256.0 * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[:, 1] / 256.0 * anchors[:, 3] + anchors[:, 1]
    w = raw_boxes[:, 2] / 256.0 * anchors[:, 2]
    h = raw_boxes[:, 3] / 256.0 * anchors[:, 3]
    
    # decode keypoints
    keypoints = np.zeros((raw_boxes.shape[0], 14), dtype=np.float32)
    for k in range(7):
        offset = 4 + k * 2
        keypoints[:, k*2] = raw_boxes[:, offset] / 256.0 * anchors[:, 2] + anchors[:, 0]
        keypoints[:, k*2+1] = raw_boxes[:, offset+1] / 256.0 * anchors[:, 3] + anchors[:, 1]
        
    return np.column_stack([x_center, y_center, w, h, keypoints])

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> List[int]:
    """Non-maximum suppression."""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

def get_rotated_crop_matrix(x_center, y_center, width, height, rotation, target_size=256):
    """Calculates affine transform matrix for rotating and cropping hand ROI."""
    # Convert angle to degrees for cv2
    angle_deg = rotation * 180.0 / np.pi
    
    # Scale box up (MediaPipe scales palm box by 2.6 to fit full hand)
    scale = 2.6
    crop_w = width * scale
    crop_h = height * scale
    
    # Get rotation matrix around center
    matrix = cv2.getRotationMatrix2D((x_center, y_center), angle_deg, 1.0)
    
    # Adjust matrix to map to target_size x target_size
    matrix[0, 2] += target_size / 2 - x_center
    matrix[1, 2] += target_size / 2 - y_center
    
    # Scale matrix to crop rect
    scale_x = target_size / crop_w
    scale_y = target_size / crop_h
    matrix[0] *= scale_x
    matrix[1] *= scale_y
    
    return matrix

class HandPipeline(InferenceBackend):
    """Full MediaPipe Hand Landmark pipeline running entirely via ONNX."""
    
    def __init__(self, palm_model_path: str = "models/hands/palm_detection.onnx", 
                 landmark_model_path: str = "models/hands/hand_landmark.onnx",
                 device: str = "npu"):
        self.palm_model_path = palm_model_path
        self.landmark_model_path = landmark_model_path
        self.device = device
        self.palm_session = None
        self.landmark_session = None
        self._backend = "Uninitialized"
        
        # Precompute anchors
        self.anchors = generate_anchors(256, 256)
        
        # Tracking state
        self.tracking = False
        self.prev_rect = None  # (x_center, y_center, width, height, rotation)

    def load(self, model_dir: str = "") -> None:
        try:
            self.palm_session, ep1 = create_session(self.palm_model_path, self.device)
            self.landmark_session, ep2 = create_session(self.landmark_model_path, self.device)
            self._backend = f"QNN ({ep1})"
            logger.info("Hand pipeline loaded on %s", self._backend)
        except Exception as e:
            logger.error("Failed to load Hand models: %s", e)
            self._backend = "Dummy/Failed"

    def warmup(self, n: int = 1) -> None:
        if not self.palm_session: return
        dummy_img = np.zeros((1, 256, 256, 3), dtype=np.float32)
        for _ in range(n):
            self.palm_session.run(None, {self.palm_session.get_inputs()[0].name: dummy_img})
            self.landmark_session.run(None, {self.landmark_session.get_inputs()[0].name: dummy_img})

    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Runs palm detection -> landmark detection. 
        Args:
            inputs: {'image': cv2 numpy image (H, W, 3) BGR}
        Returns:
            {'left': landmarks, 'right': landmarks, 'handedness': str, 'timestamp': float}
        """
        if self.palm_session is None:
            return {"left": None, "right": None, "handedness": None, "timestamp": time.time()}

        frame = inputs["image"]
        h_img, w_img, _ = frame.shape
        
        # 1. Tracking or Detection
        rect = None
        if self.tracking and self.prev_rect is not None:
            rect = self.prev_rect
        else:
            # Run Palm Detection
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256))
            img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
            img_input = np.expand_dims(img_normalized, axis=0)
            
            # The exact output names depend on the ONNX export, assume standard MediaPipe shapes
            out = self.palm_session.run(None, {self.palm_session.get_inputs()[0].name: img_input})
            # Assume out[0] is regressors (1, 2016, 18), out[1] is classificators (1, 2016, 1)
            regressors, classificators = out[0][0], out[1][0]
            
            scores = 1.0 / (1.0 + np.exp(-classificators.squeeze()))
            mask = scores > 0.5
            
            if not np.any(mask):
                self.tracking = False
                return {"left": None, "right": None, "handedness": None, "timestamp": time.time()}
                
            filtered_boxes = decode_bboxes(regressors[mask], self.anchors[mask])
            filtered_scores = scores[mask]
            
            keep = nms(filtered_boxes, filtered_scores, iou_threshold=0.3)
            if not keep:
                self.tracking = False
                return {"left": None, "right": None, "handedness": None, "timestamp": time.time()}
                
            best_box = filtered_boxes[keep[0]]
            
            # Convert normalized coordinates to pixel scale (256x256 image -> real image size)
            x_center = best_box[0] * w_img
            y_center = best_box[1] * h_img
            width = best_box[2] * w_img
            height = best_box[3] * h_img
            
            # Rotation is calculated from palm keypoints (wrist and middle finger MCP)
            kp_wrist = (best_box[4] * w_img, best_box[5] * h_img)
            kp_mid = (best_box[8] * w_img, best_box[9] * h_img)
            rotation = np.arctan2(kp_mid[1] - kp_wrist[1], kp_mid[0] - kp_wrist[0]) - np.pi/2
            
            rect = (x_center, y_center, width, height, rotation)

        # 2. Hand Landmarker
        x_center, y_center, width, height, rotation = rect
        matrix = get_rotated_crop_matrix(x_center, y_center, width, height, rotation, target_size=256)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop_img = cv2.warpAffine(img_rgb, matrix, (256, 256), flags=cv2.INTER_LINEAR)
        crop_normalized = crop_img.astype(np.float32) / 255.0
        crop_input = np.expand_dims(crop_normalized, axis=0)
        
        out = self.landmark_session.run(None, {self.landmark_session.get_inputs()[0].name: crop_input})
        
        # Standard landmark output is usually out[0] -> (1, 63)
        landmarks = out[0][0].reshape(-1, 3)
        hand_flag = out[1][0][0] if len(out) > 1 else 1.0 # hand presence score
        handedness = out[2][0][0] if len(out) > 2 else 0.5 # handedness score
        
        if hand_flag < 0.5:
            self.tracking = False
            return {"left": None, "right": None, "handedness": None, "timestamp": time.time()}
            
        # Transform landmarks back to original image space
        inv_matrix = cv2.invertAffineTransform(matrix)
        world_landmarks = []
        for i in range(21):
            lx, ly, lz = landmarks[i]
            # lx, ly are in 0-256 scale
            p = np.array([lx, ly, 1.0])
            orig_p = np.dot(inv_matrix, p)
            world_landmarks.append([orig_p[0], orig_p[1], lz])
            
        world_landmarks = np.array(world_landmarks)
        
        # Calculate new rect from landmarks for next frame
        min_x, min_y = np.min(world_landmarks[:, 0]), np.min(world_landmarks[:, 1])
        max_x, max_y = np.max(world_landmarks[:, 0]), np.max(world_landmarks[:, 1])
        new_w = max_x - min_x
        new_h = max_y - min_y
        new_cx = min_x + new_w / 2
        new_cy = min_y + new_h / 2
        
        kp_wrist = world_landmarks[0]
        kp_mid = world_landmarks[9]
        new_rotation = np.arctan2(kp_mid[1] - kp_wrist[1], kp_mid[0] - kp_wrist[0]) - np.pi/2
        
        self.prev_rect = (new_cx, new_cy, new_w, new_h, new_rotation)
        self.tracking = True
        
        # handedness score > 0.5 is usually Left (mirrored)
        hand_name = "Left" if handedness > 0.5 else "Right"
        
        result = {"left": None, "right": None, "handedness": hand_name, "timestamp": time.time()}
        if hand_name == "Left":
            result["left"] = world_landmarks
        else:
            result["right"] = world_landmarks
            
        return result

    @property
    def backend_name(self) -> str:
        return self._backend
