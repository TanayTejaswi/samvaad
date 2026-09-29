import numpy as np
from typing import Dict, Any, Tuple

def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """
    Centers hand on the wrist (landmark 0) and scales to a fixed palm size.
    Args:
        landmarks: np.ndarray of shape (21, 3)
    Returns:
        normalized_landmarks: np.ndarray of shape (21, 3)
    """
    if landmarks is None or np.all(landmarks == 0):
        return np.zeros((21, 3), dtype=np.float32)
        
    # Center on wrist
    wrist = landmarks[0].copy()
    centered = landmarks - wrist
    
    # Scale based on palm size (distance from wrist to middle finger MCP)
    middle_mcp = centered[9]
    palm_size = np.linalg.norm(middle_mcp)
    
    if palm_size > 1e-5:
        scaled = centered / palm_size
    else:
        scaled = centered
        
    return scaled

def resample_sequence(sequence: np.ndarray, target_length: int = 48) -> np.ndarray:
    """
    Linearly resamples a sequence of frames to a fixed target length.
    Args:
        sequence: np.ndarray of shape (T, F)
        target_length: int
    Returns:
        resampled: np.ndarray of shape (target_length, F)
    """
    T, F = sequence.shape
    if T == 0:
        return np.zeros((target_length, F), dtype=np.float32)
    if T == target_length:
        return sequence
        
    # Create original timeline and target timeline
    orig_steps = np.linspace(0, 1, T)
    target_steps = np.linspace(0, 1, target_length)
    
    resampled = np.zeros((target_length, F), dtype=np.float32)
    for f in range(F):
        resampled[:, f] = np.interp(target_steps, orig_steps, sequence[:, f])
        
    return resampled

def extract_features(sequence: np.ndarray, target_length: int = 48) -> np.ndarray:
    """
    Full feature extraction pipeline used for both training and live inference.
    Args:
        sequence: np.ndarray of shape (T, 2, 21, 3) 
                  where 2 is [left_hand, right_hand]
    Returns:
        features: np.ndarray of shape (target_length, 254)
                  [left_norm(63), right_norm(63), left_vel(63), right_vel(63), left_mask(1), right_mask(1)]
    """
    T = sequence.shape[0]
    
    # Initialize arrays
    left_norm = np.zeros((T, 63), dtype=np.float32)
    right_norm = np.zeros((T, 63), dtype=np.float32)
    left_mask = np.zeros((T, 1), dtype=np.float32)
    right_mask = np.zeros((T, 1), dtype=np.float32)
    
    for t in range(T):
        # Left hand
        lh = sequence[t, 0]
        if not np.all(lh == 0):
            left_norm[t] = normalize_hand(lh).flatten()
            left_mask[t] = 1.0
            
        # Right hand
        rh = sequence[t, 1]
        if not np.all(rh == 0):
            right_norm[t] = normalize_hand(rh).flatten()
            right_mask[t] = 1.0
            
    # Calculate velocity (delta between frames)
    # prepend a zero frame to maintain length T
    left_vel = np.vstack([np.zeros((1, 63)), np.diff(left_norm, axis=0)])
    right_vel = np.vstack([np.zeros((1, 63)), np.diff(right_norm, axis=0)])
    
    # Concatenate all features for time T
    # Shape: (T, 63 + 63 + 63 + 63 + 1 + 1) = (T, 254)
    raw_features = np.concatenate([left_norm, right_norm, left_vel, right_vel, left_mask, right_mask], axis=1)
    
    # Resample to fixed temporal length
    final_features = resample_sequence(raw_features, target_length=target_length)
    
    return final_features
