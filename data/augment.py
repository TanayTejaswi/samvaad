import numpy as np
from typing import Optional

def augment_time_warp(sequence: np.ndarray, speed_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """Randomly speeds up or slows down the sequence using linear interpolation."""
    T, F = sequence.shape
    speed = np.random.uniform(speed_range[0], speed_range[1])
    target_T = int(T * speed)
    
    if target_T < 5:
        return sequence
        
    orig_steps = np.linspace(0, 1, T)
    target_steps = np.linspace(0, 1, target_T)
    
    warped = np.zeros((target_T, F), dtype=np.float32)
    for f in range(F):
        warped[:, f] = np.interp(target_steps, orig_steps, sequence[:, f])
        
    return warped

def augment_spatial_jitter(landmarks: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """Adds small Gaussian noise to the normalized landmark coordinates."""
    noise = np.random.normal(0, noise_std, landmarks.shape).astype(np.float32)
    return landmarks + noise

def augment_hand_dropout(sequence: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
    """Randomly zeros out hands for a few frames to simulate tracking loss."""
    T, H, L, C = sequence.shape # Assuming shape (T, 2, 21, 3) before normalization
    augmented = sequence.copy()
    
    for t in range(T):
        if np.random.rand() < drop_prob:
            # Drop left hand
            augmented[t, 0] = 0
        if np.random.rand() < drop_prob:
            # Drop right hand
            augmented[t, 1] = 0
            
    return augmented

def augment_mirror(sequence: np.ndarray, is_symmetric: bool = False) -> np.ndarray:
    """
    Mirrors the sign left-to-right. 
    Only valid for signs that mean the same thing regardless of handedness.
    """
    if not is_symmetric or np.random.rand() > 0.5:
        return sequence
        
    # sequence shape assumed to be (T, 2, 21, 3) where dim 1 is [left, right]
    mirrored = sequence.copy()
    
    # Swap left and right hands
    mirrored[:, [0, 1]] = mirrored[:, [1, 0]]
    
    # Invert X coordinate (assuming X is index 0 in the last dim)
    mirrored[:, :, :, 0] *= -1.0
    
    return mirrored
