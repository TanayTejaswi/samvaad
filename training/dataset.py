import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from inference.features import extract_features
from data.augment import augment_hand_dropout, augment_spatial_jitter, augment_mirror

class ISLDataset(Dataset):
    def __init__(self, npz_dir: str, splits_json: str, split: str = "train", is_training: bool = False):
        self.npz_dir = npz_dir
        self.is_training = is_training
        
        with open(splits_json, 'r') as f:
            splits = json.load(f)
            
        if split not in splits:
            raise ValueError(f"Split {split} not found in {splits_json}")
            
        self.files = splits[split]
        
        # Build vocabulary
        all_files = []
        for s in splits.values():
            all_files.extend(s)
            
        self.vocab = sorted(list(set([f.split('_')[0] for f in all_files]))) # e.g. "Hello_Aditi.npz" -> "Hello"
        self.label_to_idx = {label: i for i, label in enumerate(self.vocab)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.num_classes = len(self.vocab)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.npz_dir, self.files[idx])
        data = np.load(file_path)
        
        landmarks = data['landmarks'].astype(np.float32) # shape: (T, 2, 21, 3)
        label_str = str(data['label'])
        
        # Apply augmentations on raw landmarks
        if self.is_training:
            landmarks = augment_hand_dropout(landmarks, drop_prob=0.05)
            landmarks = augment_mirror(landmarks, is_symmetric=False) # Only mirror if true, keeping false for safety
            landmarks = augment_spatial_jitter(landmarks, noise_std=0.005)
            
        # Extract features (normalizes and resamples to fixed T=48)
        features = extract_features(landmarks, target_length=48)
        
        label_idx = self.label_to_idx[label_str]
        
        return torch.tensor(features), torch.tensor(label_idx, dtype=torch.long)
