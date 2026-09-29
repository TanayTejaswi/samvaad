import os
import glob
import numpy as np
import json
import argparse
from collections import defaultdict
import random

def generate_splits(npz_dir: str, out_dir: str, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Generates strict signer-independent train/val/test splits.
    If a signer is in the train set, they will NEVER appear in val or test.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    if not npz_files:
        print(f"No .npz files found in {npz_dir}")
        return
        
    # Group files by signer
    signer_to_files = defaultdict(list)
    for f in npz_files:
        data = np.load(f)
        signer_id = str(data['signer_id'])
        signer_to_files[signer_id].append(f)
        
    signers = list(signer_to_files.keys())
    random.shuffle(signers)
    
    num_signers = len(signers)
    num_val = max(1, int(num_signers * val_ratio))
    num_test = max(1, int(num_signers * test_ratio))
    
    # Strictly isolate signers
    test_signers = signers[:num_test]
    val_signers = signers[num_test:num_test + num_val]
    train_signers = signers[num_test + num_val:]
    
    def get_files_for_signers(signer_list):
        files = []
        for s in signer_list:
            files.extend(signer_to_files[s])
        return files
        
    train_files = get_files_for_signers(train_signers)
    val_files = get_files_for_signers(val_signers)
    test_files = get_files_for_signers(test_signers)
    
    # Save splits
    splits = {
        "train": [os.path.basename(f) for f in train_files],
        "val": [os.path.basename(f) for f in val_files],
        "test": [os.path.basename(f) for f in test_files]
    }
    
    out_path = os.path.join(out_dir, "dataset_splits.json")
    with open(out_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Splits generated at {out_path}")
    print(f"Train: {len(train_files)} files ({len(train_signers)} signers)")
    print(f"Val:   {len(val_files)} files ({len(val_signers)} signers)")
    print(f"Test:  {len(test_files)} files ({len(test_signers)} signers)")
    
    # Verify strict isolation
    assert set(train_signers).isdisjoint(set(val_signers)), "Leakage: Train/Val signers overlap!"
    assert set(train_signers).isdisjoint(set(test_signers)), "Leakage: Train/Test signers overlap!"
    assert set(val_signers).isdisjoint(set(test_signers)), "Leakage: Val/Test signers overlap!"
    print("Signer-independent isolation strictly verified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, default="data/raw_npz")
    parser.add_argument("--out_dir", type=str, default="data/splits")
    args = parser.parse_args()
    generate_splits(args.npz_dir, args.out_dir)
