import os
import glob
import numpy as np
import json
import argparse
from collections import Counter

def report_dataset(npz_dir: str, splits_json: str):
    print("--- Samvaad Dataset Analytics Report ---\n")
    
    if not os.path.exists(splits_json):
        print(f"Error: {splits_json} not found. Run splits.py first.")
        return
        
    with open(splits_json, 'r') as f:
        splits = json.load(f)
        
    all_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    print(f"Total extracted .npz files: {len(all_files)}")
    
    for split_name, file_names in splits.items():
        print(f"\n[{split_name.upper()} SPLIT]")
        print(f"Total sequences: {len(file_names)}")
        
        labels = []
        frame_counts = []
        signer_ids = set()
        
        for fname in file_names:
            fpath = os.path.join(npz_dir, fname)
            if not os.path.exists(fpath):
                continue
                
            data = np.load(fpath)
            labels.append(str(data['label']))
            signer_ids.add(str(data['signer_id']))
            
            # check shape of landmarks to get frame count
            frames = data['landmarks'].shape[0]
            frame_counts.append(frames)
            
        if not labels:
            print("No valid data files found in this split.")
            continue
            
        print(f"Unique Signers: {len(signer_ids)}")
        print(f"Unique Classes (Words): {len(set(labels))}")
        
        avg_frames = np.mean(frame_counts)
        min_frames = np.min(frame_counts)
        max_frames = np.max(frame_counts)
        print(f"Frame Count - Avg: {avg_frames:.1f}, Min: {min_frames}, Max: {max_frames}")
        
        # Print top 5 classes
        counter = Counter(labels)
        top5 = counter.most_common(5)
        print("Top 5 classes by frequency:")
        for label, count in top5:
            print(f"  - {label}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, default="data/raw_npz")
    parser.add_argument("--splits_json", type=str, default="data/splits/dataset_splits.json")
    args = parser.parse_args()
    
    report_dataset(args.npz_dir, args.splits_json)
