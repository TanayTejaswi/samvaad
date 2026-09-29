import os
import glob
import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from inference.hands import HandPipeline

def extract_video(video_path: str, pipeline: HandPipeline) -> np.ndarray:
    """Extracts landmarks for a single video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return None
        
    frames_data = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        result = pipeline.infer({"image": frame})
        
        # Structure: [left_hand(21,3), right_hand(21,3)]
        frame_landmarks = np.zeros((2, 21, 3), dtype=np.float32)
        
        if result["left"] is not None:
            frame_landmarks[0] = result["left"]
        if result["right"] is not None:
            frame_landmarks[1] = result["right"]
            
        frames_data.append(frame_landmarks)
        
    cap.release()
    
    if len(frames_data) == 0:
        return None
        
    return np.array(frames_data, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Extract ISL keypoints from INCLUDE dataset videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to raw INCLUDE videos")
    parser.add_argument("--out_dir", type=str, default="data/raw_npz", help="Output directory for .npz files")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "npu"], help="Inference device")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    pipeline = HandPipeline(device=args.device)
    pipeline.load()
    
    # INCLUDE dataset structure: video_dir/Label/Video.mp4
    video_files = glob.glob(os.path.join(args.video_dir, "**", "*.mp4"), recursive=True)
    print(f"Found {len(video_files)} videos to process.")
    
    success_count = 0
    fail_count = 0
    
    for i, video_path in enumerate(video_files):
        # Extract metadata from INCLUDE structure
        path_parts = Path(video_path).parts
        label = path_parts[-2]
        video_name = path_parts[-1]
        
        # In INCLUDE, video names typically contain signer info, e.g. 'Aditi_Hello.mp4'
        # We extract the first part as signer_id. If standard format differs, adjust here.
        signer_id = video_name.split('_')[0] if '_' in video_name else "Unknown"
        
        print(f"[{i+1}/{len(video_files)}] Processing {label}/{video_name}...")
        
        landmarks = extract_video(video_path, pipeline)
        
        if landmarks is None or len(landmarks) == 0:
            fail_count += 1
            continue
            
        # Calculate mask: 1 if hand is present (not all zeros)
        # shape: (T, 2)
        presence_mask = np.any(landmarks != 0, axis=(2,3)).astype(np.float32)
        
        out_filename = f"{label}_{video_name.replace('.mp4', '.npz')}"
        out_path = os.path.join(args.out_dir, out_filename)
        
        np.savez_compressed(
            out_path,
            landmarks=landmarks,
            mask=presence_mask,
            label=label,
            signer_id=signer_id,
            fps=30.0 # Standardize or extract from cv2.CAP_PROP_FPS
        )
        success_count += 1
        
    print("\nExtraction Complete!")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    main()
