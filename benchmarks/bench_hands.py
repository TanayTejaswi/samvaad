import time
import numpy as np
import sys
import os
from pathlib import Path
import csv

sys.path.append(str(Path(__file__).parent.parent))
from inference.hands import HandPipeline

def main():
    print("Benchmarking Hand Pipeline...")
    
    # Create dummy video frames (500 frames of 640x480 noise)
    print("Generating dummy video frames for strict benchmarking...")
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(500)]
    
    pipeline_npu = HandPipeline(device="npu")
    pipeline_cpu = HandPipeline(device="cpu")
    
    pipeline_npu.load()
    pipeline_cpu.load()
    
    results = []
    
    for name, pipeline in [("CPU", pipeline_cpu), ("NPU", pipeline_npu)]:
        print(f"\nBenchmarking {name}...")
        pipeline.warmup(10)
        
        latencies = []
        for i, frame in enumerate(frames):
            start = time.perf_counter()
            res = pipeline.infer({"image": frame})
            latencies.append((time.perf_counter() - start) * 1000)
            
            if i % 100 == 0:
                print(f"Processed {i}/500 frames...")
                
        median = np.median(latencies)
        p90 = np.percentile(latencies, 90)
        fps = 1000.0 / median
        print(f"[{name}] Median Latency: {median:.2f} ms | P90: {p90:.2f} ms | FPS: {fps:.1f}")
        
        results.append({
            "Device": name,
            "Frames": 500,
            "Median_ms": round(median, 2),
            "P90_ms": round(p90, 2),
            "FPS": round(fps, 1)
        })
        
    # Save CSV
    os.makedirs("benchmarks/results", exist_ok=True)
    csv_path = f"benchmarks/results/hands_benchmark_{int(time.time())}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Device", "Frames", "Median_ms", "P90_ms", "FPS"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nBenchmark saved to {csv_path}")

if __name__ == "__main__":
    main()
