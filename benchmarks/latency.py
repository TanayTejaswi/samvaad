"""Latency Benchmarking Utility.

Tests the NPU and CPU engines over multiple iterations to determine
P95 and P99 latency statistics, saving results to a CSV log.
"""

import csv
import logging
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from inference.whisper_cpu import WhisperCPUEngine
from inference.whisper_npu import WhisperNPUEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("samvaad.benchmarks.latency")


def run_latency_test(iterations: int = 50) -> None:
    """Executes a latency benchmark on both engines."""
    results = []
    dummy_mel = np.zeros((1, 80, 3000), dtype=np.float32)
    
    # 1. Test CPU Engine
    logger.info("Initializing CPU Engine for benchmark...")
    cpu_engine = WhisperCPUEngine()
    cpu_engine.load("models/whisper")
    cpu_engine.warmup()
    
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        cpu_engine.infer(dummy_mel)
        cpu_times.append((time.perf_counter() - start) * 1000)
        
    cpu_p95 = np.percentile(cpu_times, 95)
    cpu_p99 = np.percentile(cpu_times, 99)
    results.append({
        "engine": "CPU",
        "iterations": iterations,
        "mean_ms": np.mean(cpu_times),
        "p95_ms": cpu_p95,
        "p99_ms": cpu_p99
    })
    
    # 2. Test NPU Engine
    logger.info("Initializing NPU Engine for benchmark...")
    npu_engine = WhisperNPUEngine()
    npu_engine.load("models/whisper")
    npu_engine.warmup()
    
    npu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        npu_engine.infer(dummy_mel)
        npu_times.append((time.perf_counter() - start) * 1000)
        
    npu_p95 = np.percentile(npu_times, 95)
    npu_p99 = np.percentile(npu_times, 99)
    results.append({
        "engine": "NPU",
        "iterations": iterations,
        "mean_ms": np.mean(npu_times),
        "p95_ms": npu_p95,
        "p99_ms": npu_p99
    })
    
    # 3. Export to CSV
    export_path = Path("data/benchmarks.csv")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = export_path.exists()
    
    with open(export_path, mode='a', newline='') as csvfile:
        fieldnames = ["timestamp", "engine", "iterations", "mean_ms", "p95_ms", "p99_ms"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for res in results:
            res["timestamp"] = timestamp
            writer.writerow(res)
            
    logger.info("Latency benchmark complete. Results saved to %s", export_path)


if __name__ == "__main__":
    run_latency_test()
