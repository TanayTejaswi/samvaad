"""Word Error Rate (WER) Evaluation.

Uses Levenshtein distance against a tiny hardcoded ground-truth set
to evaluate model accuracy in isolated offline execution.
"""

import csv
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("samvaad.benchmarks.wer")


def levenshtein_distance(ref: list[str], hyp: list[str]) -> int:
    """Calculates Levenshtein distance between two lists of words."""
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    for i in range(len(ref) + 1):
        d[i, 0] = i
    for j in range(len(hyp) + 1):
        d[0, j] = j
        
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
                
    return d[len(ref), len(hyp)]


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculates WER for a single pair of sentences."""
    import re
    ref_words = re.sub(r'[^\w\s]', '', reference.lower()).split()
    hyp_words = re.sub(r'[^\w\s]', '', hypothesis.lower()).split()
    
    if len(ref_words) == 0:
        return 0.0
        
    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist) / len(ref_words)


def run_wer_test() -> None:
    """Executes a hardcoded WER benchmark test."""
    # Since we are mocking NPU inference when weights are missing,
    # we simulate the evaluation process.
    ground_truth = [
        ("Hello, welcome to Samvaad.", "[MOCK NPU] Hello, welcome to Samvaad."),
        ("This is a test of the offline system.", "[MOCK NPU] Hello, welcome to Samvaad.")
    ]
    
    total_wer = 0.0
    for ref, hyp in ground_truth:
        # In Python, str.replace() doesn't take regex directly without re module.
        # Let's fix the regex logic.
        import re
        ref_clean = re.sub(r'[^\w\s]', '', ref.lower())
        hyp_clean = re.sub(r'[^\w\s]', '', hyp.lower())
        
        wer = calculate_wer(ref_clean, hyp_clean)
        logger.info("Ref: '%s' | Hyp: '%s' | WER: %.2f", ref, hyp, wer)
        total_wer += wer
        
    avg_wer = total_wer / len(ground_truth)
    logger.info("Average WER: %.2f", avg_wer)
    
    # Export to CSV
    export_path = Path("data/wer_results.csv")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = export_path.exists()
    
    with open(export_path, mode='a', newline='') as csvfile:
        fieldnames = ["timestamp", "dataset_size", "average_wer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(ground_truth),
            "average_wer": avg_wer
        })


if __name__ == "__main__":
    run_wer_test()
