import sys
from pathlib import Path
import yaml
import time
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from language.gloss2text import GlossToTextEngine

def evaluate(args):
    print("--- Samvaad: Gloss2Text Evaluation ---")
    
    try:
        with open(args.eval_file, 'r') as f:
            data = yaml.safe_load(f)
            sequences = data.get("eval_sequences", [])
    except Exception as e:
        print(f"Failed to load eval file: {e}")
        return
        
    if not sequences:
        print("No evaluation sequences found.")
        return
        
    engine = GlossToTextEngine(backend=args.backend)
    
    total = len(sequences)
    fallback_count = 0
    latencies = []
    
    print(f"\nEvaluating {total} sequences using {args.backend}...\n")
    
    for seq in sequences:
        gloss = seq["gloss"]
        ref = seq["reference"]
        
        result = engine.translate(gloss)
        out_text = result["text"]
        backend_used = result["backend"]
        lat = result["latency_ms"]
        
        latencies.append(lat)
        if backend_used == "rule_fallback":
            fallback_count += 1
            
        print(f"Gloss: {gloss}")
        print(f"Ref:   {ref}")
        print(f"Out:   {out_text}  [{backend_used}] ({lat} ms)\n")
        
    avg_latency = sum(latencies) / len(latencies)
    rejection_rate = (fallback_count / total) * 100
    
    print("--- Evaluation Summary ---")
    print(f"Total Sequences: {total}")
    print(f"Average Latency: {avg_latency:.1f} ms")
    print(f"Guardrail Rejection Rate: {rejection_rate:.1f}% ({fallback_count} fallbacks)")
    print("(Exact-meaning accuracy requires human review of the 'Out' vs 'Ref' above)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default="data/gloss_eval.yaml")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "rule_fallback"])
    args = parser.parse_args()
    
    evaluate(args)
