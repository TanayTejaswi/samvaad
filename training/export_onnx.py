import os
import json
import torch
import numpy as np
import argparse

from training.models import SignNetGRU, SignNetTransformer

def export_onnx(args):
    print(f"Exporting model from {args.run_dir}")
    
    # Load vocabulary
    vocab_path = os.path.join(args.run_dir, "vocab.json")
    with open(vocab_path, "r") as f:
        idx_to_label = json.load(f)
        
    num_classes = len(idx_to_label)
    
    # Load PyTorch model
    model_path = os.path.join(args.run_dir, "best_model.pt")
    if args.model == "gru":
        model = SignNetGRU(input_dim=254, hidden_dim=128, num_classes=num_classes)
    else:
        model = SignNetTransformer(input_dim=254, d_model=128, num_classes=num_classes)
        
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    # Define static input shape (Batch=1, Time=48, Features=254)
    # The Hexagon NPU strongly prefers purely static shapes
    dummy_input = torch.randn(1, 48, 254)
    
    onnx_path = os.path.join(args.run_dir, f"signnet_{args.model}.onnx")
    
    print("Tracing and exporting to ONNX (opset 17)...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=17, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['logits', 'embedding'],
        # Dynamic axes are omitted on purpose to enforce static shapes for NPU
    )
    
    print(f"Successfully exported ONNX model to {onnx_path}")
    
    # Mathematical Verification using ONNX Runtime
    try:
        import onnxruntime as ort
        print("\nVerifying mathematical equivalence between PyTorch and ONNX Runtime CPU...")
        
        # PyTorch output
        with torch.no_grad():
            pt_logits, pt_emb = model(dummy_input)
            
        # ONNX output
        ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        
        onnx_logits = ort_outs[0]
        onnx_emb = ort_outs[1]
        
        # Compare
        max_diff_logits = np.max(np.abs(pt_logits.numpy() - onnx_logits))
        max_diff_emb = np.max(np.abs(pt_emb.numpy() - onnx_emb))
        
        print(f"Max absolute difference in logits: {max_diff_logits:.8f}")
        print(f"Max absolute difference in embedding: {max_diff_emb:.8f}")
        
        if max_diff_logits < 1e-4 and max_diff_emb < 1e-4:
            print("Verification PASSED: Models match within numerical tolerance.")
        else:
            print("WARNING: Verification FAILED. Outputs differ significantly.")
            
    except ImportError:
        print("onnxruntime not installed. Skipping numerical verification.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--model", type=str, choices=["gru", "transformer"], default="gru")
    args = parser.parse_args()
    
    export_onnx(args)
