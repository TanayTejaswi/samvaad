import os
import argparse
import qai_hub as hub

def compile_model(args):
    print("--- Samvaad: Snapdragon AI Hub NPU Compiler ---")
    print(f"Targeting model: {args.onnx_path}")
    print("WARNING: This script requires 'qai-hub-models' and an active Qualcomm AI Hub token.\n")
    
    if not os.path.exists(args.onnx_path):
        print(f"Error: Could not find {args.onnx_path}. Run Phase 4 training and export first.")
        return
        
    # Compile for Hexagon NPU using QNN ONNX Runtime backend
    # For Snapdragon X Elite series laptops
    target_device = hub.Device("Snapdragon X Elite CRD")
    
    # 1. Upload Model
    print("1. Uploading ONNX model to AI Hub...")
    model = hub.upload_model(args.onnx_path)
    
    # 2. Submit Compile Job
    print("2. Submitting Compile Job targeting Hexagon NPU (w8a16 quantized)...")
    compile_options = (
        "--target_runtime qnn_lib_aarch64_windows "
        "--quantize_full_type int8 "
        "--quantize_io"
    )
    
    compile_job = hub.submit_compile_job(
        model=model,
        device=target_device,
        input_specs={"input": ((1, 48, 254), "float32")},
        options=compile_options,
    )
    
    # Wait for completion
    print("Waiting for compilation to finish (this may take several minutes)...")
    target_model = compile_job.get_target_model()
    
    # 3. Submit Profile Job
    print("3. Submitting Profile Job to verify NPU execution...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=target_device,
    )
    
    print("Waiting for profiling to finish...")
    profile_data = profile_job.get_profile()
    
    # Check if layers fell back to CPU
    compute_units = profile_data.execution_summary.compute_units
    print("\n--- Profile Summary ---")
    print(f"NPU Layers: {compute_units.get('NPU', 0)}")
    print(f"CPU Layers: {compute_units.get('CPU', 0)}")
    
    if compute_units.get('CPU', 0) > 0:
        print("WARNING: Some layers fell back to CPU! Model is not fully optimized for NPU.")
    else:
        print("SUCCESS: Model runs 100% on the Hexagon NPU!")
        
    # 4. Download Compiled Model
    out_dir = "models/signnet"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n4. Downloading compiled QNN model to {out_dir}...")
    target_model.download(out_dir)
    print("Done! You can now load this model in Phase 6.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, default="training/runs/latest/signnet_gru.onnx")
    args = parser.parse_args()
    
    compile_model(args)
