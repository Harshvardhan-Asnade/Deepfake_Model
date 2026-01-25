import torch
import torch.nn as nn
import os
import argparse
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import DeepfakeDetector
from src.config import Config

def export_model(output_path="deepguard.onnx", opset_version=17, quantization=False):
    """
    Export the DeepfakeDetector model to ONNX format.
    """
    print(f"🔄 Initializing DeepGuard Model...")
    device = torch.device('cpu') # Export on CPU is usually safer/standard
    
    # helper for loading weights if needed, though for export we might just want the architecture
    # checking if we need to load specific checkpoint or just pretrained structure
    # The user request implies exporting the "trained" model, but since I don't have the checkpoint path explicitly passed here,
    # I will default to loading the architecture with 'pretrained=True' (ImageNet weights) 
    # OR better, allow loading a checkpoint.
    
    model = DeepfakeDetector(pretrained=True)
    model.eval()
    model.to(device)
    
    # Try to load the best model if it exists
    checkpoint_dir = Config.CHECKPOINT_DIR
    checkpoint_name = "algro_markv2.safetensors" # Based on app.py
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    if os.path.exists(checkpoint_path):
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        try:
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint, using default weights: {e}")
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                print("✅ Checkpoint loaded successfully (torch.load)")
            except:
                pass
    else:
        print(f"⚠️ Checkpoint {checkpoint_name} not found. Exporting with default weights.")

    # Create dummy input
    # Shape: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE, device=device)
    
    print(f"⚙️  Exporting to ONNX (Opset {opset_version})...")
    
    # Dynamic axes allowed for flexible batch sizes
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"✅ Model exported to: {output_path}")
    
    # Verification
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX Model Check Passed")
    
    if quantization:
        print("📉 Performing Dynamic Quantization (INT8)...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = output_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(
            output_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )
        print(f"✅ Quantized model saved to: {quantized_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DeepGuard to ONNX")
    parser.add_argument("--output", type=str, default="deepguard.onnx", help="Output filename")
    parser.add_argument("--opset", type=int, default=17, help="ONNX Opset version")
    parser.add_argument("--quantize", action="store_true", help="Enable generic INT8 dynamic quantization")
    
    args = parser.parse_args()
    
    export_model(args.output, args.opset, args.quantize)
