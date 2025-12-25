import torch
import os
import sys
# Add 'model' directory to path so we can import src
sys.path.append(os.path.join(os.getcwd(), 'model'))

from safetensors.torch import load_file
from src.models import DeepfakeDetector
from src.config import Config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_checkpoint(path):
    print(f"Analyzing: {os.path.basename(path)}")
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    try:
        # Load Architecture
        model = DeepfakeDetector(pretrained=False) # Don't download weights, we will load file
        
        # Load Weights
        if path.endswith(".safetensors"):
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu")
            
        model.load_state_dict(state_dict, strict=False)
        
        params = count_parameters(model)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        print(f" - Parameters: {params:,}")
        print(f" - File Size: {size_mb:.2f} MB")
        print(f" - Keys in State Dict: {len(state_dict)}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    base_dir = "model/results/checkpoints"
    files = [
        "best_finetuned_largest.safetensors", 
        "best_model.safetensors", 
        "patched_model.safetensors"
    ]
    
    Config.setup()
    
    for f in files:
        if len(sys.argv) > 1:
            # Allow custom paths
            path = sys.argv[1]
        else:
            path = os.path.join(base_dir, f)
            
        analyze_checkpoint(path)
        print("-" * 30)
