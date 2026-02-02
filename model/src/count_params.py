import os
import sys
import torch
import textwrap

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
model_root = os.path.join(project_root, "model")
sys.path.insert(0, model_root)

from src.models import DeepfakeDetector

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fmt_params(num):
    return f"{num/1e6:.2f}M"

def main():
    print("📦 Instantiating Mark-V Architecture...")
    model = DeepfakeDetector(pretrained=False)
    
    total = count_parameters(model)
    rgb = count_parameters(model.rgb_branch)
    freq = count_parameters(model.freq_branch)
    patch = count_parameters(model.patch_branch)
    vit = count_parameters(model.vit_branch)
    
    print("\n" + "="*40)
    print("📊 MODEL PARAMETER COUNT (Mark-V)")
    print("="*40)
    print(f"Total Parameters:    {fmt_params(total)}")
    print("-" * 40)
    print(f" • RGB Branch (EffNet-V2-S): {fmt_params(rgb)}")
    print(f" • ViT Branch (Swin-V2-T):   {fmt_params(vit)}")
    print(f" • Frequency Branch:         {fmt_params(freq)}")
    print(f" • Patch Branch:             {fmt_params(patch)}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
