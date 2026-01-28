import os
import sys
import torch
import numpy as np
import collections
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # 'Deepfake Project /Morden Detections system'
model_root = os.path.join(project_root, "model")
sys.path.insert(0, model_root)

from src.config import Config
from src.models import DeepfakeDetector
from src.dataset import DeepfakeDataset
from safetensors.torch import load_model

def load_detector(model_path, device):
    """Loads a model instance."""
    print(f"📦 Loading {os.path.basename(model_path)}...")
    model = DeepfakeDetector(pretrained=False).to(device)
    load_model(model, model_path, strict=False)
    model.eval()
    return model

def compare_models(limit=1000):
    Config.setup()
    device = torch.device(Config.DEVICE)
    
    # Paths
    model2_path = os.path.join(Config.CHECKPOINT_DIR, "Mark-II.safetensors")
    model5_path = os.path.join(Config.CHECKPOINT_DIR, "Mark-V.safetensors")
    
    # 1. Load Data
    print(f"\n📂 Loading Universal Dataset (Limit: {limit})...")
    # We scan the dataset root
    files, labels = DeepfakeDataset.scan_directory(Config.DATA_DIR)
    
    # Shuffle and select subset
    combined = list(zip(files, labels))
    import random
    random.shuffle(combined)
    subset = combined[:limit]
    
    val_files, val_labels = zip(*subset)
    
    dataset = DeepfakeDataset(file_paths=list(val_files), labels=list(val_labels), phase='val')
    loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    
    print(f"🔹 Testing on {len(val_files)} images.")
    
    # 2. Load Models
    mark2 = load_detector(model2_path, device)
    mark5 = load_detector(model5_path, device)
    
    # 3. Battle Loop
    m2_preds = []
    m5_preds = []
    true_labels = []
    
    print("\n⚔️  Starting Battle: Mark-II vs Mark-V  ⚔️")
    
    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Battling"):
            images = images.to(device)
            lbls = lbls.numpy()
            
            # Mark-II Inference
            out2 = mark2(images)
            prob2 = torch.sigmoid(out2).cpu().numpy()
            pred2 = (prob2 > 0.5).astype(int)
            
            # Mark-V Inference
            out5 = mark5(images)
            prob5 = torch.sigmoid(out5).cpu().numpy()
            pred5 = (prob5 > 0.5).astype(int)
            
            m2_preds.extend(pred2)
            m5_preds.extend(pred5)
            true_labels.extend(lbls)
            
    # 4. Results
    acc2 = accuracy_score(true_labels, m2_preds)
    acc5 = accuracy_score(true_labels, m5_preds)
    
    print("\n" + "="*40)
    print("🏆 BATTLE RESULTS")
    print("="*40)
    print(f"Dataset Size: {limit} Images")
    print("-" * 40)
    print(f"🤖 MARK-II Accuracy: {acc2*100:.2f}%")
    print(f"🦅 MARK-V  Accuracy: {acc5*100:.2f}%")
    print("-" * 40)
    
    if acc5 > acc2:
        diff = (acc5 - acc2) * 100
        print(f"🎉 WINNER: MARK-V (+{diff:.2f}%)")
    elif acc2 > acc5:
        diff = (acc2 - acc5) * 100
        print(f"🎉 WINNER: MARK-II (+{diff:.2f}%)")
    else:
        print("🤝 DRAW")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Check for CLI arg
    import sys
    limit = 10000 # Default to 10k for speed
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    
    compare_models(limit)
