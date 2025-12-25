import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file
import pandas as pd

# Add src to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, 'src'))

from dataset import DeepfakeDataset
from models import DeepfakeDetector
from config import Config

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_test_images.py <dataset_path> <model_path>")
        return

    dataset_path = sys.argv[1]
    model_path = sys.argv[2]
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    # 1. Load Dataset
    print(f"Scanning images in: {dataset_path}")
    # Use 'val' phase to get simple resize/normalize transforms without augmentation
    dataset = DeepfakeDataset(root_dir=dataset_path, phase='val')
    
    if len(dataset) == 0:
        print("No images found.")
        return
        
    print(f"Found {len(dataset)} images.")
    
    # 2. Load Model
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    model = DeepfakeDetector(pretrained=True)
    
    # Handle model path
    if not os.path.exists(model_path):
        # Try finding it in checkpoints dir
        chk_path = os.path.join(Config.CHECKPOINT_DIR, model_path)
        if os.path.exists(chk_path):
            model_path = chk_path
        else:
            print(f"Model not found: {model_path}")
            return
            
    print(f"Loading Model: {model_path}")
    if model_path.endswith(".safetensors"):
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location=device)
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 3. Inference
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4) # Shuffle true for random sample
    
    correct = 0
    total = 0
    fp = 0 
    fn = 0 
    tp = 0
    tn = 0
    
    limit = 2000 # Check max 2000 images if too large
    if len(sys.argv) > 3:
        limit = int(sys.argv[3])
        
    print("-" * 50)
    print(f"Running Inference (Limit: {limit})...")
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            if total >= limit:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            
            if labels.dim() > 1: labels = labels.squeeze()
            
            matches = (preds == labels)
            correct += matches.sum().item()
            total += labels.size(0)
            
            t_pos = ((preds == 1) & (labels == 1)).sum().item()
            t_neg = ((preds == 0) & (labels == 0)).sum().item()
            f_pos = ((preds == 1) & (labels == 0)).sum().item()
            f_neg = ((preds == 0) & (labels == 1)).sum().item()
            
            tp += t_pos
            tn += t_neg
            fp += f_pos
            fn += f_neg
            
    # 4. Results
    acc = (correct / total) * 100
    
    print("\n" + "=" * 50)
    print(f" RESULTS: {os.path.basename(dataset_path)}")
    print("=" * 50)
    print(f"Total Images: {total}")
    print(f"Accuracy:     {acc:.2f}%")
    print("-" * 20)
    print(f"True Positives (Fake caught): {tp}")
    print(f"True Negatives (Real OK):     {tn}")
    print(f"False Positives (Real->Fake): {fp}")
    print(f"False Negatives (Fake->Real): {fn}")
    print("-" * 20)
    if (tp + fn) > 0:
        print(f"Sensitivity (Recall): {(tp / (tp+fn))*100:.2f}%")
    if (tn + fp) > 0:
        print(f"Specificity:          {(tn / (tn+fp))*100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()
