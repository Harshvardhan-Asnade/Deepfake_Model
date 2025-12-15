import argparse
import torch
import cv2
import os
import glob
import numpy as np
import ssl
# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models import DeepfakeDetector
from src.config import Config

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

def get_transform():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def load_models(checkpoints_arg, device):
    """
    Load one or multiple models for ensemble inference.
    checkpoints_arg: Comma-separated list of paths, or single path, or directory.
    """
    paths = []
    if os.path.isdir(checkpoints_arg):
        paths = glob.glob(os.path.join(checkpoints_arg, "*.safetensors"))
        if not paths:
            paths = glob.glob(os.path.join(checkpoints_arg, "*.pth"))
    else:
        paths = checkpoints_arg.split(',')
    
    models = []
    print(f"Loading {len(paths)} model(s) for ensemble inference...")
    
    for path in paths:
        path = path.strip()
        if not path: continue
        
        print(f"Loading: {path}")
        model = DeepfakeDetector(pretrained=False) # Structure only
        model.to(device)
        model.eval()
        
        try:
            if path.endswith(".safetensors") and SAFETENSORS_AVAILABLE:
                state_dict = load_file(path)
            else:
                state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            models.append(model)
            print(f"✅ Successfully loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
            import traceback
            traceback.print_exc()
            
    if not models:
        # Fallback for testing if no checkpoint exists yet
        print("Warning: No valid checkoints loaded. Using random initialization for testing flow.")
        model = DeepfakeDetector(pretrained=False).to(device)
        model.eval()
        models.append(model)
        
    return models

def predict_ensemble(models, image_path, device, transform):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Could not read image"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return None, str(e)

    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    probs = []
    with torch.no_grad():
        for model in models:
            logits = model(image_tensor)
            prob = torch.sigmoid(logits).item()
            probs.append(prob)
            
    # Ensemble Strategy: Average Probability
    avg_prob = sum(probs) / len(probs)
    return avg_prob, None

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference (Ensemble Support)")
    parser.add_argument("--source", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--checkpoints", type=str, default="results/checkpoints", help="Path to checkpoint file, list of files (comma-separated), or directory")
    parser.add_argument("--device", type=str, default=Config.DEVICE, help="Device to use (cuda/mps/cpu)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load Models
    models = load_models(args.checkpoints, device)
    transform = get_transform()
    
    # Process Source
    if os.path.isdir(args.source):
        files = glob.glob(os.path.join(args.source, "*.*"))
        # Filter images
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    else:
        files = [args.source]
        
    print(f"Processing {len(files)} images with {len(models)} model(s)...")
    print("-" * 65)
    print(f"{'Image Name':<40} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 65)
    
    for file_path in files:
        prob, error = predict_ensemble(models, file_path, device, transform)
        if error:
            print(f"{os.path.basename(file_path):<40} | ERROR: {error}")
            continue
            
        is_fake = prob > 0.5
        label = "FAKE" if is_fake else "REAL"
        confidence = prob if is_fake else 1 - prob
        
        print(f"{os.path.basename(file_path):<40} | {label:<10} | {confidence:.2%}")

if __name__ == "__main__":
    main()
