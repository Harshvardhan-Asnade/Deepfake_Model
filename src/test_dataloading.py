import os
import random
from src.config import Config
from src.dataset import DeepfakeDataset

def test_dataloading():
    print("Testing Data Loading & Splitting Logic...")
    Config.setup()
    
    print(f"Data Path: {Config.TRAIN_DATA_PATH}")
    
    # 1. Test Scan
    paths, labels = DeepfakeDataset.scan_directory(Config.TRAIN_DATA_PATH)
    total_files = len(paths)
    print(f"Total images found: {total_files}")
    
    if total_files == 0:
        print("[FAIL] No images found! Check path.")
        return

    # 2. Simulate Split Logic
    combined = list(zip(paths, labels))
    random.shuffle(combined)
    split_idx = int(len(combined) * 0.8)
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]
    
    print(f"Train Split: {len(train_data)} images")
    print(f"Val Split:   {len(val_data)} images")
    
    # 3. Test Dataset Initialization
    try:
        train_paths, train_labels = zip(*train_data)
        ds = DeepfakeDataset(file_paths=list(train_paths), labels=list(train_labels), phase='train')
        print(f"[Pass] Dataset initialized with {len(ds)} samples.")
        
        # Test Get Item
        img, lbl = ds[0]
        print(f"[Pass] Loaded sample image. Shape: {img.shape}, Label: {lbl}")
    except Exception as e:
        print(f"[FAIL] Dataset initialization or loading error: {e}")
        return

    print("\nSUCCESS: Data loading verification passed!")

if __name__ == "__main__":
    test_dataloading()
