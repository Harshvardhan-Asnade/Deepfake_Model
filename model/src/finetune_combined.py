
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import ssl

# Add src to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CURRENT_DIR))

# Disable SSL verification for downloading pretrained weights
ssl._create_default_https_context = ssl._create_unverified_context

from src.config import Config
from src.models import DeepfakeDetector
from src.dataset import DeepfakeDataset

try:
    from safetensors.torch import save_file, load_model, save_model as save_model_st
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not installed. Checkpoints will be saved as .pt")

# Combined Dataset Configuration
DATASET_PATHS = [
    "/Users/harshvardhan/Developer/Deepfake Project /DataSet/Dataset A",
    "/Users/harshvardhan/Developer/Deepfake Project /DataSet/DataSet B",
    "/Users/harshvardhan/Developer/Deepfake Project /DataSet/new Dataset",
    "/Users/harshvardhan/Developer/Deepfake Project /DataSet/Largest Dataset"
]

# Fine-tuning hyperparameters
FINETUNE_LR = 1e-5  # Low learning rate for fine-tuning
FINETUNE_EPOCHS = 1  # 1 epoch constraint

def finetune_combined():
    """Fine-tune the existing model on ALL Combined Datasets"""
    
    # Setup
    Config.setup()
    device = torch.device(Config.DEVICE)
    
    print(f"\n{'='*80}")
    print("FINE-TUNING ON COMBINED DATASETS")
    print(f"{'='*80}\n")
    
    # --- Data Loading ---
    all_paths = []
    all_labels = []
    
    print("Aggregating data from sources:")
    for path in DATASET_PATHS:
        if os.path.exists(path):
            print(f"   Scanning: {path}...")
            paths, labels = DeepfakeDataset.scan_directory(path)
            all_paths.extend(paths)
            all_labels.extend(labels)
            print(f"   -> Found {len(paths)} images")
        else:
            print(f"❌ Warning: Path not found: {path}")

    if len(all_paths) == 0:
        print("❌ Error: No images found in any dataset path!")
        return

    print(f"\n✅ Total Images Found: {len(all_paths)}")

    # Shuffle and split 80/20
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    
    split_idx = int(len(combined) * 0.8)
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]
    
    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)
    
    print(f"✅ Training samples: {len(train_paths)}")
    print(f"✅ Validation samples: {len(val_paths)}")
    
    # Create datasets
    train_dataset = DeepfakeDataset(file_paths=list(train_paths), labels=list(train_labels), phase='train')
    val_dataset = DeepfakeDataset(file_paths=list(val_paths), labels=list(val_labels), phase='val')
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True if device.type=='cuda' else False,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True if device.type=='cuda' else False,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    # Load pre-trained model
    print("\n🔄 Loading pre-trained model (best_model)...")
    model = DeepfakeDetector(pretrained=False).to(device)
    
    # Load best_model.safetensors
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.safetensors")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    
    if os.path.exists(checkpoint_path):
        try:
            if checkpoint_path.endswith(".safetensors"):
                load_model(model, checkpoint_path, strict=False)
            else:
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("   Starting from random weights")
    else:
        print("⚠️ No checkpoint found! Starting from random weights.")
    
    model.to(device)
    
    # Fine-tuning settings
    print(f"\n📝 Fine-tuning settings:")
    print(f"   Learning Rate: {FINETUNE_LR}")
    print(f"   Epochs: {FINETUNE_EPOCHS}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Datasets: {len(DATASET_PATHS)} sources combined")
    
    # Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Loop
    best_acc = 0.0
    
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{FINETUNE_EPOCHS}")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct = (preds == labels).sum().item()
            train_correct += correct
            train_total += labels.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=correct/labels.size(0) if labels.size(0) > 0 else 0)
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, epoch+1, train_acc, name=f"combined_finetuned_ep{epoch+1}")
        
        # Validation
        if len(val_dataset) > 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"⭐ New best model! Validation Accuracy: {val_acc:.4f}")
                save_checkpoint(model, epoch+1, val_acc, name="best_model_combined")
    
    print(f"\n🎉 Fine-tuning Complete!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"\n💾 Checkpoints saved in: {Config.CHECKPOINT_DIR}")

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return val_loss / len(loader), correct / total

def save_checkpoint(model, epoch, acc, name="checkpoint"):
    state_dict = model.state_dict()
    filename = f"{name}.safetensors"
    path = os.path.join(Config.CHECKPOINT_DIR, filename)
    
    if SAFETENSORS_AVAILABLE:
        try:
            save_model_st(model, path)
            print(f"✅ Saved: {filename}")
        except Exception as e:
            print(f"SafeTensors save failed, falling back to .pth: {e}")
            torch.save(state_dict, path.replace(".safetensors", ".pth"))
    else:
        torch.save(state_dict, path.replace(".safetensors", ".pth"))

if __name__ == "__main__":
    finetune_combined()
