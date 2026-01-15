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

# FaceForensics++ Configuration
FF_DATASET_ROOT = "/Users/harshvardhan/Developer/Deepfake Project /DataSet/FaceForencis++ extracted frames"

# Fine-tuning hyperparameters
FINETUNE_LR = 5e-6  # Very low learning rate (even lower than before)
FINETUNE_EPOCHS = 1  # 1 epoch as requested by user

def finetune_faceforensics():
    """Fine-tune the existing model on FaceForensics++ frames"""
    
    # Setup
    Config.setup()
    device = torch.device(Config.DEVICE)
    
    print(f"\n{'='*80}")
    print("FINE-TUNING ON FACEFORENSICS++ DATASET")
    print(f"{'='*80}\n")
    
    # Load data
    print(f"Loading data from: {FF_DATASET_ROOT}")
    
    # Scan directories - FaceForensics++ structure has real/ and fake/ at root
    train_real_path = os.path.join(FF_DATASET_ROOT, "real")
    train_fake_path = os.path.join(FF_DATASET_ROOT, "fake")
    
    if not os.path.exists(train_real_path) or not os.path.exists(train_fake_path):
        print(f"❌ Error: Real or Fake folders not found!")
        print(f"   Checked: {train_real_path}")
        print(f"   Checked: {train_fake_path}")
        return
    
    # Get all file paths
    train_real_files, train_real_labels = DeepfakeDataset.scan_directory(train_real_path)
    train_fake_files, train_fake_labels = DeepfakeDataset.scan_directory(train_fake_path)
    
    
    print(f"✅ Real images: {len(train_real_files)}")
    print(f"✅ Fake images: {len(train_fake_files)}")
    
    train_paths = list(train_real_files) + list(train_fake_files)
    train_labels = list(train_real_labels) + list(train_fake_labels)
    
    
    # Shuffle and create 80/20 split for validation
    combined = list(zip(train_paths, train_labels))
    random.shuffle(combined)
    
    split_idx = int(len(combined) * 0.8)
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]
    
    train_paths_split, train_labels_split = zip(*train_data) if train_data else ([], [])
    val_paths, val_labels = zip(*val_data) if val_data else ([], [])
    
    print(f"✅ Training samples: {len(train_paths_split)}")
    print(f"✅ Validation samples: {len(val_paths)}")
    
    
    # Create datasets
    train_dataset = DeepfakeDataset(file_paths=list(train_paths_split), labels=list(train_labels_split), phase='train')
    
    if len(val_paths) > 0:
        val_dataset = DeepfakeDataset(file_paths=list(val_paths), labels=list(val_labels), phase='val')
    else:
        val_dataset = None
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True if device.type=='cuda' else False,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    if val_dataset:
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
    
    # Try to load the best model
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
    print(f"   Learning Rate: {FINETUNE_LR} (Very low for fine-tuning)")
    print(f"   Epochs: {FINETUNE_EPOCHS}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Dataset: FaceForensics++ (4 manipulation methods)")
    
    # Optimizer and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Training loop
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
        
        # Save checkpoint after every epoch
        save_checkpoint(model, epoch+1, train_acc, name=f"ff_finetuned_ep{epoch+1}")
        
        # Validation
        if val_dataset and len(val_dataset) > 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"⭐ New best model! Validation Accuracy: {val_acc:.4f}")
                save_checkpoint(model, epoch+1, val_acc, name="best_model_ff")
    
    print(f"\n🎉 Fine-tuning Complete!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"\n💾 Checkpoints saved in: {Config.CHECKPOINT_DIR}")
    print(f"\n📊 Next steps:")
    print(f"   1. Test the model: python model/evaluate_custom.py")
    print(f"   2. Compare models: python model/compare_models.py")
    print(f"   3. Update best_model: Copy best_model_ff.safetensors to best_model.safetensors")

def validate(model, loader, criterion, device):
    """Validation function"""
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
    """Save model checkpoint"""
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
    finetune_faceforensics()
