import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import ssl
# Disable SSL verification for downloading pretrained weights
ssl._create_default_https_context = ssl._create_unverified_context

from src.config import Config
from src.models import DeepfakeDetector
from src.dataset import DeepfakeDataset

try:
    from safetensors.torch import save_file, load_model, save_model as save_model_safe
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not installed. Checkpoints will be saved as .pt")

def finetune_datasetb():
    # Setup
    Config.setup()
    device = torch.device(Config.DEVICE)
    
    # Dataset B paths
    TRAIN_PATH = "/Users/harshvardhan/Developer/dataset/DataSet B/Train"
    VAL_PATH = "/Users/harshvardhan/Developer/dataset/DataSet B/Validation"
    
    print(f"\n{'='*80}")
    print("FINE-TUNING ON DATASET B (140K IMAGES)")
    print(f"{'='*80}\n")
    
    # --- Data Loading ---
    print(f"Loading training data from: {TRAIN_PATH}")
    train_dataset = DeepfakeDataset(root_dir=TRAIN_PATH, phase='train')
    
    print(f"Loading validation data from: {VAL_PATH}")
    val_dataset = DeepfakeDataset(root_dir=VAL_PATH, phase='val')
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, 
                              pin_memory=True if device.type=='cuda' else False,
                              persistent_workers=True if Config.NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, 
                            pin_memory=True if device.type=='cuda' else False,
                            persistent_workers=True if Config.NUM_WORKERS > 0 else False)
    
    # Load pre-trained model from Dataset A
    print("\n🔄 Loading best model from Dataset A...")
    model = DeepfakeDetector(pretrained=False).to(device)
    
    checkpoint_path = "results/checkpoints/best_model.safetensors"
    if os.path.exists(checkpoint_path):
        load_model(model, checkpoint_path, strict=False)
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print("⚠️ No checkpoint found! Starting from random weights.")
    
    model.to(device)
    
    # Optimization with LOWER learning rate for fine-tuning
    FINETUNE_LR = 5e-6  # Very low for fine-tuning on large dataset
    FINETUNE_EPOCHS = 1  # Start with 1 epoch to test
    
    print(f"\n📝 Fine-tuning settings:")
    print(f"   Learning Rate: {FINETUNE_LR} (very low for fine-tuning)")
    print(f"   Epochs: {FINETUNE_EPOCHS}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
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
            
            loop.set_postfix(loss=loss.item(), acc=correct/labels.size(0))
            
        train_acc = train_correct / train_total if train_total > 0 else 0
        print(f"\nEpoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.4f}")
        
        # Save checkpoint after every epoch
        save_checkpoint(model, epoch+1, train_acc, name=f"finetuned_datasetB_ep{epoch+1}")
        
        # Validation
        if len(val_dataset) > 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # Save best model if validation accuracy improved
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"⭐ New best model! Validation Accuracy: {val_acc:.4f}")
                save_checkpoint(model, epoch+1, val_acc, name="best_finetuned_datasetB")
        
        scheduler.step()
    
    print(f"\n🎉 Fine-tuning Complete!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"\n💾 Checkpoints saved in: results/checkpoints/")
    print(f"\n🔍 Next: Test on Dataset B Test set to see final performance!")

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
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
            save_model_safe(model, path)
            print(f"✅ Saved: {filename}")
        except Exception as e:
            print(f"SafeTensors save failed, falling back to .pth: {e}")
            torch.save(state_dict, path.replace(".safetensors", ".pth"))
    else:
        torch.save(state_dict, path.replace(".safetensors", ".pth"))

if __name__ == "__main__":
    finetune_datasetb()
