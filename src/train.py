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
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not installed. Checkpoints will be saved as .pt")

def train():
    # Setup
    Config.setup()
    device = torch.device(Config.DEVICE)
    
    # --- Data Loading with Automatic Split ---
    if Config.TRAIN_DATA_PATH == Config.TEST_DATA_PATH:
        print("Train and Test paths are identical. Performing automatic 80/20 shuffle split...")
        all_paths, all_labels = DeepfakeDataset.scan_directory(Config.TRAIN_DATA_PATH)
        
        if len(all_paths) == 0:
            print(f"No images found in {Config.TRAIN_DATA_PATH}")
            return

        # Combine and shuffle
        combined = list(zip(all_paths, all_labels))
        random.shuffle(combined)
        
        split_idx = int(len(combined) * 0.8)
        train_data = combined[:split_idx]
        val_data = combined[split_idx:]
        
        train_paths, train_labels = zip(*train_data)
        val_paths, val_labels = zip(*val_data)
        
        train_dataset = DeepfakeDataset(file_paths=list(train_paths), labels=list(train_labels), phase='train')
        val_dataset = DeepfakeDataset(file_paths=list(val_paths), labels=list(val_labels), phase='val')
    else:
        # Standard folder-based loading
        train_dataset = DeepfakeDataset(root_dir=Config.TRAIN_DATA_PATH, phase='train')
        val_dataset = DeepfakeDataset(root_dir=Config.TEST_DATA_PATH, phase='val')
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, 
                              pin_memory=True if device.type=='cuda' else False,
                              persistent_workers=True if Config.NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, 
                            pin_memory=True if device.type=='cuda' else False,
                            persistent_workers=True if Config.NUM_WORKERS > 0 else False)
    
    # Model
    print("Initializing Multi-Branch DeepfakeDetector...")
    model = DeepfakeDetector(pretrained=True).to(device)
    
    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Resume from checkpoint if exists
    start_epoch = 0
    best_acc = 0.0
    resume_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.safetensors")
    if not os.path.exists(resume_path):
        resume_path = resume_path.replace(".safetensors", ".pth")
    
    if os.path.exists(resume_path):
        print(f"\n🔄 Found existing checkpoint: {resume_path}")
        response = input("Resume training from this checkpoint? (y/n): ").strip().lower()
        if response == 'y':
            if resume_path.endswith(".safetensors") and SAFETENSORS_AVAILABLE:
                state_dict = load_file(resume_path)
            else:
                state_dict = torch.load(resume_path, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Resumed from checkpoint. Starting from where you left off.")
            # Note: We don't track epoch number in checkpoint, so we continue counting from 0
            # If you want to track epoch, we'd need to save optimizer state too
    
    # Loop
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
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
        print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.4f}")
        
        # Save checkpoint after every epoch
        save_checkpoint(model, epoch+1, train_acc, best=False)
        
        # Validation
        if len(val_dataset) > 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # Save best model if validation accuracy improved
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"⭐ New best model! Validation Accuracy: {val_acc:.4f}")
                save_checkpoint(model, epoch+1, val_acc, best=True)
        
        scheduler.step()
    
    print(f"\n🎉 Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

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

def save_checkpoint(model, epoch, acc, best=False):
    state_dict = model.state_dict()
    name = "best_model.safetensors" if best else f"checkpoint_ep{epoch}.safetensors"
    path = os.path.join(Config.CHECKPOINT_DIR, name)
    
    if SAFETENSORS_AVAILABLE:
        try:
            # Try with shared tensors support
            from safetensors.torch import save_model
            save_model(model, path)
            print(f"Saved Checkpoint: {path}")
        except Exception as e:
            # Fallback to regular torch save if safetensors fails
            print(f"SafeTensors save failed ({e}), falling back to .pth format")
            torch.save(state_dict, path.replace(".safetensors", ".pth"))
            print(f"Saved Checkpoint (Legacy): {path.replace('.safetensors', '.pth')}")
    else:
        torch.save(state_dict, path.replace(".safetensors", ".pth"))
        print(f"Saved Checkpoint (Legacy): {path}")

if __name__ == "__main__":
    train()
