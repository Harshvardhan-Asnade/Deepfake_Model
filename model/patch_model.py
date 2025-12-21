import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import glob
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config
from src.models import DeepfakeDetector

try:
    from safetensors.torch import save_file, load_model, save_model as save_model_safe
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class PatchDataset(Dataset):
    def __init__(self, fake_paths, real_paths):
        self.image_paths = []
        self.labels = []
        
        # Add Fakes (Label 1.0)
        for p in fake_paths:
            self.image_paths.append(p)
            self.labels.append(1.0)
            
        # Add Reals (Label 0.0)
        for p in real_paths:
            self.image_paths.append(p)
            self.labels.append(0.0)
            
        self.transform = A.Compose([
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=image)
        image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.float32)

def patch_model():
    Config.setup()
    device = torch.device(Config.DEVICE)
    
    # 1. Identify Data
    fake_images = [
        "model/test_images/image1.jpg",
        "model/test_images/image2.jpg",
        "model/test_images/image3.jpg"
    ]
    
    # Get random reals
    real_dir = "/Users/harshvardhan/Developer/dataset/Largest Dataset/Train/Real"
    all_reals = glob.glob(os.path.join(real_dir, "*.*"))
    real_images = random.sample(all_reals, len(fake_images)) # Balance it 1:1
    
    print(f"PATCHING MODEL")
    print(f"Target Fakes: {fake_images}")
    print(f"Support Reals: {len(real_images)} images")
    
    # 2. Dataset & Loader
    dataset = PatchDataset(fake_images, real_images)
    loader = DataLoader(dataset, batch_size=2, shuffle=True) # Small batch size
    
    # 3. Load Model
    model = DeepfakeDetector(pretrained=False).to(device)
    
    checkpoint_path = "model/results/checkpoints/best_finetuned_largest.safetensors"
    load_model(model, checkpoint_path, strict=False)
    print(f"✅ Loaded base model: {checkpoint_path}")
    
    # 4. Training Loop
    EPOCHS = 5
    LR = 1e-4 # Aggressive LR for patching
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    model.train()
    
    print("\nStarting Patch Training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f} | Acc: {correct/total:.2%}")

    # 5. Save
    save_path = "model/results/checkpoints/patched_model.safetensors"
    if SAFETENSORS_AVAILABLE:
        save_model_safe(model, save_path)
    else:
        torch.save(model.state_dict(), save_path.replace(".safetensors", ".pth"))
        
    print(f"\n✅ Patched model saved to: {save_path}")

if __name__ == "__main__":
    patch_model()
