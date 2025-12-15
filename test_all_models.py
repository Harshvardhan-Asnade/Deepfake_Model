import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from src.models import DeepfakeDetector
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config
import os

# Device
device = torch.device('mps')

# Transform
transform = A.Compose([
    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Test images
images_paths = ['test_images/image1.jpg', 'test_images/image2.jpg', 'test_images/image3.jpg']
checkpoints = {
    'Best Model (Dataset A)': 'results/checkpoints/best_model.safetensors',
    'Best Finetuned (Dataset B)': 'results/checkpoints/best_finetuned_datasetB.safetensors',
    'Finetuned Dataset B - Epoch 1': 'results/checkpoints/finetuned_datasetB_ep1.safetensors',
    'Best Finetuned (Dataset C)': 'results/checkpoints/best_finetuned_datasetC.safetensors',
    'Finetuned Dataset C - Epoch 1': 'results/checkpoints/finetuned_datasetC_ep1.safetensors',
    'Finetuned Dataset C - Epoch 2': 'results/checkpoints/finetuned_datasetC_ep2.safetensors',
}

print("\n" + "="*90)
print("DEEPFAKE DETECTION - ALL MODELS COMPARISON")
print("="*90)

# Test each checkpoint
for ckpt_name, ckpt_path in checkpoints.items():
    print(f"\n🔹 Testing: {ckpt_name}")
    print("─" * 90)
    
    # Load model
    model = DeepfakeDetector(pretrained=False).to(device)
    model.eval()
    
    from safetensors.torch import load_model
    load_model(model, ckpt_path, strict=False)
    
    # Test all images
    print(f"{'Image':<20} | {'Prediction':<15} | {'Confidence':<15} | {'Fake Probability'}")
    print("─" * 90)
    
    for img_path in images_paths:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(image_tensor)
            prob_fake = torch.sigmoid(logits).item()
        
        is_fake = prob_fake > 0.5
        label = "FAKE" if is_fake else "REAL"
        confidence = prob_fake if is_fake else 1 - prob_fake
        
        print(f"{os.path.basename(img_path):<20} | {label:<15} | {confidence:>13.2%} | {prob_fake:>13.4f}")

print("\n" + "="*90)
print("✅ Testing Complete!")
print("="*90)
