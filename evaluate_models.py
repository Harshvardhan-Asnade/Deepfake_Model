import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from src.models import DeepfakeDetector
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config
import os
import glob
from tqdm import tqdm

# Device
device = torch.device('mps')

# Transform
transform = A.Compose([
    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Test dataset path
test_dir = "/Users/harshvardhan/Developer/deepfake/Dataset/Image Dataset/Test"

# Get all images with labels
print("📂 Loading test dataset...")
real_images = glob.glob(os.path.join(test_dir, "Real/*.*"))
fake_images = glob.glob(os.path.join(test_dir, "Fake/*.*"))

real_images = [f for f in real_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
fake_images = [f for f in fake_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"✅ Found {len(real_images)} REAL images")
print(f"✅ Found {len(fake_images)} FAKE images")
print(f"Total: {len(real_images) + len(fake_images)} images\n")

# Checkpoints
checkpoints = {
    'Epoch 1': 'results/checkpoints/checkpoint_ep1.safetensors',
    'Epoch 2 (Best)': 'results/checkpoints/checkpoint_ep2.safetensors',
    'Epoch 3': 'results/checkpoints/checkpoint_ep3.safetensors',
}

print("="*90)
print("EVALUATION ON FULL TEST DATASET")
print("="*90)

# Test each checkpoint
for ckpt_name, ckpt_path in checkpoints.items():
    print(f"\n🔹 Evaluating: {ckpt_name}")
    print("─" * 90)
    
    # Load model
    model = DeepfakeDetector(pretrained=False).to(device)
    model.eval()
    
    from safetensors.torch import load_model
    load_model(model, ckpt_path, strict=False)
    
    # Counters
    true_positives = 0  # Correctly identified as FAKE
    true_negatives = 0  # Correctly identified as REAL
    false_positives = 0  # Real labeled as FAKE
    false_negatives = 0  # Fake labeled as REAL
    
    # Test REAL images
    print(f"Testing {len(real_images)} REAL images...")
    for img_path in tqdm(real_images, desc="Real"):
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            augmented = transform(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(image_tensor)
                prob_fake = torch.sigmoid(logits).item()
            
            if prob_fake > 0.5:
                false_positives += 1  # Wrongly classified as fake
            else:
                true_negatives += 1  # Correctly classified as real
        except:
            continue
    
    # Test FAKE images
    print(f"Testing {len(fake_images)} FAKE images...")
    for img_path in tqdm(fake_images, desc="Fake"):
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            augmented = transform(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(image_tensor)
                prob_fake = torch.sigmoid(logits).item()
            
            if prob_fake > 0.5:
                true_positives += 1  # Correctly classified as fake
            else:
                false_negatives += 1  # Wrongly classified as real
        except:
            continue
    
    # Calculate metrics
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\n📊 RESULTS:")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%} (of detected fakes, how many were actually fake)")
    print(f"  Recall:    {recall:.2%} (of all fakes, how many were detected)")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  True Positives:  {true_positives:>5} (Fake detected as Fake)")
    print(f"  True Negatives:  {true_negatives:>5} (Real detected as Real)")
    print(f"  False Positives: {false_positives:>5} (Real detected as Fake)")
    print(f"  False Negatives: {false_negatives:>5} (Fake detected as Real)")

print("\n" + "="*90)
print("✅ Evaluation Complete!")
print("="*90)
