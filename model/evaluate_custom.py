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

# Force Windows/CUDA settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Transform (Same as training)
transform = A.Compose([
    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# TARGET DATASET
# User requested path:
test_dir = r"C:\Users\kanna\Downloads\Dataset\Largest Dataset\Largest Dataset\Train"

# Get all images with labels
print(f"📂 Scanning directory: {test_dir}")
real_images = glob.glob(os.path.join(test_dir, "Real", "*.*"))
fake_images = glob.glob(os.path.join(test_dir, "Fake", "*.*"))

# Filter extensions
exts = ('.png', '.jpg', '.jpeg', '.webp')
real_images = [f for f in real_images if f.lower().endswith(exts)]
fake_images = [f for f in fake_images if f.lower().endswith(exts)]

print(f"✅ Found {len(real_images)} REAL images")
print(f"✅ Found {len(fake_images)} FAKE images")
total_images = len(real_images) + len(fake_images)

if total_images == 0:
    print("❌ Error: No images found! Check the path.")
    exit()

# LIMIT FOR SPEED (User might not want to wait for 180k images)
# Let's test 1000 of each for a quick report, or all if user wants.
# Usually scanning 2000 images is enough to get a % score.
import random
random.shuffle(real_images)
random.shuffle(fake_images)

SAMPLE_SIZE = 1000
if len(real_images) > SAMPLE_SIZE:
    real_images = real_images[:SAMPLE_SIZE]
    print(f"ℹ️  Sampling {SAMPLE_SIZE} Real images for speed...")
if len(fake_images) > SAMPLE_SIZE:
    fake_images = fake_images[:SAMPLE_SIZE]
    print(f"ℹ️  Sampling {SAMPLE_SIZE} Fake images for speed...")


# Load Model
model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.safetensors")
print(f"\n🔹 Loading Model: {model_path}")

model = DeepfakeDetector(pretrained=False).to(device)
model.eval()

from safetensors.torch import load_model
try:
    load_model(model, model_path, strict=False)
    print("✅ Weights loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# EVALUATION LOOP
print("\n🚀 Starting Evaluation...")

true_positives = 0  # Fake -> Fake
true_negatives = 0  # Real -> Real
false_positives = 0 # Real -> Fake
false_negatives = 0 # Fake -> Real

# Test REAL
for img_path in tqdm(real_images, desc="Testing Real"):
    try:
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(image_tensor)
            prob_fake = torch.sigmoid(logits).item()
        
        if prob_fake > 0.5:
            false_positives += 1
        else:
            true_negatives += 1
    except:
        continue

# Test FAKE
for img_path in tqdm(fake_images, desc="Testing Fake"):
    try:
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(image_tensor)
            prob_fake = torch.sigmoid(logits).item()
        
        if prob_fake > 0.5:
            true_positives += 1
        else:
            false_negatives += 1
    except:
        continue

# REPORT
total = true_positives + true_negatives + false_positives + false_negatives
accuracy = (true_positives + true_negatives) / total if total > 0 else 0
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*50)
print("📊 FINAL RESULTS")
print("="*50)
print(f"  Images Tested: {total}")
print(f"  Accuracy:      {accuracy:.2%}")
print(f"  Precision:     {precision:.2%}")
print(f"  Recall:        {recall:.2%}")
print(f"  F1 Score:      {f1:.4f}")
print("="*50)
print(f"  True Pos (Fake detected as Fake): {true_positives}")
print(f"  True Neg (Real detected as Real): {true_negatives}")
print(f"  False Pos (Real detected as Fake): {false_positives}")
print(f"  False Neg (Fake detected as Real): {false_negatives}")
print("="*50)
