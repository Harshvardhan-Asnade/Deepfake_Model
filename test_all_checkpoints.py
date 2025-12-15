import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from src.models import DeepfakeDetector
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config
import os
from collections import defaultdict

# Device
device = torch.device('mps')

# Transform
transform = A.Compose([
    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Test images
test_images = [
    "/Users/harshvardhan/.gemini/antigravity/brain/3695209a-df0b-4c31-9447-5bb22b1d6430/uploaded_image_0_1765728422125.jpg",
    "/Users/harshvardhan/.gemini/antigravity/brain/3695209a-df0b-4c31-9447-5bb22b1d6430/uploaded_image_1_1765728422125.jpg",
    "/Users/harshvardhan/.gemini/antigravity/brain/3695209a-df0b-4c31-9447-5bb22b1d6430/uploaded_image_2_1765728422125.jpg",
    "/Users/harshvardhan/.gemini/antigravity/brain/3695209a-df0b-4c31-9447-5bb22b1d6430/uploaded_image_3_1765728422125.jpg",
    "/Users/harshvardhan/.gemini/antigravity/brain/3695209a-df0b-4c31-9447-5bb22b1d6430/uploaded_image_4_1765728422125.jpg",
]

# All checkpoints
checkpoints = {
    'Best Model': 'results/checkpoints/best_model.safetensors',
    'Epoch 1': 'results/checkpoints/checkpoint_ep1.safetensors',
    'Epoch 2': 'results/checkpoints/checkpoint_ep2.safetensors',
    'Epoch 3': 'results/checkpoints/checkpoint_ep3.safetensors',
    'Base Model': 'results/checkpoints/model.safetensors',
}

print("=" * 120)
print("COMPREHENSIVE MODEL CHECKPOINT TESTING")
print("=" * 120)
print(f"Testing {len(test_images)} images across {len(checkpoints)} model checkpoints\n")

# Store results for summary
results_matrix = defaultdict(dict)

# Test each checkpoint
for ckpt_name, ckpt_path in checkpoints.items():
    print(f"\n{'='*120}")
    print(f"🔹 CHECKPOINT: {ckpt_name}")
    print(f"📁 Path: {ckpt_path}")
    print('='*120)
    
    # Check if checkpoint exists
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        continue
    
    # Load model
    print("⏳ Loading model...")
    model = DeepfakeDetector(pretrained=False).to(device)
    model.eval()
    
    from safetensors.torch import load_model
    try:
        load_model(model, ckpt_path, strict=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        continue
    
    # Test each image
    print(f"\n{'─'*120}")
    for idx, img_path in enumerate(test_images, 1):
        image_name = f"Image {idx}"
        
        try:
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ {image_name}: Could not load image")
                results_matrix[ckpt_name][image_name] = "ERROR"
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = transform(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(image_tensor)
                prob_fake = torch.sigmoid(logits).item()
            
            # Determine prediction
            prediction = "FAKE" if prob_fake > 0.5 else "REAL"
            confidence = prob_fake if prob_fake > 0.5 else (1 - prob_fake)
            
            # Store result
            results_matrix[ckpt_name][image_name] = {
                'prediction': prediction,
                'confidence': confidence,
                'prob_fake': prob_fake
            }
            
            # Display result
            emoji = "🔴" if prediction == "FAKE" else "🟢"
            print(f"{emoji} {image_name:12} → {prediction:4} (Confidence: {confidence:.2%}, Fake Prob: {prob_fake:.4f})")
            
        except Exception as e:
            print(f"❌ {image_name}: Error - {e}")
            results_matrix[ckpt_name][image_name] = "ERROR"

# Print comprehensive summary
print("\n\n" + "=" * 120)
print("📊 COMPREHENSIVE RESULTS MATRIX")
print("=" * 120)

# Header
print(f"\n{'Checkpoint':<20}", end="")
for i in range(1, len(test_images) + 1):
    print(f"Image {i:<15}", end="")
print()
print("─" * 120)

# Results
for ckpt_name in checkpoints.keys():
    print(f"{ckpt_name:<20}", end="")
    for i in range(1, len(test_images) + 1):
        image_name = f"Image {i}"
        result = results_matrix.get(ckpt_name, {}).get(image_name, "N/A")
        
        if result == "ERROR" or result == "N/A":
            print(f"{result:<15}", end="")
        else:
            pred = result['prediction']
            conf = result['confidence']
            emoji = "🔴" if pred == "FAKE" else "🟢"
            print(f"{emoji} {pred} {conf:.1%}   ", end="")
    print()

# Ensemble prediction (majority vote)
print("\n" + "=" * 120)
print("🎯 ENSEMBLE PREDICTIONS (Majority Vote)")
print("=" * 120)

for i in range(1, len(test_images) + 1):
    image_name = f"Image {i}"
    fake_votes = 0
    real_votes = 0
    total_prob_fake = 0
    valid_models = 0
    
    for ckpt_name in checkpoints.keys():
        result = results_matrix.get(ckpt_name, {}).get(image_name)
        if result and result != "ERROR" and result != "N/A":
            valid_models += 1
            total_prob_fake += result['prob_fake']
            if result['prediction'] == "FAKE":
                fake_votes += 1
            else:
                real_votes += 1
    
    if valid_models > 0:
        avg_prob_fake = total_prob_fake / valid_models
        ensemble_pred = "FAKE" if fake_votes > real_votes else "REAL"
        emoji = "🔴" if ensemble_pred == "FAKE" else "🟢"
        
        print(f"{emoji} {image_name}: {ensemble_pred}")
        print(f"   └─ Votes: {fake_votes} FAKE, {real_votes} REAL")
        print(f"   └─ Average Fake Probability: {avg_prob_fake:.4f}")
        print(f"   └─ Consensus: {max(fake_votes, real_votes)}/{valid_models} models")
        print()

print("=" * 120)
print("✅ TESTING COMPLETE!")
print("=" * 120)
