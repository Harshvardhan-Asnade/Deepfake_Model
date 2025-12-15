import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from src.models import DeepfakeDetector
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config
import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Device
device = torch.device('mps')

# Transform
transform = A.Compose([
    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Test images - ALL ARE FAKE (Ground Truth)
test_images = {
    'image1.jpg': {'path': 'test_images/image1.jpg', 'ground_truth': 'FAKE'},
    'image2.jpg': {'path': 'test_images/image2.jpg', 'ground_truth': 'FAKE'},
    'image3.jpg': {'path': 'test_images/image3.jpg', 'ground_truth': 'FAKE'},
}

# All model checkpoints
checkpoints = {
    'Best Model (Dataset A)': 'results/checkpoints/best_model.safetensors',
    'Best Finetuned (Dataset B)': 'results/checkpoints/best_finetuned_datasetB.safetensors',
    'Finetuned Dataset B - Epoch 1': 'results/checkpoints/finetuned_datasetB_ep1.safetensors',
    'Best Finetuned (Dataset C)': 'results/checkpoints/best_finetuned_datasetC.safetensors',
    'Finetuned Dataset C - Epoch 1': 'results/checkpoints/finetuned_datasetC_ep1.safetensors',
    'Finetuned Dataset C - Epoch 2': 'results/checkpoints/finetuned_datasetC_ep2.safetensors',
}

print("=" * 120)
print("🔬 DEEPFAKE DETECTION - COMPREHENSIVE ERROR ANALYSIS")
print("=" * 120)
print(f"Ground Truth: ALL images are FAKE")
print(f"Testing {len(test_images)} images across {len(checkpoints)} models")
print("=" * 120)

# Store comprehensive results
analysis_results = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'num_images': len(test_images),
        'num_models': len(checkpoints),
        'ground_truth': 'ALL FAKE'
    },
    'predictions': defaultdict(dict),
    'model_performance': {},
    'image_difficulty': {},
}


# Test each model
for model_idx, (ckpt_name, ckpt_path) in enumerate(checkpoints.items(), 1):
    print(f"\n{'='*120}")
    print(f"🔹 MODEL {model_idx}/{len(checkpoints)}: {ckpt_name}")
    print(f"📁 Path: {ckpt_path}")
    print('='*120)
    
    # Load model
    print("⏳ Loading model...")
    model = DeepfakeDetector(pretrained=False).to(device)
    model.eval()
    
    from safetensors.torch import load_model
    load_model(model, ckpt_path, strict=False)
    print("✅ Model loaded successfully!")

    
    model_predictions = []
    model_errors = []
    
    # Test each image
    print(f"\n{'─'*120}")
    print(f"{'Image':<15} | {'Prediction':<12} | {'Ground Truth':<12} | {'Result':<10} | {'Fake Prob':<12} | {'Confidence'}")
    print('─'*120)
    
    for img_name, img_info in test_images.items():
        img_path = img_info['path']
        ground_truth = img_info['ground_truth']
        
        try:
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ {img_name}: Could not load image")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = transform(image=image_rgb)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(image_tensor)
                prob_fake = torch.sigmoid(logits).item()

            
            # Determine prediction
            prediction = "FAKE" if prob_fake > 0.5 else "REAL"
            confidence = prob_fake if prob_fake > 0.5 else (1 - prob_fake)
            
            # Check if correct
            is_correct = (prediction == ground_truth)
            result_emoji = "✅" if is_correct else "❌"
            result_text = "CORRECT" if is_correct else "WRONG"
            
            # Store results
            analysis_results['predictions'][ckpt_name][img_name] = {
                'prediction': prediction,
                'ground_truth': ground_truth,
                'prob_fake': prob_fake,
                'confidence': confidence,
                'correct': is_correct,
                'logits': logits.item(),
            }

            
            model_predictions.append(is_correct)
            if not is_correct:
                model_errors.append(img_name)
            
            # Display result
            print(f"{result_emoji} {img_name:<12} | {prediction:<12} | {ground_truth:<12} | {result_text:<10} | {prob_fake:>11.4f} | {confidence:>10.2%}")
            
        except Exception as e:
            print(f"❌ {img_name}: Error - {e}")
            continue
    
    # Calculate model performance metrics
    accuracy = sum(model_predictions) / len(model_predictions) if model_predictions else 0
    error_rate = 1 - accuracy
    
    analysis_results['model_performance'][ckpt_name] = {
        'accuracy': accuracy,
        'error_rate': error_rate,
        'correct_count': sum(model_predictions),
        'total_count': len(model_predictions),
        'failed_images': model_errors,
    }
    
    print('─'*120)
    print(f"📊 Model Performance: {accuracy:.2%} accuracy ({sum(model_predictions)}/{len(model_predictions)} correct)")
    print(f"❌ Failed on: {', '.join(model_errors) if model_errors else 'None'}")

# Calculate image difficulty scores
print("\n\n" + "=" * 120)
print("📊 IMAGE DIFFICULTY ANALYSIS")
print("=" * 120)
print("(How many models got each image wrong)")
print()

for img_name in test_images.keys():
    errors_count = 0
    total_models = 0
    prob_fake_values = []
    
    for ckpt_name in checkpoints.keys():
        if img_name in analysis_results['predictions'][ckpt_name]:
            total_models += 1
            if not analysis_results['predictions'][ckpt_name][img_name]['correct']:
                errors_count += 1
            prob_fake_values.append(analysis_results['predictions'][ckpt_name][img_name]['prob_fake'])
    
    difficulty_score = errors_count / total_models if total_models > 0 else 0
    avg_prob_fake = np.mean(prob_fake_values) if prob_fake_values else 0
    std_prob_fake = np.std(prob_fake_values) if prob_fake_values else 0
    
    analysis_results['image_difficulty'][img_name] = {
        'errors_count': errors_count,
        'total_models': total_models,
        'difficulty_score': difficulty_score,
        'avg_prob_fake': avg_prob_fake,
        'std_prob_fake': std_prob_fake,
        'min_prob_fake': min(prob_fake_values) if prob_fake_values else 0,
        'max_prob_fake': max(prob_fake_values) if prob_fake_values else 0,
    }
    
    difficulty_emoji = "🔴" if difficulty_score > 0.8 else "🟡" if difficulty_score > 0.5 else "🟢"
    print(f"{difficulty_emoji} {img_name:<15} | Failed by: {errors_count}/{total_models} models ({difficulty_score:.1%})")
    print(f"   └─ Avg Fake Probability: {avg_prob_fake:.4f} (±{std_prob_fake:.4f})")
    print(f"   └─ Range: [{min(prob_fake_values):.4f}, {max(prob_fake_values):.4f}]")
    print(f"   └─ Difficulty: {'VERY HARD' if difficulty_score > 0.8 else 'HARD' if difficulty_score > 0.5 else 'MODERATE'}")
    print()

# Model ranking
print("\n" + "=" * 120)
print("🏆 MODEL RANKING (by accuracy)")
print("=" * 120)

sorted_models = sorted(
    analysis_results['model_performance'].items(),
    key=lambda x: x[1]['accuracy'],
    reverse=True
)

for rank, (model_name, performance) in enumerate(sorted_models, 1):
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
    acc_bar = "█" * int(performance['accuracy'] * 20)
    print(f"{medal} {model_name:<40} | {performance['accuracy']:>6.2%} {acc_bar}")
    print(f"   └─ Correct: {performance['correct_count']}/{performance['total_count']} | Failed on: {', '.join(performance['failed_images']) if performance['failed_images'] else 'None'}")

# Confusion analysis
print("\n\n" + "=" * 120)
print("🔍 CONFUSION ANALYSIS")
print("=" * 120)
print("Why are models classifying FAKE images as REAL?")
print()

for img_name in test_images.keys():
    print(f"\n{'─'*120}")
    print(f"📸 {img_name} (Ground Truth: FAKE)")
    print('─'*120)
    
    fake_probs = []
    for ckpt_name in checkpoints.keys():
        if img_name in analysis_results['predictions'][ckpt_name]:
            pred_data = analysis_results['predictions'][ckpt_name][img_name]
            fake_probs.append(pred_data['prob_fake'])
            
            emoji = "✅" if pred_data['correct'] else "❌"
            print(f"{emoji} {ckpt_name:<40} → Fake Prob: {pred_data['prob_fake']:.4f} ({pred_data['prediction']})")
    
    if fake_probs:
        avg_fake_prob = np.mean(fake_probs)
        print(f"\n   💡 Insight: Average fake probability is {avg_fake_prob:.4f}")
        if avg_fake_prob < 0.3:
            print(f"   ⚠️  Models are VERY confident this is real (avg {avg_fake_prob:.1%} fake)")
            print(f"   🔬 This suggests high-quality deepfake that models haven't seen in training")
        elif avg_fake_prob < 0.5:
            print(f"   ⚠️  Models lean towards real ({avg_fake_prob:.1%} fake)")
            print(f"   🔬 Borderline case - models are uncertain")
        else:
            print(f"   ✅ Some models correctly identify this as fake")

# Save results
output_dir = "results/error_analysis"
os.makedirs(output_dir, exist_ok=True)

# Save JSON report
report_path = os.path.join(output_dir, f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(report_path, 'w') as f:
    # Convert defaultdict to dict for JSON serialization
    analysis_results['predictions'] = dict(analysis_results['predictions'])
    json.dump(analysis_results, f, indent=2)

print("\n\n" + "=" * 120)
print("📝 SAVING VISUALIZATIONS")
print("=" * 120)

# Create visualization: Heatmap of predictions
plt.figure(figsize=(14, 8))
heatmap_data = []
model_names = []
image_names = list(test_images.keys())

for ckpt_name in checkpoints.keys():
    model_names.append(ckpt_name.replace(' ', '\n'))
    row = []
    for img_name in image_names:
        if img_name in analysis_results['predictions'][ckpt_name]:
            row.append(analysis_results['predictions'][ckpt_name][img_name]['prob_fake'])
        else:
            row.append(0)
    heatmap_data.append(row)

sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn_r', 
            xticklabels=image_names, yticklabels=model_names,
            cbar_kws={'label': 'Fake Probability'}, vmin=0, vmax=1)
plt.title('Fake Probability Heatmap\n(Ground Truth: ALL FAKE)\nGreen = Correctly detected as fake, Red = Incorrectly classified as real', 
          fontsize=12, fontweight='bold')
plt.xlabel('Images', fontsize=11)
plt.ylabel('Models', fontsize=11)
plt.tight_layout()
heatmap_path = os.path.join(output_dir, f"probability_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved heatmap: {heatmap_path}")

# Model accuracy comparison
plt.figure(figsize=(12, 6))
models = [name.replace(' ', '\n') for name, _ in sorted_models]
accuracies = [perf['accuracy'] * 100 for _, perf in sorted_models]
colors = ['#2ecc71' if acc > 20 else '#e74c3c' for acc in accuracies]

bars = plt.bar(range(len(models)), accuracies, color=colors, edgecolor='black', linewidth=1.5)
plt.axhline(y=33.33, color='orange', linestyle='--', label='33.33% (1 out of 3 correct)', linewidth=2)
plt.axhline(y=66.67, color='green', linestyle='--', label='66.67% (2 out of 3 correct)', linewidth=2)
plt.xlabel('Models', fontsize=11, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('Model Performance Comparison\n(All test images are FAKE)', fontsize=13, fontweight='bold')
plt.xticks(range(len(models)), [m.split('\n')[0] + '\n' + m.split('\n')[-1] for m in models], 
           rotation=45, ha='right', fontsize=9)
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
accuracy_path = os.path.join(output_dir, f"model_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved accuracy chart: {accuracy_path}")

print(f"\n✅ JSON report saved: {report_path}")
print("\n" + "=" * 120)
print("✅ ERROR ANALYSIS COMPLETE!")
print("=" * 120)
print(f"\n📁 All results saved to: {output_dir}/")
print(f"   - JSON Report: {os.path.basename(report_path)}")
print(f"   - Heatmap: {os.path.basename(heatmap_path)}")
print(f"   - Accuracy Chart: {os.path.basename(accuracy_path)}")
