import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import sys

# Add src to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CURRENT_DIR))

from src.config import Config
from src.models import DeepfakeDetector
from src.dataset import DeepfakeDataset
from safetensors.torch import load_model


def generate_report(model_filename="Mark-III.safetensors", val_loader=None, device_str=None, output_dir=None):
    if device_str:
        device = torch.device(device_str)
    else:
        Config.setup()
        device = torch.device(Config.DEVICE)
    
    # Setup Output Directory
    if output_dir is None:
        report_plots_dir = os.path.join(Config.RESULTS_DIR, "plots")
    else:
        report_plots_dir = output_dir
    os.makedirs(report_plots_dir, exist_ok=True)

    # 1. Load Model (Only if not passed? Actually we pass filename, so we load it)
    # Ideally we should pass the MODEL OBJECT if it's already in memory to save time
    # But for now, let's keep it loading from file to ensure we test the saved artifact.
    print(f"🔄 Loading {model_filename}...")
    model = DeepfakeDetector(pretrained=False).to(device)
    
    # Handle full path vs filename
    if os.path.isabs(model_filename):
        checkpoint_path = model_filename
    else:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, model_filename)
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Model not found at {checkpoint_path}")
        return

    load_model(model, checkpoint_path, strict=False)
    model.eval()
    
    # 2. Load Validation Data
    # If val_loader is provided, use it. Otherwise, load default.
    if val_loader is None:
        print("📂 Loading Default Validation Dataset (FF++)...")
        FF_DATASET_ROOT = "/Users/harshvardhan/Developer/Deepfake Project /DataSet/FaceForencis++ extracted frames"
        
        train_real_path = os.path.join(FF_DATASET_ROOT, "real")
        train_fake_path = os.path.join(FF_DATASET_ROOT, "fake")
        
        real_files, real_labels = DeepfakeDataset.scan_directory(train_real_path)
        fake_files, fake_labels = DeepfakeDataset.scan_directory(train_fake_path)
        
        all_paths = list(real_files) + list(fake_files)
        all_labels = list(real_labels) + list(fake_labels)
        
        # Quick subset for reporting if loading from scratch
        import random
        combined = list(zip(all_paths, all_labels))
        random.shuffle(combined)
        val_data = combined[:2000] # 2000 samples
        val_paths, val_labels = zip(*val_data)
        
        val_dataset = DeepfakeDataset(file_paths=list(val_paths), labels=list(val_labels), phase='val')
        val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, shuffle=False)
    
    # 3. Inference
    all_preds = []
    all_labels_list = []
    all_probs = []
    
    print("⚡ Running Inference for Report...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Reporting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels_list.extend(labels.numpy())
            
    all_labels_np = np.array(all_labels_list)
    all_preds_np = np.array(all_preds).flatten()
    all_probs_np = np.array(all_probs).flatten()
    
    # 4. Metrics
    try:
        acc = accuracy_score(all_labels_np, all_preds_np)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels_np, all_preds_np, average='binary', zero_division=0)
        
        # Safe ROC Curve Calculation
        try:
            fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
            roc_auc = auc(fpr, tpr)
        except IndexError:
            print("⚠️ Warning: ROC Curve generation failed due to sklearn IndexError. Skipping ROC plot.")
            fpr, tpr, roc_auc = None, None, 0.0

        cm = confusion_matrix(all_labels_np, all_preds_np)
        
        print(f"\n📊 Report Metrics:")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        
        # 5. Visuals
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {os.path.basename(model_filename)}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(report_plots_dir, "confusion_matrix.png"))
        plt.close()
        
        # ROC Curve
        if fpr is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC - {os.path.basename(model_filename)}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(report_plots_dir, "roc_curve.png"))
            plt.close()
            
        print(f"\n✅ Visuals saved to {report_plots_dir}")
        return acc, roc_auc
        
    except Exception as e:
        print(f"🚨 Error during report generation metrics: {e}")
        return 0.0, 0.0

if __name__ == "__main__":
    generate_report()

