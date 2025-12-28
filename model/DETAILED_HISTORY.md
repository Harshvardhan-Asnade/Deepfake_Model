# Detailed Model History Log

This file contains a comprehensive record of every training run, including detailed configuration, dataset statistics, and hardware usage.

---

## Model: v4-unified-training (Active Best)

| Feature | Detail |
| :--- | :--- |
| **Filename** | `best_model.safetensors` |
| **Created On** | Dec 28, 2025 at 09:42:11 AM |
| **Model Architecture** | EfficientNet-B0 + Vision Transformer (ViT) Hybrid |
| **Training Duration** | ~10 hours (Epoch 1 only) |
| **Training Hardware** | Mac M4 (MPS Acceleration) |

### 🧠 Training Data
This model was trained on a combined dataset of **420,508 images**:
| Dataset Name | Role | Image Count |
| :--- | :--- | :--- |
| `dataset A` | Fine-tuning (High quality) | ~80,000 |
| `DataSet B` | Generalization | ~20,000 |
| `Largest Dataset` | Volume / Diversity | ~60,000 |
| `new Dataset` | Core Knowledge | ~260,000 |

### 🎯 Performance Benchmarks
| Test Data | Accuracy | Precision | Verdict |
| :--- | :--- | :--- | :--- |
| **new Dataset** | **99.15%** | **98.52%** | 🚀 Solved (Previously 50%) |
| **Dataset A** | ~98% | High | ✅ Retained Knowledge |
| **DataSet B** | ~98% | High | ✅ Retained Knowledge |
| **Largest** | ~99% | High | ✅ Retained Knowledge |

### ⚙️ Technical Config used
*   **Batch Size:** 32
*   **Optimizer:** AdamW (`lr=1e-4`)
*   **Loss Function:** BCEWithLogitsLoss
*   **Precision:** Standard (Float32) - *AMP disabled for MPS stability*

---
