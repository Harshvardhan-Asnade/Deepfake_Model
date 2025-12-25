# Deepfake Guard - Model Registry

This document tracks the lineage, training data, and performance of the Deepfake Detection models.

## Active Model
**Filename:** `best_model.safetensors` (Needs to be deployed)
**Current Version:** v3-fine-tuned-round2
**Last Updated:** Dec 24, 2025 (18:44)

---

## Version History

### v3-fine-tuned-round2 (Newest)
*   **Base Model:** v2-fine-tuned
*   **Training Date:** Dec 24, 2025 (Evening Run)
*   **Training Data:** Same 63k (`/DataSet/new Dataset`)
*   **Hyperparameters:**
    *   Epochs: 1 (Additional epoch, total 2 on this dataset)
*   **Performance:**
    *   Validation Accuracy: **99.70%** (+0.27%)
    *   Training Loss: 0.0708
*   **Status:** Currently stored as `best_model.safetensors`. To make active, overwrite `patched_model`.

### Benchmark Results (v3 - Dec 24)
We tested the model on 4 different datasets (500 samples each).

| Dataset | Accuracy | Sensitivity (Catch Fakes) | Specificity (Avoid FP) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **New Dataset** (Training) | **100.00%** | 100.00% | 100.00% | Perfect fit. |
| **Dataset A** | 48.63% | 2.24% | **99.59%** | Fails to detect fakes. |
| **DataSet B** | 56.84% | 11.61% | **92.01%** | Fails to detect fakes. |
| **Largest Dataset** | 48.83% | 9.56% | **93.33%** | Fails to detect fakes. |

**Observation:** The model is highly specialized for "New Dataset" patterns but struggles to generalize to the other datasets. It is very conservative (high specificity), confusing most unknown fakes for real.

### v2-fine-tuned (Deployed)
*   **Base Model:** v1-legacy
*   **Training Date:** Dec 24, 2025 (Afternoon Run)
*   **Training Data:** 63,792 Images
*   **Performance:**
    *   Validation Accuracy: **99.43%**
*   **Status:** Currently stored as `patched_model.safetensors`.

### v1-legacy (Original)
*   **Base Model:** Pre-trained ImageNet Weights (EfficientNet/ViT)
*   **Training Date:** Unknown (Pre-existing)
*   **Training Data:**
    *   **Source:** Original `open-deepfake-detection` dataset.
    *   **Size:** Unknown (Files missing from disk).
*   **Performance:**
    *   Good general accuracy but low sensitivity on compressed videos (18% detection rate on `Video Dataset`).
    *   Struggled with face-specific artifacts.

---

## How to Track Future Models
When running `train.py`, manually add a new entry to this file with:
1.  **Date**: When you ran it.
2.  **Dataset**: Which folder you pointed `config.py` to.
3.  **Changes**: Did you run 1 epoch? 10 epochs? Changing learning rate?
4.  **Results**: Copy the "Best Validation Accuracy" from the terminal output.
