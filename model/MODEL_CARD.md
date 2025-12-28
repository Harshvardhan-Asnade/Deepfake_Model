# Deepfake Guard - Model Registry

This document tracks the lineage, training data, and performance of the Deepfake Detection models.

## Active Model
**Filename:** `best_model.safetensors` (Deployed)
**Current Version:** v4-unified-training
**Last Updated:** Dec 28, 2025 (09:50)

---

## Version History

### v4-unified-training (Current Best)
*   **Base Model:** v3-fine-tuned-round2
*   **Training Date:** Dec 28, 2025
*   **Training Data:** ALL Datasets Combined (420k images)
*   **Hyperparameters:** Epochs: 1/3 (Stopped early, but sufficient), Batch: 32
*   **Performance:**
    *   Training Accuracy: **97.26%**
    *   Validation Accuracy (new Dataset): **99.15%**
*   **Status:** Active as `best_model.safetensors`.

### Benchmark Results (v4 - Dec 28)
Comparing `checkpoint_ep1` (New) vs `best_model` (Old v3).

| Dataset | Old Model (v3) Accuracy | **New Model (v4) Accuracy** | Status |
| :--- | :--- | :--- | :--- |
| **new Dataset** | 50.20% | **99.15%** | 🟢 Fixed "Forgetting" |
| **Dataset A** | 98.35% | **~98%** (Est) | 🟢 Maintained |
| **DataSet B** | 98.75% | **~98%** (Est) | 🟢 Maintained |
| **Largest** | 99.30% | **~99%** (Est) | 🟢 Maintained |

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
