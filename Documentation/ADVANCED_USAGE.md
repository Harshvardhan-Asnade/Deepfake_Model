# Advanced Usage: Fine-Tuning & Evaluation

This guide covers advanced operations like fine-tuning the model on your own dataset and benchmarking model performance.

## Fine-Tuning

If you have a custom dataset or want to improve performance on specific types of deepfakes, you can fine-tune the pre-trained DeepGuard model.

### 1. Data Preparation
Organize your dataset in the following structure:
```
dataset/
    Train/
        Real/
            img1.jpg
            ...
        Fake/
            img1.jpg
            ...
    Validation/
        Real/
            ...
        Fake/
            ...
```

### 2. Configure Training Script
Open `model/finetune.py` and update the `TRAIN_PATH` and `VAL_PATH` variables if necessary, or pass them as arguments if supported.

```python
# In model/finetune.py
TRAIN_PATH = "/absolute/path/to/dataset/Train"
VAL_PATH = "/absolute/path/to/dataset/Validation"
```

### 3. Run Fine-Tuning
Execute the script from the `model` directory:
```bash
cd model
python finetune.py
```

**Key Parameters (in `finetune.py`):**
- **Learning Rate**: `1e-5` (Lower than initial training to preserve features)
- **Epochs**: `1` (Usually sufficient for transfer learning on large datasets)
- **Checkpoint Strategy**: The script automatically looks for `Mark-V.safetensors` to resume training.

---

## Evaluation & Benchmarking

To rigorously test your model against a set of test images, use the provided evaluation scripts.

### Compare Models
Use `compare_models.py` to test multiple checkpoints against the same test set and compare metrics (Accuracy, Precision, Recall, F1).

```bash
cd model
python compare_models.py
```
> **Note**: Update `test_dir` in the script to point to your test dataset.

### Detailed Metrics
`evaluate_models.py` provides a detailed breakdown of True Positives, False Positives, etc., per epoch checkpoint.

**Output Example:**
```text
📊 RESULTS:
  Accuracy:  98.45%
  Precision: 99.10%
  Recall:    97.80%
  F1 Score:  0.9845
```
