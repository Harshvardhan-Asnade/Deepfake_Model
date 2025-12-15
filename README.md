# Deepfake Detection System

A state-of-the-art image authenticity detection system designed to distinguish between real (camera-captured) and AI-generated (Deepfake/GAN/Diffusion) images using a **Multi-Branch Deep Learning Architecture**.

## 🚀 Key Features
*   **Multi-Branch Architecture**:
    *   **RGB Branch**: EfficientNetV2-S for spatial feature extraction.
    *   **Frequency Branch**: Custom CNN processing the Log-Magnitude Spectrum (FFT) to detect spectral artifacts.
    *   **Patch Branch**: Shared CNN processing local image patches to identify inconsistent textures.
    *   **ViT Branch**: Swin Transformer V2 (Tiny) for global semantic consistency.
*   **Probabilistic Output**: Returns a confidence score (0.0 - 1.0) rather than a binary label.
*   **Ensemble Inference**: Supports averaging predictions from multiple model checkpoints for robustness.
*   **Robust Data Augmentation**: Trained with compression, resize, blur, and noise to handle diverse image qualities.
*   **SafeTensor Support**: Secure model weight saving and loading.

## 🛠️ Installation
1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd morden-detections-system
    ```
2.  **Install Dependencies**:
    ```bash
    pip install torch torchvision numpy opencv-python albumentations tqdm safetensors
    ```

## 🏃 Usage
### 1. Training
Configure your dataset path in `src/config.py` or ensure `data/train` exists.
```bash
python src/train.py
```
*   **Automatic Splitting**: If `TRAIN` and `TEST` paths are identical, the script automatically verifies data and creates an 80/20 train/val split.

### 2. Inference
Run detection on a single image or a directory of images:
```bash
python src/inference.py --source path/to/image.jpg
```
**Ensemble Inference**:
To use multiple checkpoints for higher reliability:
```bash
python src/inference.py --source path/to/image.jpg --checkpoints results/checkpoints/
```

## ⚠️ Limitations & Uncertainty (Requirement #8)
While this system strives for high generalization, 100% detection accuracy is theoretically impossible due to the rapid evolution of generative AI.

### Known Failure Scenarios
1.  **High Compression**: Images saved with extremely low JPEG quality (Q<50) may lose the high-frequency artifacts (fingerprints) relied upon by the Frequency and Patch branches.
2.  **Adversarial Perturbations**: Images intentionally modified with adversarial noise to fool classifiers may result in incorrect high-confidence scores.
3.  **Perfectly Generated Anatomy**: Older models struggled with hands/eyes; newer models (e.g., SDXL, Midjourney v6) are correcting this. The ViT branch aims to catch global semantic errors, but subtle improvements may reduce detection rates.
4.  **Unknown Generators**: The model is trained on known generators (SD, DALL-E, etc.). A completely novel architecture with different spectral characteristics might yield lower confidence until retrained.

### Interpreting Scores
*   **0.0 - 0.2**: High confidence **REAL**.
*   **0.8 - 1.0**: High confidence **FAKE**.
*   **0.3 - 0.7**: **Uncertain Region**. The model detects conflicting signals (e.g., realistic frequency spectrum but semantic oddities).
    *   *Action*: Manual Verification recommended. Check for logical inconsistencies (lighting, reflections, text).

## 📂 Project Structure
*   `src/models.py`: 4-Branch Network (RGB, Freq, Patch, ViT).
*   `src/dataset.py`: Data loading & Albumentations pipeline.
*   `src/utils.py`: FFT/Frequency domain helpers.
*   `src/train.py`: Training loop with SafeTensors.
*   `src/inference.py`: Probabilistic & Ensemble inference.
