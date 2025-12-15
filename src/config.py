import os
import torch

class Config:
    # System
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    
    # Model Architecture
    IMAGE_SIZE = 256
    NUM_CLASSES = 1  # Logic: 0=Real, 1=Fake (Sigmoid output)
    
    # Component Flags
    USE_RGB = True
    USE_FREQ = True
    USE_PATCH = True
    USE_VIT = True
    
    # Training Hyperparameters
    BATCH_SIZE = 32  # Optimal for M4 MPS (48 caused extreme slowdown)
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4  # Optimal for M4 (matches 4 Performance cores)
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Paths
    # User provided path: /Users/harshvardhan/Developer/Deepfake private/open-deepfake-detection/dataset
    # This directory contains 'real' and 'fake' folders directly.
    DATA_DIR = "/Users/harshvardhan/Developer/Deepfake private/open-deepfake-detection/dataset"
    
    # Since the user has a single 'dataset' folder with real/fake, we point both train and test to it.
    # The training script will handle the split.
    TRAIN_DATA_PATH = DATA_DIR 
    TEST_DATA_PATH = DATA_DIR 
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

    @classmethod
    def setup(cls):
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        print(f"Project initialized at {cls.PROJECT_ROOT}")
        print(f"Using device: {cls.DEVICE}")

if __name__ == "__main__":
    Config.setup()
