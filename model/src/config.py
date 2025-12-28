import os
import torch
import platform

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
    BATCH_SIZE = 32  # Optimized for Mac M4 (Unified Memory)
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 8  # Leverage M4 Performance Cores
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Paths
    if platform.system() == "Windows":
        # Specific path requested by user for Epoch 2
        DATA_DIR = r"C:\Users\kanna\Downloads\Dataset\Largest Dataset\Largest Dataset"
    else:
        # Mac Path
        DATA_DIR = "/Users/harshvardhan/Developer/DataSet"
    
    # Since we are using the root folder, the script will recursively find ALL images
    # in all sub-datasets and split them 80/20 for training/validation.
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
