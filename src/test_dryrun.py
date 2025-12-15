import torch
import torch.nn as nn
from src.models import DeepfakeDetector
from src.config import Config

def test_model_architecture():
    print("Testing DeepfakeDetector Architecture...")
    
    # Check device
    device = torch.device("cpu") # Test on CPU for simplicity or Config.DEVICE
    print(f"Device: {device}")
    
    # Initialize Model
    try:
        model = DeepfakeDetector(pretrained=False).to(device)
        print("[Pass] Model Initialization")
    except Exception as e:
        print(f"[Fail] Model Initialization: {e}")
        return

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    print(f"Input Shape: {x.shape}")
    
    # Forward Pass
    try:
        out = model(x)
        print(f"Output Shape: {out.shape}")
        
        if out.shape == (batch_size, 1):
            print("[Pass] Output Shape Correct")
        else:
            print(f"[Fail] Output Shape Incorrect. Expected ({batch_size}, 1), got {out.shape}")
    except Exception as e:
        print(f"[Fail] Forward Pass: {e}")
        # Debug trace
        import traceback
        traceback.print_exc()
        return

    # Loss and Backward
    try:
        criterion = nn.BCEWithLogitsLoss()
        target = torch.ones(batch_size, 1).to(device)
        loss = criterion(out, target)
        loss.backward()
        print(f"[Pass] Backward Pass (Loss: {loss.item():.4f})")
    except Exception as e:
        print(f"[Fail] Backward Pass: {e}")
        return

    print("\nSUCCESS: Model architecture verification passed!")

if __name__ == "__main__":
    test_model_architecture()
