import sys
import os

# Adjust path to include project root (one level up from src)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import Config
from src.models import DeepfakeDetector
import torch
try:
    from safetensors.torch import load_file
    print("Safetensors imported successfully.")
except ImportError:
    print("Safetensors import FAILED.")

print(f"Config Active Model Path: {Config.ACTIVE_MODEL_PATH}")

if not os.path.exists(Config.ACTIVE_MODEL_PATH):
    print("❌ File does NOT exist at path.")
else:
    print("✅ File exists at path.")
    print(f"File size: {os.path.getsize(Config.ACTIVE_MODEL_PATH)} bytes")

print("Attempting to load...")
try:
    model = DeepfakeDetector(pretrained=False)
    state_dict = load_file(Config.ACTIVE_MODEL_PATH)
    model.load_state_dict(state_dict)
    print("✅ Successfully loaded state dict!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
