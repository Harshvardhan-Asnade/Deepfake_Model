from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os

# Add model directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
import datetime
import torch
import cv2
import os
import numpy as np
import ssl
import base64
from werkzeug.utils import secure_filename
import io
from PIL import Image

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch import ToTensorV2
from src.models import DeepfakeDetector
from src.config import Config
import database

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
HISTORY_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'history_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model and transform
device = torch.device(Config.DEVICE)
model = None
transform = None

def get_transform():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def load_model():
    """Load the trained deepfake detection model"""
    global model, transform
    
    checkpoint_dir = Config.CHECKPOINT_DIR
    # Explicitly target the model requested by the user
    target_model_name = "patched_model.safetensors"
    checkpoint_path = os.path.join(checkpoint_dir, target_model_name)
    
    print(f"Using device: {device}")
    
    # Initialize with pretrained=True to ensure missing keys (frozen layers) have valid ImageNet weights
    # instead of random noise. This fixes the "random prediction" issue when the checkpoint 
    # only contains finetuned layers.
    model = DeepfakeDetector(pretrained=True)
    model.to(device)
    model.eval()
    
    # Check if file exists first
    if not os.path.exists(checkpoint_path):
        print(f"❌ CRITICAL ERROR: Model file not found at: {checkpoint_path}")
        print(f"Please ensure '{target_model_name}' exists in '{checkpoint_dir}'")
        model = None
        transform = get_transform()
        return model, transform

    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        if checkpoint_path.endswith(".safetensors") and SAFETENSORS_AVAILABLE:
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)
            
        # Use strict=False because the checkpoint might be a partial save (e.g. only finetuned layers)
        # or there might be minor architecture mismatches.
        # Since we use pretrained=True, the missing keys will remain as ImageNet weights (valid features).
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"✅ Model loaded successfully!")
        if missing_keys:
            print(f"ℹ️  {len(missing_keys)} keys missing from checkpoint (using pretrained defaults).")
        if unexpected_keys:
            print(f"ℹ️  {len(unexpected_keys)} unexpected keys in checkpoint.")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("Predictions will fail until this is resolved.")
        model = None
    
    transform = get_transform()
    return model, transform

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Make prediction on a single image"""
    if model is None:
        return None, "Error: Model not loaded. Check backend logs for 'best_finetuned_datasetB.safetensors' error."

    try:
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Could not read image"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        # Make prediction
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).item()
        
        # Generate Heatmap
        heatmap = model.get_heatmap(image_tensor)
        
        # Process Heatmap for Visualization
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose
        # Heatmap is BGR (from cv2), Image is RGB. Convert Image to BGR.
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        superimposed_img = heatmap * 0.4 + image_bgr * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # Encode to Base64
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        
        is_fake = prob > 0.5
        label = "FAKE" if is_fake else "REAL"
        confidence = prob if is_fake else 1 - prob
        
        return {
            'prediction': label,
            'confidence': float(confidence),
            'fake_probability': float(prob),
            'real_probability': float(1 - prob),
            'heatmap': heatmap_b64
        }, None
    except Exception as e:
        return None, str(e)


@app.route('/')
def index():
    """Serve the frontend"""
    # Use absolute path to avoid CWD issues
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/history_uploads/<path:filename>')
def serve_history_image(filename):
    """Serve history images"""
    return send_from_directory(HISTORY_FOLDER, filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg, webp'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result, error = predict_image(filepath)
        
        # Save to History
        import shutil
        history_filename = f"scan_{int(datetime.datetime.now().timestamp())}_{filename}"
        history_path = os.path.join(HISTORY_FOLDER, history_filename)
        
        # Copy original file to history folder
        # We need to read the file again or just copy if we haven't deleted it?
        # We read via cv2, the file is still at filepath.
        shutil.copy(filepath, history_path)
        
        # Relative path for frontend
        relative_path = f"history_uploads/{history_filename}"

        database.add_scan(
            filename=filename,
            prediction=result['prediction'],
            confidence=result['confidence'],
            fake_prob=result['fake_probability'],
            real_prob=result['real_probability'],
            image_path=relative_path
        )
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get all past scans"""
    history = database.get_history()
    history = database.get_history()
    return jsonify(history)

@app.route('/api/history/<int:scan_id>', methods=['DELETE'])
def delete_scan(scan_id):
    """Delete a specific scan"""
    if database.delete_scan(scan_id):
        return jsonify({'message': 'Scan deleted'})
    return jsonify({'error': 'Failed to delete scan'}), 500

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear all history"""
    if database.clear_history():
        return jsonify({'message': 'History cleared'})
    return jsonify({'error': 'Failed to clear history'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model information"""
    return jsonify({
        'model_name': 'DeepGuard: Advanced Deepfake Detector',
        'architecture': 'Hybrid CNN-ViT',
        'components': {
            'RGB Analysis': Config.USE_RGB,
            'Frequency Domain': Config.USE_FREQ,
            'Patch-based Detection': Config.USE_PATCH,
            'Vision Transformer': Config.USE_VIT
        },
        'image_size': Config.IMAGE_SIZE,
        'device': str(device),
        'threshold': 0.5
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 DeepGuard - Deepfake Detection System")
    print("=" * 60)
    
    # Load model
    load_model()
    
    print("=" * 60)
    print("🌐 Starting server on http://localhost:5001")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
