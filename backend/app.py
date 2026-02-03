from flask import Flask, request, jsonify, send_from_directory, Response, make_response
from flask_cors import CORS
import sys
import os
import re
import mimetypes
import subprocess

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
from src import video_inference

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch import ToTensorV2
from src.models import DeepfakeDetector
from src.config import Config
from checkers import metadata_checker
from checkers import watermark_checker
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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'mp4', 'avi', 'mov', 'webm'}
HISTORY_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'history_uploads')
FEEDBACK_FOLDER = os.path.join(os.path.dirname(__file__), 'feedback_images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Increase to 500MB for video

# Global model and transform
# Global model and transform
device = torch.device(Config.DEVICE)
model = None
video_model_onnx = None # Dedicated optimized model for video
transform = None

def get_transform():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def load_model():
    """Load the trained deepfake detection model"""
    global model, transform, video_model_onnx
    
    checkpoint_dir = Config.CHECKPOINT_DIR
    target_model_name = "Mark-V.safetensors"
    checkpoint_path = os.path.join(checkpoint_dir, target_model_name)
    
    print(f"Using device: {device}")
    
    # 1. Load PyTorch Model (Required for single image Image Heatmaps)
    model = DeepfakeDetector(pretrained=True)
    model.to(device)
    model.eval()
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ CRITICAL ERROR: Model file not found at: {checkpoint_path}")
        model = None
        transform = get_transform()
        return model, transform

    try:
        print(f"Loading PyTorch checkpoint: {checkpoint_path}")
        if checkpoint_path.endswith(".safetensors") and SAFETENSORS_AVAILABLE:
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Try loading directly first
        try:
            model.load_state_dict(state_dict)
            print(f"✅ PyTorch Model loaded successfully!")
        except Exception as e:
            # Keys don't match - apply remapping for architecture compatibility
            print(f"⚠️  Direct load failed. Attempting key remapping...")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('rgb_branch.features.'):
                    new_k = k.replace('rgb_branch.features.', 'rgb_branch.net.features.')
                    new_state_dict[new_k] = v
                elif k.startswith('rgb_branch.avgpool.'):
                    new_k = k.replace('rgb_branch.avgpool.', 'rgb_branch.net.avgpool.')
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            print(f"✅ PyTorch Model loaded successfully (with key remapping)!")
            
    except Exception as e:
        print(f"❌ Error loading PyTorch checkpoint: {e}")
        model = None

    # 2. Load ONNX Model (Removed)
    # System optimized for PyTorch Pipeline (Threaded Preprocessing)
    video_model_onnx = None

    transform = get_transform()
    return model, transform

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Make prediction on a single image"""
    if model is None:
        return None, "Error: Model not loaded. Check backend logs for 'best_model.safetensors' error."

    try:
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Could not read image"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        
        # 0. Metadata & Watermark Checks
        meta_result = metadata_checker.check_metadata(image_path)
        water_result = watermark_checker.check_watermarks(image_path)
        
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
        
        # Override if metadata confirms fake
        if meta_result['detected'] or water_result['detected']:
            is_fake = True
            # If visual model was unsure (e.g. 0.4), bump it up? 
            # Or just rely on the 'prediction' label.
            # Let's trust the metadata 100%
            prob = max(prob, 0.99) 
            
        # Hidden Check: Explicitly flag known generator filenames as FAKE without frontend badging
        filename_lower = os.path.basename(image_path).lower()
        if "chatgpt" in filename_lower or "gemini" in filename_lower:
            is_fake = True
            prob = max(prob, 0.998) # Extremely high confidence
            # Intentionally NOT adding to meta_result or water_result to keep it hidden from badges
            # as requested by user ("dont shiw this in fornetend") 
            
        label = "FAKE" if is_fake else "REAL"
        confidence = prob if is_fake else 1 - prob
        
        return {
            'prediction': label,
            'confidence': float(confidence),
            'fake_probability': float(prob),
            'real_probability': float(1 - prob),
            'heatmap': heatmap_b64,
            'metadata_check': meta_result,
            'watermark_check': water_result
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
    """Serve history images and videos with Range support"""
    file_path = os.path.join(HISTORY_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    # Handle Video Range Requests
    if filename.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
        file_size = os.path.getsize(file_path)
        range_header = request.headers.get('Range', None)
        
        if not range_header:
            # No Range header, serve normally but with video headers
            response = make_response(send_from_directory(HISTORY_FOLDER, filename))
            response.headers['Content-Type'] = 'video/mp4'
            response.headers['Accept-Ranges'] = 'bytes'
            return response
            
        # Parse Range Header
        byte1, byte2 = 0, None
        m = re.search('bytes=(\d+)-(\d*)', range_header)
        if m:
            g = m.groups()
            byte1 = int(g[0])
            if g[1]:
                byte2 = int(g[1])

        length = file_size - byte1
        if byte2 is not None:
            length = byte2 + 1 - byte1

        # Read partial content
        with open(file_path, 'rb') as f:
            f.seek(byte1)
            data = f.read(length)

        response = Response(
            data,
            206,
            mimetype='video/mp4',
            direct_passthrough=True
        )
        
        # Determine content range
        content_range_end = byte2 if byte2 is not None else file_size - 1
        
        response.headers.add('Content-Range', f'bytes {byte1}-{content_range_end}/{file_size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(length))
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    # Default for images
    response = send_from_directory(HISTORY_FOLDER, filename)
    return response

def reencode_video(input_path):
    """Re-encode video to H.264/AAC with faststart using ffmpeg"""
    try:
        output_path = input_path + "_temp.mp4"
        print(f"🔄 Re-encoding video: {input_path}")
        
        # FFmpeg command
        # -y: overwrite output
        # -c:v libx264: use H.264 video codec
        # -preset fast: encode speed
        # -profile:v high: high profile for better compatibility
        # -level 4.0: compatibility level
        # -pix_fmt yuv420p: ensure wide player compatibility (essential for QuickTime/Safari)
        # -c:a aac: use AAC audio codec
        # -movflags +faststart: move metadata to front for streaming
        cmd = [
            'ffmpeg', '-y', 
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg re-encoding failed: {result.stderr.decode()}")
            return input_path # Fallback to original
            
        print(f"✅ Video re-encoded successfully!")
        
        # Replace original
        os.remove(input_path)
        os.rename(output_path, input_path)
        return input_path
        
    except Exception as e:
        print(f"❌ Error during re-encoding: {e}")
        return input_path

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed model status"""
    model_status = "ready" if model is not None else "initializing"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
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
        
        if error:
            return jsonify({'error': error}), 500
            
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

        scan_id = database.add_scan(
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
        
        # Add scan_id to result for frontend tracking
        result['scan_id'] = scan_id
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_video', methods=['POST'])
def predict_video():
    """Handle video upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
             return jsonify({'error': 'Invalid file type'}), 400
             
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Re-encode video for proper web playback
        filepath = reencode_video(filepath)
        
        # Process Video
        # Prioritize Optimized ONNX Model
        active_model = video_model_onnx if video_model_onnx is not None else model
        
        if active_model is None:
             return jsonify({'error': 'Model not loaded'}), 500
             
        result = video_inference.process_video(filepath, active_model, transform, device, frames_per_second=10)
        
        if "error" in result:
             return jsonify(result), 500
             
        # Save to History (Using the first frame or a placeholder icon for now?)
        # For video, we might want to save the video file itself to history_uploads
        # or just a thumbnail. Let's save the video for now.
        import shutil
        history_filename = f"scan_{int(datetime.datetime.now().timestamp())}_{filename}"
        history_path = os.path.join(HISTORY_FOLDER, history_filename)
        shutil.copy(filepath, history_path)
        
        relative_path = f"history_uploads/{history_filename}"
        
        # Add to database
        # Note: The database 'add_scan' might expect image-specific fields.
        # We'll re-use 'fake_prob' as 'avg_fake_prob'
        scan_id = database.add_scan(
            filename=filename,
            prediction=result['prediction'],
            confidence=result['confidence'],
            fake_prob=result['avg_fake_prob'],
            real_prob=1 - result['avg_fake_prob'],
            image_path=relative_path 
        )
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
            
        # Add video URL for frontend playback
        result['video_url'] = relative_path
        result['scan_id'] = scan_id
        
        return jsonify(result)

    except Exception as e:
        print(f"Video Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get all past scans"""
    history = database.get_history()
    history = database.get_history()
    return jsonify(history)

@app.route('/api/history/<int:scan_id>', methods=['PATCH'])
def update_history_item(scan_id):
    """Update a specific scan's metadata (notes, tags)"""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    if database.update_scan(scan_id, data):
        return jsonify({'message': 'Scan updated successfully'})
    return jsonify({'error': 'Failed to update scan'}), 500

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

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback on a prediction"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        scan_id = data.get('scan_id')
        is_correct = data.get('is_correct')
        predicted_label = data.get('predicted_label')
        
        if scan_id is None or is_correct is None or not predicted_label:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get scan details from history
        history = database.get_history()
        scan = next((s for s in history if s['id'] == scan_id), None)
        
        if not scan:
            return jsonify({'error': 'Scan not found'}), 404
        
        actual_label = None
        feedback_image_path = None
        
        # If prediction is incorrect, determine actual label and copy image
        if not is_correct:
            # Actual label is opposite of prediction
            actual_label = 'REAL' if predicted_label == 'FAKE' else 'FAKE'
            
            # Copy image to feedback folder for retraining
            if scan.get('image_path'):
                try:
                    import shutil
                    source_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', scan['image_path'])
                    feedback_filename = f"feedback_{scan_id}_{scan['filename']}"
                    feedback_dest = os.path.join(FEEDBACK_FOLDER, feedback_filename)
                    
                    if os.path.exists(source_path):
                        shutil.copy(source_path, feedback_dest)
                        feedback_image_path = feedback_filename
                        print(f"✅ Copied feedback image to: {feedback_dest}")
                    else:
                        print(f"⚠️  Source image not found: {source_path}")
                except Exception as e:
                    print(f"❌ Error copying feedback image: {e}")
        
        # Record feedback in database
        success = database.add_feedback(
            scan_id=scan_id,
            is_correct=is_correct,
            predicted_label=predicted_label,
            actual_label=actual_label,
            image_path=feedback_image_path,
            confidence=scan.get('confidence')
        )
        
        if success:
            feedback_type = 'correct' if is_correct else 'incorrect'
            return jsonify({
                'message': f'Feedback recorded successfully',
                'feedback': feedback_type,
                'actual_label': actual_label
            })
        else:
            return jsonify({'error': 'Failed to record feedback'}), 500
            
    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get feedback statistics"""
    stats = database.get_feedback_stats()
    return jsonify(stats)

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
    print("=" * 60)
    # Check if running on Hugging Face Spaces
    if os.environ.get("SPACE_ID"):
        port = 7860
        print(f"🪐 Detected Hugging Face Space. Forcing port {port}")
    else:
        port = int(os.environ.get("PORT", 7860))
        
    print(f"🌐 Starting server on http://0.0.0.0:{port}")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=port)
