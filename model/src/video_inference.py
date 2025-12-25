import cv2
import torch
import numpy as np
import os
from PIL import Image

def process_video(video_path, model, transform, device, frames_per_second=1):
    """
    Process a video file frame-by-frame using the deepfake detection model.
    
    Args:
        video_path (str): Path to the video file.
        model (torch.nn.Module): Loaded PyTorch model.
        transform (callable): Albumentations transform pipeline.
        device (torch.device): Device to run inference on.
        frames_per_second (int): Number of frames to sample per second of video. 
                                 Default is 1 to keep processing fast.
    
    Returns:
        dict: Aggregated results including verdict, average confidence, and frame-level details.
    """
    if model is None:
        return {"error": "Model not loaded"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    # specific video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30 # Fallback
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate sampling interval (step size)
    # If we want 1 frame per second, we step by 'fps' frames
    step = int(fps / frames_per_second)
    if step < 1: step = 1

    frame_indices = []
    probs = []
    
    print(f"Processing video: {video_path}")
    print(f"Duration: {duration:.2f}s, FPS: {fps}, Total Frames: {total_frames}")
    print(f"Sampling every {step} frames...")

    count = 0
    processed_count = 0
    
    suspicious_frames = [] # Store frames with high fake probability

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % step == 0:
            # Process this frame
            try:
                # Convert BGR (OpenCV) to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- Face Extraction ---
                # Load Haar Cascade (lazy load)
                if not hasattr(process_video, "face_cascade"):
                    try:
                        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        process_video.face_cascade = cv2.CascadeClassifier(cascade_path)
                    except:
                        process_video.face_cascade = None

                face_crop = None
                if process_video.face_cascade:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = process_video.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                    )
                    
                    if len(faces) > 0:
                        # Find largest face
                        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                        x, y, w, h = largest_face
                        
                        # Add margin (20%)
                        margin = int(max(w, h) * 0.2)
                        x_start = max(x - margin, 0)
                        y_start = max(y - margin, 0)
                        x_end = min(x + w + margin, frame.shape[1])
                        y_end = min(y + h + margin, frame.shape[0])
                        
                        face_crop = image[y_start:y_end, x_start:x_end]
                
                # Use face crop if found, otherwise use full image
                input_image = face_crop if face_crop is not None else image
                
                # Apply transforms
                augmented = transform(image=input_image)
                image_tensor = augmented['image'].unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    logits = model(image_tensor)
                    prob = torch.sigmoid(logits).item()
                
                probs.append(prob)
                frame_indices.append(count)
                processed_count += 1
                
                # If highly fake, store metadata (timestamp)
                if prob > 0.5:
                    timestamp = count / fps
                    suspicious_frames.append({
                        "timestamp": round(timestamp, 2),
                        "frame_index": count,
                        "fake_prob": round(prob, 4)
                    })
                    
            except Exception as e:
                print(f"Error processing frame {count}: {e}")
        
        count += 1

    cap.release()

    if processed_count == 0:
        return {"error": "No frames processed"}

    # Aggregation
    avg_prob = sum(probs) / len(probs)
    max_prob = max(probs)
    fake_frame_count = len([p for p in probs if p > 0.6]) # Stricter frame threshold
    fake_ratio = fake_frame_count / processed_count
    
    # Verdict Logic (Tuned for High Efficiency Model)
    # The new model is detecting everything as fake, so we need stricter rules.
    
    # 1. Standard Average Check (shifted)
    cond1 = avg_prob > 0.65
    
    # 2. Density Check: Require at least 15% of frames to be strictly fake
    # Was 5%, which is too low for a sensitive model
    cond2 = fake_ratio > 0.15 and max_prob > 0.7
    
    # 3. Peak Check: Only flag single-frame anomalies if EXTREMELY suspicious
    cond3 = max_prob > 0.95
    
    is_fake = cond1 or cond2 or cond3
    
    verdict = "FAKE" if is_fake else "REAL"
    
    # Confidence Calculation
    if is_fake:
        confidence = max(max_prob, 0.6)
    else:
        confidence = 1 - avg_prob
    
    return {
        "type": "video",
        "prediction": verdict,
        "confidence": float(confidence),
        "avg_fake_prob": float(avg_prob),
        "max_fake_prob": float(max_prob),
        "fake_frame_ratio": float(fake_ratio),
        "processed_frames": processed_count,
        "duration": float(duration),
        "timeline": [
            {"time": round(i / fps, 2), "prob": round(p, 3)} 
            for i, p in zip(frame_indices, probs)
        ],
        "suspicious_frames": suspicious_frames[:10] # Top 10 suspicious moments
    }
