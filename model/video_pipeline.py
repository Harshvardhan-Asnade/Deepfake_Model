import cv2
import numpy as np
import threading
import queue
import time
import sys
import os

try:
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

import albumentations as A
from albumentations.pytorch import ToTensorV2

class VideoPipeline:
    def __init__(self, model_wrapper, transform, batch_size=8, frame_stride=5):
        """
        Args:
            model_wrapper: Instance of DeepGuardONNX
            transform: Albumentations transform
            batch_size: Number of frames to infer at once
            frame_stride: Process 1 out of N frames
        """
        self.model = model_wrapper
        self.transform = transform
        self.batch_size = batch_size
        self.frame_stride = frame_stride
        
        # Queues
        # frame_queue: stores (frame_idx, raw_image_bgr)
        self.frame_queue = queue.Queue(maxsize=128)
        # batch_queue: stores (start_idx, batch_tensor_numpy, original_frames_for_viz)
        self.batch_queue = queue.Queue(maxsize=16)
        # result_queue: stores results
        self.result_queue = queue.Queue()
        
        self.stop_event = threading.Event()
        self.video_info = {}

    def _decode_worker(self, video_path):
        """Reads video frames and puts them in frame_queue"""
        try:
            if DECORD_AVAILABLE:
                self._decode_decord(video_path)
            else:
                print("⚠️  Decord not installed. Falling back to OpenCV decoder.")
                self._decode_opencv(video_path)
        except Exception as e:
            print(f"❌ Decode Error: {e}")
        finally:
            self.stop_event.set()

    def _decode_decord(self, video_path):
        # Prefer CPU for decoding to leave GPU for inference, unless distinct GPU decoding is needed
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        self.video_info = {'total_frames': total_frames, 'fps': fps, 'duration': total_frames/fps}
        
        indices = list(range(0, total_frames, self.frame_stride))
        
        print(f"🎞️  Processing {len(indices)} frames (Stride: {self.frame_stride}) w/ Decord")
        
        # Batch read for Decord is faster
        read_batch_size = 32
        for i in range(0, len(indices), read_batch_size):
            if self.stop_event.is_set(): break
            
            batch_indices = indices[i : i + read_batch_size]
            frames = vr.get_batch(batch_indices).asnumpy() # (N, H, W, C) RGB
            
            for j, frame in enumerate(frames):
                # Decord returns RGB, OpenCV usually expects BGR for existing pipeline consistency
                # checking what transform expects. 
                # Our standard transform (albumentations) expects RGB.
                # So we keep it RGB.
                
                # However, original pipeline might have used CV2 BGR.
                # Let's standardize on RGB for 'raw_image' in queue
                idx = batch_indices[j]
                self.frame_queue.put((idx, frame)) # Block if full
        
    def _decode_opencv(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_info = {'total_frames': total_frames, 'fps': fps, 'duration': total_frames/fps}
        
        frame_idx = 0
        while cap.isOpened():
            if self.stop_event.is_set(): break
            
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % self.frame_stride == 0:
                # CV2 is BGR, convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_queue.put((frame_idx, frame_rgb))
            
            frame_idx += 1
        cap.release()

    def _preprocess_worker(self):
        """Aggregates frames into batches and applies Transform"""
        current_batch_imgs = []
        current_batch_indices = []
        current_batch_raw = []
        
        while not (self.stop_event.is_set() and self.frame_queue.empty()):
            try:
                # 1 second timeout to check stop_event
                item = self.frame_queue.get(timeout=1)
                idx, frame_rgb = item
                
                # Preprocess
                # Transform expects 'image' kwarg
                augmented = self.transform(image=frame_rgb)
                # Albumentations ToTensorV2 returns Tensor (C, H, W)
                # Converting to numpy for ONNX Runtime: (C, H, W)
                img_tensor = augmented['image'].numpy()
                
                current_batch_imgs.append(img_tensor)
                current_batch_indices.append(idx)
                current_batch_raw.append(frame_rgb)
                
                if len(current_batch_imgs) == self.batch_size:
                    # Stack: (B, C, H, W)
                    batch_np = np.stack(current_batch_imgs)
                    self.batch_queue.put({
                        'indices': current_batch_indices,
                        'data': batch_np,
                        'raw': current_batch_raw
                    })
                    current_batch_imgs = []
                    current_batch_indices = []
                    current_batch_raw = []
                    
            except queue.Empty:
                continue
                
        # Process remaining
        if current_batch_imgs:
            batch_np = np.stack(current_batch_imgs)
            self.batch_queue.put({
                'indices': current_batch_indices,
                'data': batch_np,
                'raw': current_batch_raw
            })

    def run(self, video_path):
        """Main execution method"""
        
        # Start Threads
        t_dec = threading.Thread(target=self._decode_worker, args=(video_path,))
        t_pre = threading.Thread(target=self._preprocess_worker)
        
        t_dec.start()
        t_pre.start()
        
        # Main Thread: Inference Consumer
        results_agg = {
            'probs': [],
            'indices': [],
            'thumbnails': [] # base64 thumbnails
        }
        
        import base64
        
        while True:
            # Check exit condition: threads done and batch queue empty
            if not t_dec.is_alive() and not t_pre.is_alive() and self.batch_queue.empty():
                break
                
            try:
                batch_item = self.batch_queue.get(timeout=1)
            except queue.Empty:
                continue
                
            # Run Inference
            # input shape: (B, 3, 256, 256)
            logits = self.model.predict(batch_item['data'])
            
            # Sigmoid
            probs = 1 / (1 + np.exp(-logits)) # (B, 1) or (B,)
            if probs.ndim == 2: probs = probs.flatten()
            
            # Store results
            for k in range(len(probs)):
                idx = batch_item['indices'][k]
                prob = float(probs[k])
                raw_img = batch_item['raw'][k] # RGB
                
                results_agg['probs'].append(prob)
                results_agg['indices'].append(idx)
                
                # Generate thumbnail for suspicious frames or timeline
                # Resize to small
                thumb = cv2.resize(raw_img, (160, 90))
                # Convert back to BGR for encoding
                thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', thumb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                b64 = base64.b64encode(buffer).decode('utf-8')
                
                results_agg['thumbnails'].append(b64)
                
        t_dec.join()
        t_pre.join()
        
        # Final aggregation logic (similar to video_inference.py)
        if not results_agg['probs']:
             return {"error": "No frames processed"}
             
        probs = results_agg['probs']
        indices = results_agg['indices']
        thumbnails = results_agg['thumbnails']
        
        avg_prob = sum(probs) / len(probs)
        max_prob = max(probs)
        fake_ratio = len([p for p in probs if p > 0.6]) / len(probs)
        
        is_fake = (avg_prob > 0.65) or (fake_ratio > 0.15 and max_prob > 0.7) or (max_prob > 0.95)
        
        # Build timeline
        timeline = []
        fps = self.video_info.get('fps', 30)
        
        for i in range(len(probs)):
            timeline.append({
                "time": round(indices[i] / fps, 2),
                "prob": round(probs[i], 3),
                "thumbnail": thumbnails[i]
            })
            
        return {
            "prediction": "FAKE" if is_fake else "REAL",
            "confidence": float(max(max_prob, 0.6) if is_fake else 1 - avg_prob),
            "avg_fake_prob": float(avg_prob),
            "processed_frames": len(probs),
            "duration": self.video_info.get('duration', 0),
            "timeline": timeline
        }
