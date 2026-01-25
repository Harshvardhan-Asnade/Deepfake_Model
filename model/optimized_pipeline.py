
import threading
import queue
import time
import numpy as np
import torch
import cv2
import onnxruntime as ort
import sys

# Try importing decord for hardware-accelerated/optimized decoding
try:
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

class CascadedInferenceWrapper:
    """
    Wraps two ONNX models (Fast/Filter and Slow/Accurate) to implement
    cascaded inference logic.
    """
    def __init__(self, fast_model_path, slow_model_path=None, provider='CUDAExecutionProvider'):
        self.fast_sess = self._create_session(fast_model_path, provider)
        self.slow_sess = self._create_session(slow_model_path, provider) if slow_model_path else None
        
        self.fast_input_name = self.fast_sess.get_inputs()[0].name
        self.fast_output_name = self.fast_sess.get_outputs()[0].name
        
        if self.slow_sess:
            self.slow_input_name = self.slow_sess.get_inputs()[0].name
            self.slow_output_name = self.slow_sess.get_outputs()[0].name

    def _create_session(self, path, provider):
        options = ort.SessionOptions()
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Check providers
        available = ort.get_available_providers()
        providers = []
        if provider in available:
            providers.append(provider)
        providers.append('CPUExecutionProvider')
        
        return ort.InferenceSession(path, sess_options=options, providers=providers)

    def predict_batch(self, batch_tensor):
        """
        Args:
            batch_tensor (torch.Tensor): Pinned CPU tensor or GPU tensor (B, C, H, W)
        """
        # Convert to numpy for ONNX Runtime (unless using IOBinding, keeping it simple/compatible first)
        # Note: For maximum speed with Pinned Memory, we'd use IOBinding. 
        # Here we assume batch_tensor is a pinned CPU tensor.
        
        # 1. Run Fast Model
        batch_np = batch_tensor.numpy() # No copy if already on CPU (Pinned)
        
        fast_logits = self.fast_sess.run([self.fast_output_name], {self.fast_input_name: batch_np})[0]
        # Assume logits -> sigmoid
        fast_probs = 1.0 / (1.0 + np.exp(-fast_logits)) # (B, 1) or (B,)
        if fast_probs.ndim == 2: fast_probs = fast_probs.flatten()
        
        results = np.zeros_like(fast_probs)
        final_confs = np.zeros_like(fast_probs)
        
        # 2. Filter Logic
        # > 90% confidence means: prob > 0.9 (Fake) OR prob < 0.1 (Real, which is 0.9 confidence in Real)
        # We process 'uncertain' frames with the Slow Model
        
        uncertain_indices = []
        uncertain_inputs = []
        
        for i, prob in enumerate(fast_probs):
            contrast = abs(prob - 0.5) * 2 # 0.0 to 1.0 confidence magnitude
            
            # If confidence > 90% (i.e. contrast > 0.8), accept fast model
            if contrast > 0.8 or self.slow_sess is None:
                results[i] = prob
                final_confs[i] = contrast
            else:
                uncertain_indices.append(i)
                uncertain_inputs.append(batch_np[i])
        
        # 3. Run Slow Model on Uncertain Frames
        if uncertain_indices:
            uncertain_batch = np.stack(uncertain_inputs)
            slow_logits = self.slow_sess.run([self.slow_output_name], {self.slow_input_name: uncertain_batch})[0]
            slow_probs = 1.0 / (1.0 + np.exp(-slow_logits))
            if slow_probs.ndim == 2: slow_probs = slow_probs.flatten()
            
            for k, original_idx in enumerate(uncertain_indices):
                results[original_idx] = slow_probs[k]
                final_confs[original_idx] = abs(slow_probs[k] - 0.5) * 2
                
        return results, final_confs


class OptimizedVideoPipeline:
    def __init__(self, 
                 fast_model_path, 
                 slow_model_path, 
                 transform,
                 batch_size=8, 
                 frame_stride=6, # 30fps / 6 = 5fps
                 device='cuda',
                 queue_size=128):
        """
        Production-ready Pipeline for Video Inference.
        
        Args:
            fast_model_path (str): Path to EfficientNetV2 ONNX
            slow_model_path (str): Path to Swin/Ensemble ONNX
            transform: Albumentations/Torchvision transform
            batch_size (int): Inference batch size
            frame_stride (int): Process every Nth frame (Sparse Sampling)
            device (str): 'cuda' or 'cpu'
        """
        self.device = device
        self.batch_size = batch_size
        self.frame_stride = frame_stride
        self.transform = transform
        
        # Initialize Cascaded Model
        self.model_wrapper = CascadedInferenceWrapper(
            fast_model_path, 
            slow_model_path, 
            provider='TensorrtExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'
        )
        
        # Pinned Memory Pool (Reusable buffers could be added here for further opt)
        
        # Queues
        # raw_queue: (frame_idx, numpy_frame_rgb)
        self.raw_queue = queue.Queue(maxsize=queue_size)
        
        # batch_queue: (indices, pinned_tensor_batch, raw_frames_for_viz)
        self.batch_queue = queue.Queue(maxsize=queue_size // batch_size + 2)
        
        # result_queue
        self.result_queue = queue.Queue()
        
        self.stop_event = threading.Event()
        self.video_info = {}

    def _producer_decord(self, video_path):
        """Decodes video and puts raw frames into raw_queue using Decord."""
        try:
            # CPU context for decoding is often sufficient and prevents GPU context fighting
            # But if we want pinned memory, we originate in CPU RAM anyway.
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            self.video_info['fps'] = fps
            self.video_info['total_frames'] = total_frames
            self.video_info['duration'] = total_frames / fps
            
            # Sparse Sampling Logic
            # We want approx 5 FPS? Or just use stride?
            # User requirement: "Sample only N frames per second (e.g. 5 FPS)"
            # dynamic stride calculation:
            target_fps = 5
            stride = max(1, int(fps / target_fps))
            
            # Or use fixed stride from init if preferred. Let's use the dynamic calculation to respect '5 FPS' requirement
            # strictly. Or fallback to self.frame_stride if fps is weird.
            if fps > 0:
                final_stride = max(1, int(fps / 5)) 
            else:
                final_stride = self.frame_stride
            
            print(f"🎞️  Sampling Stride: {final_stride} (~5 FPS)")
                
            indices = list(range(0, total_frames, final_stride))
            
            # Batch Read
            chunk_size = 32
            for i in range(0, len(indices), chunk_size):
                if self.stop_event.is_set(): break
                
                batch_indices = indices[i : i + chunk_size]
                # Decord returns RGB (H, W, C)
                frames = vr.get_batch(batch_indices).asnumpy()
                
                for j, frame in enumerate(frames):
                    idx = batch_indices[j]
                    self.raw_queue.put((idx, frame))
                    
        except Exception as e:
            print(f"❌ Producer Error: {e}")
        finally:
            self.raw_queue.put(None) # Sentinel

    def _producer_opencv(self, video_path):
        """Fallback if decord is missing."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_info = {'fps': fps, 'total_frames': total_frames, 'duration': total_frames/fps if fps else 0}
        
        target_fps = 5
        stride = max(1, int(fps / target_fps)) if fps > 0 else self.frame_stride
        
        frame_idx = 0
        while cap.isOpened():
            if self.stop_event.is_set(): break
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % stride == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.raw_queue.put((frame_idx, frame_rgb))
            
            frame_idx += 1
        cap.release()
        self.raw_queue.put(None)

    def _batch_worker(self):
        """
        Consumes raw frames, applies transform, pins memory, packs batches.
        """
        current_batch = []
        current_indices = []
        current_raw = []
        
        while True:
            item = self.raw_queue.get()
            if item is None:
                # Flush remaining
                if current_batch:
                    self._dispatch_batch(current_batch, current_indices, current_raw)
                self.batch_queue.put(None) # Sentinel
                break
                
            idx, frame_rgb = item
            
            # Transform
            # User albumentations transform expects 'image'
            try:
                augmented = self.transform(image=frame_rgb)
                # Ensure tensor is float32
                tensor = augmented['image'] # Should be torch tensor (C, H, W)
                
                # If transform returns numpy, convert to torch
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                
                current_batch.append(tensor)
                current_indices.append(idx)
                current_raw.append(frame_rgb)
                
                if len(current_batch) == self.batch_size:
                    self._dispatch_batch(current_batch, current_indices, current_raw)
                    current_batch = []
                    current_indices = []
                    current_raw = []
                    
            except Exception as e:
                print(f"Preprocessing Error: {e}")

    def _dispatch_batch(self, batch_tensors, indices, raw_frames):
        """Stack and Pin Memory"""
        # Stack: (B, C, H, W)
        batch_stack = torch.stack(batch_tensors)
        
        # MEMORY PINNING: Critical Step
        # This locks the page in RAM, allowing faster DMA transfer to GPU
        if torch.cuda.is_available():
            batch_stack = batch_stack.pin_memory()
            
        self.batch_queue.put({
            'data': batch_stack,
            'indices': indices,
            'raw': raw_frames
        })

    def run(self, video_path):
        """Starts the pipeline"""
        
        # 1. Start Producer
        if DECORD_AVAILABLE:
            t_prod = threading.Thread(target=self._producer_decord, args=(video_path,))
        else:
            t_prod = threading.Thread(target=self._producer_opencv, args=(video_path,))
            
        t_prod.start()
        
        # 2. Start Batch/Pin Worker
        t_batch = threading.Thread(target=self._batch_worker)
        t_batch.start()
        
        # 3. Consumer Inference Loop (Main Thread)
        results_agg = {'probs': [], 'indices': [], 'confidences': []}
        
        while True:
            batch_item = self.batch_queue.get()
            if batch_item is None:
                break
                
            # Input is Pinned CPU Tensor
            # For ONNX Runtime with CUDA, passing this Pinned Tensor (converted to numpy) 
            # is efficient if ORT supports it, or we move to device if using torch models.
            # Our Wrapper expects the tensor.
            
            probs, confs = self.model_wrapper.predict_batch(batch_item['data'])
            
            # Store
            for k in range(len(probs)):
                results_agg['probs'].append(float(probs[k]))
                results_agg['indices'].append(batch_item['indices'][k])
                results_agg['confidences'].append(float(confs[k]))
                
        t_prod.join()
        t_batch.join()
        
        return self._finalize_results(results_agg)

    def _finalize_results(self, agg):
        """Simple aggregation stats"""
        if not agg['probs']:
            return {"verdict": "ERROR", "confidence": 0.0}
            
        probs = agg['probs']
        max_prob = max(probs)
        avg_prob = sum(probs) / len(probs)
        fake_ratio = len([p for p in probs if p > 0.6]) / len(probs)
        
        is_fake = (avg_prob > 0.65) or (fake_ratio > 0.15 and max_prob > 0.7)
        
        return {
            "verdict": "FAKE" if is_fake else "REAL",
            "confidence": avg_prob if is_fake else 1 - avg_prob,
            "details": agg
        }

