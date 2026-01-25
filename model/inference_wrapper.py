import onnxruntime as ort
import numpy as np
import os
import sys

class DeepGuardONNX:
    def __init__(self, model_path, use_quantized=False):
        """
        Initialize the DeepGuard ONNX Inference Session.
        
        Args:
            model_path (str): Path to the .onnx file.
            use_quantized (bool): If True, tries to load the _int8 version if available.
        """
        self.providers = []
        
        # Hardware Detection & Provider Selection
        available_providers = ort.get_available_providers()
        print(f"🖥️  Available ONNX Providers: {available_providers}")
        
        # Priority List
        # 1. NVIDIA TensorRT (Fastest for NVIDIA)
        if 'TensorrtExecutionProvider' in available_providers:
            self.providers.append(('TensorrtExecutionProvider', {
                'trt_fp16_enable': True,       # Enable FP16
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_cache'
            }))
            self.providers.append(('CUDAExecutionProvider', {
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))
            print("🚀 Specific Mode: NVIDIA TensorRT/CUDA Selected")
            
        # 2. Apple Silicon CoreML (Fastest for Mac)
        elif 'CoreMLExecutionProvider' in available_providers:
            self.providers.append('CoreMLExecutionProvider')
            print("🍎 Specific Mode: Apple CoreML Selected")
            
        # 3. Intel OpenVINO (Fastest for Intel CPUs/iGPUs)
        elif 'OpenVINOExecutionProvider' in available_providers:
            self.providers.append('OpenVINOExecutionProvider')
            print("🔵 Specific Mode: Intel OpenVINO Selected")
            
        # 4. Fallback CPU
        else:
            self.providers.append('CPUExecutionProvider')
            print("🐢 Specific Mode: CPU Fallback")

        # Load appropriate model file
        target_path = model_path
        if use_quantized:
            quantized_path = model_path.replace(".onnx", "_int8.onnx")
            if os.path.exists(quantized_path):
                print(f"📉 Loading Quantized Model: {quantized_path}")
                target_path = quantized_path
            else:
                print(f"ℹ️  Quantized model not found at {quantized_path}, falling back to FP32.")

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Model not found at: {target_path}")

        # Create Session
        try:
            self.session = ort.InferenceSession(target_path, providers=self.providers)
        except Exception as e:
            print(f"❌ Error creating InferenceSession with preferred providers: {e}")
            print("⚠️  Falling back to CPUExecutionProvider...")
            self.session = ort.InferenceSession(target_path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_batch):
        """
        Run inference on a batch of images.
        
        Args:
            image_batch (np.ndarray): Preprocessed batch of images (B, 3, H, W). 
                                      Must be float32 in range [0, 1] usually, or normalized.
        
        Returns:
            np.ndarray: Logits or probabilities depending on model output.
        """
        # Ensure input is float32
        if image_batch.dtype != np.float32:
            image_batch = image_batch.astype(np.float32)
            
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: image_batch}
        )
        return outputs[0]
