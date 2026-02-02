import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from src.utils import get_fft_feature

class RGBBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # EfficientNet V2 Small: Robust and efficient spatial features
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.net = models.efficientnet_v2_s(weights=weights)
        # Extract features before classification head
        self.features = self.net.features
        self.avgpool = self.net.avgpool
        self.out_dim = 1280 

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class FreqBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN to analyze frequency domain patterns
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.out_dim = 128
    
    def forward(self, x):
        return torch.flatten(self.net(x), 1)

class PatchBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Analyzes local patches for inconsistencies
        # Shared lightweight CNN for each patch
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.out_dim = 64

    def forward(self, x):
        # x: (B, 3, 256, 256)
        # Create 4x4=16 patches of size 64x64
        # Unfold logic: kernel_size=64, stride=64
        patches = x.unfold(2, 64, 64).unfold(3, 64, 64)
        # patches shape: (B, 3, 4, 4, 64, 64)
        B, C, H_grid, W_grid, H_patch, W_patch = patches.shape
        
        # Merge batch and grid dimensions for parallel processing
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B * H_grid * W_grid, C, H_patch, W_patch)
        
        # Encode
        feats = self.patch_encoder(patches) # (B*16, 64, 1, 1)
        feats = torch.flatten(feats, 1) # (B*16, 64)
        
        # Aggregate back to B
        feats = feats.view(B, H_grid * W_grid, -1) # (B, 16, 64)
        
        # Max pool over patches to capture the "most fake" patch signal
        feats_max, _ = torch.max(feats, dim=1) # (B, 64)
        
        return feats_max

class ViTBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Swin Transformer Tiny: Capture long-range dependencies
        weights = models.Swin_V2_T_Weights.DEFAULT if pretrained else None
        self.net = models.swin_v2_t(weights=weights)
        
        # Replace head with Identity to get features
        self.out_dim = self.net.head.in_features
        self.net.head = nn.Identity()
    
    def forward(self, x):
        return self.net(x)

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.rgb_branch = RGBBranch(pretrained)
        self.freq_branch = FreqBranch()
        self.patch_branch = PatchBranch()
        self.vit_branch = ViTBranch(pretrained)
        
        input_dim = (self.rgb_branch.out_dim + 
                     self.freq_branch.out_dim + 
                     self.patch_branch.out_dim + 
                     self.vit_branch.out_dim)
        
        # Confidence-based fusion head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        # 1. Spatial Analysis
        rgb_feat = self.rgb_branch(x)
        
        # 2. Frequency Analysis
        freq_img = get_fft_feature(x)
        freq_feat = self.freq_branch(freq_img)
        
        # 3. Patch Analysis (Local Inconsistencies)
        patch_feat = self.patch_branch(x)
        
        # 4. Global Consistency (ViT)
        vit_feat = self.vit_branch(x)
        
        # 5. Feature Fusion
        combined = torch.cat([rgb_feat, freq_feat, patch_feat, vit_feat], dim=1)
        
        return self.classifier(combined)

    def get_heatmap(self, x):
        """Generate Grad-CAM heatmap for the input image"""
        # We'll use the RGB branch for visualization as it contains spatial features
        # Enable gradients for the input if needed, though typically we hook into layers
        
        # 1. Forward pass through RGB branch
        # We need to register a hook on the last conv layer of the efficientnet features
        # Target layer: self.rgb_branch.features[-1] (the last block)
        
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
            
        def forward_hook(module, input, output):
            activations.append(output)
            
        # Register hooks on the last convolutional layer of RGB branch
        target_layer = self.rgb_branch.features[-1]
        hook_b = target_layer.register_full_backward_hook(backward_hook)
        hook_f = target_layer.register_forward_hook(forward_hook)
        
        # Forward pass
        logits = self(x)
        pred_idx = 0 # Binary classification, output is scalar logic
        
        # Backward pass
        self.zero_grad()
        logits.backward(retain_graph=True)
        
        # Get gradients and activations
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
        activation = activations[0][0]
        
        # Weight activations by gradients (Grad-CAM)
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        
        # Normalize
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        # Remove hooks
        hook_b.remove()
        hook_f.remove()
        
        return heatmap
