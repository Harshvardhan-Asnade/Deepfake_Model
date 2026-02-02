import torch
import numpy as np
import cv2

def get_fft_feature(x):
    """
    Computes the Log-Magnitude Spectrum of the input images.
    Args:
        x (torch.Tensor): Input images of shape (B, C, H, W)
    Returns:
        torch.Tensor: Log-magnitude spectrum of shape (B, C, H, W)
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
        
    # Compute 2D FFT
    fft = torch.fft.fft2(x, norm='ortho')
    
    # Compute magnitude
    mag = torch.abs(fft)
    
    # Apply log scale (add epsilon for stability)
    mag = torch.log(mag + 1e-6)
    
    # Shift zero-frequency component to the center of the spectrum
    mag = torch.fft.fftshift(mag, dim=(-2, -1))
    
    return mag

def min_max_normalize(tensor):
    """
    Min-max normalization for visualization or stable training provided tensor.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)

