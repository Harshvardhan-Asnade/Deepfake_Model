import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir=None, file_paths=None, labels=None, phase='train', max_samples=None):
        """
        Args:
            root_dir (str): Directory with subfolders containing images. (Optional if file_paths provided)
            file_paths (list): List of absolute paths to images.
            labels (list): List of labels corresponding to file_paths.
            phase (str): 'train' or 'val'.
            max_samples (int): Optional limit for quick debugging.
        """
        self.phase = phase
        
        if file_paths is not None and labels is not None:
            self.image_paths = file_paths
            self.labels = labels
        elif root_dir is not None:
            self.image_paths, self.labels = self.scan_directory(root_dir)
        else:
            raise ValueError("Either root_dir or (file_paths, labels) must be provided.")
            
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.labels = self.labels[:max_samples]
            
        self.transform = self._get_transforms()
        
        print(f"Initialized {self.phase} dataset with {len(self.image_paths)} samples.")

    @staticmethod
    def scan_directory(root_dir):
        image_paths = []
        labels = []
        print(f"Scanning dataset at {root_dir}...")
        
        # Valid extensions
        exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif')
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(exts):
                    path = os.path.join(root, file)
                    # Label inference based on full path
                    path_lower = path.lower()
                    
                    label = None
                    # Prioritize explicit folder names
                    if "real" in path_lower:
                        label = 0.0
                    elif any(x in path_lower for x in ["fake", "df", "synthesis", "generated", "ai"]):
                        label = 1.0
                    
                    if label is not None:
                        image_paths.append(path)
                        labels.append(label)
        
        return image_paths, labels

    def _get_transforms(self):
        size = Config.IMAGE_SIZE
        if self.phase == 'train':
            return A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                # A.GaussianBlur(p=0.1),
                # Fixed for newer albumentations versions
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(size, size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("Image not found or corrupt")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # print(f"Error loading {path}: {e}")
            # Fallback to next image
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.float32)
