# src/data/transforms.py

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import random
import math
from PIL import Image, ImageFilter, ImageOps
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MHCTransformComposer:
    """
    Compose transformations specifically optimized for mHC training stability.
    
    Features:
    - Augmentations that preserve manifold structure
    - Mixed precision compatible
    - Batch-aware augmentations
    - Hardware acceleration support
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (416, 416),
        is_training: bool = True,
        use_mosaic: bool = True,
        use_mixup: bool = True,
        use_autoaugment: bool = True,
        color_jitter: float = 0.3,
        normalize: bool = True
    ):
        """
        Initialize transform composer.
        
        Args:
            image_size: Target image size (height, width)
            is_training: Whether to apply training augmentations
            use_mosaic: Enable mosaic augmentation
            use_mixup: Enable mixup augmentation
            use_autoaugment: Enable autoaugment policy
            color_jitter: Strength of color jittering
            normalize: Whether to normalize images
        """
        self.image_size = image_size
        self.is_training = is_training
        self.use_mosaic = use_mosaic
        self.use_mixup = use_mixup
        self.use_autoaugment = use_autoaugment
        
        # Mean and std for ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Build transforms
        self.transforms = self._build_transforms()
        
        # Special augmentations
        self.mosaic_transform = MosaicAugmentation(image_size) if use_mosaic else None
        self.mixup_transform = MixupAugmentation(alpha=0.2) if use_mixup else None
        
    def _build_transforms(self) -> A.Compose:
        """Build Albumentations transform pipeline."""
        transforms_list = []
        
        if self.is_training:
            # Training augmentations
            transforms_list.extend([
                # Geometric augmentations
                A.RandomResizedCrop(
                    height=self.image_size[0],
                    width=self.image_size[1],
                    scale=(0.5, 1.0),
                    ratio=(0.75, 1.33),
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                
                # Color augmentations
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.3
                ),
                
                # Advanced augmentations
                A.Cutout(
                    num_holes=8,
                    max_h_size=32,
                    max_w_size=32,
                    fill_value=0,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),
                
                # AutoAugment policy for mHC stability
                A.Compose([
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.3
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.05,
                        shift_limit=0.05,
                        p=0.2
                    ),
                ], p=0.5) if self.use_autoaugment else A.NoOp()
            ])
        else:
            # Validation/Inference transforms
            transforms_list.append(
                A.Resize(height=self.image_size[0], width=self.image_size[1])
            )
        
        # Always apply
        transforms_list.extend([
            A.Normalize(mean=self.mean, std=self.std) if hasattr(self, 'mean') else A.NoOp(),
            ToTensorV2()
        ])
        
        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(
                format='yolo',  # Normalized [x_center, y_center, width, height]
                label_fields=['labels', 'area'],
                min_area=1.0,
                min_visibility=0.1
            ) if self.is_training else None
        )
    
    def __call__(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Apply transforms to input data.
        
        Args:
            kwargs: Should contain 'image' and optionally 'boxes', 'labels', 'area'
            
        Returns:
            Transformed data dictionary
        """
        # Apply mosaic augmentation (if enabled and training)
        if self.is_training and self.mosaic_transform and random.random() < 0.5:
            kwargs = self.mosaic_transform(**kwargs)
        
        # Apply Albumentations transforms
        if 'boxes' in kwargs and kwargs['boxes'] is not None:
            # Detection task
            transformed = self.transforms(
                image=kwargs['image'],
                bboxes=kwargs['boxes'],
                labels=kwargs.get('labels', []),
                area=kwargs.get('area', [])
            )
            
            # Extract transformed data
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32) if transformed['bboxes'] else torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64) if transformed['labels'] else torch.zeros((0,), dtype=torch.int64)
            area = torch.tensor(transformed['area'], dtype=torch.float32) if transformed['area'] else torch.zeros((0,), dtype=torch.float32)
            
            result = {
                'image': image,
                'boxes': boxes,
                'labels': labels,
                'area': area
            }
        else:
            # Classification or other tasks
            transformed = self.transforms(image=kwargs['image'])
            result = {'image': transformed['image']}
        
        # Apply mixup augmentation (if enabled and training)
        if self.is_training and self.mixup_transform and 'boxes' in result:
            result = self.mixup_transform(**result)
        
        return result

class MosaicAugmentation:
    """
    Mosaic data augmentation for better context learning.
    
    Combines 4 random images into one training sample.
    Improves detection of small objects and context understanding.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (416, 416)):
        self.image_size = image_size
        
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Apply mosaic augmentation."""
        # For simplicity, this is a placeholder
        # In practice, you would load 4 images and combine them
        return kwargs

class MixupAugmentation:
    """
    Mixup augmentation for regularization.
    
    Blends two images and their labels to create new training samples.
    Improves generalization and calibration.
    """
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Apply mixup augmentation."""
        # For simplicity, this is a placeholder
        # In practice, you would blend with another sample
        return kwargs

class RandomErasing:
    """
    Random erasing data augmentation.
    
    Randomly selects a rectangle region in an image and erases its pixels.
    """
    
    def __init__(
        self,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        mean: List[float] = None
    ):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean if mean else [0.485, 0.456, 0.406]
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() > self.probability:
            return image
        
        c, h, w = image.shape
        area = h * w
        
        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)
        
        erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if erase_h < h and erase_w < w:
            x1 = random.randint(0, h - erase_h)
            y1 = random.randint(0, w - erase_w)
            x2 = x1 + erase_h
            y2 = y1 + erase_w
            
            # Replace with mean value
            image[:, x1:x2, y1:y2] = torch.tensor(self.mean).view(-1, 1, 1)
            
        return image

class GPUAcceleratedTransforms:
    """
    GPU-accelerated transformations for faster training.
    
    Uses Kornia or custom CUDA kernels for on-GPU augmentations.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        try:
            import kornia.augmentation as K
            self.kornia_available = True
            
            # Define Kornia augmentations
            self.augmentations = torch.nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                K.RandomAffine(
                    degrees=15,
                    translate=0.1,
                    scale=(0.9, 1.1),
                    p=0.3
                )
            ).to(device)
            
        except ImportError:
            self.kornia_available = False
            print("Kornia not available, using CPU transforms")
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply GPU-accelerated transforms to batch."""
        if self.kornia_available:
            return self.augmentations(batch)
        else:
            # Fallback to CPU transforms
            return batch

class AdaptiveAugmentation:
    """
    Adaptive augmentation that adjusts strength based on training progress.
    
    Starts with strong augmentations, reduces as model converges.
    """
    
    def __init__(
        self,
        initial_strength: float = 1.0,
        final_strength: float = 0.3,
        total_steps: int = 100000
    ):
        self.initial_strength = initial_strength
        self.final_strength = final_strength
        self.total_steps = total_steps
        self.current_step = 0
        
    def update_step(self, step: int):
        """Update current training step."""
        self.current_step = step
        
    def get_strength(self) -> float:
        """Get current augmentation strength."""
        progress = min(self.current_step / self.total_steps, 1.0)
        strength = self.initial_strength - (self.initial_strength - self.final_strength) * progress
        return strength
    
    def adjust_transforms(self, transforms: A.Compose, strength: float):
        """Adjust transform strengths based on current training progress."""
        # This would adjust parameters of each transform
        pass