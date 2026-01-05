# src/data/dataset.py

import torch
from torch.utils.data import Dataset
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseVisionDataset(Dataset):
    """
    Base dataset class for vision tasks with mHC-compatible data handling.
    
    Features:
    - Multi-task support (detection, classification, segmentation)
    - Lazy loading for memory efficiency
    - Streaming support
    - Automatic cache management
    """
    
    def __init__(
        self,
        root_dir: str,
        annotation_file: Optional[str] = None,
        transforms: Optional[callable] = None,
        task: str = 'detection',
        streaming: bool = False,
        cache_size: int = 1000
    ):
        """
        Initialize base dataset.
        
        Args:
            root_dir: Root directory containing images
            annotation_file: Path to annotation file (JSON/COCO format)
            transforms: Transformations to apply
            task: 'detection', 'classification', or 'segmentation'
            streaming: Enable streaming mode for large datasets
            cache_size: Size of LRU cache for loaded items
        """
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.task = task
        self.streaming = streaming
        
        # Validate task
        valid_tasks = ['detection', 'classification', 'segmentation', 'multi_task']
        if task not in valid_tasks:
            raise ValueError(f"Task must be one of {valid_tasks}, got {task}")
        
        # Load annotations
        self.annotations = self._load_annotations(annotation_file)
        
        # Create image paths
        self.image_paths = self._collect_image_paths()
        
        # Setup cache for streaming
        if streaming:
            from functools import lru_cache
            self._get_item_cached = lru_cache(maxsize=cache_size)(self._get_item_uncached)
        
        logger.info(f"Dataset initialized with {len(self)} samples")
        logger.info(f"Task: {task}, Streaming: {streaming}")
        
    def _load_annotations(self, annotation_file: Optional[str]) -> Dict:
        """
        Load annotations from file.
        
        Supports:
        - COCO format
        - Custom JSON format
        - YOLO format
        """
        if annotation_file is None:
            return {}
        
        annotation_path = Path(annotation_file)
        if not annotation_path.exists():
            logger.warning(f"Annotation file not found: {annotation_file}")
            return {}
        
        with open(annotation_path, 'r') as f:
            if annotation_path.suffix == '.json':
                annotations = json.load(f)
            else:
                raise ValueError(f"Unsupported annotation format: {annotation_path.suffix}")
        
        # Convert to standard format
        return self._standardize_annotations(annotations)
    
    def _standardize_annotations(self, raw_annotations: Dict) -> Dict:
        """
        Convert raw annotations to standard format.
        
        Standard format:
        {
            'images': [
                {
                    'id': int,
                    'file_name': str,
                    'width': int,
                    'height': int,
                    'annotations': [
                        {
                            'id': int,
                            'category_id': int,
                            'bbox': [x, y, width, height],
                            'area': float,
                            'segmentation': list,  # For segmentation
                            'iscrowd': int
                        }
                    ]
                }
            ],
            'categories': [
                {
                    'id': int,
                    'name': str,
                    'supercategory': str
                }
            ]
        }
        """
        # Default implementation - override for custom formats
        return raw_annotations
    
    def _collect_image_paths(self) -> List[Path]:
        """Collect all image paths from root directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(self.root_dir.rglob(f'*{ext}'))
            image_paths.extend(self.root_dir.rglob(f'*{ext.upper()}'))
        
        # Sort for reproducibility
        image_paths.sort()
        
        return image_paths
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample with proper formatting for mHC training.
        
        Returns:
            Dictionary containing:
            - 'image': Image tensor [C, H, W]
            - 'target': Task-specific targets
            - 'image_id': Unique image identifier
            - 'original_size': Original image dimensions
        """
        if self.streaming:
            return self._get_item_cached(idx)
        else:
            return self._get_item_uncached(idx)
    
    def _get_item_uncached(self, idx: int) -> Dict[str, torch.Tensor]:
        """Uncached version of __getitem__."""
        # Get image path
        image_path = self.image_paths[idx]
        
        # Load image
        image = self._load_image(image_path)
        
        # Get annotations for this image
        image_id = self._get_image_id(image_path)
        annotations = self._get_image_annotations(image_id)
        
        # Prepare targets based on task
        target = self._prepare_targets(annotations, image.size)
        
        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(image=image, **target)
            image = transformed['image']
            target = {k: transformed[k] for k in target.keys()}
        
        # Convert to tensor
        image_tensor = self._image_to_tensor(image)
        
        # Prepare final output
        sample = {
            'image': image_tensor,
            'target': target,
            'image_id': torch.tensor([image_id], dtype=torch.int64),
            'original_size': torch.tensor([image.height, image.width], dtype=torch.int32)
        }
        
        return sample
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load image with error handling."""
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='black')
    
    def _get_image_id(self, image_path: Path) -> int:
        """Get unique image ID."""
        # Use hash of path for consistency
        return hash(str(image_path)) % (2**31)
    
    def _get_image_annotations(self, image_id: int) -> List[Dict]:
        """Get annotations for specific image."""
        # Find image in annotations
        for img_info in self.annotations.get('images', []):
            if img_info.get('id') == image_id:
                return img_info.get('annotations', [])
        return []
    
    def _prepare_targets(self, annotations: List[Dict], image_size: Tuple[int, int]) -> Dict:
        """
        Prepare targets for specific task.
        
        Returns format depends on task:
        - Detection: bounding boxes and labels
        - Classification: single label
        - Segmentation: mask
        - Multi-task: combination
        """
        if self.task == 'detection':
            return self._prepare_detection_targets(annotations, image_size)
        elif self.task == 'classification':
            return self._prepare_classification_targets(annotations)
        elif self.task == 'segmentation':
            return self._prepare_segmentation_targets(annotations, image_size)
        elif self.task == 'multi_task':
            return self._prepare_multi_task_targets(annotations, image_size)
        else:
            return {}
    
    def _prepare_detection_targets(
        self,
        annotations: List[Dict],
        image_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Prepare detection targets in YOLO format."""
        if not annotations:
            # Return empty targets
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'area': torch.zeros((0,), dtype=torch.float32)
            }
        
        boxes = []
        labels = []
        areas = []
        
        img_h, img_w = image_size
        
        for ann in annotations:
            # Get bounding box
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                # Convert to normalized coordinates [0, 1]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                width = w / img_w
                height = h / img_h
                
                boxes.append([x_center, y_center, width, height])
                labels.append(ann.get('category_id', 0))
                areas.append(w * h)
        
        if boxes:
            return {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'area': torch.tensor(areas, dtype=torch.float32)
            }
        else:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'area': torch.zeros((0,), dtype=torch.float32)
            }
    
    def _prepare_classification_targets(self, annotations: List[Dict]) -> Dict[str, torch.Tensor]:
        """Prepare classification targets."""
        if annotations:
            # Use first annotation's category
            label = annotations[0].get('category_id', 0)
        else:
            label = 0
        
        return {
            'label': torch.tensor([label], dtype=torch.int64)
        }
    
    def _prepare_segmentation_targets(
        self,
        annotations: List[Dict],
        image_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Prepare segmentation targets."""
        # Create blank mask
        img_h, img_w = image_size
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        for ann in annotations:
            if 'segmentation' in ann:
                # Simplified segmentation - in practice, use proper mask decoding
                # This is a placeholder for COCO segmentation format
                pass
        
        return {
            'mask': torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        }
    
    def _prepare_multi_task_targets(
        self,
        annotations: List[Dict],
        image_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Prepare multi-task targets (detection + classification)."""
        detection_targets = self._prepare_detection_targets(annotations, image_size)
        classification_targets = self._prepare_classification_targets(annotations)
        
        return {**detection_targets, **classification_targets}
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # HWC to CHW
        img_array = img_array.transpose(2, 0, 1)
        
        return torch.from_numpy(img_array)
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for variable-sized targets.
        
        Handles:
        - Variable number of bounding boxes per image
        - Padding for batch processing
        - Stacking of images
        """
        if len(batch) == 0:
            return {}
        
        # Stack images
        images = torch.stack([item['image'] for item in batch])
        image_ids = torch.stack([item['image_id'] for item in batch])
        original_sizes = torch.stack([item['original_size'] for item in batch])
        
        # Handle targets (variable length)
        if 'boxes' in batch[0]['target']:
            # Detection targets
            max_boxes = max(len(item['target']['boxes']) for item in batch)
            
            padded_boxes = []
            padded_labels = []
            padded_areas = []
            box_masks = []
            
            for item in batch:
                boxes = item['target']['boxes']
                labels = item['target']['labels']
                areas = item['target']['area']
                
                num_boxes = len(boxes)
                
                # Pad boxes
                if num_boxes < max_boxes:
                    padding = max_boxes - num_boxes
                    padded_boxes.append(
                        torch.cat([boxes, torch.zeros(padding, 4)], dim=0)
                    )
                    padded_labels.append(
                        torch.cat([labels, -1 * torch.ones(padding, dtype=torch.int64)])
                    )
                    padded_areas.append(
                        torch.cat([areas, torch.zeros(padding)], dim=0)
                    )
                    box_masks.append(
                        torch.cat([
                            torch.ones(num_boxes, dtype=torch.bool),
                            torch.zeros(padding, dtype=torch.bool)
                        ])
                    )
                else:
                    padded_boxes.append(boxes)
                    padded_labels.append(labels)
                    padded_areas.append(areas)
                    box_masks.append(torch.ones(num_boxes, dtype=torch.bool))
            
            targets = {
                'boxes': torch.stack(padded_boxes),
                'labels': torch.stack(padded_labels),
                'area': torch.stack(padded_areas),
                'box_mask': torch.stack(box_masks)
            }
        elif 'label' in batch[0]['target']:
            # Classification targets
            targets = {
                'labels': torch.stack([item['target']['label'] for item in batch])
            }
        else:
            targets = {}
        
        return {
            'images': images,
            'targets': targets,
            'image_ids': image_ids,
            'original_sizes': original_sizes
        }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names from annotations."""
        categories = self.annotations.get('categories', [])
        class_names = [cat.get('name', f'class_{i}') for i, cat in enumerate(categories)]
        
        if not class_names:
            # Default COCO classes if none provided
            class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
        
        return class_names
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """Visualize a sample with annotations."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        sample = self[idx]
        image = sample['image'].permute(1, 2, 0).numpy()
        target = sample['target']
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        if 'boxes' in target:
            boxes = target['boxes'].numpy()
            labels = target['labels'].numpy()
            
            class_names = self.get_class_names()
            
            for box, label in zip(boxes, labels):
                if label < 0:  # Padding
                    continue
                
                # Convert normalized to pixel coordinates
                x_center, y_center, width, height = box
                x = (x_center - width / 2) * image.shape[1]
                y = (y_center - height / 2) * image.shape[0]
                w = width * image.shape[1]
                h = height * image.shape[0]
                
                # Create rectangle patch
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                class_name = class_names[label] if label < len(class_names) else f'Class {label}'
                ax.text(
                    x, y - 5,
                    class_name,
                    color='white',
                    fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.8, edgecolor='none')
                )
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()