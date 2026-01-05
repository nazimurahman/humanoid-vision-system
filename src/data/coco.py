# src/data/coco.py

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import logging
from .dataset import BaseVisionDataset

logger = logging.getLogger(__name__)

class COCODataset(BaseVisionDataset):
    """
    COCO dataset adapter with full mHC compatibility.
    
    Supports:
    - COCO 2017/2014 datasets
    - Detection, segmentation, captioning
    - Custom splits and filtering
    """
    
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        split: str = 'train2017',
        transforms: Optional[callable] = None,
        task: str = 'detection',
        streaming: bool = False,
        cache_size: int = 1000,
        filter_empty: bool = True,
        max_samples: Optional[int] = None,
        class_filter: Optional[List[int]] = None
    ):
        """
        Initialize COCO dataset.
        
        Args:
            root_dir: Root directory containing images
            annotation_file: Path to COCO annotation file
            split: Dataset split (train2017, val2017, etc.)
            transforms: Transformations to apply
            task: 'detection' or 'segmentation'
            streaming: Enable streaming mode
            cache_size: Cache size for streaming
            filter_empty: Filter images without annotations
            max_samples: Maximum number of samples to load
            class_filter: List of class IDs to include
        """
        self.split = split
        self.filter_empty = filter_empty
        self.max_samples = max_samples
        self.class_filter = class_filter
        
        # Initialize COCO API
        self.coco = COCO(annotation_file)
        
        # Get image IDs
        self.image_ids = self._get_image_ids()
        
        # Get category mapping
        self.cat_ids = self.coco.getCatIds()
        self.categories = self.coco.loadCats(self.cat_ids)
        
        # Create ID to index mapping
        self.cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(self.categories)}
        
        # Initialize parent
        super().__init__(
            root_dir=root_dir,
            annotation_file=annotation_file,
            transforms=transforms,
            task=task,
            streaming=streaming,
            cache_size=cache_size
        )
        
    def _get_image_ids(self) -> List[int]:
        """Get filtered image IDs."""
        # Get all image IDs
        img_ids = self.coco.getImgIds()
        
        # Filter by classes if specified
        if self.class_filter:
            cat_ids = [cat_id for cat_id in self.class_filter if cat_id in self.cat_ids]
            if cat_ids:
                img_ids = self.coco.getImgIds(catIds=cat_ids)
        
        # Filter empty images if requested
        if self.filter_empty:
            filtered_ids = []
            for img_id in img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                if ann_ids:
                    filtered_ids.append(img_id)
            img_ids = filtered_ids
        
        # Limit number of samples if specified
        if self.max_samples:
            img_ids = img_ids[:self.max_samples]
        
        return img_ids
    
    def _collect_image_paths(self) -> List[Path]:
        """Collect image paths from COCO IDs."""
        image_paths = []
        
        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            
            # Look for image in possible locations
            possible_paths = [
                self.root_dir / self.split / file_name,
                self.root_dir / 'images' / self.split / file_name,
                self.root_dir / file_name,
            ]
            
            for path in possible_paths:
                if path.exists():
                    image_paths.append(path)
                    break
            else:
                logger.warning(f"Image not found: {file_name}")
        
        return image_paths
    
    def _get_image_annotations(self, image_id: int) -> List[Dict]:
        """Get annotations for specific image from COCO."""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Filter by class if specified
        if self.class_filter:
            annotations = [ann for ann in annotations if ann['category_id'] in self.class_filter]
        
        return annotations
    
    def _standardize_annotations(self, raw_annotations: Dict) -> Dict:
        """Convert COCO annotations to standard format."""
        # COCO annotations are already in standard format
        return {
            'images': [
                {
                    'id': img_info['id'],
                    'file_name': img_info['file_name'],
                    'width': img_info['width'],
                    'height': img_info['height'],
                    'annotations': self.coco.loadAnns(
                        self.coco.getAnnIds(imgIds=img_info['id'])
                    )
                }
                for img_info in self.coco.loadImgs(self.image_ids)
            ],
            'categories': self.categories
        }
    
    def get_class_names(self) -> List[str]:
        """Get COCO class names."""
        return [cat['name'] for cat in self.categories]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in dataset."""
        distribution = {}
        
        for cat in self.categories:
            cat_id = cat['id']
            img_ids = self.coco.getImgIds(catIds=[cat_id])
            distribution[cat['name']] = len(img_ids)
        
        return distribution
    
    def visualize_coco_sample(
        self,
        idx: int,
        show_segmentation: bool = False,
        save_path: Optional[str] = None
    ):
        """Visualize COCO sample with annotations."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from pycocotools import mask as maskUtils
        
        sample = self[idx]
        image = sample['image'].permute(1, 2, 0).numpy()
        target = sample['target']
        
        # Get image info
        img_id = sample['image_id'].item()
        img_info = self.coco.loadImgs(img_id)[0]
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        if 'boxes' in target:
            boxes = target['boxes'].numpy()
            labels = target['labels'].numpy()
            
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
                cat_info = self.categories[label]
                ax.text(
                    x, y - 5,
                    cat_info['name'],
                    color='white',
                    fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.8, edgecolor='none')
                )
        
        if show_segmentation:
            # Show segmentation masks
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(ann_ids)
            
            for ann in annotations:
                if 'segmentation' in ann:
                    # Decode segmentation
                    if isinstance(ann['segmentation'], list):
                        # Polygon format
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape(-1, 2)
                            poly = patches.Polygon(
                                poly,
                                fill=True,
                                alpha=0.3,
                                color='green'
                            )
                            ax.add_patch(poly)
                    else:
                        # RLE format
                        mask = maskUtils.decode(ann['segmentation'])
                        mask = np.ma.masked_where(mask == 0, mask)
                        ax.imshow(mask, alpha=0.3, cmap='tab20c')
        
        ax.axis('off')
        plt.title(f"Image ID: {img_id} | {img_info['file_name']}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class COCODataModule:
    """
    Complete COCO data module for PyTorch Lightning compatibility.
    
    Handles:
    - Train/val/test splits
    - Distributed training
    - Automatic download and setup
    """
    
    def __init__(
        self,
        data_dir: str = './data/coco',
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (416, 416),
        task: str = 'detection',
        download: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.task = task
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset if requested
        if download:
            self._download_coco()
        
        # Setup paths
        self.train_dir = self.data_dir / 'train2017'
        self.val_dir = self.data_dir / 'val2017'
        self.train_ann = self.data_dir / 'annotations' / 'instances_train2017.json'
        self.val_ann = self.data_dir / 'annotations' / 'instances_val2017.json'
        
    def _download_coco(self):
        """Download COCO dataset if not present."""
        # This is a placeholder - in practice, you'd download from official sources
        logger.info("COCO dataset download placeholder")
        # Actual implementation would download from:
        # http://images.cocodataset.org/zips/train2017.zip
        # http://images.cocodataset.org/zips/val2017.zip
        # http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation."""
        from .transforms import MHCTransformComposer
        
        # Training transforms
        train_transforms = MHCTransformComposer(
            image_size=self.image_size,
            is_training=True,
            use_mosaic=True,
            use_mixup=True
        )
        
        # Validation transforms
        val_transforms = MHCTransformComposer(
            image_size=self.image_size,
            is_training=False
        )
        
        # Create datasets
        self.train_dataset = COCODataset(
            root_dir=self.data_dir,
            annotation_file=self.train_ann,
            split='train2017',
            transforms=train_transforms,
            task=self.task,
            filter_empty=True
        )
        
        self.val_dataset = COCODataset(
            root_dir=self.data_dir,
            annotation_file=self.val_ann,
            split='val2017',
            transforms=val_transforms,
            task=self.task,
            filter_empty=False
        )
        
        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Val dataset: {len(self.val_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        from .dataloader import MHCDataLoader
        
        return MHCDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        from .dataloader import MHCDataLoader
        
        return MHCDataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )
    
    def get_class_names(self) -> List[str]:
        """Get COCO class names."""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.get_class_names()
        else:
            # Default COCO classes
            return [
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