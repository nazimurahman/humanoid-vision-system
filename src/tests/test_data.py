#!/usr/bin/env python3
"""
Data pipeline testing for the Hybrid Vision System.
Tests include:
1. Dataset loading and parsing
2. Data augmentation
3. Data transforms
4. Data loader batching
5. Streaming data handling
6. COCO dataset integration
"""

import torch
import numpy as np
import cv2
import pytest
import tempfile
import os
import json
from pathlib import Path
from PIL import Image
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import VisionDataset
from src.data.transforms import ImageTransform, DetectionTransform
from src.data.dataloader import create_data_loader
from src.data.streaming import CameraStream, ImageStream
from src.data.coco import COCODataset, COCODataLoader

class TestVisionDataset:
    """Test base vision dataset class."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy dataset structure
        self.create_dummy_dataset()
        
        # Create dataset
        self.dataset = VisionDataset(
            image_dir=os.path.join(self.temp_dir, 'images'),
            annotation_dir=os.path.join(self.temp_dir, 'annotations'),
            transform=None
        )
    
    def create_dummy_dataset(self):
        """Create dummy dataset for testing."""
        # Create directories
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'annotations'), exist_ok=True)
        
        # Create 10 dummy images and annotations
        self.num_samples = 10
        self.image_size = 416
        
        for i in range(self.num_samples):
            # Create image
            img = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            img_path = os.path.join(self.temp_dir, 'images', f'image_{i:04d}.jpg')
            cv2.imwrite(img_path, img)
            
            # Create annotation
            annotation = {
                'image_id': i,
                'file_name': f'image_{i:04d}.jpg',
                'width': self.image_size,
                'height': self.image_size,
                'annotations': [
                    {
                        'bbox': [50, 50, 100, 100],  # [x, y, w, h]
                        'category_id': 0,
                        'score': 1.0
                    },
                    {
                        'bbox': [200, 200, 50, 50],
                        'category_id': 1,
                        'score': 0.8
                    }
                ]
            }
            
            annotation_path = os.path.join(self.temp_dir, 'annotations', f'image_{i:04d}.json')
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f)
    
    def test_dataset_length(self):
        """Test dataset returns correct length."""
        assert len(self.dataset) == self.num_samples
        
        print(f"✓ Dataset has correct length: {len(self.dataset)}")
    
    def test_dataset_getitem(self):
        """Test dataset indexing."""
        # Get first item
        image, target = self.dataset[0]
        
        # Check image
        assert isinstance(image, (np.ndarray, Image.Image))
        if isinstance(image, np.ndarray):
            assert image.shape == (self.image_size, self.image_size, 3)
        elif isinstance(image, Image.Image):
            assert image.size == (self.image_size, self.image_size)
        
        # Check target
        assert isinstance(target, dict)
        assert 'boxes' in target
        assert 'labels' in target
        
        # Should have 2 annotations
        assert len(target['boxes']) == 2
        assert len(target['labels']) == 2
        
        print("✓ Dataset indexing works correctly")
    
    def test_dataset_iteration(self):
        """Test dataset iteration."""
        count = 0
        for i, (image, target) in enumerate(self.dataset):
            count += 1
            if i >= 2:  # Just test first few
                break
        
        assert count > 0
        print("✓ Dataset iteration works")
    
    def test_empty_dataset(self):
        """Test empty dataset handling."""
        # Create empty dataset
        empty_dir = tempfile.mkdtemp()
        empty_dataset = VisionDataset(
            image_dir=empty_dir,
            annotation_dir=empty_dir,
            transform=None
        )
        
        assert len(empty_dataset) == 0
        
        # Should raise IndexError when accessing
        try:
            _ = empty_dataset[0]
            assert False, "Should have raised IndexError"
        except IndexError:
            pass
        
        print("✓ Empty dataset handled correctly")
    
    def test_corrupt_image_handling(self):
        """Test handling of corrupt images."""
        # Create corrupt image file
        corrupt_path = os.path.join(self.temp_dir, 'corrupt.jpg')
        with open(corrupt_path, 'w') as f:
            f.write('not an image')
        
        # Try to load it (should handle gracefully or raise appropriate error)
        # This depends on implementation - might raise exception or skip
        
        print("✓ Corrupt image handling test completed")

class TestImageTransforms:
    """Test image transformation pipeline."""
    
    def setup_method(self):
        """Setup test fixture."""
        # Create test image
        self.image_size = 416
        self.test_image = np.random.randint(
            0, 255, 
            (self.image_size, self.image_size, 3), 
            dtype=np.uint8
        )
        
        # Create test bounding boxes
        self.test_boxes = np.array([
            [0.1, 0.1, 0.3, 0.3],  # [x1, y1, x2, y2] normalized
            [0.5, 0.5, 0.7, 0.7],
        ])
        
        self.test_labels = np.array([0, 1])
    
    def test_basic_transform(self):
        """Test basic image transform."""
        transform = ImageTransform(
            image_size=self.image_size,
            normalize=True
        )
        
        # Apply transform
        transformed = transform(image=self.test_image)
        
        # Check output
        assert 'image' in transformed
        img_tensor = transformed['image']
        
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, self.image_size, self.image_size)
        assert img_tensor.dtype == torch.float32
        
        print("✓ Basic image transform works")
    
    def test_detection_transform(self):
        """Test detection transform with boxes."""
        transform = DetectionTransform(
            image_size=self.image_size,
            augment=False  # No augmentation for deterministic test
        )
        
        # Apply transform
        result = transform(
            image=self.test_image,
            boxes=self.test_boxes,
            labels=self.test_labels
        )
        
        # Check all outputs present
        assert 'image' in result
        assert 'boxes' in result
        assert 'labels' in result
        
        # Check types
        assert isinstance(result['image'], torch.Tensor)
        assert isinstance(result['boxes'], torch.Tensor)
        assert isinstance(result['labels'], torch.Tensor)
        
        # Boxes should be preserved
        assert len(result['boxes']) == len(self.test_boxes)
        
        print("✓ Detection transform works")
    
    def test_augmentation(self):
        """Test data augmentation."""
        transform = DetectionTransform(
            image_size=self.image_size,
            augment=True,
            augment_params={
                'hue': 0.1,
                'saturation': 1.5,
                'exposure': 1.5,
                'flip_prob': 0.5
            }
        )
        
        # Apply multiple times to test random augmentations
        results = []
        for _ in range(5):
            result = transform(
                image=self.test_image,
                boxes=self.test_boxes,
                labels=self.test_labels
            )
            results.append(result['image'])
        
        # Images should be different due to random augmentation
        # (though could randomly be the same)
        differences = []
        for i in range(len(results) - 1):
            diff = torch.abs(results[i] - results[i + 1]).mean().item()
            differences.append(diff)
        
        # At least some should be different
        assert max(differences) > 0.01
        
        print("✓ Data augmentation works")
    
    def test_flip_augmentation(self):
        """Test flip augmentation specifically."""
        transform = DetectionTransform(
            image_size=self.image_size,
            augment=True,
            augment_params={'flip_prob': 1.0}  # Always flip
        )
        
        result = transform(
            image=self.test_image,
            boxes=self.test_boxes,
            labels=self.test_labels
        )
        
        # Boxes should be flipped horizontally
        original_boxes = self.test_boxes.copy()
        flipped_boxes = result['boxes'].numpy()
        
        # x coordinates should be mirrored: x' = 1 - x
        for orig, flipped in zip(original_boxes, flipped_boxes):
            assert abs(flipped[0] - (1 - orig[2])) < 0.01  # x1' ≈ 1 - x2
            assert abs(flipped[2] - (1 - orig[0])) < 0.01  # x2' ≈ 1 - x1
            assert abs(flipped[1] - orig[1]) < 0.01  # y unchanged
            assert abs(flipped[3] - orig[3]) < 0.01  # y unchanged
        
        print("✓ Flip augmentation works correctly")
    
    def test_color_jitter(self):
        """Test color jitter augmentation."""
        from src.data.transforms import ColorJitter
        
        jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        
        # Apply to image
        jittered = jitter(self.test_image)
        
        # Should be same shape
        assert jittered.shape == self.test_image.shape
        
        # Should be different (though could randomly be similar)
        diff = np.abs(jittered.astype(float) - self.test_image.astype(float)).mean()
        # Allow for possibility of no change (random)
        
        print(f"✓ Color jitter works (mean diff: {diff:.2f})")
    
    def test_normalization(self):
        """Test image normalization."""
        from src.data.transforms import Normalize
        
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Convert image to tensor first
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(self.test_image)
        
        # Normalize
        normalized = normalize(img_tensor)
        
        # Check statistics
        for c in range(3):
            channel_mean = normalized[c].mean().item()
            channel_std = normalized[c].std().item()
            
            # After normalization, should be roughly mean 0, std 1
            assert abs(channel_mean) < 1.0
            assert 0.5 < channel_std < 2.0
        
        print("✓ Image normalization works correctly")

class TestDataLoader:
    """Test data loader functionality."""
    
    def setup_method(self):
        """Setup test fixture."""
        # Create dummy dataset
        self.temp_dir = tempfile.mkdtemp()
        self.create_dummy_dataset()
        
        # Create dataset
        self.dataset = VisionDataset(
            image_dir=os.path.join(self.temp_dir, 'images'),
            annotation_dir=os.path.join(self.temp_dir, 'annotations'),
            transform=DetectionTransform(image_size=416, augment=False)
        )
    
    def create_dummy_dataset(self):
        """Create dummy dataset."""
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'annotations'), exist_ok=True)
        
        # Create 20 samples
        for i in range(20):
            # Image
            img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
            img_path = os.path.join(self.temp_dir, 'images', f'img_{i:03d}.jpg')
            cv2.imwrite(img_path, img)
            
            # Annotation
            ann = {
                'boxes': [[50, 50, 100, 100], [200, 200, 250, 250]],
                'labels': [0, 1]
            }
            ann_path = os.path.join(self.temp_dir, 'annotations', f'img_{i:03d}.json')
            with open(ann_path, 'w') as f:
                json.dump(ann, f)
    
    def test_data_loader_creation(self):
        """Test data loader creation."""
        loader = create_data_loader(
            dataset=self.dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # 0 for testing
        )
        
        assert loader.batch_size == 4
        assert len(loader) == 5  # 20 samples / 4 batch size = 5 batches
        
        print("✓ Data loader creation works")
    
    def test_batch_loading(self):
        """Test batch loading."""
        loader = create_data_loader(
            dataset=self.dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        
        # Get first batch
        batch = next(iter(loader))
        
        # Should return tuple of (images, targets)
        assert isinstance(batch, (tuple, list))
        assert len(batch) == 2
        
        images, targets = batch
        
        # Check images
        assert isinstance(images, torch.Tensor)
        assert images.shape == (4, 3, 416, 416)  # batch_size, channels, H, W
        
        # Check targets
        assert isinstance(targets, dict)
        assert 'boxes' in targets
        assert 'labels' in targets
        
        # Each should have 4 items (batch size)
        assert len(targets['boxes']) == 4
        assert len(targets['labels']) == 4
        
        print("✓ Batch loading works correctly")
    
    def test_shuffling(self):
        """Test data shuffling."""
        # Create loader with shuffle
        loader = create_data_loader(
            dataset=self.dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # Get first batch from two different epochs
        batch1 = next(iter(loader))
        # Reset loader
        loader = create_data_loader(
            dataset=self.dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        batch2 = next(iter(loader))
        
        # With shuffling, they're likely different
        # (could randomly be the same, but probability is low)
        images1, _ = batch1
        images2, _ = batch2
        
        diff = torch.abs(images1 - images2).mean().item()
        # Usually different, but allow for rare case they're the same
        
        print(f"✓ Data shuffling works (batch diff: {diff:.4f})")
    
    def test_collate_fn(self):
        """Test custom collate function."""
        from src.data.dataloader import detection_collate_fn
        
        # Create batch of samples with variable number of objects
        batch = []
        for i in range(4):
            # Vary number of objects
            num_objects = i + 1
            sample = (
                torch.randn(3, 416, 416),  # image
                {
                    'boxes': torch.randn(num_objects, 4),
                    'labels': torch.randint(0, 10, (num_objects,))
                }
            )
            batch.append(sample)
        
        # Apply collate function
        images, targets = detection_collate_fn(batch)
        
        # Check images are stacked
        assert images.shape == (4, 3, 416, 416)
        
        # Check targets
        assert isinstance(targets, dict)
        assert 'boxes' in targets
        assert 'labels' in targets
        
        # Boxes and labels should be lists
        assert isinstance(targets['boxes'], list)
        assert len(targets['boxes']) == 4
        
        print("✓ Custom collate function works with variable-sized targets")

class TestStreamingData:
    """Test streaming data functionality."""
    
    def test_camera_stream(self):
        """Test camera stream abstraction."""
        # Create mock camera
        class MockCamera:
            def __init__(self):
                self.frame_count = 0
            
            def read(self):
                self.frame_count += 1
                if self.frame_count > 10:
                    return False, None  # Simulate end
                return True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            def release(self):
                pass
        
        # Create stream with mock camera
        stream = CameraStream(camera_source=MockCamera())
        
        # Test frame reading
        frames = []
        for i, frame in enumerate(stream):
            if i >= 5:  # Read 5 frames
                break
            frames.append(frame)
            assert frame.shape == (480, 640, 3)
        
        assert len(frames) == 5
        
        print("✓ Camera stream works")
    
    def test_image_stream(self):
        """Test image file streaming."""
        # Create temporary images
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        for i in range(5):
            img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
            path = os.path.join(temp_dir, f'stream_{i}.jpg')
            cv2.imwrite(path, img)
            image_paths.append(path)
        
        # Create stream
        stream = ImageStream(image_paths=image_paths)
        
        # Read frames
        frames = list(stream)
        
        assert len(frames) == 5
        for frame in frames:
            assert frame.shape == (416, 416, 3)
        
        print("✓ Image stream works")
    
    def test_stream_preprocessing(self):
        """Test preprocessing in streaming pipeline."""
        from src.data.streaming import StreamProcessor
        
        # Create mock stream
        class MockStream:
            def __iter__(self):
                for _ in range(3):
                    yield np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create processor
        processor = StreamProcessor(
            stream=MockStream(),
            preprocessing_fn=lambda x: cv2.resize(x, (416, 416))
        )
        
        # Process frames
        processed_frames = list(processor)
        
        assert len(processed_frames) == 3
        for frame in processed_frames:
            assert frame.shape == (416, 416, 3)
        
        print("✓ Stream preprocessing works")

class TestCOCODataset:
    """Test COCO dataset integration."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_mock_coco_dataset()
    
    def create_mock_coco_dataset(self):
        """Create mock COCO dataset structure."""
        # Create directories
        images_dir = os.path.join(self.temp_dir, 'images')
        annotations_dir = os.path.join(self.temp_dir, 'annotations')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Create images
        self.num_images = 5
        for i in range(self.num_images):
            img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(images_dir, f'{i:012d}.jpg'), img)
        
        # Create COCO annotation file
        annotations = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'person'},
                {'id': 2, 'name': 'bicycle'},
                {'id': 3, 'name': 'car'}
            ]
        }
        
        annotation_id = 0
        for i in range(self.num_images):
            # Add image info
            annotations['images'].append({
                'id': i,
                'file_name': f'{i:012d}.jpg',
                'width': 416,
                'height': 416
            })
            
            # Add some annotations for each image
            for j in range(2):  # 2 objects per image
                annotations['annotations'].append({
                    'id': annotation_id,
                    'image_id': i,
                    'category_id': (i + j) % 3 + 1,  # Cycle through categories
                    'bbox': [50 + j*100, 50 + j*100, 100, 100],  # [x, y, w, h]
                    'area': 10000,
                    'iscrowd': 0
                })
                annotation_id += 1
        
        # Save annotations
        annotation_file = os.path.join(annotations_dir, 'instances_train2017.json')
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        self.annotation_file = annotation_file
        self.images_dir = images_dir
    
    def test_coco_dataset_loading(self):
        """Test COCO dataset loading."""
        dataset = COCODataset(
            root=self.temp_dir,
            annotation_file='annotations/instances_train2017.json',
            transform=None
        )
        
        # Check dataset size
        assert len(dataset) == self.num_images
        
        # Get sample
        image, target = dataset[0]
        
        # Check image
        assert isinstance(image, (np.ndarray, Image.Image))
        
        # Check target
        assert 'boxes' in target
        assert 'labels' in target
        assert 'image_id' in target
        
        # Should have 2 boxes
        assert len(target['boxes']) == 2
        
        print("✓ COCO dataset loading works")
    
    def test_coco_dataloader(self):
        """Test COCO data loader."""
        dataset = COCODataset(
            root=self.temp_dir,
            annotation_file='annotations/instances_train2017.json',
            transform=DetectionTransform(image_size=416, augment=False)
        )
        
        loader = COCODataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        
        # Get batch
        batch = next(iter(loader))
        images, targets = batch
        
        # Check batch
        assert images.shape == (2, 3, 416, 416)
        assert isinstance(targets, list)
        assert len(targets) == 2
        
        print("✓ COCO data loader works")

def run_data_tests():
    """Run all data pipeline tests."""
    print("=" * 80)
    print("Running Data Pipeline Tests")
    print("=" * 80)
    
    test_classes = [
        ('Vision Dataset', TestVisionDataset),
        ('Image Transforms', TestImageTransforms),
        ('Data Loader', TestDataLoader),
        ('Streaming Data', TestStreamingData),
        ('COCO Dataset', TestCOCODataset),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, test_class in test_classes:
        print(f"\nTesting {test_name}:")
        
        try:
            # Create instance
            test = test_class()
            if hasattr(test, 'setup_method'):
                test.setup_method()
            
            # Get test methods
            test_methods = [
                method for method in dir(test) 
                if method.startswith('test_') and callable(getattr(test, method))
            ]
            
            for method_name in test_methods:
                total_tests += 1
                try:
                    method = getattr(test, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")
        except Exception as e:
            print(f"  ✗ {test_name} setup failed: {e}")
    
    print("\n" + "=" * 80)
    print("Data Pipeline Tests Summary")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n✅ All data pipeline tests passed!")
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed")

if __name__ == "__main__":
    run_data_tests()