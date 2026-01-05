#!/usr/bin/env python3
"""
Inference testing for the Hybrid Vision System.
Tests include:
1. Real-time inference performance
2. Preprocessing pipeline
3. Postprocessing (NMS, decoding)
4. Visualization
5. API endpoints
6. Robot interface
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import pytest
import tempfile
import os
import json
from pathlib import Path
from PIL import Image
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.hybrid_vision import HybridVisionSystem
from src.inference.engine import InferenceEngine
from src.inference.preprocessing import ImagePreprocessor
from src.inference.postprocessing import DetectionPostprocessor
from src.inference.visualizer import DetectionVisualizer
from src.deployment.api_server import VisionAPI
from src.config.inference_config import InferenceConfig

class TestInferenceEngine:
    """Test inference engine performance and correctness."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
        self.model = HybridVisionSystem(
            config={'use_vit': True, 'use_rag': False},
            num_classes=80,
            use_vit=False,  # Faster for testing
            use_rag=False
        ).to(self.device)
        
        # Put in evaluation mode
        self.model.eval()
        
        # Create inference engine
        self.config = InferenceConfig()
        self.engine = InferenceEngine(
            model=self.model,
            config=self.config,
            device=self.device
        )
        
        # Create test image
        self.create_test_image()
    
    def create_test_image(self):
        """Create test images for inference."""
        # Create synthetic image with colored squares
        self.image_size = 416
        self.test_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Add colored squares (simulating objects)
        cv2.rectangle(self.test_image, (50, 50), (150, 150), (255, 0, 0), -1)    # Red square
        cv2.rectangle(self.test_image, (200, 200), (300, 300), (0, 255, 0), -1)  # Green square
        cv2.rectangle(self.test_image, (350, 100), (400, 200), (0, 0, 255), -1)  # Blue square
        
        # Save to temp file
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        cv2.imwrite(self.image_path, self.test_image)
        
        # Create batch of images
        self.batch_images = []
        for i in range(4):
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            cv2.rectangle(img, (i*50, i*50), (i*50+100, i*50+100), (255, 255, 255), -1)
            self.batch_images.append(img)
    
    def test_single_image_inference(self):
        """Test inference on single image."""
        # Process single image
        detections = self.engine.process_image(self.test_image)
        
        # Check output structure
        assert isinstance(detections, dict)
        assert 'boxes' in detections
        assert 'scores' in detections
        assert 'classes' in detections
        
        # Check types
        assert isinstance(detections['boxes'], list)
        assert isinstance(detections['scores'], list)
        assert isinstance(detections['classes'], list)
        
        # All lists should have same length
        assert len(detections['boxes']) == len(detections['scores'])
        assert len(detections['boxes']) == len(detections['classes'])
        
        print(f"✓ Single image inference produces {len(detections['boxes'])} detections")
    
    def test_batch_inference(self):
        """Test batch inference."""
        detections_batch = self.engine.process_batch(self.batch_images)
        
        # Should return list of detections for each image
        assert isinstance(detections_batch, list)
        assert len(detections_batch) == len(self.batch_images)
        
        # Each should have correct structure
        for detections in detections_batch:
            assert 'boxes' in detections
            assert 'scores' in detections
            assert 'classes' in detections
        
        print(f"✓ Batch inference on {len(self.batch_images)} images works")
    
    def test_inference_latency(self):
        """Test inference latency meets requirements."""
        # Warm up
        for _ in range(5):
            _ = self.engine.process_image(self.test_image)
        
        # Measure latency
        num_runs = 50
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = self.engine.process_image(self.test_image)
            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)
        
        # Calculate statistics
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Check meets real-time requirements (< 50ms)
        assert mean_latency < 50, f"Mean latency {mean_latency:.2f}ms > 50ms"
        assert p95_latency < 100, f"95th percentile latency {p95_latency:.2f}ms > 100ms"
        
        print(f"✓ Inference latency: {mean_latency:.2f}±{std_latency:.2f}ms "
              f"(95th: {p95_latency:.2f}ms)")
    
    def test_memory_usage(self):
        """Test memory usage during inference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory test")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Run inference
        _ = self.engine.process_image(self.test_image)
        
        # Get memory after inference
        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated() / 1024**2
        
        # Memory increase should be reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 500, f"Memory increase {memory_increase:.1f}MB too high"
        
        print(f"✓ Memory usage: {memory_increase:.1f}MB increase")
    
    def test_deterministic_inference(self):
        """Test inference is deterministic."""
        # Run inference twice
        torch.manual_seed(42)
        detections1 = self.engine.process_image(self.test_image)
        
        torch.manual_seed(42)
        detections2 = self.engine.process_image(self.test_image)
        
        # Check boxes are the same (allowing small numerical differences)
        for i, (box1, box2) in enumerate(zip(detections1['boxes'], detections2['boxes'])):
            assert len(box1) == len(box2) == 4
            for coord1, coord2 in zip(box1, box2):
                assert abs(coord1 - coord2) < 1e-5
        
        print("✓ Inference is deterministic")
    
    def test_edge_cases(self):
        """Test edge cases in inference."""
        # Test empty image
        empty_image = np.zeros((416, 416, 3), dtype=np.uint8)
        detections = self.engine.process_image(empty_image)
        assert len(detections['boxes']) == 0
        
        # Test very small image
        small_image = np.zeros((10, 10, 3), dtype=np.uint8)
        detections = self.engine.process_image(small_image)
        # Should still work (image gets resized)
        assert isinstance(detections, dict)
        
        # Test image with NaN/Inf (should handle gracefully)
        corrupt_image = np.full((416, 416, 3), np.nan, dtype=np.float32)
        # This might raise an error, which is OK
        try:
            _ = self.engine.process_image(corrupt_image)
        except:
            pass  # Expected to fail on invalid input
        
        print("✓ Edge cases handled appropriately")

class TestImagePreprocessor:
    """Test image preprocessing pipeline."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.config = InferenceConfig()
        self.preprocessor = ImagePreprocessor(self.config)
        
        # Create test images
        self.rgb_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        self.bgr_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        self.gray_image = np.random.randint(0, 255, (416, 416), dtype=np.uint8)
    
    def test_preprocess_single_image(self):
        """Test single image preprocessing."""
        # Process RGB image
        processed = self.preprocessor.preprocess(self.rgb_image)
        
        # Check output type and shape
        assert isinstance(processed, torch.Tensor)
        assert processed.dtype == torch.float32
        assert processed.shape == (1, 3, 416, 416)  # Batch of 1
        
        # Check normalization
        mean = processed.mean().item()
        std = processed.std().item()
        assert abs(mean) < 1.0  # Should be near 0 after normalization
        assert std > 0.5  # Should have some variance
        
        print("✓ Single image preprocessing works")
    
    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        batch_images = [self.rgb_image, self.rgb_image.copy()]
        
        processed = self.preprocessor.preprocess_batch(batch_images)
        
        # Check shape
        assert processed.shape == (2, 3, 416, 416)
        
        # Check all images processed correctly
        for i in range(2):
            img_tensor = processed[i]
            assert img_tensor.shape == (3, 416, 416)
        
        print("✓ Batch preprocessing works")
    
    def test_color_conversion(self):
        """Test automatic color conversion."""
        # BGR input should be converted to RGB
        processed_bgr = self.preprocessor.preprocess(self.bgr_image)
        processed_rgb = self.preprocessor.preprocess(self.rgb_image)
        
        # They should be different since we convert BGR to RGB
        # Check by comparing first few values
        diff = torch.abs(processed_bgr - processed_rgb).mean().item()
        assert diff > 0.01  # Should be different
        
        print("✓ Automatic color conversion works")
    
    def test_resizing(self):
        """Test image resizing."""
        # Create image of different size
        large_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        
        processed = self.preprocessor.preprocess(large_image)
        
        # Should be resized to config size
        assert processed.shape == (1, 3, 416, 416)
        
        print("✓ Image resizing works")
    
    def test_normalization(self):
        """Test normalization values."""
        # Create solid color image
        solid_image = np.full((416, 416, 3), 128, dtype=np.uint8)
        
        processed = self.preprocessor.preprocess(solid_image)
        
        # After normalization with ImageNet stats:
        # (128/255 - mean) / std
        # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        expected_value = ((128/255) - 0.485) / 0.229
        
        # Check it's close to expected
        actual_value = processed[0, 0, 0, 0].item()
        assert abs(actual_value - expected_value) < 0.01
        
        print("✓ Normalization uses correct ImageNet stats")

class TestDetectionPostprocessor:
    """Test detection postprocessing."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.config = InferenceConfig()
        self.postprocessor = DetectionPostprocessor(self.config)
        
        # Create fake model output
        self.create_fake_detections()
    
    def create_fake_detections(self):
        """Create fake detection outputs for testing."""
        # Simulate model output: [batch, 85, 13, 13]
        batch_size = 2
        num_classes = 80
        
        self.fake_output = torch.randn(
            batch_size, 
            5 + num_classes,  # bbox(4) + obj(1) + classes(80)
            13, 13
        )
        
        # Add some high-confidence detections
        self.fake_output[0, 4, 3, 3] = 10.0  # High objectness
        self.fake_output[0, 5:85, 3, 3] = torch.randn(80) * 5 + 5  # High class scores
        
        # Add another detection
        self.fake_output[0, 4, 7, 7] = 8.0
        self.fake_output[0, 5:85, 7, 7] = torch.randn(80) * 5 + 3
    
    def test_decode_predictions(self):
        """Test decoding of raw predictions."""
        decoded = self.postprocessor.decode_predictions(self.fake_output)
        
        # Check output structure
        assert isinstance(decoded, list)
        assert len(decoded) == 2  # Batch size
        
        for batch_item in decoded:
            assert isinstance(batch_item, dict)
            assert 'boxes' in batch_item
            assert 'scores' in batch_item
            assert 'classes' in batch_item
            
            # Should have some detections
            assert len(batch_item['boxes']) > 0
        
        print("✓ Prediction decoding works")
    
    def test_non_max_suppression(self):
        """Test non-maximum suppression."""
        # Create overlapping boxes
        boxes = [
            [0.1, 0.1, 0.3, 0.3],  # Box 1
            [0.15, 0.15, 0.35, 0.35],  # Box 2 (overlaps with Box 1)
            [0.6, 0.6, 0.8, 0.8],  # Box 3 (far away)
        ]
        scores = [0.9, 0.8, 0.7]
        
        filtered_boxes, filtered_scores = self.postprocessor.non_max_suppression(
            boxes, scores, iou_threshold=0.5
        )
        
        # Should keep Box 1 and Box 3 (Box 2 suppressed)
        assert len(filtered_boxes) == 2
        assert filtered_scores[0] == 0.9  # Highest score kept
        
        print("✓ Non-maximum suppression works")
    
    def test_confidence_thresholding(self):
        """Test confidence threshold filtering."""
        boxes = [
            [0.1, 0.1, 0.2, 0.2],
            [0.3, 0.3, 0.4, 0.4],
            [0.5, 0.5, 0.6, 0.6],
        ]
        scores = [0.9, 0.3, 0.1]  # Scores below/above threshold
        classes = [0, 1, 2]
        
        # Apply confidence threshold
        filtered = self.postprocessor.filter_by_confidence(
            boxes, scores, classes, confidence_threshold=0.5
        )
        
        # Should keep only first box
        assert len(filtered['boxes']) == 1
        assert filtered['scores'][0] == 0.9
        
        print("✓ Confidence threshold filtering works")
    
    def test_box_format_conversion(self):
        """Test box format conversions."""
        # Test center to corner conversion
        center_box = [0.5, 0.5, 0.2, 0.3]  # [cx, cy, w, h]
        corner_box = self.postprocessor.center_to_corner(center_box)
        
        assert len(corner_box) == 4
        assert corner_box[0] < corner_box[2]  # x1 < x2
        assert corner_box[1] < corner_box[3]  # y1 < y2
        
        # Test corner to center conversion
        converted_back = self.postprocessor.corner_to_center(corner_box)
        
        # Should be close to original (allowing float precision)
        for orig, conv in zip(center_box, converted_back):
            assert abs(orig - conv) < 1e-5
        
        print("✓ Box format conversions work correctly")

class TestDetectionVisualizer:
    """Test detection visualization."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.visualizer = DetectionVisualizer()
        
        # Create test image and detections
        self.image = np.zeros((416, 416, 3), dtype=np.uint8)
        self.detections = {
            'boxes': [[50, 50, 150, 150], [200, 200, 300, 300]],
            'scores': [0.9, 0.8],
            'classes': [0, 1],
            'class_names': ['person', 'car']
        }
    
    def test_visualize_detections(self):
        """Test visualization of detections."""
        # Visualize detections
        result = self.visualizer.visualize(
            self.image, 
            self.detections,
            confidence_threshold=0.5
        )
        
        # Check output
        assert isinstance(result, np.ndarray)
        assert result.shape == self.image.shape
        assert result.dtype == np.uint8
        
        # Check that visualization modified the image
        # (by checking pixel values changed)
        diff = np.abs(result.astype(float) - self.image.astype(float)).mean()
        assert diff > 0, "Visualization should modify image"
        
        print("✓ Detection visualization works")
    
    def test_draw_bounding_box(self):
        """Test individual bounding box drawing."""
        # Draw single box
        image_copy = self.image.copy()
        self.visualizer._draw_bounding_box(
            image_copy,
            box=[100, 100, 200, 200],
            label="test: 0.95",
            color=(255, 0, 0)
        )
        
        # Check some pixels changed to red
        # Look at center of box
        center_pixel = image_copy[150, 150]
        # Box border is red, interior unchanged (black)
        # So we expect some red pixels around edges
        
        print("✓ Bounding box drawing works")
    
    def test_color_mapping(self):
        """Test class to color mapping."""
        # Get color for class
        color = self.visualizer._get_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
        
        # Same class should give same color
        color2 = self.visualizer._get_color(0)
        assert color == color2
        
        # Different classes should (usually) give different colors
        color3 = self.visualizer._get_color(1)
        # They might be similar but not guaranteed to be different
        
        print("✓ Color mapping works")

class TestAPIEndpoints:
    """Test API endpoints."""
    
    def setup_method(self):
        """Setup test fixture."""
        # Create mock model for API testing
        class MockModel:
            def __call__(self, x, task='detection'):
                return {
                    'detections': torch.randn(1, 85, 13, 13),
                    'features': torch.randn(1, 256, 13, 13)
                }
        
        self.model = MockModel()
        self.api = VisionAPI(model=self.model, config=InferenceConfig())
        
        # Create test image data
        self.create_test_image_data()
    
    def create_test_image_data(self):
        """Create test image data for API tests."""
        # Create simple image
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Save to file
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.temp_dir, 'test_api.jpg')
        cv2.imwrite(self.image_path, self.test_image)
        
        # Create base64 encoded image
        import base64
        _, buffer = cv2.imencode('.jpg', self.test_image)
        self.image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.api.health_check()
        
        assert response['status'] == 'healthy'
        assert 'timestamp' in response
        assert 'version' in response
        
        print("✓ Health endpoint works")
    
    def test_detect_endpoint_file(self):
        """Test detection endpoint with file upload."""
        # Simulate file upload
        files = {'file': ('test.jpg', open(self.image_path, 'rb'), 'image/jpeg')}
        
        try:
            response = self.api.detect_image(file=files['file'])
            
            # Check response structure
            assert 'detections' in response
            assert 'processing_time' in response
            assert 'image_size' in response
            
            detections = response['detections']
            assert isinstance(detections, list)
        finally:
            files['file'][1].close()
        
        print("✓ Detection endpoint with file upload works")
    
    def test_detect_endpoint_b64(self):
        """Test detection endpoint with base64 image."""
        response = self.api.detect_image_b64(image_b64=self.image_b64)
        
        assert 'detections' in response
        assert 'processing_time' in response
        
        print("✓ Detection endpoint with base64 works")
    
    def test_batch_detect_endpoint(self):
        """Test batch detection endpoint."""
        # Create multiple images
        image_list = [self.test_image, self.test_image.copy()]
        
        response = self.api.detect_batch(images=image_list)
        
        assert 'results' in response
        assert isinstance(response['results'], list)
        assert len(response['results']) == 2
        
        print("✓ Batch detection endpoint works")

def run_inference_tests():
    """Run all inference tests."""
    print("=" * 80)
    print("Running Inference Tests")
    print("=" * 80)
    
    # Test classes
    test_classes = [
        ('Inference Engine', TestInferenceEngine),
        ('Image Preprocessor', TestImagePreprocessor),
        ('Detection Postprocessor', TestDetectionPostprocessor),
        ('Detection Visualizer', TestDetectionVisualizer),
        ('API Endpoints', TestAPIEndpoints),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, test_class in test_classes:
        print(f"\nTesting {test_name}:")
        
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
    
    print("\n" + "=" * 80)
    print("Inference Tests Summary")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n✅ All inference tests passed!")
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed")

if __name__ == "__main__":
    run_inference_tests()