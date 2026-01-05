#!/usr/bin/env python3
"""
Comprehensive model testing for the Hybrid Vision System.
Tests include:
1. MHC layer stability and constraints
2. Model forward/backward passes
3. Gradient flow
4. Mixed precision compatibility
5. Parameter counting
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.manifold_layers import (
    SinkhornKnoppProjection,
    ManifoldHyperConnection
)
from src.models.vision_backbone import HybridVisionBackbone
from src.models.hybrid_vision import HybridVisionSystem
from src.config.model_config import ModelConfig

class TestSinkhornKnoppProjection:
    """Test Sinkhorn-Knopp projection for doubly stochastic matrices."""
    
    def test_basic_properties(self):
        """Test that projection produces doubly stochastic matrices."""
        sk = SinkhornKnoppProjection(num_iterations=20)
        
        # Random matrix
        batch_size = 4
        n, m = 8, 8
        matrix = torch.randn(batch_size, n, m)
        
        # Apply projection
        projected = sk(matrix)
        
        # Check positivity
        assert torch.all(projected >= 0), "All values must be ≥ 0"
        
        # Check row sums ≈ 1
        row_sums = projected.sum(dim=2)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-4)
        
        # Check column sums ≈ 1
        col_sums = projected.sum(dim=1)
        assert torch.allclose(col_sums, torch.ones_like(col_sums), rtol=1e-4)
        
        print("✓ Sinkhorn-Knopp projection produces doubly stochastic matrices")
    
    def test_gradient_flow(self):
        """Test that gradients flow through projection."""
        sk = SinkhornKnoppProjection(num_iterations=5)
        
        # Create learnable matrix
        matrix = nn.Parameter(torch.randn(3, 3))
        optimizer = torch.optim.SGD([matrix], lr=0.01)
        
        # Forward pass
        projected = sk(matrix)
        
        # Compute loss
        target = torch.eye(3)
        loss = torch.nn.functional.mse_loss(projected, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert matrix.grad is not None, "Gradients should flow through projection"
        assert not torch.allclose(matrix.grad, torch.zeros_like(matrix.grad))
        
        # Optimizer step (should not crash)
        optimizer.step()
        
        print("✓ Gradients flow through Sinkhorn-Knopp projection")
    
    def test_deterministic(self):
        """Test that projection is deterministic."""
        sk = SinkhornKnoppProjection(num_iterations=10)
        
        matrix = torch.randn(2, 5, 5)
        
        # Two forward passes should produce same result
        torch.manual_seed(42)
        result1 = sk(matrix)
        
        torch.manual_seed(42)
        result2 = sk(matrix)
        
        assert torch.allclose(result1, result2), "Projection should be deterministic"
        
        print("✓ Sinkhorn-Knopp projection is deterministic")

class TestManifoldHyperConnection:
    """Test Manifold Hyper-Connection layer."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.batch_size = 4
        self.input_dim = 64
        self.expansion_rate = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.mhc = ManifoldHyperConnection(
            input_dim=self.input_dim,
            expansion_rate=self.expansion_rate,
            sk_iterations=5  # Fewer iterations for faster tests
        ).to(self.device)
    
    def test_initialization(self):
        """Test that MHC initializes correctly."""
        # Check parameter shapes
        assert self.mhc.H_pre_raw.shape == (self.input_dim, self.input_dim * self.expansion_rate)
        assert self.mhc.H_post_raw.shape == (self.input_dim * self.expansion_rate, self.input_dim)
        assert self.mhc.H_res_raw.shape == (self.input_dim, self.input_dim)
        
        # Check MLP layers
        assert len(self.mhc.mlp) == 5  # Linear -> GELU -> Dropout -> Linear -> GELU
        
        print("✓ MHC layer initializes correctly")
    
    def test_forward_pass(self):
        """Test forward pass maintains shape."""
        x = torch.randn(self.batch_size, self.input_dim).to(self.device)
        
        output = self.mhc(x)
        
        # Check shape preservation
        assert output.shape == x.shape
        
        # Check no NaN/Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        print("✓ MHC forward pass maintains shape and produces valid outputs")
    
    def test_constrained_matrices(self):
        """Test matrix constraints are satisfied."""
        H_pre, H_post, H_res = self.mhc.constrained_matrices()
        
        # H_pre: sigmoid for [0, 1] range
        assert torch.all(H_pre >= 0) and torch.all(H_pre <= 1)
        
        # H_post: 2 * sigmoid for [0, 2] range
        assert torch.all(H_post >= 0) and torch.all(H_post <= 2)
        
        # H_res: doubly stochastic (approximately)
        row_sums = H_res.sum(dim=1)
        col_sums = H_res.sum(dim=0)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-3)
        assert torch.allclose(col_sums, torch.ones_like(col_sums), rtol=1e-3)
        
        print("✓ MHC matrices satisfy constraints")
    
    def test_gradient_stability(self):
        """Test gradient flow and stability."""
        x = torch.randn(self.batch_size, self.input_dim).to(self.device)
        x.requires_grad = True
        
        # Forward pass
        output = self.mhc(x)
        
        # Create dummy loss
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # Check gradient norm
        grad_norm = x.grad.norm().item()
        assert 0 < grad_norm < 100, f"Gradient norm {grad_norm} suspicious"
        
        print(f"✓ Gradients flow stably through MHC (norm: {grad_norm:.4f})")
    
    def test_mixed_precision(self):
        """Test mixed precision compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        self.mhc.use_mixed_precision = True
        
        # Create mixed precision input
        with torch.cuda.amp.autocast():
            x = torch.randn(self.batch_size, self.input_dim, dtype=torch.bfloat16).cuda()
            output = self.mhc(x)
            
            # Should produce bfloat16 output in autocast context
            assert output.dtype == torch.bfloat16
        
        print("✓ MHC supports mixed precision training")
    
    def test_stability_monitoring(self):
        """Test stability metrics collection."""
        # Do a forward pass to populate metrics
        x = torch.randn(self.batch_size, self.input_dim).to(self.device)
        _ = self.mhc(x)
        
        # Get metrics
        metrics = self.mhc.get_stability_metrics()
        
        # Check expected metrics
        assert 'max_eigenvalue' in metrics
        assert 'min_eigenvalue' in metrics
        assert 'gradient_norms' in metrics
        
        # Eigenvalues should be reasonable
        assert -1 <= metrics['min_eigenvalue'] <= 1
        assert -1 <= metrics['max_eigenvalue'] <= 1
        
        print("✓ MHC collects stability metrics correctly")

class TestHybridVisionBackbone:
    """Test the Hybrid Vision Backbone."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.batch_size = 2
        self.channels = 3
        self.height = 416
        self.width = 416
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.backbone = HybridVisionBackbone(
            input_channels=self.channels,
            base_channels=32,
            num_blocks=[2, 3, 4, 2],
            use_mhc=True
        ).to(self.device)
        
        # Create test input
        self.x = torch.randn(
            self.batch_size, 
            self.channels, 
            self.height, 
            self.width
        ).to(self.device)
    
    def test_forward_pass(self):
        """Test backbone forward pass produces multi-scale features."""
        features = self.backbone(self.x)
        
        # Should return dict with three scales
        assert isinstance(features, dict)
        assert 'scale_small' in features
        assert 'scale_medium' in features
        assert 'scale_large' in features
        
        # Check shapes
        B, C, H, W = self.x.shape
        
        # Scale calculations
        # Initial stride 2 in stem, then stages 2-4 each have stride 2
        # scale_small: H/4, W/4
        # scale_medium: H/8, W/8
        # scale_large: H/16, W/16
        
        small = features['scale_small']
        assert small.shape == (B, 64, H // 4, W // 4)  # base_channels * 2
        
        medium = features['scale_medium']
        assert medium.shape == (B, 128, H // 8, W // 8)  # base_channels * 4
        
        large = features['scale_large']
        assert large.shape == (B, 256, H // 16, W // 16)  # base_channels * 8
        
        print("✓ Backbone produces correct multi-scale features")
    
    def test_no_nan_inf(self):
        """Test no NaN or Inf values in output."""
        features = self.backbone(self.x)
        
        for name, feature in features.items():
            assert not torch.isnan(feature).any(), f"NaN in {name}"
            assert not torch.isinf(feature).any(), f"Inf in {name}"
        
        print("✓ Backbone produces no NaN/Inf values")
    
    def test_gradient_flow(self):
        """Test gradients flow through backbone."""
        x = self.x.clone()
        x.requires_grad = True
        
        features = self.backbone(x)
        
        # Create dummy loss on each scale
        total_loss = 0
        for feature in features.values():
            target = torch.randn_like(feature)
            loss = torch.nn.functional.mse_loss(feature, target)
            total_loss += loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        print("✓ Gradients flow through backbone")
    
    def test_parameter_count(self):
        """Test reasonable parameter count."""
        total_params = sum(p.numel() for p in self.backbone.parameters())
        
        # Should be in reasonable range for backbone
        # 32 base channels, 4 stages: ~2-5 million parameters
        assert 2_000_000 < total_params < 5_000_000
        
        print(f"✓ Backbone has reasonable parameter count: {total_params:,}")
    
    def test_training_mode(self):
        """Test behavior in training vs evaluation mode."""
        # In training mode
        self.backbone.train()
        features_train = self.backbone(self.x)
        
        # In evaluation mode
        self.backbone.eval()
        with torch.no_grad():
            features_eval = self.backbone(self.x)
        
        # Features should be different due to dropout/batch norm
        # but shapes should be same
        for name in features_train.keys():
            assert features_train[name].shape == features_eval[name].shape
        
        print("✓ Backbone properly handles train/eval modes")

class TestHybridVisionSystem:
    """Test complete Hybrid Vision System."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.batch_size = 2
        self.channels = 3
        self.height = 416
        self.width = 416
        self.num_classes = 80
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create config
        self.config = {
            'use_vit': True,
            'use_rag': False,
            'detection': {
                'anchor_sizes': [[10, 13], [16, 30], [33, 23]],
                'num_classes': self.num_classes
            }
        }
        
        self.model = HybridVisionSystem(
            config=self.config,
            num_classes=self.num_classes,
            use_vit=True,
            use_rag=False
        ).to(self.device)
        
        # Create test input
        self.x = torch.randn(
            self.batch_size, 
            self.channels, 
            self.height, 
            self.width
        ).to(self.device)
    
    def test_detection_forward(self):
        """Test detection task forward pass."""
        outputs = self.model(self.x, task='detection')
        
        # Should return dict with features and detections
        assert 'features' in outputs
        assert 'detections' in outputs
        
        # Check feature shape
        features = outputs['features']
        assert features.dim() == 4  # [B, C, H, W]
        
        # Check detection shape
        detections = outputs['detections']
        # [B, 5+num_classes, H', W'] where H',W' are feature map dimensions
        assert detections.shape[1] == 5 + self.num_classes
        
        print("✓ Detection forward pass works correctly")
    
    def test_classification_forward(self):
        """Test classification task forward pass."""
        outputs = self.model(self.x, task='classification')
        
        assert 'features' in outputs
        assert 'classifications' in outputs
        
        classifications = outputs['classifications']
        assert classifications.shape == (self.batch_size, self.num_classes)
        
        print("✓ Classification forward pass works correctly")
    
    def test_feature_extraction(self):
        """Test feature extraction task."""
        outputs = self.model(self.x, task='features')
        
        assert 'features' in outputs
        assert outputs['features'].dim() == 4
        
        print("✓ Feature extraction works correctly")
    
    def test_gradient_flow_full_model(self):
        """Test gradients flow through entire model."""
        x = self.x.clone()
        x.requires_grad = True
        
        # Test detection task
        outputs = self.model(x, task='detection')
        
        # Create dummy loss
        target = torch.randn_like(outputs['detections'])
        loss = torch.nn.functional.mse_loss(outputs['detections'], target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        print("✓ Gradients flow through full model")
    
    def test_mixed_precision_compatibility(self):
        """Test model works with mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        with torch.cuda.amp.autocast():
            # Create bfloat16 input
            x = torch.randn(
                self.batch_size, 
                self.channels, 
                self.height, 
                self.width,
                dtype=torch.bfloat16
            ).cuda()
            
            outputs = self.model(x, task='detection')
            
            # Outputs should be bfloat16 in autocast context
            assert outputs['detections'].dtype == torch.bfloat16
        
        print("✓ Full model supports mixed precision")
    
    def test_stability_metrics(self):
        """Test stability metrics collection."""
        # Do a forward pass
        _ = self.model(self.x, task='detection')
        
        # Get stability metrics
        metrics = self.model.get_stability_metrics()
        
        # Should have metrics from MHC layers
        assert len(metrics) > 0
        
        # Check some key metrics
        for key in metrics.keys():
            if 'eigenvalue' in key:
                value = metrics[key]
                assert isinstance(value, (int, float))
        
        print("✓ Model collects stability metrics from all MHC layers")
    
    def test_model_serialization(self):
        """Test model can be saved and loaded."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            save_path = os.path.join(tmpdir, 'model.pt')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }, save_path)
            
            # Load model
            checkpoint = torch.load(save_path, map_location=self.device)
            
            # Create new model
            new_model = HybridVisionSystem(
                config=checkpoint['config'],
                num_classes=self.num_classes
            ).to(self.device)
            
            # Load state dict
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Verify same outputs
            self.model.eval()
            new_model.eval()
            
            with torch.no_grad():
                outputs1 = self.model(self.x, task='detection')
                outputs2 = new_model(self.x, task='detection')
            
            # Should be close (allowing for small numerical differences)
            for key in outputs1.keys():
                if torch.is_tensor(outputs1[key]):
                    assert torch.allclose(
                        outputs1[key], 
                        outputs2[key], 
                        rtol=1e-5
                    )
        
        print("✓ Model can be serialized and deserialized correctly")
    
    def test_inference_mode(self):
        """Test model in inference mode."""
        # Switch to eval mode
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(self.x, task='detection')
        
        # Should not have gradients
        for param in self.model.parameters():
            assert param.grad is None
        
        # Outputs should be valid
        assert not torch.isnan(outputs['detections']).any()
        
        print("✓ Model works correctly in inference mode")
    
    def test_parameter_stats(self):
        """Test parameter statistics are reasonable."""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        # Should have reasonable number of parameters
        # Hybrid model with ViT: ~10-20 million parameters
        assert 10_000_000 < total_params < 30_000_000
        
        # Most parameters should be trainable
        assert trainable_params > 0.95 * total_params
        
        print(f"✓ Model has {total_params:,} total parameters, "
              f"{trainable_params:,} trainable")

def run_all_tests():
    """Run all model tests."""
    print("=" * 80)
    print("Running Model Tests")
    print("=" * 80)
    
    # Create test instances
    test_classes = [
        TestSinkhornKnoppProjection(),
        TestManifoldHyperConnection(),
        TestHybridVisionBackbone(),
        TestHybridVisionSystem()
    ]
    
    # Run all test methods
    for test_class in test_classes:
        test_class.setup_method()
        
        # Get all test methods (starting with 'test_')
        test_methods = [
            method for method in dir(test_class) 
            if method.startswith('test_') and callable(getattr(test_class, method))
        ]
        
        print(f"\nTesting {test_class.__class__.__name__}:")
        for method_name in test_methods:
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name} failed: {e}")
                raise
    
    print("\n" + "=" * 80)
    print("All Model Tests Passed!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()