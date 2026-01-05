#!/usr/bin/env python3
"""
Test suite initialization for the Humanoid Vision System.
This module exports all test classes and provides a unified test runner.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

__version__ = "1.0.0"
__author__ = "Humanoid Vision System Team"
__email__ = "vision-team@example.com"

# Export test functions
def run_all_tests():
    """Run all test suites."""
    from .test_models import run_all_tests as run_model_tests
    from .test_training import run_training_tests
    from .test_inference import run_inference_tests
    from .test_data import run_data_tests
    from .test_deployment import run_deployment_tests
    
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE - HUMANOID VISION SYSTEM")
    print("=" * 80)
    
    # Run all test suites
    run_model_tests()
    print()
    
    run_training_tests()
    print()
    
    run_inference_tests()
    print()
    
    run_data_tests()
    print()
    
    run_deployment_tests()
    
    print("\n" + "=" * 80)
    print("ALL TEST SUITES COMPLETED")
    print("=" * 80)

def run_specific_tests(test_type):
    """Run specific type of tests.
    
    Args:
        test_type: One of 'models', 'training', 'inference', 'data', 'deployment'
    """
    if test_type == 'models':
        from .test_models import run_all_tests
        run_all_tests()
    elif test_type == 'training':
        from .test_training import run_training_tests
        run_training_tests()
    elif test_type == 'inference':
        from .test_inference import run_inference_tests
        run_inference_tests()
    elif test_type == 'data':
        from .test_data import run_data_tests
        run_data_tests()
    elif test_type == 'deployment':
        from .test_deployment import run_deployment_tests
        run_deployment_tests()
    else:
        raise ValueError(f"Unknown test type: {test_type}. "
                         f"Must be one of: models, training, inference, data, deployment")

# Export test classes for direct import
from .test_models import (
    TestSinkhornKnoppProjection,
    TestManifoldHyperConnection,
    TestHybridVisionBackbone,
    TestHybridVisionSystem
)

from .test_training import (
    TestTrainingStability,
    TestTrainingConfig
)

from .test_inference import (
    TestInferenceEngine,
    TestImagePreprocessor,
    TestDetectionPostprocessor,
    TestDetectionVisualizer,
    TestAPIEndpoints
)

from .test_data import (
    TestVisionDataset,
    TestImageTransforms,
    TestDataLoader,
    TestStreamingData,
    TestCOCODataset
)

from .test_deployment import (
    TestDockerConfig,
    TestKubernetesConfig,
    TestAPIServer,
    TestGRPCServer,
    TestModelServer,
    TestHealthChecker,
    TestRobotInterface,
    TestIntegration
)

# Utility functions for testing
def create_test_image(width=416, height=416, channels=3):
    """Create a test image for inference tests."""
    import numpy as np
    return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)

def create_test_batch(batch_size=4, width=416, height=416, channels=3):
    """Create a batch of test images."""
    return [create_test_image(width, height, channels) for _ in range(batch_size)]

def create_test_detections(num_detections=3):
    """Create test detection outputs."""
    import numpy as np
    return {
        'boxes': [[i*50, i*50, i*50+100, i*50+100] for i in range(num_detections)],
        'scores': [0.9 - i*0.1 for i in range(num_detections)],
        'classes': list(range(num_detections)),
        'class_names': ['class_{}'.format(i) for i in range(num_detections)]
    }

if __name__ == "__main__":
    # If run directly, run all tests
    run_all_tests()