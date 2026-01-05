# src/inference/__init__.py
"""
Real-time Inference Module for Humanoid Vision System

This module provides production-grade inference capabilities for robotic vision.
Features:
1. Low-latency inference engine with manifold constraints
2. Camera streaming and preprocessing pipeline
3. Real-time detection postprocessing
4. Robot communication interfaces
5. Visualization and debugging tools

Author: AI Systems Architect
Date: 2024
"""

from .engine import InferenceEngine, AsyncInferenceEngine
from .preprocessing import ImagePreprocessor, VideoStreamer, CameraManager
from .postprocessing import DetectionPostprocessor, NMSFilter, DetectionTracker
from .visualizer import DetectionVisualizer, PerformanceMonitor, DebugVisualizer
from .robot_interface import RobotCommunication, RobotState, CommandHandler

__version__ = "1.0.0"
__all__ = [
    # Engine
    'InferenceEngine',
    'AsyncInferenceEngine',
    
    # Preprocessing
    'ImagePreprocessor',
    'VideoStreamer',
    'CameraManager',
    
    # Postprocessing
    'DetectionPostprocessor',
    'NMSFilter',
    'DetectionTracker',
    
    # Visualization
    'DetectionVisualizer',
    'PerformanceMonitor',
    'DebugVisualizer',
    
    # Robot Interface
    'RobotCommunication',
    'RobotState',
    'CommandHandler',
]