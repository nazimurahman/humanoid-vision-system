# src/inference/engine.py
"""
Inference Engine with Manifold-Constrained Hyper-Connection Support

Features:
1. Real-time inference with mixed precision
2. Batch processing optimization
3. Async inference support
4. Memory management for edge devices
5. Stability monitoring for mHC layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import threading
import queue
import asyncio
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_half_precision: bool = True
    use_tensorrt: bool = False
    tensorrt_precision: str = "FP16"
    
    # Performance settings
    batch_size: int = 1
    max_batch_size: int = 16
    warmup_iterations: int = 10
    enable_batching: bool = True
    
    # Model settings
    model_path: str = ""
    input_height: int = 416
    input_width: int = 416
    num_classes: int = 80
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Monitoring settings
    enable_profiling: bool = False
    log_interval: int = 100
    memory_monitoring: bool = True
    
    # Robot-specific settings
    max_latency_ms: int = 50  # Maximum allowed latency
    target_fps: int = 30  # Target frames per second

class InferenceMode(Enum):
    """Inference modes for different scenarios."""
    REAL_TIME = "real_time"      # Lowest latency, single frame
    BATCHED = "batched"          # Batched processing
    STREAMING = "streaming"      # Continuous stream
    ASYNC = "async"              # Asynchronous processing

class InferenceEngine:
    """
    Main inference engine with manifold constraint support.
    
    Optimized for robotic deployment with:
    1. Real-time performance guarantees
    2. Memory-efficient execution
    3. Stability monitoring for mHC layers
    4. Edge device compatibility
    """
    
    def __init__(self, config: InferenceConfig, model: Optional[nn.Module] = None):
        """
        Initialize inference engine.
        
        Args:
            config: Inference configuration
            model: Optional pre-initialized model
        """
        self.config = config
        self.device = torch.device(config.device)
        self.use_half = config.use_half_precision and self.device.type == "cuda"
        
        # Initialize model
        self.model = model
        if model is None and config.model_path:
            self._load_model()
        elif model is not None:
            self.model = model.to(self.device)
            
        # Set model to evaluation mode
        if self.model:
            self.model.eval()
        
        # Mixed precision
        self.scaler = GradScaler(enabled=self.use_half)
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.stability_metrics = []
        
        # Batch processing
        self.batch_queue = queue.Queue(maxsize=config.max_batch_size)
        self.batch_thread = None
        self.batch_running = False
        
        # TensorRT optimization (if enabled)
        if config.use_tensorrt and self.device.type == "cuda":
            self._optimize_with_tensorrt()
        
        # Warmup
        if config.warmup_iterations > 0:
            self._warmup()
        
        logger.info(f"InferenceEngine initialized on {self.device}")
        logger.info(f"Half precision: {self.use_half}")
        logger.info(f"Batch size: {config.batch_size}")
    
    def _load_model(self):
        """Load model from checkpoint with proper initialization."""
        try:
            logger.info(f"Loading model from {self.config.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(
                self.config.model_path,
                map_location=self.device,
                weights_only=False
            )
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume checkpoint is the model itself
                self.model = checkpoint
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _optimize_with_tensorrt(self):
        """Optimize model with TensorRT for inference."""
        if not torch.cuda.is_available():
            logger.warning("TensorRT optimization requires CUDA")
            return
            
        try:
            import tensorrt as trt
            from torch2trt import torch2trt
            
            logger.info("Starting TensorRT optimization...")
            
            # Create sample input
            sample_input = torch.randn(
                1, 3, self.config.input_height, self.config.input_width
            ).to(self.device)
            
            # Convert to TensorRT
            self.model_trt = torch2trt(
                self.model,
                [sample_input],
                fp16_mode=(self.config.tensorrt_precision == "FP16"),
                max_workspace_size=1 << 25,  # 32MB
                log_level=trt.Logger.WARNING
            )
            
            logger.info("TensorRT optimization completed")
            
        except ImportError:
            logger.warning("TensorRT not available, using PyTorch backend")
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
    
    def _warmup(self):
        """Warmup the model for stable performance."""
        logger.info(f"Warming up model with {self.config.warmup_iterations} iterations")
        
        # Create dummy input
        dummy_input = torch.randn(
            self.config.batch_size,
            3,
            self.config.input_height,
            self.config.input_width
        ).to(self.device)
        
        # Warmup iterations
        with torch.no_grad():
            for i in range(self.config.warmup_iterations):
                if self.config.use_tensorrt and hasattr(self, 'model_trt'):
                    _ = self.model_trt(dummy_input)
                else:
                    with autocast(enabled=self.use_half):
                        _ = self.model(dummy_input, task='detection')
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        logger.info("Warmup completed")
    
    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess batch of images for inference.
        
        Args:
            images: List of images in BGR format
            
        Returns:
            Preprocessed batch tensor
        """
        from .preprocessing import ImagePreprocessor
        preprocessor = ImagePreprocessor(
            target_size=(self.config.input_height, self.config.input_width)
        )
        
        # Process each image
        processed_images = []
        for img in images:
            processed = preprocessor.process(img)
            processed_images.append(processed)
        
        # Stack into batch
        batch = torch.stack(processed_images).to(self.device)
        
        # Convert to half precision if enabled
        if self.use_half:
            batch = batch.half()
        
        return batch
    
    @torch.no_grad()
    def infer(self, image: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Perform inference on single image.
        
        Args:
            image: Input image (numpy array or tensor)
            
        Returns:
            Dictionary containing detections and metadata
        """
        start_time = time.perf_counter()
        
        # Preprocess if needed
        if isinstance(image, np.ndarray):
            from .preprocessing import ImagePreprocessor
            preprocessor = ImagePreprocessor(
                target_size=(self.config.input_height, self.config.input_width)
            )
            input_tensor = preprocessor.process(image).to(self.device)
        else:
            input_tensor = image.to(self.device)
        
        # Add batch dimension if needed
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Mixed precision inference
        with autocast(enabled=self.use_half):
            # Forward pass
            if hasattr(self, 'model_trt') and self.config.use_tensorrt:
                outputs = self.model_trt(input_tensor)
            else:
                outputs = self.model(input_tensor, task='detection')
        
        # Synchronize if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Calculate inference time
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # Track memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            self.memory_usage.append(memory_allocated)
        
        # Collect stability metrics from mHC layers
        if hasattr(self.model, 'get_stability_metrics'):
            stability = self.model.get_stability_metrics()
            self.stability_metrics.append(stability)
        
        # Prepare results
        results = {
            'outputs': outputs,
            'inference_time_ms': inference_time,
            'batch_size': input_tensor.shape[0],
            'input_shape': input_tensor.shape,
            'device': str(self.device)
        }
        
        # Add stability metrics if available
        if 'stability' in locals():
            results['stability_metrics'] = stability
        
        return results
    
    @torch.no_grad()
    def infer_batch(self, images: List[Union[np.ndarray, torch.Tensor]]) -> List[Dict[str, Any]]:
        """
        Perform batched inference on multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of inference results for each image
        """
        batch_start = time.perf_counter()
        
        # Preprocess all images
        input_tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                from .preprocessing import ImagePreprocessor
                preprocessor = ImagePreprocessor(
                    target_size=(self.config.input_height, self.config.input_width)
                )
                tensor = preprocessor.process(img)
            else:
                tensor = img
            
            input_tensors.append(tensor)
        
        # Create batch
        batch = torch.stack(input_tensors).to(self.device)
        if self.use_half:
            batch = batch.half()
        
        # Perform inference
        with autocast(enabled=self.use_half):
            if hasattr(self, 'model_trt') and self.config.use_tensorrt:
                outputs = self.model_trt(batch)
            else:
                outputs = self.model(batch, task='detection')
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        batch_time = (time.perf_counter() - batch_start) * 1000
        
        # Split batch results
        results = []
        for i in range(len(images)):
            # Extract individual outputs
            batch_outputs = {}
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    batch_outputs[key] = value[i:i+1]  # Keep batch dimension
                elif isinstance(value, dict):
                    # Handle nested outputs
                    batch_outputs[key] = {
                        k: v[i:i+1] if isinstance(v, torch.Tensor) else v
                        for k, v in value.items()
                    }
                else:
                    batch_outputs[key] = value
            
            results.append({
                'outputs': batch_outputs,
                'batch_index': i,
                'batch_inference_time_ms': batch_time / len(images),
                'total_inference_time_ms': batch_time
            })
        
        return results
    
    def start_async_inference(self):
        """Start asynchronous inference processing."""
        if self.batch_thread is not None:
            logger.warning("Async inference already running")
            return
        
        self.batch_running = True
        self.batch_thread = threading.Thread(
            target=self._async_inference_loop,
            daemon=True
        )
        self.batch_thread.start()
        logger.info("Async inference started")
    
    def stop_async_inference(self):
        """Stop asynchronous inference."""
        self.batch_running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)
            self.batch_thread = None
        logger.info("Async inference stopped")
    
    def _async_inference_loop(self):
        """Async inference processing loop."""
        batch_buffer = []
        batch_timestamps = []
        
        while self.batch_running:
            try:
                # Wait for items with timeout
                try:
                    item = self.batch_queue.get(timeout=0.1)
                    batch_buffer.append(item)
                    batch_timestamps.append(time.time())
                except queue.Empty:
                    continue
                
                # Check if we should process batch
                batch_ready = (
                    len(batch_buffer) >= self.config.batch_size or
                    (batch_buffer and time.time() - batch_timestamps[0] > 0.033)  # ~30 FPS
                )
                
                if batch_ready:
                    # Extract images and callbacks
                    images = [item['image'] for item in batch_buffer]
                    callbacks = [item.get('callback') for item in batch_buffer]
                    
                    # Process batch
                    results = self.infer_batch(images)
                    
                    # Execute callbacks
                    for result, callback in zip(results, callbacks):
                        if callback:
                            try:
                                callback(result)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                    
                    # Clear buffer
                    batch_buffer.clear()
                    batch_timestamps.clear()
                    
            except Exception as e:
                logger.error(f"Error in async inference loop: {e}")
                time.sleep(0.01)  # Prevent tight loop on error
    
    def async_infer(self, image: np.ndarray, callback: Optional[callable] = None):
        """
        Queue image for async inference.
        
        Args:
            image: Input image
            callback: Optional callback function for results
        """
        if not self.batch_running:
            raise RuntimeError("Async inference not started")
        
        self.batch_queue.put({
            'image': image,
            'callback': callback,
            'timestamp': time.time()
        })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {}
        
        times = list(self.inference_times)
        stats = {
            'inference_time': {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99),
                'latest': times[-1] if times else 0,
            },
            'throughput_fps': 1000 / np.mean(times) if times else 0,
            'total_inferences': len(times),
        }
        
        if self.memory_usage:
            mem_stats = list(self.memory_usage)
            stats['memory_usage_mb'] = {
                'mean': np.mean(mem_stats),
                'max': np.max(mem_stats),
                'current': mem_stats[-1] if mem_stats else 0,
            }
        
        # Check latency constraints
        stats['latency_constraint_met'] = (
            stats['inference_time']['p95'] <= self.config.max_latency_ms
        )
        
        return stats
    
    def get_stability_report(self) -> Dict[str, Any]:
        """
        Get stability report from mHC layers.
        
        Returns:
            Dictionary with stability metrics
        """
        if not self.stability_metrics:
            return {}
        
        # Aggregate stability metrics
        report = {
            'eigenvalues': [],
            'signal_ratios': [],
            'gradient_norms': [],
            'total_checks': len(self.stability_metrics)
        }
        
        for metrics in self.stability_metrics[-100:]:  # Last 100 samples
            if 'max_eigenvalue' in metrics:
                report['eigenvalues'].append(metrics['max_eigenvalue'])
            if 'signal_ratio_mean' in metrics:
                report['signal_ratios'].append(metrics['signal_ratio_mean'])
            if 'gradient_norms' in metrics:
                report['gradient_norms'].extend(metrics['gradient_norms'])
        
        # Compute statistics
        if report['eigenvalues']:
            report['eigenvalue_stats'] = {
                'mean': np.mean(report['eigenvalues']),
                'std': np.std(report['eigenvalues']),
                'max': np.max(report['eigenvalues']),
            }
            # Check stability condition: eigenvalues ≤ 1
            report['is_stable'] = all(e <= 1.0 + 1e-3 for e in report['eigenvalues'])
        
        if report['signal_ratios']:
            report['signal_ratio_stats'] = {
                'mean': np.mean(report['signal_ratios']),
                'std': np.std(report['signal_ratios']),
            }
        
        return report
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_times.clear()
        self.memory_usage.clear()
        self.stability_metrics.clear()
        logger.info("Performance statistics reset")

class AsyncInferenceEngine(InferenceEngine):
    """
    Asynchronous inference engine with non-blocking operations.
    
    Features:
    1. Non-blocking inference calls
    2. Async/await support
    3. Concurrent request handling
    4. Rate limiting
    """
    
    def __init__(self, config: InferenceConfig, model: Optional[nn.Module] = None):
        super().__init__(config, model)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.request_queue = asyncio.Queue(maxsize=100)
        self.result_callbacks = {}
        self.request_counter = 0
        self.processing_task = None
        
    async def start_processing(self):
        """Start async processing loop."""
        if self.processing_task is not None:
            return
        
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Async processing started")
    
    async def stop_processing(self):
        """Stop async processing."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
        logger.info("Async processing stopped")
    
    async def _processing_loop(self):
        """Async processing loop."""
        while True:
            try:
                # Get request from queue
                request_id, image, future = await self.request_queue.get()
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self.infer,
                    image
                )
                
                # Set result on future
                future.set_result(result)
                
                # Mark task as done
                self.request_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    async def infer_async(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform async inference.
        
        Args:
            image: Input image
            
        Returns:
            Inference results
        """
        # Create future for result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Generate request ID
        request_id = self.request_counter
        self.request_counter += 1
        
        # Add to queue
        await self.request_queue.put((request_id, image, future))
        
        # Wait for result
        result = await future
        return result
    
    def infer_sync(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Synchronous inference wrapper for async engine.
        
        Args:
            image: Input image
            
        Returns:
            Inference results
        """
        # Run in current event loop or create new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run async inference
        return loop.run_until_complete(self.infer_async(image))