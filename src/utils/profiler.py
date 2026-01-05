# src/utils/profiler.py
"""
Advanced Profiler for Robotic Vision System.

This module provides:
1. Comprehensive performance profiling (CPU, GPU, memory)
2. Bottleneck identification and analysis
3. Resource utilization monitoring
4. Automatic optimization suggestions
5. Real-time performance dashboards
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
import psutil
import GPUtil
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, field
from enum import Enum
import json
import matplotlib.pyplot as plt
from pathlib import Path

from .logging import get_logger

logger = get_logger()

class ProfilerState(Enum):
    """Profiler states."""
    IDLE = "idle"
    PROFILING = "profiling"
    ANALYZING = "analyzing"
    ERROR = "error"


@dataclass
class ProfileEvent:
    """Single profiling event."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    memory_allocated: Optional[float] = None
    memory_reserved: Optional[float] = None
    cpu_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_memory: Optional[float] = None
    children: List['ProfileEvent'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get event duration in seconds."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'duration': self.duration,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'memory_allocated': self.memory_allocated,
            'memory_reserved': self.memory_reserved,
            'cpu_percent': self.cpu_percent,
            'gpu_utilization': self.gpu_utilization,
            'gpu_memory': self.gpu_memory,
            'children': [child.to_dict() for child in self.children],
            'metadata': self.metadata
        }


class ResourceMonitor:
    """
    Monitor system resources (CPU, GPU, memory).
    
    Runs in a separate thread to continuously monitor resource usage.
    """
    
    def __init__(self, update_interval: float = 0.1):
        """
        Initialize resource monitor.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Resource histories
        self.cpu_percent_history = deque(maxlen=1000)
        self.memory_percent_history = deque(maxlen=1000)
        self.gpu_utilization_history = deque(maxlen=1000)
        self.gpu_memory_history = deque(maxlen=1000)
        
        # Process-specific tracking
        self.process = psutil.Process()
        
        # GPU availability
        self.has_gpu = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.has_gpu else 0
    
    def start(self):
        """Start resource monitoring thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Resource monitor started")
    
    def stop(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Resource monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = self.process.cpu_percent() / 100.0  # Normalize to [0, 1]
                self.cpu_percent_history.append(cpu_percent)
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_percent = memory_info.rss / psutil.virtual_memory().total
                self.memory_percent_history.append(memory_percent)
                
                # GPU usage
                if self.has_gpu:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # First GPU
                            self.gpu_utilization_history.append(gpu.load)
                            self.gpu_memory_history.append(gpu.memoryUtil)
                    except Exception as e:
                        logger.warning(f"GPU monitoring failed: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def get_current_stats(self) -> Dict[str, float]:
        """
        Get current resource statistics.
        
        Returns:
            Dictionary with current resource usage
        """
        stats = {
            'cpu_percent': self.cpu_percent_history[-1] if self.cpu_percent_history else 0.0,
            'memory_percent': self.memory_percent_history[-1] if self.memory_percent_history else 0.0,
        }
        
        if self.has_gpu:
            stats.update({
                'gpu_utilization': self.gpu_utilization_history[-1] if self.gpu_utilization_history else 0.0,
                'gpu_memory': self.gpu_memory_history[-1] if self.gpu_memory_history else 0.0,
            })
        
        return stats
    
    def get_history_stats(self, window: int = 100) -> Dict[str, Any]:
        """
        Get historical resource statistics.
        
        Args:
            window: Window size for statistics
            
        Returns:
            Dictionary with historical statistics
        """
        stats = {}
        
        # CPU statistics
        if self.cpu_percent_history:
            cpu_values = list(self.cpu_percent_history)[-window:]
            stats['cpu'] = {
                'mean': np.mean(cpu_values),
                'std': np.std(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
            }
        
        # Memory statistics
        if self.memory_percent_history:
            mem_values = list(self.memory_percent_history)[-window:]
            stats['memory'] = {
                'mean': np.mean(mem_values),
                'std': np.std(mem_values),
                'max': np.max(mem_values),
                'min': np.min(mem_values),
            }
        
        # GPU statistics
        if self.has_gpu:
            if self.gpu_utilization_history:
                gpu_util_values = list(self.gpu_utilization_history)[-window:]
                stats['gpu_utilization'] = {
                    'mean': np.mean(gpu_util_values),
                    'std': np.std(gpu_util_values),
                    'max': np.max(gpu_util_values),
                    'min': np.min(gpu_util_values),
                }
            
            if self.gpu_memory_history:
                gpu_mem_values = list(self.gpu_memory_history)[-window:]
                stats['gpu_memory'] = {
                    'mean': np.mean(gpu_mem_values),
                    'std': np.std(gpu_mem_values),
                    'max': np.max(gpu_mem_values),
                    'min': np.min(gpu_mem_values),
                }
        
        return stats


class ModelProfiler:
    """
    Profiler for PyTorch models.
    
    Provides detailed profiling of model execution including:
    - Layer-by-layer timing
    - Memory usage per layer
    - GPU utilization
    - Bottleneck identification
    """
    
    def __init__(self, model: nn.Module, use_cuda: bool = True):
        """
        Initialize model profiler.
        
        Args:
            model: PyTorch model to profile
            use_cuda: Whether to profile GPU usage
        """
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Hooks for profiling
        self.hooks = []
        self.profile_data = defaultdict(list)
        
        # Current profiling state
        self.current_event = None
        self.event_stack = []
        
        # Resource monitor
        self.resource_monitor = ResourceMonitor()
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward/backward hooks for all modules."""
        def make_hook(name):
            def forward_hook(module, input, output):
                self._record_event_start(f"{name}_forward", module, input)
                return output
            
            def backward_hook(module, grad_input, grad_output):
                self._record_event_start(f"{name}_backward", module, grad_input)
                return grad_input
            
            return forward_hook, backward_hook
        
        for name, module in self.model.named_modules():
            if name:  # Skip empty name (root module)
                forward_hook, backward_hook = make_hook(name)
                
                # Register hooks
                fhook = module.register_forward_hook(forward_hook)
                bhook = module.register_backward_hook(backward_hook)
                
                self.hooks.extend([fhook, bhook])
    
    def _record_event_start(self, event_name: str, module: nn.Module, input_data):
        """Record start of a profiling event."""
        # Get resource stats
        stats = self.resource_monitor.get_current_stats() if self.resource_monitor.monitoring else {}
        
        # Create new event
        event = ProfileEvent(
            name=event_name,
            start_time=time.time(),
            memory_allocated=torch.cuda.memory_allocated() / 1024**3 if self.use_cuda else None,
            memory_reserved=torch.cuda.memory_reserved() / 1024**3 if self.use_cuda else None,
            cpu_percent=stats.get('cpu_percent'),
            gpu_utilization=stats.get('gpu_utilization'),
            gpu_memory=stats.get('gpu_memory'),
            metadata={
                'module_type': module.__class__.__name__,
                'module_params': sum(p.numel() for p in module.parameters()),
            }
        )
        
        # Handle event hierarchy
        if self.current_event:
            self.current_event.children.append(event)
            self.event_stack.append(self.current_event)
        
        self.current_event = event
    
    def _record_event_end(self, output_data):
        """Record end of current profiling event."""
        if self.current_event:
            # Update end time and resource stats
            self.current_event.end_time = time.time()
            
            stats = self.resource_monitor.get_current_stats() if self.resource_monitor.monitoring else {}
            if self.current_event.memory_allocated is None and self.use_cuda:
                self.current_event.memory_allocated = torch.cuda.memory_allocated() / 1024**3
            
            # Add output metadata
            if output_data is not None:
                if isinstance(output_data, torch.Tensor):
                    self.current_event.metadata.update({
                        'output_shape': list(output_data.shape),
                        'output_dtype': str(output_data.dtype),
                    })
                elif isinstance(output_data, (tuple, list)):
                    shapes = []
                    for out in output_data:
                        if isinstance(out, torch.Tensor):
                            shapes.append(list(out.shape))
                    if shapes:
                        self.current_event.metadata['output_shapes'] = shapes
            
            # Restore previous event
            if self.event_stack:
                self.current_event = self.event_stack.pop()
            else:
                # Store completed event
                self.profile_data['events'].append(self.current_event)
                self.current_event = None
    
    def profile_forward(self, input_data, **kwargs) -> Tuple[Any, ProfileEvent]:
        """
        Profile forward pass.
        
        Args:
            input_data: Model input
            **kwargs: Additional forward arguments
            
        Returns:
            Tuple of (model_output, profile_event)
        """
        # Start resource monitoring
        if not self.resource_monitor.monitoring:
            self.resource_monitor.start()
        
        # Create root event
        root_event = ProfileEvent(
            name="forward_pass",
            start_time=time.time(),
            metadata={'input_shape': self._get_shape(input_data)}
        )
        
        self.current_event = root_event
        
        try:
            # Run forward pass
            with torch.no_grad():
                output = self.model(input_data, **kwargs)
            
            # Record end of forward pass
            self._record_event_end(output)
            root_event.end_time = time.time()
            
            # Ensure all events are properly closed
            while self.current_event:
                self._record_event_end(None)
            
        except Exception as e:
            root_event.metadata['error'] = str(e)
            root_event.end_time = time.time()
            raise
        finally:
            # Clean up
            self.current_event = None
            self.event_stack.clear()
        
        return output, root_event
    
    def profile_training_step(
        self,
        input_data,
        target_data,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        **kwargs
    ) -> Tuple[Dict[str, Any], ProfileEvent]:
        """
        Profile complete training step.
        
        Args:
            input_data: Model input
            target_data: Target data
            loss_fn: Loss function
            optimizer: Optimizer
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (training_metrics, profile_event)
        """
        # Start resource monitoring
        if not self.resource_monitor.monitoring:
            self.resource_monitor.start()
        
        # Create root event
        root_event = ProfileEvent(
            name="training_step",
            start_time=time.time()
        )
        
        self.current_event = root_event
        
        try:
            # Forward pass
            with self._profile_sub_event("forward"):
                output = self.model(input_data, **kwargs)
            
            # Loss computation
            with self._profile_sub_event("loss"):
                loss = loss_fn(output, target_data)
            
            # Backward pass
            with self._profile_sub_event("backward"):
                optimizer.zero_grad()
                loss.backward()
            
            # Optimization step
            with self._profile_sub_event("optimizer_step"):
                optimizer.step()
            
            # Record metrics
            root_event.metadata.update({
                'loss': loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0,
            })
            
            # Record end
            root_event.end_time = time.time()
            
        except Exception as e:
            root_event.metadata['error'] = str(e)
            root_event.end_time = time.time()
            raise
        finally:
            # Clean up
            self.current_event = None
            self.event_stack.clear()
        
        metrics = {
            'loss': loss.item() if 'loss' in root_event.metadata else 0.0,
        }
        
        return metrics, root_event
    
    def _profile_sub_event(self, name: str):
        """Context manager for profiling sub-events."""
        class SubEventContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
            
            def __enter__(self):
                # Create sub-event
                stats = self.profiler.resource_monitor.get_current_stats() if self.profiler.resource_monitor.monitoring else {}
                
                event = ProfileEvent(
                    name=self.name,
                    start_time=time.time(),
                    cpu_percent=stats.get('cpu_percent'),
                    gpu_utilization=stats.get('gpu_utilization'),
                    gpu_memory=stats.get('gpu_memory'),
                )
                
                # Add to current event
                if self.profiler.current_event:
                    self.profiler.current_event.children.append(event)
                    self.profiler.event_stack.append(self.profiler.current_event)
                
                self.profiler.current_event = event
                self.event = event
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                # End event
                self.event.end_time = time.time()
                
                # Restore previous event
                if self.profiler.event_stack:
                    self.profiler.current_event = self.profiler.event_stack.pop()
                else:
                    self.profiler.current_event = None
        
        return SubEventContext(self, name)
    
    def _get_shape(self, data) -> Any:
        """Get shape of input data."""
        if isinstance(data, torch.Tensor):
            return list(data.shape)
        elif isinstance(data, (tuple, list)):
            return [self._get_shape(d) for d in data]
        elif isinstance(data, dict):
            return {k: self._get_shape(v) for k, v in data.items()}
        else:
            return str(type(data))
    
    def analyze_bottlenecks(self, profile_event: ProfileEvent) -> Dict[str, Any]:
        """
        Analyze profiling data to identify bottlenecks.
        
        Args:
            profile_event: Root profile event
            
        Returns:
            Dictionary with bottleneck analysis
        """
        analysis = {
            'total_time': profile_event.duration,
            'bottlenecks': [],
            'recommendations': [],
            'layer_breakdown': [],
        }
        
        # Recursively collect all events
        def collect_events(event, depth=0):
            events = [(event, depth)]
            for child in event.children:
                events.extend(collect_events(child, depth + 1))
            return events
        
        all_events = collect_events(profile_event)
        
        # Find bottlenecks (events taking >10% of parent time)
        for event, depth in all_events:
            if depth > 0:  # Skip root
                # Calculate percentage of parent time
                parent = self._find_parent(event, profile_event)
                if parent and parent.duration > 0:
                    percentage = event.duration / parent.duration * 100
                    
                    if percentage > 10:  # Bottleneck threshold
                        analysis['bottlenecks'].append({
                            'name': event.name,
                            'duration': event.duration,
                            'percentage_of_parent': percentage,
                            'depth': depth,
                            'metadata': event.metadata,
                        })
        
        # Generate layer breakdown
        for event, depth in all_events:
            if 'module_type' in event.metadata:
                analysis['layer_breakdown'].append({
                    'name': event.name,
                    'module_type': event.metadata['module_type'],
                    'duration': event.duration,
                    'parameters': event.metadata.get('module_params', 0),
                    'memory_allocated': event.memory_allocated,
                })
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _find_parent(self, event: ProfileEvent, root: ProfileEvent) -> Optional[ProfileEvent]:
        """Find parent event for a given event."""
        if event in root.children:
            return root
        
        for child in root.children:
            parent = self._find_parent(event, child)
            if parent:
                return parent
        
        return None
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Check for GPU memory issues
        gpu_events = [e for e in analysis['layer_breakdown'] if e.get('memory_allocated', 0) > 1.0]  # >1GB
        if gpu_events:
            recommendations.append(
                "High GPU memory usage detected. Consider: "
                "1. Using gradient checkpointing "
                "2. Reducing batch size "
                "3. Using mixed precision training"
            )
        
        # Check for slow layers
        slow_layers = sorted(analysis['layer_breakdown'], key=lambda x: x['duration'], reverse=True)[:3]
        for layer in slow_layers:
            if layer['duration'] > 0.1:  # >100ms
                recommendations.append(
                    f"Slow layer detected: {layer['name']} ({layer['duration']:.3f}s). "
                    f"Consider optimizing {layer['module_type']} implementation."
                )
        
        # Check for parameter-heavy layers
        param_layers = sorted(analysis['layer_breakdown'], key=lambda x: x.get('parameters', 0), reverse=True)[:3]
        for layer in param_layers:
            if layer.get('parameters', 0) > 1_000_000:  # >1M parameters
                recommendations.append(
                    f"Parameter-heavy layer: {layer['name']} ({layer['parameters']:,} parameters). "
                    "Consider pruning or using more efficient architecture."
                )
        
        return recommendations
    
    def generate_report(self, profile_event: ProfileEvent, output_dir: str = "./profiler_reports"):
        """
        Generate comprehensive profiling report.
        
        Args:
            profile_event: Root profile event
            output_dir: Directory for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate analysis
        analysis = self.analyze_bottlenecks(profile_event)
        
        # Save JSON report
        report = {
            'profile_event': profile_event.to_dict(),
            'analysis': analysis,
            'resource_stats': self.resource_monitor.get_history_stats(),
            'timestamp': time.time(),
            'model_info': {
                'name': self.model.__class__.__name__,
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
        }
        
        json_path = output_dir / f"profile_report_{int(time.time())}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualization
        self._generate_visualization(profile_event, output_dir)
        
        # Print summary
        self._print_summary(analysis)
        
        logger.info(f"Profiling report saved to {json_path}")
        
        return report
    
    def _generate_visualization(self, profile_event: ProfileEvent, output_dir: Path):
        """Generate visualization plots."""
        try:
            # Timeline plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Collect events for plotting
            events_data = []
            def collect_for_plot(event, start_offset=0):
                events_data.append({
                    'name': event.name,
                    'start': event.start_time,
                    'end': event.end_time or event.start_time,
                    'duration': event.duration,
                    'depth': len(event.name.split('.')) - 1,
                })
                for child in event.children:
                    collect_for_plot(child, start_offset)
            
            collect_for_plot(profile_event)
            
            # Timeline plot
            ax = axes[0, 0]
            for i, event in enumerate(events_data):
                ax.barh(i, event['duration'], left=event['start'] - profile_event.start_time)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Event')
            ax.set_title('Execution Timeline')
            
            # Duration distribution
            ax = axes[0, 1]
            durations = [e['duration'] for e in events_data if e['duration'] > 0]
            if durations:
                ax.hist(durations, bins=20, edgecolor='black')
                ax.set_xlabel('Duration (s)')
                ax.set_ylabel('Count')
                ax.set_title('Duration Distribution')
            
            # Resource usage
            ax = axes[1, 0]
            resource_stats = self.resource_monitor.get_history_stats()
            if 'cpu' in resource_stats:
                times = np.arange(len(self.resource_monitor.cpu_percent_history))
                ax.plot(times, list(self.resource_monitor.cpu_percent_history), label='CPU')
            if 'gpu_utilization' in resource_stats:
                times = np.arange(len(self.resource_monitor.gpu_utilization_history))
                ax.plot(times, list(self.resource_monitor.gpu_utilization_history), label='GPU')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Utilization')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Memory usage
            ax = axes[1, 1]
            if self.resource_monitor.memory_percent_history:
                times = np.arange(len(self.resource_monitor.memory_percent_history))
                ax.plot(times, list(self.resource_monitor.memory_percent_history), label='CPU Memory')
            if self.resource_monitor.gpu_memory_history:
                times = np.arange(len(self.resource_monitor.gpu_memory_history))
                ax.plot(times, list(self.resource_monitor.gpu_memory_history), label='GPU Memory')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Memory Usage')
            ax.set_title('Memory Usage')
            ax.legend()
            
            plt.tight_layout()
            plot_path = output_dir / f"profile_plots_{int(time.time())}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")
    
    def _print_summary(self, analysis: Dict[str, Any]):
        """Print profiling summary to console."""
        print("\n" + "="*60)
        print("PROFILING SUMMARY")
        print("="*60)
        print(f"Total time: {analysis['total_time']:.3f}s")
        
        if analysis['bottlenecks']:
            print("\nBOTTLENECKS:")
            for bottleneck in analysis['bottlenecks'][:5]:  # Top 5
                print(f"  {bottleneck['name']}: {bottleneck['duration']:.3f}s "
                      f"({bottleneck['percentage_of_parent']:.1f}% of parent)")
        
        if analysis['recommendations']:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(analysis['recommendations'][:3], 1):  # Top 3
                print(f"  {i}. {rec}")
        
        print("="*60)
    
    def cleanup(self):
        """Clean up profiler resources."""
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Stop resource monitor
        if self.resource_monitor.monitoring:
            self.resource_monitor.stop()


class InferenceProfiler:
    """
    Specialized profiler for inference performance.
    
    Focuses on:
    - End-to-end latency breakdown
    - Batch size optimization
    - Memory footprint
    - Hardware utilization
    """
    
    def __init__(self, model: nn.Module, warmup_iterations: int = 10):
        """
        Initialize inference profiler.
        
        Args:
            model: Model to profile
            warmup_iterations: Number of warmup iterations
        """
        self.model = model
        self.warmup_iterations = warmup_iterations
        self.use_cuda = torch.cuda.is_available()
        
        # Profiling data
        self.latency_data = {
            'preprocess': [],
            'inference': [],
            'postprocess': [],
            'total': []
        }
        
        self.memory_data = {
            'peak_gpu_memory': [],
            'cpu_memory': [],
        }
        
        self.batch_size_data = {}
    
    def profile_inference(
        self,
        input_data,
        batch_sizes: List[int] = None,
        num_iterations: int = 100,
        profile_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Profile inference performance.
        
        Args:
            input_data: Sample input data
            batch_sizes: List of batch sizes to profile
            num_iterations: Number of iterations per batch size
            profile_memory: Whether to profile memory usage
            
        Returns:
            Dictionary with profiling results
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        results = {}
        
        # Warmup
        logger.info(f"Warming up for {self.warmup_iterations} iterations...")
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = self.model(input_data)
        
        if self.use_cuda:
            torch.cuda.synchronize()
        
        # Profile each batch size
        for batch_size in batch_sizes:
            logger.info(f"Profiling batch size {batch_size}...")
            
            # Create batched input
            if isinstance(input_data, torch.Tensor):
                batched_input = input_data.repeat(batch_size, *[1]*(input_data.dim()-1))
            elif isinstance(input_data, (tuple, list)):
                batched_input = [x.repeat(batch_size, *[1]*(x.dim()-1)) for x in input_data]
            else:
                raise TypeError(f"Unsupported input type: {type(input_data)}")
            
            # Profile
            batch_results = self._profile_single_batch(
                batched_input, num_iterations, profile_memory
            )
            
            self.batch_size_data[batch_size] = batch_results
            results[batch_size] = batch_results
        
        # Analyze results
        analysis = self._analyze_inference_results(results)
        
        return {
            'results': results,
            'analysis': analysis,
            'optimal_batch_size': analysis.get('optimal_batch_size', 1),
        }
    
    def _profile_single_batch(
        self,
        input_data,
        num_iterations: int,
        profile_memory: bool
    ) -> Dict[str, Any]:
        """
        Profile inference for a single batch size.
        
        Args:
            input_data: Batched input data
            num_iterations: Number of iterations
            profile_memory: Whether to profile memory
            
        Returns:
            Dictionary with profiling results
        """
        latencies = []
        memory_usage = []
        
        # Clear GPU cache
        if self.use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        for i in range(num_iterations):
            # Start timing
            start_time = time.time()
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_data)
            
            # Synchronize if using CUDA
            if self.use_cuda:
                torch.cuda.synchronize()
            
            # Record latency
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            # Record memory usage
            if profile_memory and self.use_cuda:
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                memory_usage.append(peak_memory)
            
            # Clear cache every 10 iterations
            if self.use_cuda and i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Compute statistics
        latencies = np.array(latencies)
        
        results = {
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'throughput': 1.0 / np.mean(latencies),
        }
        
        if memory_usage:
            results.update({
                'mean_memory_gb': np.mean(memory_usage),
                'max_memory_gb': np.max(memory_usage),
                'memory_std_gb': np.std(memory_usage),
            })
        
        return results
    
    def _analyze_inference_results(self, results: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Analyze inference profiling results.
        
        Args:
            results: Profiling results per batch size
            
        Returns:
            Analysis dictionary
        """
        analysis = {
            'batch_size_analysis': [],
            'optimal_batch_size': 1,
            'bottlenecks': [],
            'recommendations': [],
        }
        
        # Analyze batch size scaling
        batch_sizes = sorted(results.keys())
        throughputs = [results[bs]['throughput'] for bs in batch_sizes]
        latencies = [results[bs]['mean_latency'] for bs in batch_sizes]
        
        # Find optimal batch size (throughput/latency tradeoff)
        efficiency = [t / l for t, l in zip(throughputs, latencies)]
        optimal_idx = np.argmax(efficiency)
        analysis['optimal_batch_size'] = batch_sizes[optimal_idx]
        
        # Batch size analysis
        for i, bs in enumerate(batch_sizes):
            analysis['batch_size_analysis'].append({
                'batch_size': bs,
                'throughput': throughputs[i],
                'latency_ms': latencies[i] * 1000,
                'efficiency': efficiency[i],
                'memory_gb': results[bs].get('mean_memory_gb', 0),
            })
        
        # Identify bottlenecks
        if len(batch_sizes) > 1:
            # Check for sublinear scaling
            scaling_factors = []
            for i in range(1, len(throughputs)):
                scaling = throughputs[i] / throughputs[i-1]
                expected_scaling = batch_sizes[i] / batch_sizes[i-1]
                scaling_factors.append(scaling / expected_scaling)
            
            # Bottleneck if scaling factor < 0.8
            bottleneck_indices = [i for i, factor in enumerate(scaling_factors) if factor < 0.8]
            for idx in bottleneck_indices:
                bs1, bs2 = batch_sizes[idx], batch_sizes[idx + 1]
                analysis['bottlenecks'].append({
                    'type': 'sublinear_scaling',
                    'batch_sizes': f"{bs1}->{bs2}",
                    'scaling_factor': scaling_factors[idx],
                    'description': f"Throughput scaling is sublinear when increasing batch size from {bs1} to {bs2}"
                })
        
        # Generate recommendations
        if analysis['bottlenecks']:
            analysis['recommendations'].append(
                "Sublinear scaling detected. Consider: "
                "1. Increasing GPU memory bandwidth "
                "2. Optimizing data loading pipeline "
                "3. Using gradient accumulation for larger effective batches"
            )
        
        # Memory-based recommendations
        memory_limited = any(results[bs].get('max_memory_gb', 0) > 0.9 * self._get_available_gpu_memory() 
                           for bs in batch_sizes if bs in results)
        if memory_limited:
            analysis['recommendations'].append(
                "GPU memory limited. Consider: "
                "1. Reducing model size "
                "2. Using gradient checkpointing "
                "3. Implementing model parallelism"
            )
        
        return analysis
    
    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        if not self.use_cuda:
            return 0.0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryFree / 1024  # Convert MB to GB
        except:
            pass
        
        return 0.0
    
    def generate_report(self, results: Dict[str, Any], output_dir: str = "./inference_reports"):
        """
        Generate inference profiling report.
        
        Args:
            results: Profiling results
            output_dir: Output directory
            
        Returns:
            Path to generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        timestamp = int(time.time())
        json_path = output_dir / f"inference_profile_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate visualization
        self._generate_inference_plots(results, output_dir, timestamp)
        
        # Print summary
        self._print_inference_summary(results)
        
        logger.info(f"Inference profiling report saved to {json_path}")
        
        return json_path
    
    def _generate_inference_plots(self, results: Dict[str, Any], output_dir: Path, timestamp: int):
        """Generate inference profiling plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract data
            batch_analysis = results.get('analysis', {}).get('batch_size_analysis', [])
            
            if batch_analysis:
                batch_sizes = [item['batch_size'] for item in batch_analysis]
                throughputs = [item['throughput'] for item in batch_analysis]
                latencies = [item['latency_ms'] for item in batch_analysis]
                efficiencies = [item['efficiency'] for item in batch_analysis]
                memories = [item.get('memory_gb', 0) for item in batch_analysis]
                
                # Throughput vs Batch Size
                ax = axes[0, 0]
                ax.plot(batch_sizes, throughputs, 'o-', linewidth=2)
                ax.set_xlabel('Batch Size')
                ax.set_ylabel('Throughput (samples/s)')
                ax.set_title('Throughput vs Batch Size')
                ax.grid(True, alpha=0.3)
                
                # Latency vs Batch Size
                ax = axes[0, 1]
                ax.plot(batch_sizes, latencies, 'o-', linewidth=2, color='orange')
                ax.set_xlabel('Batch Size')
                ax.set_ylabel('Latency (ms)')
                ax.set_title('Latency vs Batch Size')
                ax.grid(True, alpha=0.3)
                
                # Efficiency vs Batch Size
                ax = axes[1, 0]
                ax.plot(batch_sizes, efficiencies, 'o-', linewidth=2, color='green')
                ax.set_xlabel('Batch Size')
                ax.set_ylabel('Efficiency (throughput/latency)')
                ax.set_title('Efficiency vs Batch Size')
                ax.grid(True, alpha=0.3)
                
                # Memory vs Batch Size
                if any(m > 0 for m in memories):
                    ax = axes[1, 1]
                    ax.plot(batch_sizes, memories, 'o-', linewidth=2, color='red')
                    ax.set_xlabel('Batch Size')
                    ax.set_ylabel('Memory Usage (GB)')
                    ax.set_title('Memory Usage vs Batch Size')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = output_dir / f"inference_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to generate inference plots: {e}")
    
    def _print_inference_summary(self, results: Dict[str, Any]):
        """Print inference profiling summary."""
        analysis = results.get('analysis', {})
        
        print("\n" + "="*60)
        print("INFERENCE PROFILING SUMMARY")
        print("="*60)
        
        if 'optimal_batch_size' in analysis:
            print(f"Optimal batch size: {analysis['optimal_batch_size']}")
        
        if 'batch_size_analysis' in analysis:
            print("\nBatch Size Analysis:")
            print("BS  | Throughput | Latency(ms) | Memory(GB)")
            print("-" * 45)
            for item in analysis['batch_size_analysis']:
                print(f"{item['batch_size']:3d} | "
                      f"{item['throughput']:10.2f} | "
                      f"{item['latency_ms']:11.2f} | "
                      f"{item.get('memory_gb', 0):10.2f}")
        
        if 'recommendations' in analysis and analysis['recommendations']:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("="*60)


# Global profiler instance
_global_profiler = None

def get_profiler(model: Optional[nn.Module] = None) -> ModelProfiler:
    """
    Get or create global profiler instance.
    
    Args:
        model: Model to profile (required for first call)
        
    Returns:
        ModelProfiler instance
    """
    global _global_profiler
    
    if _global_profiler is None:
        if model is None:
            raise ValueError("Model must be provided for first profiler creation")
        _global_profiler = ModelProfiler(model)
    
    return _global_profiler

def profile_function(func: Callable, num_iterations: int = 100) -> Dict[str, float]:
    """
    Profile a function's execution time.
    
    Args:
        func: Function to profile
        num_iterations: Number of iterations
        
    Returns:
        Dictionary with profiling results
    """
    times = []
    
    # Warmup
    for _ in range(10):
        func()
    
    # Profile
    for _ in range(num_iterations):
        start_time = time.time()
        func()
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'p95_time': np.percentile(times, 95),
        'p99_time': np.percentile(times, 99),
        'throughput': 1.0 / np.mean(times),
    }


def measure_memory_usage(func: Callable) -> Tuple[float, float]:
    """
    Measure memory usage of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Tuple of (memory_allocated_gb, memory_reserved_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run function
    func()
    
    # Get memory usage
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    return allocated, reserved