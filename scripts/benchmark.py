#!/usr/bin/env python3
"""
Performance benchmarking script for Hybrid Vision System.
Measures inference speed, memory usage, and power consumption.
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
import psutil
import GPUtil
from pathlib import Path
import json
import csv
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.inference_config import InferenceConfig
from models.hybrid_vision import HybridVisionSystem
from inference.engine import InferenceEngine
from inference.preprocessing import ImagePreprocessor
from utils.logging import setup_logging, get_logger
from utils.profiler import PerformanceProfiler

class SystemMonitor:
    """Monitor system resources during benchmarking."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.gpus = GPUtil.getGPUs() if GPUtil.getGPUs() else []
    
    def get_cpu_usage(self):
        """Get CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    def get_memory_usage(self):
        """Get memory usage in MB."""
        memory = self.process.memory_info()
        return {
            'rss_mb': memory.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory.vms / 1024 / 1024,  # Virtual Memory Size
            'shared_mb': memory.shared / 1024 / 1024 if hasattr(memory, 'shared') else 0
        }
    
    def get_gpu_usage(self, gpu_id=0):
        """Get GPU usage metrics."""
        if not self.gpus or gpu_id >= len(self.gpus):
            return None
        
        gpu = self.gpus[gpu_id]
        return {
            'name': gpu.name,
            'load_percent': gpu.load * 100,
            'memory_used_mb': gpu.memoryUsed,
            'memory_total_mb': gpu.memoryTotal,
            'memory_percent': gpu.memoryUtil * 100,
            'temperature_c': gpu.temperature
        }
    
    def get_all_metrics(self, gpu_id=0):
        """Get all system metrics."""
        return {
            'cpu_percent': self.get_cpu_usage(),
            'memory': self.get_memory_usage(),
            'gpu': self.get_gpu_usage(gpu_id),
            'timestamp': time.time()
        }

class BenchmarkRunner:
    """Run comprehensive benchmarks."""
    
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        
        # Create components
        self.preprocessor = ImagePreprocessor(
            image_size=config.image_size,
            normalize=config.normalize
        )
        
        self.engine = InferenceEngine(
            model=model,
            config=config,
            device=config.device
        )
        
        # System monitor
        self.monitor = SystemMonitor()
        
        # Results storage
        self.results = {
            'config': {
                'model_name': config.model_name,
                'image_size': config.image_size,
                'device': config.device,
                'precision': config.precision
            },
            'benchmarks': {}
        }
    
    def warmup(self, num_iterations=100):
        """Warm up the model."""
        
        self.logger.info(f"Warming up for {num_iterations} iterations")
        
        dummy_input = torch.randn(1, 3, self.config.image_size[0], 
                                 self.config.image_size[1]).to(self.config.device)
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize()
        self.logger.info("Warmup completed")
    
    def benchmark_inference_speed(self, num_iterations=1000, batch_sizes=None):
        """Benchmark inference speed for different batch sizes."""
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]
        
        self.logger.info("Benchmarking inference speed...")
        
        speed_results = {}
        
        for batch_size in batch_sizes:
            self.logger.info(f"  Batch size: {batch_size}")
            
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, self.config.image_size[0],
                                     self.config.image_size[1]).to(self.config.device)
            
            # Warm up for this batch size
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.perf_counter()
            
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_iterations
            avg_time_per_sample = avg_time_per_batch / batch_size
            
            fps = batch_size / avg_time_per_batch
            
            speed_results[batch_size] = {
                'total_time_seconds': total_time,
                'avg_batch_time_ms': avg_time_per_batch * 1000,
                'avg_sample_time_ms': avg_time_per_sample * 1000,
                'fps': fps,
                'iterations': num_iterations
            }
            
            self.logger.info(f"    FPS: {fps:.2f}, Time per sample: {avg_time_per_sample*1000:.2f} ms")
        
        self.results['benchmarks']['inference_speed'] = speed_results
        return speed_results
    
    def benchmark_memory_usage(self, batch_sizes=None):
        """Benchmark memory usage for different batch sizes."""
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]
        
        self.logger.info("Benchmarking memory usage...")
        
        memory_results = {}
        
        for batch_size in batch_sizes:
            self.logger.info(f"  Batch size: {batch_size}")
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Get baseline memory
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, self.config.image_size[0],
                                     self.config.image_size[1]).to(self.config.device)
            
            # Run inference
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            torch.cuda.synchronize()
            
            # Get peak memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            memory_increase = peak_memory - baseline_memory
            
            # Get system memory
            system_memory = self.monitor.get_memory_usage()
            
            memory_results[batch_size] = {
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': memory_increase,
                'system_memory_mb': system_memory['rss_mb'],
                'system_vms_mb': system_memory['vms_mb']
            }
            
            self.logger.info(f"    Peak GPU memory: {peak_memory:.2f} MB (+{memory_increase:.2f} MB)")
            
            # Clear cache for next iteration
            torch.cuda.empty_cache()
        
        self.results['benchmarks']['memory_usage'] = memory_results
        return memory_results
    
    def benchmark_power_consumption(self, duration=30, interval=0.1):
        """Benchmark power consumption during continuous inference."""
        
        self.logger.info(f"Benchmarking power consumption for {duration} seconds...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, self.config.image_size[0],
                                 self.config.image_size[1]).to(self.config.device)
        
        # Lists to store metrics
        timestamps = []
        cpu_usage = []
        memory_usage = []
        gpu_metrics = []
        
        # Benchmark loop
        start_time = time.time()
        iteration = 0
        
        with tqdm(total=duration, desc='Power benchmark') as pbar:
            while time.time() - start_time < duration:
                # Run inference
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                # Get system metrics
                metrics = self.monitor.get_all_metrics()
                
                timestamps.append(metrics['timestamp'])
                cpu_usage.append(metrics['cpu_percent'])
                memory_usage.append(metrics['memory']['rss_mb'])
                
                if metrics['gpu']:
                    gpu_metrics.append(metrics['gpu'])
                
                iteration += 1
                
                # Update progress
                elapsed = time.time() - start_time
                pbar.update(min(interval, duration - elapsed))
                
                time.sleep(max(0, interval - (time.time() - start_time - iteration * interval)))
        
        # Calculate statistics
        power_results = {
            'duration_seconds': duration,
            'iterations': iteration,
            'inference_rate_hz': iteration / duration,
            'cpu_usage_percent': {
                'mean': np.mean(cpu_usage),
                'std': np.std(cpu_usage),
                'max': np.max(cpu_usage),
                'min': np.min(cpu_usage)
            },
            'memory_usage_mb': {
                'mean': np.mean(memory_usage),
                'std': np.std(memory_usage),
                'max': np.max(memory_usage),
                'min': np.min(memory_usage)
            }
        }
        
        if gpu_metrics:
            gpu_load = [g['load_percent'] for g in gpu_metrics]
            gpu_memory = [g['memory_percent'] for g in gpu_metrics]
            
            power_results['gpu_usage'] = {
                'load_percent': {
                    'mean': np.mean(gpu_load),
                    'std': np.std(gpu_load),
                    'max': np.max(gpu_load),
                    'min': np.min(gpu_load)
                },
                'memory_percent': {
                    'mean': np.mean(gpu_memory),
                    'std': np.std(gpu_memory),
                    'max': np.max(gpu_memory),
                    'min': np.min(gpu_memory)
                }
            }
        
        self.results['benchmarks']['power_consumption'] = power_results
        return power_results
    
    def benchmark_end_to_end(self, image_path, num_iterations=100):
        """Benchmark end-to-end pipeline with real image."""
        
        self.logger.info(f"Benchmarking end-to-end pipeline with {image_path}")
        
        # Load and preprocess image
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Cannot load image: {image_path}")
            return None
        
        # Lists to store timing
        preprocess_times = []
        inference_times = []
        postprocess_times = []
        total_times = []
        
        # Benchmark loop
        for i in range(num_iterations):
            # Preprocess
            start_time = time.perf_counter()
            input_tensor = self.preprocessor.process(image)
            preprocess_time = time.perf_counter() - start_time
            
            # Inference
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = self.engine.inference(input_tensor)
            inference_time = time.perf_counter() - start_time
            
            # Postprocess (simulated)
            start_time = time.perf_counter()
            # Simulate postprocessing
            time.sleep(0.001)  # 1ms simulation
            postprocess_time = time.perf_counter() - start_time - 0.001
            
            total_time = preprocess_time + inference_time + postprocess_time
            
            preprocess_times.append(preprocess_time)
            inference_times.append(inference_time)
            postprocess_times.append(postprocess_time)
            total_times.append(total_time)
        
        # Calculate statistics
        end_to_end_results = {
            'num_iterations': num_iterations,
            'preprocess_ms': {
                'mean': np.mean(preprocess_times) * 1000,
                'std': np.std(preprocess_times) * 1000,
                'p95': np.percentile(preprocess_times, 95) * 1000
            },
            'inference_ms': {
                'mean': np.mean(inference_times) * 1000,
                'std': np.std(inference_times) * 1000,
                'p95': np.percentile(inference_times, 95) * 1000
            },
            'postprocess_ms': {
                'mean': np.mean(postprocess_times) * 1000,
                'std': np.std(postprocess_times) * 1000,
                'p95': np.percentile(postprocess_times, 95) * 1000
            },
            'total_ms': {
                'mean': np.mean(total_times) * 1000,
                'std': np.std(total_times) * 1000,
                'p95': np.percentile(total_times, 95) * 1000,
                'fps': 1000.0 / np.mean(total_times)
            }
        }
        
        self.results['benchmarks']['end_to_end'] = end_to_end_results
        
        self.logger.info(f"  End-to-end FPS: {end_to_end_results['total_ms']['fps']:.2f}")
        self.logger.info(f"  Inference time: {end_to_end_results['inference_ms']['mean']:.2f} ms")
        
        return end_to_end_results
    
    def run_all_benchmarks(self, output_dir):
        """Run all benchmarks and save results."""
        
        self.logger.info("Running all benchmarks...")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Warm up
        self.warmup()
        
        # Run benchmarks
        inference_speed = self.benchmark_inference_speed()
        memory_usage = self.benchmark_memory_usage()
        power_consumption = self.benchmark_power_consumption()
        
        # End-to-end benchmark if image provided
        if hasattr(self.config, 'benchmark_image'):
            end_to_end = self.benchmark_end_to_end(self.config.benchmark_image)
        
        # Add metadata
        self.results['metadata'] = {
            'benchmark_date': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
        
        # Save results
        self.save_results(output_dir)
        
        # Generate plots
        self.generate_plots(output_dir)
        
        self.logger.info("All benchmarks completed")
        
        return self.results
    
    def _get_system_info(self):
        """Get system information."""
        
        import platform
        
        return {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'gpus': [g.name for g in self.monitor.gpus] if self.monitor.gpus else []
        }
    
    def save_results(self, output_dir):
        """Save benchmark results to files."""
        
        # Save JSON
        json_path = output_dir / 'benchmark_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV summary
        csv_path = output_dir / 'benchmark_summary.csv'
        self._save_csv_summary(csv_path)
        
        # Save markdown report
        md_path = output_dir / 'benchmark_report.md'
        self._save_markdown_report(md_path)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _save_csv_summary(self, csv_path):
        """Save summary to CSV file."""
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Metric', 'Batch Size', 'Value', 'Unit'])
            
            # Inference speed
            if 'inference_speed' in self.results['benchmarks']:
                for batch_size, metrics in self.results['benchmarks']['inference_speed'].items():
                    writer.writerow(['FPS', batch_size, f"{metrics['fps']:.2f}", 'frames/sec'])
                    writer.writerow(['Inference Time', batch_size, 
                                   f"{metrics['avg_sample_time_ms']:.2f}", 'ms'])
            
            # Memory usage
            if 'memory_usage' in self.results['benchmarks']:
                for batch_size, metrics in self.results['benchmarks']['memory_usage'].items():
                    writer.writerow(['Peak GPU Memory', batch_size, 
                                   f"{metrics['peak_memory_mb']:.2f}", 'MB'])
    
    def _save_markdown_report(self, md_path):
        """Save markdown report."""
        
        with open(md_path, 'w') as f:
            f.write(f"# Benchmark Report\n\n")
            f.write(f"**Date:** {self.results['metadata']['benchmark_date']}\n\n")
            
            # System Info
            f.write(f"## System Information\n\n")
            f.write(f"- **System:** {self.results['metadata']['system_info']['system']}\n")
            f.write(f"- **CPU:** {self.results['metadata']['system_info']['processor']}\n")
            f.write(f"- **CPU Cores:** {self.results['metadata']['system_info']['cpu_count']}\n")
            f.write(f"- **Memory:** {self.results['metadata']['system_info']['total_memory_gb']:.1f} GB\n")
            f.write(f"- **GPUs:** {', '.join(self.results['metadata']['system_info']['gpus'])}\n\n")
            
            # Model Info
            f.write(f"## Model Information\n\n")
            f.write(f"- **Model Name:** {self.results['config']['model_name']}\n")
            f.write(f"- **Image Size:** {self.results['config']['image_size']}\n")
            f.write(f"- **Device:** {self.results['config']['device']}\n")
            f.write(f"- **Precision:** {self.results['config']['precision']}\n\n")
            
            # Inference Speed Results
            if 'inference_speed' in self.results['benchmarks']:
                f.write(f"## Inference Speed\n\n")
                f.write(f"| Batch Size | FPS | Time per Sample (ms) |\n")
                f.write(f"|------------|-----|---------------------|\n")
                
                for batch_size, metrics in self.results['benchmarks']['inference_speed'].items():
                    f.write(f"| {batch_size} | {metrics['fps']:.2f} | {metrics['avg_sample_time_ms']:.2f} |\n")
                f.write(f"\n")
            
            # Memory Usage Results
            if 'memory_usage' in self.results['benchmarks']:
                f.write(f"## Memory Usage\n\n")
                f.write(f"| Batch Size | Peak GPU Memory (MB) | Increase (MB) |\n")
                f.write(f"|------------|---------------------|---------------|\n")
                
                for batch_size, metrics in self.results['benchmarks']['memory_usage'].items():
                    f.write(f"| {batch_size} | {metrics['peak_memory_mb']:.2f} | {metrics['memory_increase_mb']:.2f} |\n")
                f.write(f"\n")
            
            # End-to-end Results
            if 'end_to_end' in self.results['benchmarks']:
                f.write(f"## End-to-End Performance\n\n")
                metrics = self.results['benchmarks']['end_to_end']
                f.write(f"- **FPS:** {metrics['total_ms']['fps']:.2f}\n")
                f.write(f"- **Total Time:** {metrics['total_ms']['mean']:.2f} ± {metrics['total_ms']['std']:.2f} ms\n")
                f.write(f"- **Inference Time:** {metrics['inference_ms']['mean']:.2f} ms\n")
                f.write(f"- **Preprocess Time:** {metrics['preprocess_ms']['mean']:.2f} ms\n")
                f.write(f"- **Postprocess Time:** {metrics['postprocess_ms']['mean']:.2f} ms\n\n")
    
    def generate_plots(self, output_dir):
        """Generate plots from benchmark results."""
        
        try:
            # Inference speed plot
            if 'inference_speed' in self.results['benchmarks']:
                plt.figure(figsize=(10, 6))
                
                batch_sizes = []
                fps_values = []
                
                for batch_size, metrics in self.results['benchmarks']['inference_speed'].items():
                    batch_sizes.append(batch_size)
                    fps_values.append(metrics['fps'])
                
                plt.plot(batch_sizes, fps_values, 'o-', linewidth=2)
                plt.xlabel('Batch Size')
                plt.ylabel('FPS')
                plt.title('Inference Speed vs Batch Size')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'inference_speed.png', dpi=150)
                plt.close()
            
            # Memory usage plot
            if 'memory_usage' in self.results['benchmarks']:
                plt.figure(figsize=(10, 6))
                
                batch_sizes = []
                memory_values = []
                
                for batch_size, metrics in self.results['benchmarks']['memory_usage'].items():
                    batch_sizes.append(batch_size)
                    memory_values.append(metrics['peak_memory_mb'])
                
                plt.plot(batch_sizes, memory_values, 's-', linewidth=2, color='red')
                plt.xlabel('Batch Size')
                plt.ylabel('Peak GPU Memory (MB)')
                plt.title('Memory Usage vs Batch Size')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'memory_usage.png', dpi=150)
                plt.close()
            
            self.logger.info(f"Plots saved to {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate plots: {e}")

def load_model_for_benchmark(checkpoint_path, config_path, device='cuda'):
    """Load model for benchmarking."""
    
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    inference_config = InferenceConfig(**config_dict.get('inference', {}))
    inference_config.device = device
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        from config.model_config import ModelConfig
        model_config = ModelConfig(**config_dict.get('model', {}))
    
    from models.hybrid_vision import HybridVisionSystem
    model = HybridVisionSystem(
        config=model_config,
        num_classes=model_config.num_classes,
        use_vit=model_config.use_vit,
        use_rag=model_config.use_rag
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, inference_config

def main(args):
    """Main benchmarking function."""
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting performance benchmark")
    
    # Load model
    model, config = load_model_for_benchmark(
        args.checkpoint,
        args.config,
        args.device
    )
    
    # Update config with command line arguments
    config.model_name = args.model_name or Path(args.checkpoint).stem
    
    if args.benchmark_image:
        config.benchmark_image = args.benchmark_image
    
    # Create benchmark runner
    runner = BenchmarkRunner(model, config, logger)
    
    # Run benchmarks
    results = runner.run_all_benchmarks(args.output_dir)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*50)
    
    if 'inference_speed' in results['benchmarks']:
        logger.info("Inference Speed (Batch Size 1):")
        metrics = results['benchmarks']['inference_speed'].get(1, {})
        if metrics:
            logger.info(f"  FPS: {metrics.get('fps', 0):.2f}")
            logger.info(f"  Time per sample: {metrics.get('avg_sample_time_ms', 0):.2f} ms")
    
    if 'memory_usage' in results['benchmarks']:
        logger.info("\nMemory Usage (Batch Size 1):")
        metrics = results['benchmarks']['memory_usage'].get(1, {})
        if metrics:
            logger.info(f"  Peak GPU Memory: {metrics.get('peak_memory_mb', 0):.2f} MB")
    
    logger.info("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Hybrid Vision System performance')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/inference.yaml',
                       help='Path to configuration file')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/benchmark',
                       help='Directory to save benchmark results')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name for reporting')
    
    # Benchmark options
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run benchmarks on')
    parser.add_argument('--benchmark-image', type=str, default=None,
                       help='Path to image for end-to-end benchmark')
    parser.add_argument('--batch-sizes', type=str, default='1,2,4,8,16',
                       help='Comma-separated list of batch sizes to test')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of iterations for speed benchmark')
    parser.add_argument('--power-duration', type=int, default=30,
                       help='Duration for power consumption benchmark (seconds)')
    
    # Precision
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'bf16'],
                       help='Precision for inference')
    
    args = parser.parse_args()
    
    main(args)