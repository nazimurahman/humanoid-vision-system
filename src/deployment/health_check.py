# src/deployment/health_check.py
"""
Comprehensive health checking and monitoring system.
Monitors model, GPU, system resources, and API endpoints.
"""

import os
import sys
import time
import psutil
import GPUtil
import threading
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
import statistics

import torch
import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.logging import setup_logger

logger = setup_logger("health_check")

# Prometheus metrics
HEALTH_STATUS = Gauge('vision_health_status', 'Health status (1=healthy, 0=unhealthy)')
GPU_MEMORY_USAGE = Gauge('vision_gpu_memory_usage_mb', 'GPU memory usage in MB', ['gpu_id'])
GPU_UTILIZATION = Gauge('vision_gpu_utilization_percent', 'GPU utilization percentage', ['gpu_id'])
CPU_USAGE = Gauge('vision_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('vision_memory_usage_mb', 'Memory usage in MB')
MODEL_LATENCY = Histogram('vision_model_latency_seconds', 'Model inference latency in seconds')
API_REQUESTS = Counter('vision_api_requests_total', 'Total API requests', ['endpoint', 'status'])

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: str  # "healthy", "warning", "critical"
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class SystemMetrics:
    """System metrics collection."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    gpu_metrics: List[Dict[str, Any]]
    disk_usage: Dict[str, float]
    network_io: Dict[str, float]
    load_average: Tuple[float, float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_mb': self.memory_mb,
            'gpu_metrics': self.gpu_metrics,
            'disk_usage': self.disk_usage,
            'network_io': self.network_io,
            'load_average': self.load_average
        }

class ModelHealthChecker:
    """Health checker for vision model."""
    
    def __init__(self, inference_engine):
        """Initialize model health checker."""
        self.inference_engine = inference_engine
        self.latency_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        self.last_check_time = time.time()
        
        # Thresholds
        self.latency_threshold_ms = 100.0  # 100ms
        self.memory_threshold_mb = 8000.0  # 8GB
        self.error_rate_threshold = 0.05  # 5%
        
    def check_model(self) -> HealthCheckResult:
        """Check model health."""
        checks = []
        
        # Check 1: Model loaded
        if self.inference_engine.model is None:
            checks.append(("model_loaded", "critical", "Model not loaded"))
        else:
            checks.append(("model_loaded", "healthy", "Model loaded"))
        
        # Check 2: GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            checks.append(("gpu_available", "healthy", f"{gpu_count} GPU(s) available"))
        else:
            checks.append(("gpu_available", "warning", "No GPU available, using CPU"))
        
        # Check 3: Memory usage
        gpu_memory = self._get_gpu_memory()
        if gpu_memory["used_mb"] > self.memory_threshold_mb:
            checks.append(("gpu_memory", "warning", 
                          f"High GPU memory usage: {gpu_memory['used_mb']:.0f}MB"))
        else:
            checks.append(("gpu_memory", "healthy", 
                          f"GPU memory OK: {gpu_memory['used_mb']:.0f}MB/{gpu_memory['total_mb']:.0f}MB"))
        
        # Check 4: Latency
        avg_latency = self._get_average_latency()
        if avg_latency > self.latency_threshold_ms:
            checks.append(("latency", "warning", 
                          f"High latency: {avg_latency:.1f}ms"))
        else:
            checks.append(("latency", "healthy", 
                          f"Latency OK: {avg_latency:.1f}ms"))
        
        # Check 5: Error rate
        error_rate = self._get_error_rate()
        if error_rate > self.error_rate_threshold:
            checks.append(("error_rate", "critical", 
                          f"High error rate: {error_rate*100:.1f}%"))
        else:
            checks.append(("error_rate", "healthy", 
                          f"Error rate OK: {error_rate*100:.1f}%"))
        
        # Determine overall status
        status_counts = {"critical": 0, "warning": 0, "healthy": 0}
        for _, status, _ in checks:
            status_counts[status] += 1
        
        if status_counts["critical"] > 0:
            overall_status = "critical"
        elif status_counts["warning"] > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Create result
        metrics = {
            "average_latency_ms": avg_latency,
            "error_rate": error_rate,
            "gpu_memory_mb": gpu_memory,
            "inference_count": len(self.latency_history)
        }
        
        details = {name: {"status": status, "message": msg} 
                  for name, status, msg in checks}
        
        return HealthCheckResult(
            component="model",
            status=overall_status,
            message=f"Model health: {overall_status}",
            metrics=metrics,
            details=details
        )
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage."""
        if not torch.cuda.is_available():
            return {"used_mb": 0, "total_mb": 0, "percent": 0}
        
        try:
            used = torch.cuda.memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            percent = (used / total) * 100
            
            return {
                "used_mb": used,
                "total_mb": total,
                "percent": percent
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
            return {"used_mb": 0, "total_mb": 0, "percent": 0}
    
    def _get_average_latency(self) -> float:
        """Get average inference latency."""
        if not self.latency_history:
            return 0.0
        return statistics.mean(self.latency_history)
    
    def _get_error_rate(self) -> float:
        """Get error rate from history."""
        if not self.error_history:
            return 0.0
        return sum(self.error_history) / len(self.error_history)
    
    def record_inference(self, latency_ms: float, success: bool = True):
        """Record inference for health monitoring."""
        self.latency_history.append(latency_ms)
        self.error_history.append(0 if success else 1)

class SystemHealthChecker:
    """Health checker for system resources."""
    
    def __init__(self, check_interval: float = 5.0):
        """Initialize system health checker."""
        self.check_interval = check_interval
        self.metrics_history = deque(maxlen=100)
        self.last_metrics = None
        
        # Thresholds
        self.cpu_threshold = 90.0  # 90%
        self.memory_threshold = 90.0  # 90%
        self.disk_threshold = 90.0  # 90%
        self.gpu_temp_threshold = 85.0  # 85°C
        
        # Start background monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        def monitor_loop():
            while True:
                try:
                    metrics = self.collect_metrics()
                    self.metrics_history.append(metrics)
                    self.last_metrics = metrics
                    
                    # Update Prometheus metrics
                    CPU_USAGE.set(metrics.cpu_percent)
                    MEMORY_USAGE.set(metrics.memory_mb)
                    
                    for gpu_metric in metrics.gpu_metrics:
                        GPU_MEMORY_USAGE.labels(gpu_id=gpu_metric['id']).set(gpu_metric['memory_used'])
                        GPU_UTILIZATION.labels(gpu_id=gpu_metric['id']).set(gpu_metric['utilization'])
                    
                except Exception as e:
                    logger.error(f"Failed to collect system metrics: {e}")
                
                time.sleep(self.check_interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info("Started system health monitoring")
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / 1024**2
        
        # GPU metrics
        gpu_metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'utilization': gpu.load * 100,
                    'temperature': gpu.temperature
                })
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
        
        # Disk metrics
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    'total_gb': usage.total / 1024**3,
                    'used_gb': usage.used / 1024**3,
                    'percent': usage.percent
                }
            except Exception as e:
                logger.warning(f"Failed to get disk usage for {partition.mountpoint}: {e}")
        
        # Network metrics
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
        # Load average
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            gpu_metrics=gpu_metrics,
            disk_usage=disk_usage,
            network_io=network_io,
            load_average=load_avg
        )
    
    def check_system(self) -> HealthCheckResult:
        """Check system health."""
        if not self.last_metrics:
            self.last_metrics = self.collect_metrics()
        
        metrics = self.last_metrics
        checks = []
        
        # Check 1: CPU usage
        if metrics.cpu_percent > self.cpu_threshold:
            checks.append(("cpu_usage", "warning", 
                          f"High CPU usage: {metrics.cpu_percent:.1f}%"))
        else:
            checks.append(("cpu_usage", "healthy", 
                          f"CPU usage OK: {metrics.cpu_percent:.1f}%"))
        
        # Check 2: Memory usage
        if metrics.memory_percent > self.memory_threshold:
            checks.append(("memory_usage", "warning", 
                          f"High memory usage: {metrics.memory_percent:.1f}%"))
        else:
            checks.append(("memory_usage", "healthy", 
                          f"Memory usage OK: {metrics.memory_percent:.1f}%"))
        
        # Check 3: GPU temperature
        for gpu in metrics.gpu_metrics:
            if gpu['temperature'] > self.gpu_temp_threshold:
                checks.append((f"gpu_{gpu['id']}_temp", "warning",
                              f"High GPU {gpu['id']} temperature: {gpu['temperature']:.1f}°C"))
        
        # Check 4: Disk space
        for mountpoint, usage in metrics.disk_usage.items():
            if usage['percent'] > self.disk_threshold:
                checks.append((f"disk_{mountpoint}", "warning",
                              f"Low disk space on {mountpoint}: {usage['percent']:.1f}% used"))
        
        # Determine overall status
        status_counts = {"critical": 0, "warning": 0, "healthy": 0}
        for _, status, _ in checks:
            status_counts[status] += 1
        
        if status_counts["warning"] > 2:  # Multiple warnings = critical
            overall_status = "critical"
        elif status_counts["warning"] > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Create result
        metrics_data = {
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_mb": metrics.memory_mb,
            "gpu_count": len(metrics.gpu_metrics),
            "disk_usage": {k: v['percent'] for k, v in metrics.disk_usage.items()}
        }
        
        details = {name: {"status": status, "message": msg} 
                  for name, status, msg in checks}
        
        return HealthCheckResult(
            component="system",
            status=overall_status,
            message=f"System health: {overall_status}",
            metrics=metrics_data,
            details=details
        )
    
    def get_metrics_history(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get metrics history for the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = []
        
        for metrics in self.metrics_history:
            if metrics.timestamp >= cutoff_time:
                recent_metrics.append(metrics.to_dict())
        
        return recent_metrics

class APIChecker:
    """Health checker for API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API checker."""
        self.base_url = base_url
        self.endpoints = [
            "/health",
            "/detect",
            "/metrics",
            "/models"
        ]
        
        self.response_times = {endpoint: deque(maxlen=100) for endpoint in self.endpoints}
        self.error_counts = {endpoint: 0 for endpoint in self.endpoints}
        self.total_requests = {endpoint: 0 for endpoint in self.endpoints}
    
    def check_endpoints(self) -> HealthCheckResult:
        """Check all API endpoints."""
        import requests
        
        checks = []
        
        for endpoint in self.endpoints:
            url = self.base_url + endpoint
            self.total_requests[endpoint] += 1
            
            try:
                start_time = time.time()
                
                # Special handling for detect endpoint (needs image)
                if endpoint == "/detect":
                    # Create dummy request
                    response = requests.post(
                        url,
                        json={
                            "image_base64": "",
                            "confidence_threshold": 0.5
                        },
                        timeout=5
                    )
                else:
                    response = requests.get(url, timeout=5)
                
                response_time = (time.time() - start_time) * 1000
                self.response_times[endpoint].append(response_time)
                
                if response.status_code == 200:
                    checks.append((endpoint, "healthy", 
                                  f"Response OK: {response_time:.1f}ms"))
                else:
                    self.error_counts[endpoint] += 1
                    checks.append((endpoint, "warning", 
                                  f"Response {response.status_code}: {response_time:.1f}ms"))
                    
            except Exception as e:
                self.error_counts[endpoint] += 1
                checks.append((endpoint, "critical", f"Failed: {str(e)}"))
        
        # Determine overall status
        status_counts = {"critical": 0, "warning": 0, "healthy": 0}
        for _, status, _ in checks:
            status_counts[status] += 1
        
        if status_counts["critical"] > 0:
            overall_status = "critical"
        elif status_counts["warning"] > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Calculate metrics
        metrics = {}
        for endpoint in self.endpoints:
            if self.response_times[endpoint]:
                avg_time = statistics.mean(self.response_times[endpoint])
                error_rate = self.error_counts[endpoint] / max(self.total_requests[endpoint], 1)
                metrics[endpoint] = {
                    "average_response_time_ms": avg_time,
                    "error_rate": error_rate,
                    "total_requests": self.total_requests[endpoint]
                }
        
        details = {name: {"status": status, "message": msg} 
                  for name, status, msg in checks}
        
        return HealthCheckResult(
            component="api",
            status=overall_status,
            message=f"API health: {overall_status}",
            metrics=metrics,
            details=details
        )

class HealthChecker:
    """Main health checker combining all components."""
    
    def __init__(self, inference_engine=None, api_base_url: str = "http://localhost:8000"):
        """Initialize comprehensive health checker."""
        self.model_checker = ModelHealthChecker(inference_engine) if inference_engine else None
        self.system_checker = SystemHealthChecker()
        self.api_checker = APIChecker(api_base_url)
        
        self.check_history = deque(maxlen=100)
        self.overall_status = "unknown"
        
        logger.info("Health checker initialized")
    
    def check_all(self) -> Dict[str, Any]:
        """Perform all health checks."""
        results = {}
        
        # Check model if available
        if self.model_checker:
            model_result = self.model_checker.check_model()
            results["model"] = model_result.to_dict()
        else:
            results["model"] = {
                "component": "model",
                "status": "unknown",
                "message": "Model checker not available",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Check system
        system_result = self.system_checker.check_system()
        results["system"] = system_result.to_dict()
        
        # Check API
        api_result = self.api_checker.check_endpoints()
        results["api"] = api_result.to_dict()
        
        # Determine overall status
        status_priority = {"critical": 3, "warning": 2, "healthy": 1, "unknown": 0}
        all_statuses = [
            results["model"]["status"],
            results["system"]["status"],
            results["api"]["status"]
        ]
        
        # Get highest priority status
        overall_status = max(all_statuses, key=lambda x: status_priority[x])
        self.overall_status = overall_status
        
        # Update Prometheus metric
        HEALTH_STATUS.set(1 if overall_status == "healthy" else 0)
        
        # Store in history
        self.check_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status,
            "results": results
        })
        
        return {
            "healthy": overall_status == "healthy",
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
            "system_metrics": self.system_checker.last_metrics.to_dict() if self.system_checker.last_metrics else {}
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        if self.system_checker.last_metrics:
            return self.system_checker.last_metrics.to_dict()
        return {}
    
    def get_check_history(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get health check history."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_checks = []
        
        for check in self.check_history:
            if datetime.fromisoformat(check["timestamp"]) >= cutoff_time:
                recent_checks.append(check)
        
        return recent_checks
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary."""
        if not self.check_history:
            return {"status": "unknown", "last_check": None}
        
        last_check = self.check_history[-1]
        
        return {
            "status": self.overall_status,
            "last_check": last_check["timestamp"],
            "components": {
                "model": last_check["results"]["model"]["status"],
                "system": last_check["results"]["system"]["status"],
                "api": last_check["results"]["api"]["status"]
            }
        }
    
    def run_continuous_monitoring(self, interval: float = 30.0):
        """Run continuous monitoring in background."""
        def monitoring_loop():
            while True:
                try:
                    self.check_all()
                    logger.debug(f"Health check completed: {self.overall_status}")
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                
                time.sleep(interval)
        
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
        logger.info(f"Started continuous health monitoring (interval: {interval}s)")

# Command-line interface
def main():
    """Command-line health checker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health Checker for Vision System")
    parser.add_argument("--check", choices=["all", "model", "system", "api"], 
                       default="all", help="What to check")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="API base URL")
    parser.add_argument("--model-path", help="Path to model checkpoint")
    parser.add_argument("--continuous", action="store_true", 
                       help="Run continuous monitoring")
    parser.add_argument("--interval", type=float, default=30.0,
                       help="Check interval in seconds (continuous mode)")
    parser.add_argument("--output", choices=["json", "text"], default="text",
                       help="Output format")
    
    args = parser.parse_args()
    
    # Load inference engine if needed
    inference_engine = None
    if args.model_path and os.path.exists(args.model_path):
        try:
            from src.inference.engine import VisionInferenceEngine
            inference_engine = VisionInferenceEngine(args.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    # Create health checker
    checker = HealthChecker(inference_engine, args.api_url)
    
    if args.continuous:
        print(f"Starting continuous health monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop")
        
        checker.run_continuous_monitoring(args.interval)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped monitoring")
            
    else:
        # Run single check
        if args.check == "all":
            result = checker.check_all()
        elif args.check == "model" and checker.model_checker:
            result = checker.model_checker.check_model().to_dict()
        elif args.check == "system":
            result = checker.system_checker.check_system().to_dict()
        elif args.check == "api":
            result = checker.api_checker.check_endpoints().to_dict()
        else:
            print(f"Check '{args.check}' not available")
            return
        
        # Output results
        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            # Pretty print
            if args.check == "all":
                print(f"Overall Status: {result['status'].upper()}")
                print(f"Timestamp: {result['timestamp']}")
                print()
                
                for component, comp_result in result['results'].items():
                    status = comp_result['status']
                    color = {
                        'healthy': '\033[92m',  # Green
                        'warning': '\033[93m',  # Yellow
                        'critical': '\033[91m',  # Red
                        'unknown': '\033[90m'   # Gray
                    }.get(status, '')
                    
                    print(f"{component.upper()}: {color}{status.upper()}\033[0m")
                    print(f"  Message: {comp_result['message']}")
                    
                    if 'metrics' in comp_result:
                        for key, value in comp_result['metrics'].items():
                            if isinstance(value, dict):
                                print(f"  {key}:")
                                for k2, v2 in value.items():
                                    print(f"    {k2}: {v2}")
                            else:
                                print(f"  {key}: {value}")
                    
                    print()
            else:
                print(f"Status: {result['status'].upper()}")
                print(f"Component: {result['component']}")
                print(f"Message: {result['message']}")
                print(f"Timestamp: {result['timestamp']}")
                
                if 'metrics' in result:
                    print("\nMetrics:")
                    for key, value in result['metrics'].items():
                        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()