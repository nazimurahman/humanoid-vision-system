# src/deployment/__init__.py
"""
Deployment module for Hybrid Vision System.

Provides:
- REST API server (FastAPI)
- gRPC server for robots
- Model serving with Triton/TorchServe
- Health monitoring and metrics
"""

__version__ = "1.0.0"
__author__ = "Hybrid Vision System Team"
__license__ = "Apache 2.0"

# Expose main classes and functions
from .api_server import VisionAPIServer, run_server as run_api_server
from .grpc_server import RobotGRPCServer, run_server as run_grpc_server
from .model_server import (
    ModelServerManager, ModelExporter, TritonModelServer,
    ModelFormat, ServingBackend, ModelConfig, export_and_serve
)
from .health_check import (
    HealthChecker, ModelHealthChecker, SystemHealthChecker, APIChecker,
    HealthCheckResult, SystemMetrics
)

__all__ = [
    # API Server
    "VisionAPIServer",
    "run_api_server",
    
    # gRPC Server
    "RobotGRPCServer",
    "run_grpc_server",
    
    # Model Serving
    "ModelServerManager",
    "ModelExporter",
    "TritonModelServer",
    "ModelFormat",
    "ServingBackend",
    "ModelConfig",
    "export_and_serve",
    
    # Health Checking
    "HealthChecker",
    "ModelHealthChecker",
    "SystemHealthChecker",
    "APIChecker",
    "HealthCheckResult",
    "SystemMetrics",
]

# Configuration file paths
import os

CONFIG_PATHS = {
    "api_server": "configs/api_server.yaml",
    "grpc_server": "configs/grpc_server.yaml",
    "triton": "configs/triton.yaml",
    "torchserve": "configs/torchserve.yaml",
    "inference": "configs/inference.yaml",
}

def get_config_path(config_name: str) -> str:
    """Get path to configuration file."""
    if config_name in CONFIG_PATHS:
        return CONFIG_PATHS[config_name]
    else:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIG_PATHS.keys())}")

def check_deployment_requirements() -> bool:
    """Check if all deployment requirements are met."""
    import subprocess
    import sys
    
    requirements = [
        ("torch", lambda: subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"], 
                                        capture_output=True, text=True).returncode == 0),
        ("fastapi", lambda: subprocess.run([sys.executable, "-c", "import fastapi"], 
                                          capture_output=True).returncode == 0),
        ("grpcio", lambda: subprocess.run([sys.executable, "-c", "import grpc"], 
                                         capture_output=True).returncode == 0),
        ("uvicorn", lambda: subprocess.run([sys.executable, "-c", "import uvicorn"], 
                                          capture_output=True).returncode == 0),
    ]
    
    missing = []
    for req, check in requirements:
        if not check():
            missing.append(req)
    
    if missing:
        print(f"Missing requirements: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

# Initialize on import
print(f"Hybrid Vision System Deployment Module v{__version__}")
print(f"Available components: {', '.join(__all__)}")