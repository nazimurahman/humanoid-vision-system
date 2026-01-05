#!/usr/bin/env python3
"""Health check script for Humanoid Vision System."""

import sys
import logging
import socket
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu() -> bool:
    """Check if GPU is available and functional."""
    try:
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available")
            return False
        
        # Check CUDA device count
        device_count = torch.cuda.device_count()
        if device_count == 0:
            logger.warning("No CUDA devices found")
            return False
        
        # Test a simple CUDA operation
        test_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        result = test_tensor.sum().item()
        
        logger.info(f"GPU check passed: {device_count} device(s), test result: {result}")
        return True
        
    except Exception as e:
        logger.error(f"GPU check failed: {e}")
        return False

def check_memory() -> bool:
    """Check if sufficient memory is available."""
    try:
        import psutil
        
        # Check system memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > 90:
            logger.warning(f"High memory usage: {memory_percent}%")
            return False
        
        # Check disk space
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        if disk_percent > 90:
            logger.warning(f"High disk usage: {disk_percent}%")
            return False
        
        logger.info(f"Memory check passed: RAM {memory_percent}%, Disk {disk_percent}%")
        return True
        
    except ImportError:
        logger.warning("psutil not available, skipping memory check")
        return True
    except Exception as e:
        logger.error(f"Memory check failed: {e}")
        return False

def check_ports() -> bool:
    """Check if required ports are available."""
    ports_to_check = [8000, 50051]  # REST API and gRPC
    
    for port in ports_to_check:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                logger.info(f"Port {port} is available")
            else:
                logger.warning(f"Port {port} is not available")
                return False
                
        except Exception as e:
            logger.error(f"Port check failed for port {port}: {e}")
            return False
    
    return True

def check_model() -> bool:
    """Check if model files exist and are valid."""
    model_path = Path('/models/vision_model.pt')
    
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}")
        return False
    
    try:
        # Try to load the model (without executing)
        model_size = model_path.stat().st_size
        if model_size < 1024:  # Less than 1KB
            logger.warning(f"Model file too small: {model_size} bytes")
            return False
        
        logger.info(f"Model check passed: {model_size / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        logger.error(f"Model check failed: {e}")
        return False

def check_dependencies() -> bool:
    """Check if required Python packages are available."""
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'fastapi',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False
    
    logger.info("Dependency check passed")
    return True

def main() -> int:
    """Run all health checks."""
    logger.info("Running health checks...")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("GPU", check_gpu),
        ("Memory", check_memory),
        ("Ports", check_ports),
        ("Model", check_model),
    ]
    
    all_passed = True
    results = {}
    
    for check_name, check_func in checks:
        try:
            passed = check_func()
            results[check_name] = passed
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{check_name}: {status}")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            logger.error(f"{check_name} check error: {e}")
            results[check_name] = False
            all_passed = False
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("HEALTH CHECK SUMMARY")
    logger.info("="*50)
    
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{check_name:<15} : {status}")
    
    logger.info("="*50)
    
    if all_passed:
        logger.info("✅ All health checks passed")
        return 0
    else:
        logger.error("❌ Some health checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())