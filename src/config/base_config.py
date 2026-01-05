"""
Base configuration classes for the Humanoid Vision System.
Provides common configuration utilities and validation.
"""

import os
import yaml
import json
import torch
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import warnings
from pathlib import Path
import numpy as np

class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    TPU = "tpu"
    AUTO = "auto"

class PrecisionType(Enum):
    """Precision modes for training/inference."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    TF32 = "tf32"
    INT8 = "int8"

class TrainingMode(Enum):
    """Training modes."""
    STANDARD = "standard"
    DISTRIBUTED = "distributed"
    GRADIENT_ACCUMULATION = "grad_accum"
    MIXED_PRECISION = "mixed_precision"

class InferenceMode(Enum):
    """Inference modes."""
    STANDARD = "standard"
    QUANTIZED = "quantized"
    TRITON = "triton"
    TENSORRT = "tensorrt"

@dataclass
class BaseConfig:
    """
    Base configuration class with common settings.
    All configuration classes should inherit from this.
    """
    
    # =============== SYSTEM SETTINGS ===============
    seed: int = 42
    """Random seed for reproducibility."""
    
    device: DeviceType = DeviceType.AUTO
    """Device to run computations on."""
    
    num_workers: int = 4
    """Number of data loading workers."""
    
    pin_memory: bool = True
    """Pin memory for faster data transfer to GPU."""
    
    # =============== LOGGING & MONITORING ===============
    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""
    
    log_dir: str = "logs"
    """Directory for log files."""
    
    checkpoint_dir: str = "checkpoints"
    """Directory for model checkpoints."""
    
    tensorboard_dir: str = "runs"
    """Directory for TensorBoard logs."""
    
    wandb_project: Optional[str] = None
    """Weights & Biases project name."""
    
    wandb_entity: Optional[str] = None
    """Weights & Biases entity/username."""
    
    # =============== PERFORMANCE SETTINGS ===============
    precision: PrecisionType = PrecisionType.FP32
    """Numerical precision."""
    
    deterministic: bool = False
    """Make operations deterministic (slower)."""
    
    benchmark: bool = True
    """Enable CUDA benchmarking for optimal algorithm selection."""
    
    compile_mode: bool = False
    """Enable torch.compile for faster execution (PyTorch 2.0+)."""
    
    # =============== DATA SETTINGS ===============
    batch_size: int = 16
    """Batch size for training."""
    
    eval_batch_size: int = 8
    """Batch size for evaluation."""
    
    # =============== ADVANCED SETTINGS ===============
    gradient_checkpointing: bool = False
    """Enable gradient checkpointing to save memory."""
    
    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients."""
    
    # =============== CLASS METHODS ===============
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate()
        self._setup_device()
        self._create_directories()
    
    def _validate(self):
        """Validate configuration parameters."""
        if self.seed < 0:
            warnings.warn(f"Seed {self.seed} is negative, using absolute value")
            self.seed = abs(self.seed)
        
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
    
    def _setup_device(self):
        """Automatically detect and setup device."""
        if self.device == DeviceType.AUTO:
            if torch.cuda.is_available():
                self.device = DeviceType.CUDA
                torch.backends.cudnn.benchmark = self.benchmark
                if self.deterministic:
                    torch.backends.cudnn.deterministic = True
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = DeviceType.MPS
            else:
                self.device = DeviceType.CPU
        elif self.device == DeviceType.CUDA and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, falling back to CPU")
            self.device = DeviceType.CPU
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.log_dir,
            self.checkpoint_dir,
            self.tensorboard_dir,
            os.path.join(self.log_dir, "training"),
            os.path.join(self.log_dir, "inference"),
            os.path.join(self.log_dir, "deployment"),
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def save(self, filepath: str):
        """Save configuration to file."""
        filepath = Path(filepath)
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseConfig':
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Convert string enums back to Enum values
        for key, value in data.items():
            if key == 'device' and isinstance(value, str):
                data[key] = DeviceType(value)
            elif key == 'precision' and isinstance(value, str):
                data[key] = PrecisionType(value)
        
        return cls(**data)
    
    def get_torch_device(self) -> torch.device:
        """Get torch device object."""
        return torch.device(self.device.value)
    
    def get_compute_dtype(self) -> torch.dtype:
        """Get compute dtype based on precision setting."""
        if self.precision == PrecisionType.FP32:
            return torch.float32
        elif self.precision == PrecisionType.FP16:
            return torch.float16
        elif self.precision == PrecisionType.BF16:
            return torch.bfloat16
        elif self.precision == PrecisionType.TF32:
            return torch.float32  # TF32 is handled by CUDA
        else:
            return torch.float32
    
    def get_autocast_kwargs(self) -> Dict[str, Any]:
        """Get arguments for torch.autocast."""
        if self.precision == PrecisionType.FP16:
            return {'enabled': True, 'dtype': torch.float16}
        elif self.precision == PrecisionType.BF16:
            return {'enabled': True, 'dtype': torch.bfloat16}
        else:
            return {'enabled': False}
    
    def display(self):
        """Display configuration in a readable format."""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        
        for category, settings in self._categorize_settings().items():
            print(f"\n{category.upper()}:")
            print("-" * 40)
            for key, value in settings.items():
                print(f"  {key}: {value}")
        
        print("="*60)
    
    def _categorize_settings(self) -> Dict[str, Dict[str, Any]]:
        """Categorize settings for display."""
        categories = {
            'system': {},
            'performance': {},
            'data': {},
            'logging': {},
            'advanced': {}
        }
        
        # Categorize each setting
        for key, value in self.to_dict().items():
            if key in ['seed', 'device', 'num_workers', 'pin_memory']:
                categories['system'][key] = value
            elif key in ['precision', 'deterministic', 'benchmark', 'compile_mode']:
                categories['performance'][key] = value
            elif key in ['batch_size', 'eval_batch_size']:
                categories['data'][key] = value
            elif key in ['log_level', 'log_dir', 'checkpoint_dir', 
                        'tensorboard_dir', 'wandb_project', 'wandb_entity']:
                categories['logging'][key] = value
            elif key in ['gradient_checkpointing', 'gradient_accumulation_steps']:
                categories['advanced'][key] = value
        
        return categories