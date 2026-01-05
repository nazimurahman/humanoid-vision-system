# src/utils/logging.py
"""
Comprehensive Logging Utilities for Robotic Vision System.

This module provides:
1. Structured logging with multiple handlers
2. Performance monitoring and profiling
3. Training/Inference metrics tracking
4. TensorBoard and WandB integration
5. Log file rotation and management
"""

import logging
import logging.handlers
import time
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
import traceback
import inspect

import torch
import numpy as np
from colorama import init, Fore, Style
import wandb
from tensorboardX import SummaryWriter

# Initialize colorama for colored output
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with colors for different log levels.
    """
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        
        # Format the message
        return super().format(record)

class StructuredLogger:
    """
    Structured logger for robotic vision system.
    
    Supports:
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Multiple outputs (console, file, TensorBoard, WandB)
    - Structured JSON logging
    - Performance metrics tracking
    - Exception handling with traceback
    """
    
    def __init__(
        self,
        name: str = "hybrid_vision",
        log_dir: str = "./logs",
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Console log level
            file_level: File log level
            use_tensorboard: Enable TensorBoard logging
            use_wandb: Enable Weights & Biases logging
            wandb_project: WandB project name
            wandb_config: WandB configuration
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create console handler
        self._setup_console_handler(console_level)
        
        # Create file handler
        self._setup_file_handler(file_level)
        
        # Create error file handler
        self._setup_error_handler()
        
        # Create JSON handler for structured logging
        self._setup_json_handler()
        
        # Initialize TensorBoard
        self.tensorboard_writer = None
        if use_tensorboard:
            self._setup_tensorboard()
        
        # Initialize WandB
        self.wandb_run = None
        if use_wandb:
            self._setup_wandb(wandb_project, wandb_config)
        
        # Performance tracking
        self.timers = {}
        self.metrics_history = {}
        self.batch_times = []
        
        # Exception tracking
        self.error_count = 0
        self.warning_count = 0
        
        # Log initialization
        self.info("Logger initialized", extra={
            'log_dir': str(self.log_dir),
            'console_level': console_level,
            'file_level': file_level
        })
    
    def _setup_console_handler(self, level: str):
        """Setup console handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Colored format
        console_format = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, level: str):
        """Setup rotating file handler."""
        log_file = self.log_dir / f"{self.name}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        self.logger.addHandler(file_handler)
    
    def _setup_error_handler(self):
        """Setup separate error log file."""
        error_file = self.log_dir / f"{self.name}_errors.log"
        
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        
        error_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s\n%(exc_info)s'
        )
        error_handler.setFormatter(error_format)
        
        self.logger.addHandler(error_handler)
    
    def _setup_json_handler(self):
        """Setup JSON handler for structured logging."""
        json_file = self.log_dir / f"{self.name}_structured.jsonl"
        
        json_handler = logging.FileHandler(json_file)
        json_handler.setLevel(logging.INFO)
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'logger': record.name,
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                }
                
                # Add extra fields
                if hasattr(record, 'extra'):
                    log_record.update(record.extra)
                
                # Add exception info
                if record.exc_info:
                    log_record['exception'] = self.formatException(record.exc_info)
                
                return json.dumps(log_record)
        
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer."""
        tb_dir = self.log_dir / "tensorboard"
        tb_dir.mkdir(exist_ok=True)
        
        self.tensorboard_writer = SummaryWriter(str(tb_dir))
        self.info(f"TensorBoard initialized at {tb_dir}")
    
    def _setup_wandb(self, project: Optional[str], config: Optional[Dict]):
        """Setup Weights & Biases."""
        try:
            wandb.login()
            
            self.wandb_run = wandb.init(
                project=project or "hybrid-vision-system",
                config=config or {},
                dir=str(self.log_dir)
            )
            
            self.info(f"WandB initialized: {self.wandb_run.name}")
        except Exception as e:
            self.warning(f"Failed to initialize WandB: {e}")
            self.wandb_run = None
    
    # Logging methods with extra context
    def debug(self, message: str, **kwargs):
        """Log debug message with extra context."""
        self.logger.debug(message, extra={'extra': kwargs})
    
    def info(self, message: str, **kwargs):
        """Log info message with extra context."""
        self.logger.info(message, extra={'extra': kwargs})
        
        # Log to TensorBoard
        if self.tensorboard_writer and 'step' in kwargs:
            self._log_to_tensorboard('info', message, kwargs)
        
        # Log to WandB
        if self.wandb_run and 'step' in kwargs:
            self._log_to_wandb(kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra={'extra': kwargs})
        self.warning_count += 1
        
        # Add to TensorBoard
        if self.tensorboard_writer and 'step' in kwargs:
            self.tensorboard_writer.add_scalar(
                'logs/warnings', 1, kwargs['step']
            )
    
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Log error message with optional exception info."""
        self.logger.error(message, extra={'extra': kwargs}, exc_info=exc_info)
        self.error_count += 1
        
        # Add to TensorBoard
        if self.tensorboard_writer and 'step' in kwargs:
            self.tensorboard_writer.add_scalar(
                'logs/errors', 1, kwargs['step']
            )
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra={'extra': kwargs})
        
        # Also print to stderr for immediate attention
        print(f"\n{Fore.RED}CRITICAL: {message}{Style.RESET_ALL}", file=sys.stderr)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra={'extra': kwargs})
        self.error_count += 1
    
    # Performance monitoring
    def start_timer(self, name: str):
        """Start a named timer."""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and return elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.timers:
            self.warning(f"Timer '{name}' not found")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        # Log timing information
        self.debug(f"Timer '{name}' completed", 
                  elapsed_seconds=elapsed,
                  timer_name=name)
        
        return elapsed
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step/iteration number
        """
        # Store in history
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        self.metrics_history[name].append((step or len(self.metrics_history[name]), value))
        
        # Log to console
        self.info(f"Metric '{name}'", value=value, step=step)
        
        # Log to TensorBoard
        if self.tensorboard_writer and step is not None:
            self.tensorboard_writer.add_scalar(name, value, step)
        
        # Log to WandB
        if self.wandb_run and step is not None:
            wandb.log({name: value}, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_batch_time(self, batch_time: float):
        """Log batch processing time."""
        self.batch_times.append(batch_time)
        
        # Keep only recent times for stats
        if len(self.batch_times) > 100:
            self.batch_times.pop(0)
        
        # Log stats periodically
        if len(self.batch_times) % 10 == 0:
            avg_time = np.mean(self.batch_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            self.debug("Batch timing statistics",
                      avg_batch_time=avg_time,
                      fps=fps,
                      num_batches=len(self.batch_times))
    
    # Model-specific logging
    def log_model_parameters(self, model: torch.nn.Module):
        """Log model parameter statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info("Model parameters",
                  total_parameters=total_params,
                  trainable_parameters=trainable_params,
                  model_name=model.__class__.__name__)
    
    def log_gradient_norms(self, model: torch.nn.Module, step: int):
        """Log gradient norms for model parameters."""
        grad_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                grad_norms[f"grad_norm/{name}"] = norm
        
        if grad_norms:
            # Log individual norms
            for name, norm in grad_norms.items():
                self.log_metric(name, norm, step)
            
            # Log statistics
            norms = list(grad_norms.values())
            self.log_metric("grad_norm/mean", np.mean(norms), step)
            self.log_metric("grad_norm/max", np.max(norms), step)
            self.log_metric("grad_norm/min", np.min(norms), step)
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """Log current learning rate(s)."""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group.get('lr', 0)
            self.log_metric(f"learning_rate/group_{i}", lr, step)
    
    # TensorBoard/WandB helpers
    def _log_to_tensorboard(self, tag: str, message: str, kwargs: Dict):
        """Log text to TensorBoard."""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_text(
                f'logs/{tag}',
                f"{datetime.now().strftime('%H:%M:%S')} - {message}",
                kwargs['step']
            )
    
    def _log_to_wandb(self, metrics: Dict):
        """Log metrics to WandB."""
        if self.wandb_run:
            # Filter out non-scalar metrics
            scalar_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float, np.number))
            }
            
            if scalar_metrics:
                wandb.log(scalar_metrics, step=metrics.get('step'))
    
    # Visualization logging
    def log_image(self, tag: str, image: np.ndarray, step: int):
        """
        Log image to TensorBoard and WandB.
        
        Args:
            tag: Image tag/name
            image: Image array (H, W, C) or (C, H, W)
            step: Step number
        """
        # Ensure correct format
        if len(image.shape) == 3 and image.shape[0] == 3:  # (C, H, W)
            image = image.transpose(1, 2, 0)  # Convert to (H, W, C)
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_image(
                tag, image, step, dataformats='HWC'
            )
        
        # Log to WandB
        if self.wandb_run:
            wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """
        Log histogram of values.
        
        Args:
            tag: Histogram tag
            values: Values to histogram
            step: Step number
        """
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(tag, values, step)
        
        # Log to WandB
        if self.wandb_run:
            wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    # Cleanup
    def close(self):
        """Close all logging handlers and writers."""
        # Close TensorBoard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # Finish WandB run
        if self.wandb_run:
            wandb.finish()
        
        # Log summary
        self.info("Logger closing", 
                  total_errors=self.error_count,
                  total_warnings=self.warning_count,
                  total_logs=len(self.metrics_history))
        
        # Remove handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# Global logger instance
_global_logger = None

def setup_global_logger(**kwargs):
    """
    Setup global logger instance.
    
    Args:
        **kwargs: Arguments for StructuredLogger
        
    Returns:
        Global logger instance
    """
    global _global_logger
    _global_logger = StructuredLogger(**kwargs)
    return _global_logger

def get_logger() -> StructuredLogger:
    """
    Get global logger instance.
    
    Returns:
        Global logger instance
    """
    global _global_logger
    if _global_logger is None:
        # Create default logger
        _global_logger = StructuredLogger()
    return _global_logger

# Convenience functions for quick logging
def log_info(message: str, **kwargs):
    """Quick info log using global logger."""
    get_logger().info(message, **kwargs)

def log_error(message: str, **kwargs):
    """Quick error log using global logger."""
    get_logger().error(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Quick warning log using global logger."""
    get_logger().warning(message, **kwargs)

def log_debug(message: str, **kwargs):
    """Quick debug log using global logger."""
    get_logger().debug(message, **kwargs)