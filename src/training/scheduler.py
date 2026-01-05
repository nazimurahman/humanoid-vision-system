# src/training/scheduler.py
"""
Learning rate schedulers with warmup and manifold stabilization.

Includes:
1. Cosine annealing with warmup
2. Plateau scheduler with reset capability
3. Manifold-aware scheduling
4. Cyclic learning rates for exploration
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List, Optional, Dict, Any
import numpy as np

class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Features:
    1. Linear warmup from warmup_lr to initial_lr
    2. Cosine annealing from initial_lr to min_lr
    3. Optional restarts
    4. Manifold stabilization during warmup
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        warmup_lr: float = 1e-5,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize cosine scheduler with warmup.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            min_lr: Minimum learning rate
            warmup_lr: Learning rate at start of warmup
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_lr = warmup_lr
        self.cosine_epochs = total_epochs - warmup_epochs
        
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> List[float]:
        """
        Compute learning rate using warmup + cosine schedule.
        
        Returns:
            List of learning rates for each parameter group
        """
        if not self._get_lr_called_within_step:
            raise RuntimeError(
                "should call scheduler.step() before scheduler.get_lr()"
            )
        
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            progress = self.last_epoch / max(self.warmup_epochs, 1)
            warmup_factor = progress
            
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            epoch_in_cosine = self.last_epoch - self.warmup_epochs
            progress = epoch_in_cosine / max(self.cosine_epochs, 1)
            
            # Cosine decay
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
    
    def step(self, epoch: Optional[int] = None):
        """
        Step the scheduler.
        
        Args:
            epoch: Epoch number (if None, uses self.last_epoch + 1)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        if self.verbose:
            print(f"Epoch {epoch}: learning rate = {self._last_lr[0]:.2e}")
    
    def get_current_stage(self) -> str:
        """Get current stage of scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return "warmup"
        else:
            return "cosine_annealing"
    
    def get_progress(self) -> Dict[str, float]:
        """Get progress metrics."""
        if self.last_epoch < self.warmup_epochs:
            progress = self.last_epoch / max(self.warmup_epochs, 1)
            stage = "warmup"
        else:
            epoch_in_cosine = self.last_epoch - self.warmup_epochs
            progress = epoch_in_cosine / max(self.cosine_epochs, 1)
            stage = "cosine_annealing"
        
        return {
            'stage': stage,
            'progress': progress,
            'current_lr': self._last_lr[0] if hasattr(self, '_last_lr') else self.base_lrs[0]
        }

class PlateauSchedulerWithReset(_LRScheduler):
    """
    Reduce learning rate on plateau with reset capability.
    
    Features:
    1. Reduce LR when metric plateaus
    2. Reset to initial LR after patience period
    3. Minimum LR bound
    4. Cooldown period after reset
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 1e-6,
        eps: float = 1e-8,
        reset_patience: int = 50,
        verbose: bool = False
    ):
        """
        Initialize plateau scheduler with reset.
        
        Args:
            optimizer: Wrapped optimizer
            mode: One of 'min', 'max'
            factor: Factor by which to reduce LR
            patience: Number of epochs with no improvement
            threshold: Threshold for measuring new optimum
            threshold_mode: One of 'rel', 'abs'
            cooldown: Number of epochs to wait before resuming normal operation
            min_lr: Lower bound on learning rate
            eps: Minimal decay applied to LR
            reset_patience: Patience before resetting to initial LR
            verbose: If True, prints a message for each update
        """
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.reset_patience = reset_patience
        
        self.cooldown_counter = 0
        self.wait = 0
        self.reset_wait = 0
        self.best = None
        self.num_bad_epochs = 0
        self.mode = mode
        self.best_epoch = 0
        
        self._reset()
        
        super().__init__(optimizer, verbose)
        
        self._init_is_better(mode, threshold, threshold_mode)
        
        # Store initial LRs for reset
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def _reset(self):
        """Reset scheduler state."""
        self.best = None
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.wait = 0
        self.reset_wait = 0
    
    def _init_is_better(self, mode, threshold, threshold_mode):
        """Initialize comparison function."""
        if mode not in {'min', 'max'}:
            raise ValueError(f"mode {mode} is unknown")
        
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError(f"threshold mode {threshold_mode} is unknown")
        
        if mode == 'min' and threshold_mode == 'rel':
            self.is_better = lambda a, best: a < best * (1 - threshold)
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
        elif mode == 'max' and threshold_mode == 'rel':
            self.is_better = lambda a, best: a > best * (1 + threshold)
        else:  # mode == 'max' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
    
    def step(self, metrics, epoch=None):
        """
        Step the scheduler based on metrics.
        
        Args:
            metrics: Metric to monitor
            epoch: Epoch number
        """
        current = float(metrics)
        
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if self.is_better(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            self.reset_wait = 0
        else:
            self.num_bad_epochs += 1
            self.reset_wait += 1
        
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        
        # Reset to initial LR if no improvement for reset_patience
        if self.reset_wait > self.reset_patience:
            self._reset_lr()
            self.reset_wait = 0
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
    def _reduce_lr(self, epoch):
        """Reduce learning rate."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                
                if self.verbose:
                    print(f"Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}")
    
    def _reset_lr(self):
        """Reset learning rate to initial values."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.initial_lrs[i]
            
            if self.verbose:
                print(f"Resetting learning rate of group {i} to {self.initial_lrs[i]:.4e}")
        
        self._reset()
    
    @property
    def in_cooldown(self):
        """Check if in cooldown period."""
        return self.cooldown_counter > 0
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get scheduler state information."""
        return {
            'best_metric': self.best,
            'best_epoch': self.best_epoch,
            'num_bad_epochs': self.num_bad_epochs,
            'reset_wait': self.reset_wait,
            'in_cooldown': self.in_cooldown,
            'current_lrs': self._last_lr if hasattr(self, '_last_lr') else self.base_lrs
        }

class ManifoldAwareScheduler:
    """
    Scheduler that adapts based on manifold stability metrics.
    
    Adjusts learning rates based on:
    1. Gradient norms in MHC layers
    2. Sinkhorn convergence error
    3. Eigenvalue stability
    4. Training loss curvature
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        stability_metrics: Dict[str, float],
        base_scheduler: _LRScheduler,
        mhc_lr_multiplier: float = 0.5,
        stability_thresholds: Dict[str, float] = None
    ):
        """
        Initialize manifold-aware scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            stability_metrics: Dictionary of stability metrics
            base_scheduler: Base learning rate scheduler
            mhc_lr_multiplier: Multiplier for MHC parameter learning rates
            stability_thresholds: Thresholds for stability metrics
        """
        self.optimizer = optimizer
        self.stability_metrics = stability_metrics
        self.base_scheduler = base_scheduler
        self.mhc_lr_multiplier = mhc_lr_multiplier
        
        if stability_thresholds is None:
            stability_thresholds = {
                'grad_norm': 1.0,
                'sk_error': 0.1,
                'eigenvalue_max': 1.1,
                'loss_curvature': 0.01
            }
        
        self.stability_thresholds = stability_thresholds
        
        # Track stability history
        self.stability_history = {
            'grad_norms': [],
            'sk_errors': [],
            'eigenvalues': [],
            'loss_curvatures': []
        }
    
    def step(self, metrics=None, stability_update=None):
        """
        Step the scheduler.
        
        Args:
            metrics: Loss/metric for base scheduler
            stability_update: Updated stability metrics
        """
        # Update stability metrics
        if stability_update is not None:
            self.stability_metrics.update(stability_update)
            
            # Add to history
            for key in self.stability_history:
                if key in stability_update:
                    self.stability_history[key].append(stability_update[key])
        
        # Adjust learning rates based on stability
        self._adjust_for_stability()
        
        # Step base scheduler
        if metrics is not None:
            self.base_scheduler.step(metrics)
    
    def _adjust_for_stability(self):
        """Adjust learning rates based on stability metrics."""
        # Check each stability metric
        adjustments = {}
        
        # Gradient norm adjustment
        if 'grad_norm' in self.stability_metrics:
            grad_norm = self.stability_metrics['grad_norm']
            threshold = self.stability_thresholds['grad_norm']
            
            if grad_norm > threshold:
                # Reduce LR if gradients are too large
                adjustments['lr_multiplier'] = max(0.5, threshold / grad_norm)
        
        # Sinkhorn convergence adjustment
        if 'sk_error' in self.stability_metrics:
            sk_error = self.stability_metrics['sk_error']
            threshold = self.stability_thresholds['sk_error']
            
            if sk_error > threshold:
                # MHC layers need more stable updates
                adjustments['mhc_multiplier'] = max(0.3, threshold / sk_error)
        
        # Apply adjustments
        if adjustments:
            self._apply_adjustments(adjustments)
    
    def _apply_adjustments(self, adjustments: Dict[str, float]):
        """Apply learning rate adjustments."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            # Check if this is an MHC parameter group
            is_mhc = any('mhc' in str(p) for p in param_group['params'])
            
            if is_mhc and 'mhc_multiplier' in adjustments:
                # Adjust MHC parameters separately
                multiplier = adjustments['mhc_multiplier']
                param_group['lr'] *= multiplier
            
            elif 'lr_multiplier' in adjustments:
                # Adjust all parameters
                multiplier = adjustments['lr_multiplier']
                param_group['lr'] *= multiplier
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Generate stability report."""
        report = {
            'current_metrics': self.stability_metrics.copy(),
            'thresholds': self.stability_thresholds.copy(),
            'history_stats': {}
        }
        
        # Compute statistics from history
        for key, values in self.stability_history.items():
            if values:
                report['history_stats'][key] = {
                    'mean': np.mean(values[-100:]),
                    'std': np.std(values[-100:]),
                    'max': max(values[-100:]),
                    'min': min(values[-100:])
                }
        
        return report
    
    def reset_stability(self):
        """Reset stability tracking."""
        self.stability_metrics.clear()
        for key in self.stability_history:
            self.stability_history[key].clear()