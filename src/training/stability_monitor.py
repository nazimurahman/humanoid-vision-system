# src/training/stability_monitor.py
"""
Training stability monitoring for mHC system.

Monitors:
1. Gradient norms and distributions
2. Eigenvalue stability of MHC matrices
3. Sinkhorn-Knopp convergence
4. Loss landscape curvature
5. Training dynamics
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from ..utils.sinkhorn import SinkhornKnoppProjection
from ..utils.manifold_ops import compute_eigenvalues

class StabilityMonitor:
    """
    Comprehensive stability monitoring for mHC training.
    
    Tracks and analyzes:
    - Gradient statistics
    - Weight updates
    - Manifold constraint satisfaction
    - Training convergence
    """
    
    def __init__(
        self,
        check_interval: int = 100,
        max_grad_norm: float = 1.0,
        max_eigenvalue: float = 1.1,
        sk_tolerance: float = 1e-4,
        buffer_size: int = 1000,
        log_dir: str = "logs/stability"
    ):
        """
        Initialize stability monitor.
        
        Args:
            check_interval: Steps between stability checks
            max_grad_norm: Maximum allowed gradient norm
            max_eigenvalue: Maximum allowed eigenvalue for H_res
            sk_tolerance: Tolerance for Sinkhorn-Knopp convergence
            buffer_size: Size of history buffer
            log_dir: Directory for saving logs
        """
        self.check_interval = check_interval
        self.max_grad_norm = max_grad_norm
        self.max_eigenvalue = max_eigenvalue
        self.sk_tolerance = sk_tolerance
        self.buffer_size = buffer_size
        
        # Create log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Monitoring buffers
        self.gradient_history = {
            'norms': [],
            'means': [],
            'stds': [],
            'maxes': [],
            'mins': []
        }
        
        self.eigenvalue_history = {
            'max': [],
            'min': [],
            'mean': [],
            'condition': []  # condition number
        }
        
        self.sk_convergence_history = {
            'errors': [],
            'iterations': [],
            'time': []
        }
        
        self.loss_history = {
            'values': [],
            'gradients': [],  # Loss gradient (difference)
            'curvature': []   # Second derivative estimate
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'gradient_explosion': 10.0,
            'gradient_vanishing': 1e-6,
            'eigenvalue_explosion': 2.0,
            'sk_divergence': 0.1,
            'loss_nan': True
        }
        
        # Alert history
        self.alerts = []
        
        # Statistics
        self.stats = {
            'checks_performed': 0,
            'alerts_triggered': 0,
            'last_check_time': time.time()
        }
    
    def check_stability(
        self,
        model: nn.Module,
        loss: Optional[float] = None,
        gradients: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive stability check.
        
        Args:
            model: Model to check
            loss: Current loss value
            gradients: Current gradients (optional)
            
        Returns:
            Stability report dictionary
        """
        self.stats['checks_performed'] += 1
        current_time = time.time()
        
        report = {
            'timestamp': current_time,
            'step': self.stats['checks_performed'],
            'unstable': False,
            'alerts': [],
            'metrics': {}
        }
        
        # 1. Check gradients
        grad_report = self._check_gradients(model, gradients)
        report['metrics'].update(grad_report)
        
        if grad_report.get('gradient_explosion', False):
            report['unstable'] = True
            report['alerts'].append('gradient_explosion')
        
        if grad_report.get('gradient_vanishing', False):
            report['alerts'].append('gradient_vanishing')
        
        # 2. Check MHC layer eigenvalues
        eigen_report = self._check_eigenvalues(model)
        report['metrics'].update(eigen_report)
        
        if eigen_report.get('eigenvalue_explosion', False):
            report['unstable'] = True
            report['alerts'].append('eigenvalue_explosion')
        
        # 3. Check Sinkhorn-Knopp convergence
        sk_report = self._check_sinkhorn_convergence(model)
        report['metrics'].update(sk_report)
        
        if sk_report.get('sk_divergence', False):
            report['alerts'].append('sk_divergence')
        
        # 4. Check loss stability
        if loss is not None:
            loss_report = self._check_loss_stability(loss)
            report['metrics'].update(loss_report)
            
            if loss_report.get('loss_nan_inf', False):
                report['unstable'] = True
                report['alerts'].append('loss_nan_inf')
        
        # Update history
        self._update_history(report['metrics'])
        
        # Save alert if unstable
        if report['unstable']:
            self.alerts.append({
                'step': report['step'],
                'time': report['timestamp'],
                'alerts': report['alerts'],
                'metrics': report['metrics']
            })
            self.stats['alerts_triggered'] += 1
            
            # Save detailed report
            self._save_alert_report(report)
        
        # Update statistics
        self.stats['last_check_time'] = current_time
        
        return report
    
    def _check_gradients(
        self,
        model: nn.Module,
        gradients: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Check gradient statistics."""
        grad_norms = []
        grad_means = []
        grad_stds = []
        
        if gradients is not None:
            # Use provided gradients
            for grad in gradients:
                if grad is not None:
                    grad_norm = grad.norm().item()
                    grad_norms.append(grad_norm)
                    grad_means.append(grad.mean().item())
                    grad_stds.append(grad.std().item())
        else:
            # Extract gradients from model parameters
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    grad_norm = grad.norm().item()
                    grad_norms.append(grad_norm)
                    grad_means.append(grad.mean().item())
                    grad_stds.append(grad.std().item())
        
        if not grad_norms:
            return {'no_gradients': True}
        
        # Compute statistics
        max_norm = max(grad_norms)
        min_norm = min(grad_norms)
        mean_norm = np.mean(grad_norms)
        std_norm = np.std(grad_norms)
        
        # Check for issues
        gradient_explosion = max_norm > self.alert_thresholds['gradient_explosion']
        gradient_vanishing = min_norm < self.alert_thresholds['gradient_vanishing']
        
        return {
            'grad_norm_max': max_norm,
            'grad_norm_min': min_norm,
            'grad_norm_mean': mean_norm,
            'grad_norm_std': std_norm,
            'gradient_explosion': gradient_explosion,
            'gradient_vanishing': gradient_vanishing,
            'grad_mean_mean': np.mean(grad_means),
            'grad_std_mean': np.mean(grad_stds)
        }
    
    def _check_eigenvalues(self, model: nn.Module) -> Dict[str, Any]:
        """Check eigenvalue stability of MHC layers."""
        eigenvalues_max = []
        eigenvalues_min = []
        eigenvalues_mean = []
        condition_numbers = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'H_res_raw'):
                # Get H_res matrix (after projection if available)
                if hasattr(module, 'H_res'):
                    H_res = module.H_res
                else:
                    H_res = module.H_res_raw
                
                # Compute eigenvalues
                eigvals = torch.linalg.eigvalsh(H_res.detach())
                eigvals_np = eigvals.cpu().numpy()
                
                eigenvalues_max.append(eigvals_np.max())
                eigenvalues_min.append(eigvals_np.min())
                eigenvalues_mean.append(eigvals_np.mean())
                
                # Condition number (max/min eigenvalue)
                condition = abs(eigvals_np.max()) / max(abs(eigvals_np.min()), 1e-8)
                condition_numbers.append(condition)
        
        if not eigenvalues_max:
            return {'no_eigenvalues': True}
        
        # Compute statistics
        max_eigenvalue = max(eigenvalues_max)
        min_eigenvalue = min(eigenvalues_min)
        mean_eigenvalue = np.mean(eigenvalues_mean)
        mean_condition = np.mean(condition_numbers)
        
        # Check for issues
        eigenvalue_explosion = max_eigenvalue > self.max_eigenvalue
        
        return {
            'eigenvalue_max': max_eigenvalue,
            'eigenvalue_min': min_eigenvalue,
            'eigenvalue_mean': mean_eigenvalue,
            'condition_mean': mean_condition,
            'eigenvalue_explosion': eigenvalue_explosion,
            'num_mhc_layers': len(eigenvalues_max)
        }
    
    def _check_sinkhorn_convergence(self, model: nn.Module) -> Dict[str, Any]:
        """Check Sinkhorn-Knopp projection convergence."""
        sk_errors = []
        sk_iterations = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'sinkhorn'):
                # Check if sinkhorn module tracks error
                if hasattr(module.sinkhorn, 'last_error'):
                    sk_errors.append(module.sinkhorn.last_error)
                
                if hasattr(module.sinkhorn, 'iterations'):
                    sk_iterations.append(module.sinkhorn.iterations)
        
        if not sk_errors:
            return {'no_sk_data': True}
        
        mean_error = np.mean(sk_errors)
        max_error = max(sk_errors)
        
        # Check for divergence
        sk_divergence = mean_error > self.sk_tolerance * 10  # 10x tolerance
        
        return {
            'sk_error_mean': mean_error,
            'sk_error_max': max_error,
            'sk_iterations_mean': np.mean(sk_iterations) if sk_iterations else 0,
            'sk_divergence': sk_divergence
        }
    
    def _check_loss_stability(self, loss: float) -> Dict[str, Any]:
        """Check loss stability and trends."""
        # Add to history
        self.loss_history['values'].append(loss)
        
        # Limit history size
        if len(self.loss_history['values']) > self.buffer_size:
            self.loss_history['values'].pop(0)
        
        # Compute gradient (first difference)
        if len(self.loss_history['values']) > 1:
            loss_gradient = self.loss_history['values'][-1] - self.loss_history['values'][-2]
            self.loss_history['gradients'].append(loss_gradient)
            
            # Compute curvature (second difference)
            if len(self.loss_history['gradients']) > 1:
                loss_curvature = self.loss_history['gradients'][-1] - self.loss_history['gradients'][-2]
                self.loss_history['curvature'].append(loss_curvature)
        
        # Check for NaN/Inf
        loss_nan_inf = not np.isfinite(loss)
        
        # Check for sudden jumps
        sudden_jump = False
        if len(self.loss_history['values']) > 10:
            recent_values = self.loss_history['values'][-10:]
            std_recent = np.std(recent_values)
            mean_recent = np.mean(recent_values)
            
            if std_recent > 0:
                z_score = abs(loss - mean_recent) / std_recent
                sudden_jump = z_score > 3.0  # 3 sigma
        
        return {
            'loss_current': loss,
            'loss_mean': np.mean(self.loss_history['values'][-10:]) if self.loss_history['values'] else loss,
            'loss_std': np.std(self.loss_history['values'][-10:]) if len(self.loss_history['values']) > 1 else 0,
            'loss_nan_inf': loss_nan_inf,
            'loss_sudden_jump': sudden_jump
        }
    
    def _update_history(self, metrics: Dict[str, Any]):
        """Update monitoring history."""
        # Gradient history
        if 'grad_norm_mean' in metrics:
            self.gradient_history['norms'].append(metrics['grad_norm_mean'])
            self.gradient_history['means'].append(metrics.get('grad_mean_mean', 0))
            self.gradient_history['stds'].append(metrics.get('grad_std_mean', 0))
            self.gradient_history['maxes'].append(metrics.get('grad_norm_max', 0))
            self.gradient_history['mins'].append(metrics.get('grad_norm_min', 0))
        
        # Eigenvalue history
        if 'eigenvalue_mean' in metrics:
            self.eigenvalue_history['max'].append(metrics.get('eigenvalue_max', 0))
            self.eigenvalue_history['min'].append(metrics.get('eigenvalue_min', 0))
            self.eigenvalue_history['mean'].append(metrics.get('eigenvalue_mean', 0))
            self.eigenvalue_history['condition'].append(metrics.get('condition_mean', 0))
        
        # SK convergence history
        if 'sk_error_mean' in metrics:
            self.sk_convergence_history['errors'].append(metrics.get('sk_error_mean', 0))
            self.sk_convergence_history['iterations'].append(metrics.get('sk_iterations_mean', 0))
            self.sk_convergence_history['time'].append(time.time())
    
    def _save_alert_report(self, report: Dict[str, Any]):
        """Save detailed alert report."""
        alert_file = self.log_dir / f"alert_step_{report['step']:06d}.json"
        
        with open(alert_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def generate_stability_report(self) -> Dict[str, Any]:
        """Generate comprehensive stability report."""
        report = {
            'statistics': self.stats.copy(),
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'current_status': {},
            'trends': {}
        }
        
        # Current status
        if self.gradient_history['norms']:
            report['current_status']['gradient_norm'] = self.gradient_history['norms'][-1]
        
        if self.eigenvalue_history['mean']:
            report['current_status']['eigenvalue_mean'] = self.eigenvalue_history['mean'][-1]
        
        if self.sk_convergence_history['errors']:
            report['current_status']['sk_error'] = self.sk_convergence_history['errors'][-1]
        
        # Trends
        for metric_name, history in [
            ('gradient_norms', self.gradient_history['norms']),
            ('eigenvalues', self.eigenvalue_history['mean']),
            ('sk_errors', self.sk_convergence_history['errors'])
        ]:
            if len(history) > 10:
                recent = history[-10:]
                report['trends'][metric_name] = {
                    'mean': np.mean(recent),
                    'std': np.std(recent),
                    'trend': self._compute_trend(recent)
                }
        
        return report
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def plot_stability_metrics(self, save_path: Optional[str] = None):
        """Plot stability metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Gradient norms
        if self.gradient_history['norms']:
            axes[0, 0].plot(self.gradient_history['norms'])
            axes[0, 0].axhline(y=self.max_grad_norm, color='r', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('Gradient Norms')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Norm')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Eigenvalues
        if self.eigenvalue_history['mean']:
            axes[0, 1].plot(self.eigenvalue_history['mean'], label='Mean')
            axes[0, 1].plot(self.eigenvalue_history['max'], label='Max', alpha=0.5)
            axes[0, 1].axhline(y=self.max_eigenvalue, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Eigenvalues')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # SK convergence
        if self.sk_convergence_history['errors']:
            axes[1, 0].plot(self.sk_convergence_history['errors'])
            axes[1, 0].axhline(y=self.sk_tolerance, color='r', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Sinkhorn-Knopp Error')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss
        if self.loss_history['values']:
            axes[1, 1].plot(self.loss_history['values'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def save_report(self, filename: str = "stability_report.json"):
        """Save comprehensive stability report."""
        report = self.generate_stability_report()
        report_path = self.log_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Stability report saved to {report_path}")

class TrainingStabilityMetrics:
    """
    Class to compute and track training stability metrics.
    
    Computes:
    1. Gradient statistics
    2. Weight update statistics
    3. Learning rate effectiveness
    4. Training efficiency metrics
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            'gradient': {},
            'weight': {},
            'learning': {},
            'efficiency': {}
        }
        
        self.history = {
            'gradient_norms': [],
            'weight_updates': [],
            'learning_rates': [],
            'loss_values': []
        }
    
    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: float,
        gradients: Optional[List[torch.Tensor]] = None
    ):
        """
        Update metrics with current training state.
        
        Args:
            model: Current model
            optimizer: Optimizer
            loss: Current loss value
            gradients: Current gradients (optional)
        """
        # Update history
        self.history['loss_values'].append(loss)
        
        # Compute gradient metrics
        self._compute_gradient_metrics(model, gradients)
        
        # Compute weight update metrics
        self._compute_weight_metrics(model)
        
        # Compute learning rate metrics
        self._compute_learning_metrics(optimizer)
        
        # Compute training efficiency
        self._compute_efficiency_metrics()
    
    def _compute_gradient_metrics(
        self,
        model: nn.Module,
        gradients: Optional[List[torch.Tensor]] = None
    ):
        """Compute gradient statistics."""
        grad_norms = []
        grad_means = []
        grad_stds = []
        
        if gradients is not None:
            grad_tensors = gradients
        else:
            grad_tensors = [p.grad for p in model.parameters() if p.grad is not None]
        
        for grad in grad_tensors:
            if grad is not None:
                grad_norms.append(grad.norm().item())
                grad_means.append(grad.mean().item())
                grad_stds.append(grad.std().item())
        
        if grad_norms:
            self.metrics['gradient'].update({
                'norm_mean': np.mean(grad_norms),
                'norm_std': np.std(grad_norms),
                'norm_max': max(grad_norms),
                'norm_min': min(grad_norms),
                'mean_mean': np.mean(grad_means),
                'std_mean': np.mean(grad_stds)
            })
            
            self.history['gradient_norms'].append(np.mean(grad_norms))
    
    def _compute_weight_metrics(self, model: nn.Module):
        """Compute weight update statistics."""
        weight_norms = []
        weight_updates = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Weight norm
                weight_norms.append(param.norm().item())
                
                # Update norm (approximation)
                if hasattr(param, 'grad'):
                    update_norm = param.grad.norm().item()
                    weight_updates.append(update_norm)
        
        if weight_norms:
            self.metrics['weight'].update({
                'norm_mean': np.mean(weight_norms),
                'norm_std': np.std(weight_norms),
                'update_mean': np.mean(weight_updates) if weight_updates else 0,
                'update_std': np.std(weight_updates) if weight_updates else 0,
                'update_ratio': np.mean(weight_updates) / np.mean(weight_norms) if weight_norms else 0
            })
            
            self.history['weight_updates'].append(np.mean(weight_updates) if weight_updates else 0)
    
    def _compute_learning_metrics(self, optimizer: torch.optim.Optimizer):
        """Compute learning rate statistics."""
        lrs = [group['lr'] for group in optimizer.param_groups]
        
        self.metrics['learning'].update({
            'lr_mean': np.mean(lrs),
            'lr_std': np.std(lrs),
            'lr_max': max(lrs),
            'lr_min': min(lrs)
        })
        
        self.history['learning_rates'].append(np.mean(lrs))
    
    def _compute_efficiency_metrics(self):
        """Compute training efficiency metrics."""
        if len(self.history['loss_values']) > 1:
            # Loss reduction rate
            recent_losses = self.history['loss_values'][-10:]
            if len(recent_losses) > 1:
                loss_reduction = recent_losses[0] - recent_losses[-1]
                loss_reduction_rate = loss_reduction / len(recent_losses)
                
                self.metrics['efficiency']['loss_reduction_rate'] = loss_reduction_rate
            
            # Gradient efficiency (loss reduction per gradient norm)
            if self.history['gradient_norms']:
                avg_gradient_norm = np.mean(self.history['gradient_norms'][-10:])
                if avg_gradient_norm > 0:
                    gradient_efficiency = loss_reduction_rate / avg_gradient_norm
                    self.metrics['efficiency']['gradient_efficiency'] = gradient_efficiency
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        summary = {
            'current': self.metrics.copy(),
            'trends': {}
        }
        
        # Compute trends
        for metric_name, history in self.history.items():
            if len(history) > 10:
                recent = history[-10:]
                summary['trends'][metric_name] = {
                    'mean': np.mean(recent),
                    'std': np.std(recent),
                    'trend': 'increasing' if recent[-1] > recent[0] else 'decreasing'
                }
        
        return summary
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'gradient': {},
            'weight': {},
            'learning': {},
            'efficiency': {}
        }
        
        self.history = {
            'gradient_norms': [],
            'weight_updates': [],
            'learning_rates': [],
            'loss_values': []
        }