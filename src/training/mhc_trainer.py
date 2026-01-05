# src/training/mhc_trainer.py
"""
Manifold-Constrained Hyper-Connection Trainer.

Implements the complete training pipeline with:
1. Mixed precision training with bfloat16/float32
2. Gradient clipping with manifold awareness
3. Sinkhorn-Knopp convergence monitoring
4. Stability-preserving optimization
5. Multi-task learning for hybrid vision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from tqdm import tqdm
import wandb
import time
import logging
from pathlib import Path
import json

from ..utils.sinkhorn import SinkhornKnoppProjection
from ..utils.manifold_ops import compute_manifold_constraints
from .loss_functions import MHCYOLOLoss, MultiTaskLoss
from .optimizer import ManifoldAwareOptimizer
from .scheduler import CosineAnnealingWithWarmup
from .stability_monitor import StabilityMonitor

logger = logging.getLogger(__name__)

class ManifoldConstrainedTrainer:
    """
    Advanced trainer implementing mHC methodology.
    
    Key Features:
    - Enforces doubly stochastic constraints via Sinkhorn-Knopp
    - Monitors gradient norms and eigenvalue stability
    - Supports mixed precision with safe coefficient casting
    - Implements curriculum learning for robotic tasks
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = 'cuda',
        experiment_name: Optional[str] = None
    ):
        """
        Initialize mHC trainer.
        
        Args:
            model: Hybrid vision model with MHC layers
            config: Training configuration dictionary
            device: Training device ('cuda' or 'cpu')
            experiment_name: Optional name for experiment tracking
        """
        self.model = model
        self.config = config
        self.device = device
        self.experiment_name = experiment_name or f"mhc_train_{int(time.time())}"
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Mixed precision setup
        self.use_amp = config.get('use_mixed_precision', True)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Manifold constraint parameters
        self.sk_iterations = config.get('sinkhorn_iterations', 20)
        self.sk_epsilon = config.get('sinkhorn_epsilon', 1e-8)
        self.sk_temperature = config.get('sinkhorn_temperature', 1.0)
        
        # Loss functions
        self.detection_loss = MHCYOLOLoss(
            num_classes=config.get('num_classes', 80),
            anchors=config.get('anchors'),
            lambda_coord=config.get('lambda_coord', 5.0),
            lambda_obj=config.get('lambda_obj', 1.0),
            lambda_noobj=config.get('lambda_noobj', 0.5),
            lambda_cls=config.get('lambda_cls', 1.0)
        )
        
        self.multitask_loss = MultiTaskLoss(
            task_weights=config.get('task_weights', {'detection': 1.0, 'classification': 0.5}),
            manifold_weight=config.get('manifold_weight', 0.01)
        )
        
        # Optimizer with manifold constraints
        self.optimizer = ManifoldAwareOptimizer(
            model_params=model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
            mhc_params=config.get('mhc_optimizer', {}),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = CosineAnnealingWithWarmup(
            optimizer=self.optimizer,
            warmup_epochs=config.get('warmup_epochs', 10),
            total_epochs=config.get('total_epochs', 100),
            min_lr=config.get('min_lr', 1e-6),
            warmup_lr=config.get('warmup_lr', 1e-5)
        )
        
        # Stability monitoring
        self.stability_monitor = StabilityMonitor(
            check_interval=config.get('stability_check_interval', 100),
            max_grad_norm=config.get('max_grad_norm', 1.0),
            max_eigenvalue=config.get('max_eigenvalue', 1.1)
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'gradient_norms': [],
            'stability_metrics': [],
            'sk_convergence': []
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        logger.info(f"Initialized mHC trainer on device: {device}")
        logger.info(f"Using mixed precision: {self.use_amp}")
        logger.info(f"Sinkhorn iterations: {self.sk_iterations}")
        
    def train_epoch(
        self,
        train_loader,
        epoch: int,
        progress_bar: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch with mHC constraints.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'detection_loss': 0.0,
            'classification_loss': 0.0,
            'manifold_loss': 0.0,
            'grad_norm': 0.0,
            'sk_error': 0.0
        }
        
        num_batches = len(train_loader)
        batch_iter = tqdm(train_loader, desc=f'Epoch {epoch}') if progress_bar else train_loader
        
        for batch_idx, batch_data in enumerate(batch_iter):
            # Unpack batch data (supports multiple formats)
            if isinstance(batch_data, (list, tuple)):
                images, targets = batch_data
            else:
                images = batch_data['images']
                targets = batch_data['targets']
            
            # Move to device
            images = images.to(self.device)
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = targets.to(self.device)
            
            # Training step
            step_metrics = self.train_step(images, targets)
            
            # Update epoch metrics
            for key in epoch_metrics:
                if key in step_metrics:
                    epoch_metrics[key] += step_metrics[key]
            
            # Update progress bar
            if progress_bar:
                batch_iter.set_postfix({
                    'loss': step_metrics['total_loss'],
                    'grad': step_metrics['grad_norm'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Global step update
            self.global_step += 1
            
            # Stability check
            if self.global_step % self.stability_monitor.check_interval == 0:
                stability_report = self.stability_monitor.check_stability(self.model)
                self.history['stability_metrics'].append(stability_report)
                
                if stability_report['unstable']:
                    logger.warning(f"Stability issue detected at step {self.global_step}")
                    logger.warning(f"Report: {stability_report}")
                    
                    # Apply corrective measures
                    self._apply_stability_corrections()
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train_step(
        self,
        images: torch.Tensor,
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Single training step with mHC constraints.
        
        Args:
            images: Batch of images [B, C, H, W]
            targets: Target tensors or dictionary
            
        Returns:
            Dictionary of step metrics
        """
        step_metrics = {}
        
        # Mixed precision forward pass
        with autocast(enabled=self.use_amp):
            # Forward pass
            outputs = self.model(images, task='detection')
            
            # Compute detection loss
            det_loss_dict = self.detection_loss(outputs['detections'], targets)
            
            # Compute manifold regularization loss
            manifold_loss = self._compute_manifold_regularization()
            
            # Total loss
            total_loss = (
                det_loss_dict['total'] +
                manifold_loss * self.config.get('manifold_weight', 0.01)
            )
            
            # Add classification loss if available
            if 'classifications' in outputs:
                cls_loss = F.cross_entropy(outputs['classifications'], targets.get('labels'))
                total_loss += cls_loss * self.config.get('classification_weight', 0.5)
                step_metrics['classification_loss'] = cls_loss.item()
        
        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        
        # Apply manifold-aware gradient clipping
        grad_norm = self._apply_manifold_gradient_clipping()
        
        # Optimizer step with scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Learning rate scheduling
        self.scheduler.step(self.global_step)
        
        # Collect metrics
        step_metrics.update({
            'total_loss': total_loss.item(),
            'detection_loss': det_loss_dict['total'].item(),
            'bbox_loss': det_loss_dict.get('bbox', 0.0),
            'obj_loss': det_loss_dict.get('obj', 0.0),
            'cls_loss': det_loss_dict.get('cls', 0.0),
            'manifold_loss': manifold_loss.item(),
            'grad_norm': grad_norm,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })
        
        # Check Sinkhorn convergence
        sk_error = self._check_sinkhorn_convergence()
        step_metrics['sk_error'] = sk_error
        
        # Update history
        self.history['gradient_norms'].append(grad_norm)
        self.history['sk_convergence'].append(sk_error)
        
        return step_metrics
    
    def _compute_manifold_regularization(self) -> torch.Tensor:
        """
        Compute manifold regularization loss.
        
        Encourages doubly stochastic constraints and stable eigenvalues.
        """
        reg_loss = 0.0
        num_mhc_layers = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'H_res_raw'):
                # Get the raw H_res parameter
                H_res_raw = module.H_res_raw
                
                # Apply Sinkhorn-Knopp projection
                H_res_projected = SinkhornKnoppProjection(
                    iterations=self.sk_iterations,
                    epsilon=self.sk_epsilon,
                    temperature=self.sk_temperature
                )(H_res_raw)
                
                # Compute deviation from doubly stochastic
                row_sum = H_res_projected.sum(dim=1)
                col_sum = H_res_projected.sum(dim=0)
                
                row_dev = torch.abs(row_sum - 1.0).mean()
                col_dev = torch.abs(col_sum - 1.0).mean()
                
                # Eigenvalue regularization (encourage eigenvalues <= 1)
                eigenvalues = torch.linalg.eigvalsh(H_res_projected)
                eigen_reg = F.relu(eigenvalues - 1.0).mean()
                
                # Total regularization for this layer
                layer_reg = row_dev + col_dev + 0.1 * eigen_reg
                reg_loss += layer_reg
                num_mhc_layers += 1
        
        # Average across layers
        if num_mhc_layers > 0:
            reg_loss = reg_loss / num_mhc_layers
        
        return reg_loss
    
    def _apply_manifold_gradient_clipping(self) -> float:
        """
        Apply gradient clipping with awareness of manifold constraints.
        
        Returns:
            Total gradient norm after clipping
        """
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        mhc_max_norm = self.config.get('mhc_max_norm', 0.5)
        
        # Separate parameters for different clipping
        mhc_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if 'mhc' in name.lower() or 'H_' in name:
                    mhc_params.append(param)
                else:
                    other_params.append(param)
        
        total_norm = 0.0
        
        # Clip MHC parameters tighter
        if mhc_params:
            mhc_norm = torch.nn.utils.clip_grad_norm_(
                mhc_params,
                max_norm=mhc_max_norm,
                norm_type=2
            )
            total_norm += mhc_norm.item() ** 2
        
        # Clip other parameters
        if other_params:
            other_norm = torch.nn.utils.clip_grad_norm_(
                other_params,
                max_norm=max_grad_norm,
                norm_type=2
            )
            total_norm += other_norm.item() ** 2
        
        return total_norm ** 0.5 if total_norm > 0 else 0.0
    
    def _check_sinkhorn_convergence(self) -> float:
        """
        Check Sinkhorn-Knopp projection convergence.
        
        Returns:
            Average convergence error across MHC layers
        """
        total_error = 0.0
        num_layers = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'sinkhorn'):
                # Check if projection is converging
                if hasattr(module.sinkhorn, 'last_error'):
                    total_error += module.sinkhorn.last_error
                    num_layers += 1
        
        return total_error / max(num_layers, 1)
    
    def _apply_stability_corrections(self):
        """Apply corrections when instability is detected."""
        logger.info("Applying stability corrections...")
        
        # 1. Reduce learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # 2. Reset Sinkhorn iterations for better convergence
        self.sk_iterations = min(self.sk_iterations + 5, 50)
        
        # 3. Increase gradient clipping thresholds
        self.stability_monitor.max_grad_norm *= 0.9
        
        logger.info(f"Reduced LR to {new_lr:.2e}, SK iter to {self.sk_iterations}")
    
    def validate(
        self,
        val_loader,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Validation phase with comprehensive metrics.
        
        Args:
            val_loader: Validation data loader
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        default_metrics = ['mAP', 'precision', 'recall', 'F1']
        metrics = metrics or default_metrics
        
        val_results = {metric: 0.0 for metric in metrics}
        val_results['loss'] = 0.0
        
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc='Validation'):
                # Unpack batch
                if isinstance(batch_data, (list, tuple)):
                    images, targets = batch_data
                else:
                    images = batch_data['images']
                    targets = batch_data['targets']
                
                images = images.to(self.device)
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                else:
                    targets = targets.to(self.device)
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images, task='detection')
                    loss_dict = self.detection_loss(outputs['detections'], targets)
                
                # Accumulate loss
                val_results['loss'] += loss_dict['total'].item()
                
                # TODO: Compute additional metrics (mAP, precision, recall)
                # This would require post-processing detections
                
                num_batches += 1
        
        # Average metrics
        for key in val_results:
            val_results[key] /= max(num_batches, 1)
        
        return val_results
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        early_stopping_patience: int = 20,
        save_checkpoints: bool = True,
        use_wandb: bool = False,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_checkpoints: Whether to save checkpoints
            use_wandb: Whether to log to Weights & Biases
            resume_from_checkpoint: Optional checkpoint to resume from
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Initialize wandb if enabled
        if use_wandb:
            wandb.init(
                project="mhc-vision-system",
                name=self.experiment_name,
                config=self.config
            )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Early stopping
        early_stop_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch + 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['learning_rates'].append(train_metrics['learning_rate'])
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Learning Rate: {train_metrics['learning_rate']:.2e}")
            logger.info(f"Gradient Norm: {train_metrics['grad_norm']:.4f}")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'epoch': self.current_epoch,
                    'train/total_loss': train_metrics['total_loss'],
                    'train/detection_loss': train_metrics['detection_loss'],
                    'train/grad_norm': train_metrics['grad_norm'],
                    'val/loss': val_metrics['loss'],
                    'learning_rate': train_metrics['learning_rate'],
                    'sk_convergence': train_metrics['sk_error']
                })
            
            # Save checkpoint if best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                early_stop_counter = 0
                
                if save_checkpoints:
                    self.save_checkpoint(
                        filename=f"best_epoch_{self.current_epoch:03d}.pt",
                        is_best=True
                    )
                    logger.info(f"New best model saved with loss: {best_val_loss:.4f}")
            else:
                early_stop_counter += 1
                logger.info(f"No improvement for {early_stop_counter} epochs")
            
            # Save regular checkpoint
            if save_checkpoints and (self.current_epoch % 5 == 0):
                self.save_checkpoint(
                    filename=f"epoch_{self.current_epoch:03d}.pt",
                    is_best=False
                )
            
            # Early stopping check
            if early_stop_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.current_epoch} epochs")
                break
        
        # Finalize
        logger.info("\nTraining completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        if use_wandb:
            wandb.finish()
        
        # Save final model and training history
        self.save_checkpoint(filename="final_model.pt", is_best=False)
        self.save_training_history()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_metric,
            'experiment_name': self.experiment_name,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            # Also save as best model
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']
        self.best_metric = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def save_training_history(self):
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir / "training_history.json"
        
        # Convert tensors to lists for JSON serialization
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                # Convert tensor items to floats
                serializable_history[key] = [
                    float(v.item()) if torch.is_tensor(v) else v 
                    for v in value
                ]
            else:
                serializable_history[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def export_model(self, format: str = "torchscript"):
        """
        Export model for deployment.
        
        Args:
            format: Export format ('torchscript', 'onnx', 'tensorrt')
        """
        self.model.eval()
        
        export_dir = Path("exported_models")
        export_dir.mkdir(exist_ok=True)
        
        if format == "torchscript":
            # Example input for tracing
            example_input = torch.randn(1, 3, 416, 416).to(self.device)
            
            # Trace model
            traced_model = torch.jit.trace(self.model, example_input)
            
            # Save traced model
            export_path = export_dir / "vision_model.pt"
            traced_model.save(export_path)
            
            logger.info(f"Model exported as TorchScript to {export_path}")
        
        elif format == "onnx":
            # TODO: Implement ONNX export
            pass
        
        elif format == "tensorrt":
            # TODO: Implement TensorRT export
            pass
        
        else:
            raise ValueError(f"Unsupported export format: {format}")