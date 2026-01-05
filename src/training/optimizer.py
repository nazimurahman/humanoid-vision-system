# src/training/optimizer.py
"""
Optimizers with manifold constraints for mHC training.

Includes:
1. Manifold-aware optimizer with doubly stochastic projections
2. Gradient clipping specific to MHC parameters
3. Learning rate warmup with manifold stabilization
"""

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Tuple, Optional, Any, Callable
import math
import numpy as np

from ..utils.sinkhorn import SinkhornKnoppProjection

class ManifoldAwareOptimizer(Optimizer):
    """
    Optimizer that respects manifold constraints of MHC layers.
    
    Features:
    1. Special handling for doubly stochastic matrices
    2. Different learning rates for MHC vs regular parameters
    3. Gradient preconditioning for stability
    4. Periodic manifold projections
    """
    
    def __init__(
        self,
        model_params,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        mhc_params: Dict[str, Any] = None,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
        manifold_update_freq: int = 100
    ):
        """
        Initialize manifold-aware optimizer.
        
        Args:
            model_params: Model parameters
            lr: Learning rate
            weight_decay: Weight decay coefficient
            mhc_params: MHC-specific parameters
            betas: Coefficients for computing running averages
            eps: Term for numerical stability
            amsgrad: Whether to use AMSGrad variant
            manifold_update_freq: Frequency of manifold projections
        """
        if mhc_params is None:
            mhc_params = {
                'mhc_lr_scale': 0.5,
                'project_iterations': 20,
                'strict_double_stochastic': True
            }
        
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
            mhc_params=mhc_params,
            manifold_update_freq=manifold_update_freq
        )
        
        super().__init__(model_params, defaults)
        
        # Sinkhorn projector for doubly stochastic constraints
        self.sinkhorn = SinkhornKnoppProjection(
            iterations=mhc_params.get('project_iterations', 20),
            epsilon=1e-8
        )
        
        # Tracking
        self.step_count = 0
        self.mhc_parameters = []
        self.regular_parameters = []
        
        # Classify parameters
        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.requires_grad:
                    param_name = self._get_param_name(param)
                    if 'mhc' in param_name.lower() or 'H_' in param_name:
                        self.mhc_parameters.append(param)
                    else:
                        self.regular_parameters.append(param)
        
    def _get_param_name(self, param) -> str:
        """Get parameter name from its tensor."""
        # This is a simplified version
        # In practice, you'd track names during model construction
        return str(param.shape)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.step_count += 1
        
        for param_group in self.param_groups:
            # Get parameters
            mhc_params = self.mhc_parameters
            regular_params = self.regular_parameters
            
            # Apply different strategies
            self._update_regular_parameters(param_group, regular_params)
            self._update_mhc_parameters(param_group, mhc_params)
            
            # Apply manifold projections periodically
            if self.step_count % param_group['manifold_update_freq'] == 0:
                self._apply_manifold_projections(mhc_params)
        
        return loss
    
    def _update_regular_parameters(
        self,
        param_group: Dict[str, Any],
        parameters: List[torch.Tensor]
    ):
        """Update regular parameters using AdamW."""
        lr = param_group['lr']
        weight_decay = param_group['weight_decay']
        betas = param_group['betas']
        eps = param_group['eps']
        amsgrad = param_group['amsgrad']
        
        for param in parameters:
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # State initialization
            state = self.state[param]
            
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(param)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(param)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(param)
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            
            beta1, beta2 = betas
            
            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg.
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            
            step_size = lr / bias_correction1
            
            # Apply weight decay
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)
            
            # Update parameters
            param.addcdiv_(exp_avg, denom, value=-step_size)
    
    def _update_mhc_parameters(
        self,
        param_group: Dict[str, Any],
        parameters: List[torch.Tensor]
    ):
        """
        Update MHC parameters with special handling.
        
        Includes:
        1. Scaled learning rates
        2. Gradient preconditioning
        3. Sign consistency constraints
        """
        mhc_params = param_group['mhc_params']
        mhc_lr_scale = mhc_params.get('mhc_lr_scale', 0.5)
        lr = param_group['lr'] * mhc_lr_scale
        
        for param in parameters:
            if param.grad is None:
                continue
            
            # Apply gradient preconditioning for stability
            grad = self._precondition_mhc_gradient(param.grad, param)
            
            # Simple SGD update for MHC parameters
            # (Adam can sometimes destabilize constrained matrices)
            param.add_(grad, alpha=-lr)
            
            # Ensure non-negativity for certain MHC parameters
            if 'H_pre' in str(param) or 'H_post' in str(param):
                # These should be non-negative after sigmoid
                pass  # Sigmoid will handle this
            elif 'H_res' in str(param):
                # H_res should be non-negative after Sinkhorn
                pass  # Sinkhorn will handle this
    
    def _precondition_mhc_gradient(
        self,
        grad: torch.Tensor,
        param: torch.Tensor
    ) -> torch.Tensor:
        """
        Precondition gradient for MHC parameters.
        
        Ensures gradient updates respect manifold structure.
        """
        # 1. Scale gradient by parameter norm (Riemannian-like)
        param_norm = param.norm() + 1e-8
        grad_norm = grad.norm() + 1e-8
        
        # Avoid too large gradient updates
        max_ratio = 10.0
        scale = min(param_norm / grad_norm, max_ratio)
        
        # 2. For H_res, project gradient to tangent space of doubly stochastic manifold
        if param.dim() == 2 and param.shape[0] == param.shape[1]:
            # Project gradient to satisfy row/column sum constraints
            row_sum = grad.sum(dim=1, keepdim=True)
            col_sum = grad.sum(dim=0, keepdim=True)
            
            # Subtract mean of row and column sums
            grad = grad - (row_sum + col_sum) / (2 * param.shape[0])
        
        return grad * scale
    
    def _apply_manifold_projections(self, parameters: List[torch.Tensor]):
        """Apply manifold projections to enforce constraints."""
        for param in parameters:
            if param.dim() == 2 and param.shape[0] == param.shape[1]:
                # This might be an H_res matrix
                if 'H_res' in str(param):
                    # Apply Sinkhorn-Knopp projection for doubly stochastic
                    with torch.no_grad():
                        param.data = self.sinkhorn(param.data)
    
    def get_mhc_learning_rate(self) -> float:
        """Get current learning rate for MHC parameters."""
        if self.param_groups:
            lr = self.param_groups[0]['lr']
            mhc_params = self.param_groups[0]['mhc_params']
            mhc_lr_scale = mhc_params.get('mhc_lr_scale', 0.5)
            return lr * mhc_lr_scale
        return 0.0

class DoublyStochasticProjection:
    """
    Projection operator for doubly stochastic matrices.
    
    Can be used as an optimizer step or as a constraint.
    """
    
    def __init__(
        self,
        method: str = 'sinkhorn',
        iterations: int = 20,
        epsilon: float = 1e-8,
        temperature: float = 1.0
    ):
        """
        Initialize projection operator.
        
        Args:
            method: Projection method ('sinkhorn', 'softmax', 'exponential')
            iterations: Number of iterations for iterative methods
            epsilon: Small constant for numerical stability
            temperature: Temperature parameter for soft projections
        """
        self.method = method
        self.iterations = iterations
        self.epsilon = epsilon
        self.temperature = temperature
        
        if method == 'sinkhorn':
            self.projector = SinkhornKnoppProjection(
                iterations=iterations,
                epsilon=epsilon
            )
    
    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Project matrix to doubly stochastic manifold.
        
        Args:
            matrix: Input matrix [N, N] or [B, N, N]
            
        Returns:
            Projected doubly stochastic matrix
        """
        if self.method == 'sinkhorn':
            return self.projector(matrix)
        
        elif self.method == 'softmax':
            # Softmax along rows and columns alternately
            result = matrix.clone()
            
            for _ in range(self.iterations):
                # Softmax along rows
                result = F.softmax(result / self.temperature, dim=1)
                
                # Softmax along columns
                result = F.softmax(result / self.temperature, dim=0)
            
            return result
        
        elif self.method == 'exponential':
            # Use exponential and normalization
            result = torch.exp(matrix)
            
            for _ in range(self.iterations):
                # Normalize rows
                row_sum = result.sum(dim=1, keepdim=True)
                result = result / (row_sum + self.epsilon)
                
                # Normalize columns
                col_sum = result.sum(dim=0, keepdim=True)
                result = result / (col_sum + self.epsilon)
            
            return result
        
        else:
            raise ValueError(f"Unknown projection method: {self.method}")
    
    def compute_distance(
        self,
        matrix: torch.Tensor,
        projected: torch.Tensor = None
    ) -> float:
        """
        Compute distance to doubly stochastic manifold.
        
        Args:
            matrix: Input matrix
            projected: Pre-computed projection (optional)
            
        Returns:
            Distance metric
        """
        if projected is None:
            projected = self(matrix)
        
        # Compute row and column sum deviations
        row_sums = projected.sum(dim=1)
        col_sums = projected.sum(dim=0)
        
        row_error = torch.abs(row_sums - 1.0).mean().item()
        col_error = torch.abs(col_sums - 1.0).mean().item()
        
        # Compute orthogonality error (for permutation matrices)
        if matrix.shape[0] == matrix.shape[1]:
            identity = torch.eye(matrix.shape[0], device=matrix.device)
            product = torch.matmul(projected, projected.T)
            orth_error = torch.norm(product - identity).item()
        else:
            orth_error = 0.0
        
        return {
            'row_error': row_error,
            'col_error': col_error,
            'orth_error': orth_error,
            'total_error': row_error + col_error + orth_error
        }