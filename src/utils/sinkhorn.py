# src/utils/sinkhorn.py
"""
Sinkhorn-Knopp Algorithm for Doubly Stochastic Matrix Projection.

This module implements the Sinkhorn-Knopp algorithm for projecting matrices
onto the Birkhoff polytope (doubly stochastic matrices).

Theory:
A doubly stochastic matrix M satisfies:
1. M_ij ≥ 0 for all i,j
2. Σ_j M_ij = 1 for all i (rows sum to 1)
3. Σ_i M_ij = 1 for all j (columns sum to 1)

The Sinkhorn-Knopp algorithm iteratively normalizes rows and columns
until convergence, guaranteeing doubly stochastic constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SinkhornKnopp:
    """
    Sinkhorn-Knopp algorithm for doubly stochastic matrix projection.
    
    Implements both the standard algorithm and a differentiable version
    for use in neural networks.
    
    References:
    - Sinkhorn, R. (1964). A relationship between arbitrary positive matrices 
      and doubly stochastic matrices.
    - Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport.
    """
    
    def __init__(
        self,
        num_iterations: int = 20,
        epsilon: float = 1e-8,
        tau: float = 1.0,
        noise_factor: float = 0.0,
        device: str = 'cuda'
    ):
        """
        Initialize Sinkhorn-Knopp projector.
        
        Args:
            num_iterations: Number of Sinkhorn iterations (≥ 20 for stability)
            epsilon: Small constant to avoid division by zero
            tau: Temperature parameter for Sinkhorn (1.0 for exact, >1.0 for smoother)
            noise_factor: Add noise for stochastic stability (0.0 for deterministic)
            device: Computation device
        """
        self.num_iterations = max(20, num_iterations)  # Enforce ≥ 20 iterations
        self.epsilon = epsilon
        self.tau = tau
        self.noise_factor = noise_factor
        self.device = device
        
        # Tracking for diagnostics
        self.convergence_history = []
        self.row_error_history = []
        self.col_error_history = []
        
    def project(
        self,
        matrix: torch.Tensor,
        target_rows: Optional[torch.Tensor] = None,
        target_cols: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Project matrix onto doubly stochastic manifold.
        
        Args:
            matrix: Input matrix of shape [..., n, m]
            target_rows: Target row sums (default: ones)
            target_cols: Target column sums (default: ones)
            
        Returns:
            Doubly stochastic matrix of same shape
        """
        original_shape = matrix.shape
        needs_reshape = len(original_shape) > 2
        
        if needs_reshape:
            # Flatten batch dimensions
            batch_dims = original_shape[:-2]
            n, m = original_shape[-2:]
            matrix = matrix.reshape(-1, n, m)
            batch_size = matrix.shape[0]
        else:
            batch_size = 1
            matrix = matrix.unsqueeze(0)
            n, m = matrix.shape[-2:]
        
        # Initialize targets if not provided
        if target_rows is None:
            target_rows = torch.ones(batch_size, n, device=self.device)
        if target_cols is None:
            target_cols = torch.ones(batch_size, m, device=self.device)
        
        # Ensure positivity while preserving gradients
        # Use softplus for differentiable positivity
        K = F.softplus(matrix) + self.epsilon
        
        # Add noise for stochastic stability (optional)
        if self.noise_factor > 0 and self.training:
            noise = torch.randn_like(K) * self.noise_factor
            K = K * torch.exp(noise)
        
        # Apply temperature
        if self.tau != 1.0:
            K = K ** (1.0 / self.tau)
        
        # Initialize scaling vectors
        u = torch.ones(batch_size, n, device=self.device) / n
        v = torch.ones(batch_size, m, device=self.device) / m
        
        # Sinkhorn iterations
        for iteration in range(self.num_iterations):
            # Store previous for convergence check
            u_prev = u.clone()
            v_prev = v.clone()
            
            # Scale rows
            K_u = K * u.unsqueeze(-1)  # [B, n, m]
            col_sums = K_u.sum(dim=1)  # [B, m]
            v = target_cols / (col_sums + self.epsilon)
            
            # Scale columns
            K_v = K * v.unsqueeze(-2)  # [B, n, m]
            row_sums = K_v.sum(dim=2)  # [B, n]
            u = target_rows / (row_sums + self.epsilon)
            
            # Check convergence (every 5 iterations)
            if iteration % 5 == 0:
                row_error = (row_sums - target_rows).abs().max().item()
                col_error = (col_sums - target_cols).abs().max().item()
                
                self.row_error_history.append(row_error)
                self.col_error_history.append(col_error)
                
                if row_error < 1e-6 and col_error < 1e-6:
                    logger.debug(f"Sinkhorn converged at iteration {iteration}")
                    break
        
        # Final scaling
        P = K * u.unsqueeze(-1) * v.unsqueeze(-2)
        
        # Normalize to ensure exact doubly stochastic
        P = self._exact_normalize(P)
        
        # Verify constraints
        if self._check_constraints:
            self._verify_doubly_stochastic(P, target_rows, target_cols)
        
        # Restore original shape
        if needs_reshape:
            P = P.reshape(original_shape)
        else:
            P = P.squeeze(0)
        
        return P
    
    def _exact_normalize(self, P: torch.Tensor) -> torch.Tensor:
        """
        Apply exact normalization to ensure doubly stochastic constraints.
        
        Args:
            P: Approximately doubly stochastic matrix
            
        Returns:
            Exactly doubly stochastic matrix
        """
        # Iterative refinement
        for _ in range(3):  # 3 refinement iterations
            # Normalize rows
            row_sums = P.sum(dim=-1, keepdim=True)
            P = P / (row_sums + self.epsilon)
            
            # Normalize columns
            col_sums = P.sum(dim=-2, keepdim=True)
            P = P / (col_sums + self.epsilon)
        
        return P
    
    def _verify_doubly_stochastic(
        self,
        P: torch.Tensor,
        target_rows: torch.Tensor,
        target_cols: torch.Tensor
    ):
        """
        Verify doubly stochastic constraints.
        
        Args:
            P: Projected matrix
            target_rows: Target row sums
            target_cols: Target column sums
        """
        with torch.no_grad():
            # Check non-negativity
            min_val = P.min().item()
            if min_val < -1e-6:
                logger.warning(f"Sinkhorn: Negative values detected: {min_val:.2e}")
            
            # Check row sums
            row_sums = P.sum(dim=-1)
            row_error = (row_sums - target_rows).abs().max().item()
            
            # Check column sums
            col_sums = P.sum(dim=-2)
            col_error = (col_sums - target_cols).abs().max().item()
            
            # Log errors
            if row_error > 1e-4 or col_error > 1e-4:
                logger.warning(
                    f"Sinkhorn constraints violated: "
                    f"row_error={row_error:.2e}, col_error={col_error:.2e}"
                )
            
            # Update convergence history
            self.convergence_history.append((row_error, col_error))
    
    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information about Sinkhorn performance.
        
        Returns:
            Dictionary with convergence metrics
        """
        if not self.convergence_history:
            return {}
        
        # Compute statistics
        row_errors = [e[0] for e in self.convergence_history[-100:]]
        col_errors = [e[1] for e in self.convergence_history[-100:]]
        
        return {
            'sinkhorn_row_error_mean': np.mean(row_errors),
            'sinkhorn_row_error_std': np.std(row_errors),
            'sinkhorn_col_error_mean': np.mean(col_errors),
            'sinkhorn_col_error_std': np.std(col_errors),
            'sinkhorn_iterations': len(self.convergence_history),
        }
    
    def reset_diagnostics(self):
        """Reset diagnostic tracking."""
        self.convergence_history.clear()
        self.row_error_history.clear()
        self.col_error_history.clear()


class DifferentiableSinkhorn(nn.Module):
    """
    Differentiable Sinkhorn-Knopp layer for neural networks.
    
    This module can be integrated into neural networks and supports
    backpropagation through the Sinkhorn iterations.
    """
    
    def __init__(
        self,
        num_iterations: int = 20,
        epsilon: float = 1e-8,
        tau: float = 1.0,
        stochastic: bool = False
    ):
        """
        Initialize differentiable Sinkhorn layer.
        
        Args:
            num_iterations: Number of Sinkhorn iterations
            epsilon: Small constant for numerical stability
            tau: Temperature parameter
            stochastic: Whether to use stochastic Sinkhorn
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.tau = tau
        self.stochastic = stochastic
        
        # Learnable parameters for adaptive scaling
        self.learnable_scale = nn.Parameter(torch.ones(1))
        self.learnable_bias = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        log_alpha: torch.Tensor,
        target_rows: Optional[torch.Tensor] = None,
        target_cols: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with automatic differentiation.
        
        Args:
            log_alpha: Log-space input matrix [..., n, m]
            target_rows: Target row sums [..., n]
            target_cols: Target column sums [..., m]
            
        Returns:
            Doubly stochastic matrix [..., n, m]
        """
        original_shape = log_alpha.shape
        
        # Add learnable scaling
        log_alpha = log_alpha * self.learnable_scale + self.learnable_bias
        
        # Apply Sinkhorn in log-space for numerical stability
        P = self._sinkhorn_log_domain(
            log_alpha, 
            target_rows, 
            target_cols
        )
        
        return P
    
    def _sinkhorn_log_domain(
        self,
        log_alpha: torch.Tensor,
        target_rows: Optional[torch.Tensor],
        target_cols: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Sinkhorn in log domain for numerical stability.
        
        Implements the log-sum-exp trick for stable computation.
        """
        batch_dims = log_alpha.shape[:-2]
        n, m = log_alpha.shape[-2:]
        
        # Initialize in log domain
        if target_rows is None:
            log_target_rows = torch.zeros(*batch_dims, n, device=log_alpha.device)
        else:
            log_target_rows = torch.log(target_rows + self.epsilon)
        
        if target_cols is None:
            log_target_cols = torch.zeros(*batch_dims, m, device=log_alpha.device)
        else:
            log_target_cols = torch.log(target_cols + self.epsilon)
        
        # Initialize scaling vectors in log domain
        log_u = torch.zeros(*batch_dims, n, device=log_alpha.device)
        log_v = torch.zeros(*batch_dims, m, device=log_alpha.device)
        
        # Sinkhorn iterations in log domain
        for _ in range(self.num_iterations):
            # Update log_u (rows)
            # log_u = log_target_rows - logsumexp(log_alpha + log_v.unsqueeze(-2), dim=-1)
            log_alpha_plus_v = log_alpha + log_v.unsqueeze(-2)  # [..., n, m]
            log_row_sum = torch.logsumexp(log_alpha_plus_v, dim=-1)  # [..., n]
            log_u = log_target_rows - log_row_sum
            
            # Update log_v (columns)
            # log_v = log_target_cols - logsumexp(log_alpha + log_u.unsqueeze(-1), dim=-2)
            log_alpha_plus_u = log_alpha + log_u.unsqueeze(-1)  # [..., n, m]
            log_col_sum = torch.logsumexp(log_alpha_plus_u, dim=-2)  # [..., m]
            log_v = log_target_cols - log_col_sum
        
        # Compute final matrix in log domain
        log_P = log_alpha + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        
        # Exponentiate to get probabilities
        P = torch.exp(log_P)
        
        return P
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"num_iterations={self.num_iterations}, "
            f"epsilon={self.epsilon}, "
            f"tau={self.tau}, "
            f"stochastic={self.stochastic}"
        )


def sinkhorn_regularization_loss(
    matrix: torch.Tensor,
    target_rows: Optional[torch.Tensor] = None,
    target_cols: Optional[torch.Tensor] = None,
    weight: float = 1.0
) -> torch.Tensor:
    """
    Compute regularization loss to encourage doubly stochastic properties.
    
    This loss can be added to training to help matrices satisfy
    doubly stochastic constraints.
    
    Args:
        matrix: Input matrix
        target_rows: Target row sums
        target_cols: Target column sums
        weight: Loss weight
        
    Returns:
        Regularization loss
    """
    batch_dims = matrix.shape[:-2]
    n, m = matrix.shape[-2:]
    
    if target_rows is None:
        target_rows = torch.ones(*batch_dims, n, device=matrix.device)
    if target_cols is None:
        target_cols = torch.ones(*batch_dims, m, device=matrix.device)
    
    # Row sum loss
    row_sums = matrix.sum(dim=-1)
    row_loss = F.mse_loss(row_sums, target_rows)
    
    # Column sum loss
    col_sums = matrix.sum(dim=-2)
    col_loss = F.mse_loss(col_sums, target_cols)
    
    # Non-negativity loss
    neg_loss = torch.relu(-matrix).mean()
    
    total_loss = weight * (row_loss + col_loss + 0.1 * neg_loss)
    
    return total_loss


# Convenience function for quick projection
def project_to_doubly_stochastic(
    matrix: torch.Tensor,
    iterations: int = 20,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Convenience function for Sinkhorn-Knopp projection.
    
    Args:
        matrix: Input matrix
        iterations: Number of Sinkhorn iterations
        epsilon: Numerical stability constant
        
    Returns:
        Doubly stochastic matrix
    """
    projector = SinkhornKnopp(
        num_iterations=iterations,
        epsilon=epsilon,
        device=matrix.device
    )
    return projector.project(matrix)